//! CPU verification for speculative decoding.
//!
//! Draft tokens are accepted with probability `min(1, p(token) / q(token))`.
//! The first rejected token is replaced with a sample from normalized `(p - q)+`.
//! This preserves the target sampling distribution; it does not guarantee the
//! same random stream as ordinary one-token decoding.

use anyhow::{Result, bail};

use crate::chat::Sampler;

#[derive(Clone, Copy, Debug)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
}

impl SamplingConfig {
    fn validate(self) -> Result<()> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            bail!("temperature must be finite and nonnegative");
        }
        if !(self.top_p.is_finite() && 0.0 < self.top_p && self.top_p <= 1.0) {
            bail!("top_p must be in (0, 1]");
        }
        Ok(())
    }
}

/// A finite, normalized categorical distribution over consecutive token IDs.
///
/// Keeping unnormalized weights and their sum avoids rounding twice and lets
/// `from_logits` retain the sampling order and weights of [`Sampler::sample`].
/// Fields are private so every instance has nonempty, finite positive mass.
#[derive(Clone, Debug)]
pub struct Distribution {
    weights: Vec<f64>,
    order: Vec<u32>,
    total: f64,
}

impl Distribution {
    pub fn from_logits(logits: &[f32], config: SamplingConfig) -> Result<Self> {
        config.validate()?;
        Self::validate_vocabulary(logits.len())?;
        if logits.iter().any(|value| !value.is_finite()) {
            bail!("logits contain non-finite values");
        }
        let mut weights = vec![0.0; logits.len()];
        if config.temperature == 0.0 {
            let best = logits
                .iter()
                .enumerate()
                .max_by(|(a, av), (b, bv)| av.total_cmp(bv).then_with(|| b.cmp(a)))
                .expect("validated nonempty logits")
                .0;
            weights[best] = 1.0;
            return Ok(Self {
                weights,
                order: vec![best as u32],
                total: 1.0,
            });
        }

        let mut order: Vec<usize> = (0..logits.len()).collect();
        let compare = |&a: &usize, &b: &usize| logits[b].total_cmp(&logits[a]).then(a.cmp(&b));
        let count = if config.top_k == 0 {
            order.len()
        } else {
            config.top_k.min(order.len())
        };
        if count < order.len() {
            order.select_nth_unstable_by(count - 1, compare);
        }
        order.truncate(count);
        order.sort_unstable_by(compare);
        let maximum = logits[order[0]] as f64;
        let ordered_weights: Vec<f64> = order
            .iter()
            .map(|&token| (((logits[token] as f64) - maximum) / config.temperature as f64).exp())
            .collect();
        let sum: f64 = ordered_weights.iter().sum();
        if !sum.is_finite() || sum <= 0.0 {
            bail!("sampling distribution is invalid");
        }
        let mut nucleus = order.len();
        let mut cumulative = 0.0;
        for (i, &weight) in ordered_weights.iter().enumerate() {
            cumulative += weight / sum;
            if cumulative >= config.top_p as f64 {
                nucleus = i + 1;
                break;
            }
        }
        let total: f64 = ordered_weights[..nucleus].iter().sum();
        let mut support = Vec::with_capacity(nucleus);
        for (&token, &weight) in order[..nucleus].iter().zip(&ordered_weights[..nucleus]) {
            weights[token] = weight;
            if weight > 0.0 {
                support.push(token as u32);
            }
        }
        Ok(Self {
            weights,
            order: support,
            total,
        })
    }

    /// Normalize finite nonnegative probability masses in token-ID order.
    /// The input need not sum to one, but must contain a positive mass.
    pub fn from_probabilities(probabilities: &[f64]) -> Result<Self> {
        Self::validate_vocabulary(probabilities.len())?;
        if probabilities
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            bail!("probability masses must be finite and nonnegative");
        }
        let maximum = probabilities.iter().copied().fold(0.0_f64, f64::max);
        if maximum == 0.0 {
            bail!("sampling distribution has zero probability mass");
        }
        // Scaling before summation handles both very small rejection residuals
        // and finite input masses whose unscaled sum would overflow.
        let weights: Vec<f64> = probabilities.iter().map(|value| value / maximum).collect();
        let total: f64 = weights.iter().sum();
        if !total.is_finite() || total <= 0.0 {
            bail!("sampling distribution is invalid");
        }
        let order = weights
            .iter()
            .enumerate()
            .filter_map(|(token, &weight)| (weight > 0.0).then_some(token as u32))
            .collect();
        Ok(Self {
            weights,
            order,
            total,
        })
    }

    fn validate_vocabulary(len: usize) -> Result<()> {
        if len == 0 {
            bail!("cannot sample an empty vocabulary");
        }
        if len > u32::MAX as usize {
            bail!("vocabulary exceeds u32 token IDs");
        }
        Ok(())
    }

    pub fn sample(&self, rng: &mut Sampler) -> Result<u32> {
        if self.order.len() == 1 {
            return Ok(self.order[0]);
        }
        let threshold = rng.uniform() * self.total;
        let mut cumulative = 0.0;
        for &token in &self.order {
            cumulative += self.weights[token as usize];
            if threshold < cumulative {
                return Ok(token);
            }
        }
        // Match the ordinary sampler's final rounding guard. The total and
        // cumulative sum use the same positive weights in the same order.
        Ok(*self.order.last().expect("validated nonempty distribution"))
    }

    /// Probability zero is returned for token IDs outside the vocabulary.
    pub fn probability(&self, token: u32) -> f64 {
        self.weights.get(token as usize).copied().unwrap_or(0.0) / self.total
    }

    fn residual(&self, draft: &Self) -> Result<Self> {
        if self.weights.len() != draft.weights.len() {
            bail!("draft and target vocabularies differ");
        }
        let masses: Vec<f64> = (0..self.weights.len())
            .map(|token| {
                (self.probability(token as u32) - draft.probability(token as u32)).max(0.0)
            })
            .collect();
        // A zero residual is an error. Replacing it with p would silently
        // change the rejection law instead of exposing an invalid state.
        Self::from_probabilities(&masses)
    }
}

#[derive(Clone, Debug)]
pub struct DraftProposal {
    pub token: u32,
    /// The actual distribution used to draw `token`, after all truncation.
    pub distribution: Distribution,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Verification {
    pub accepted_drafts: usize,
    /// Target input columns to retain: the pending anchor plus accepted drafts.
    /// The emitted correction or bonus has not yet been consumed by the model.
    pub consumed_inputs: usize,
    pub tokens: Vec<u32>,
}

/// Verify a block with one target-logit row per proposal plus a bonus row.
///
/// Row zero predicts the first draft from the pending anchor. After a rejection,
/// suffix rows and proposals are ignored and no further random numbers are used.
/// EOS, output limits and state rollback belong to the caller.
pub fn verify(
    drafts: &[DraftProposal],
    target_logits: &[Vec<f32>],
    config: SamplingConfig,
    rng: &mut Sampler,
) -> Result<Verification> {
    config.validate()?;
    if drafts.len().checked_add(1) != Some(target_logits.len()) {
        bail!("target verification needs one logit row per draft plus one bonus row");
    }
    let mut tokens = Vec::with_capacity(target_logits.len());
    for (index, proposal) in drafts.iter().enumerate() {
        if target_logits[index].len() != target_logits[0].len() {
            bail!("target vocabulary differs at proposal {index}");
        }
        let target = Distribution::from_logits(&target_logits[index], config)?;
        if target.weights.len() != proposal.distribution.weights.len() {
            bail!("draft and target vocabularies differ at proposal {index}");
        }
        let q = proposal.distribution.probability(proposal.token);
        if q <= 0.0 {
            bail!("draft proposal {index} has zero probability under its draft distribution");
        }
        let p = target.probability(proposal.token);
        let accepted = p >= q || (p > 0.0 && rng.uniform() < p / q);
        if !accepted {
            tokens.push(target.residual(&proposal.distribution)?.sample(rng)?);
            return Ok(Verification {
                accepted_drafts: index,
                consumed_inputs: index + 1,
                tokens,
            });
        }
        tokens.push(proposal.token);
    }
    let bonus = Distribution::from_logits(&target_logits[drafts.len()], config)?;
    if let Some(first) = target_logits.first()
        && first.len() != bonus.weights.len()
    {
        bail!("target bonus vocabulary differs from the first logit row");
    }
    tokens.push(bonus.sample(rng)?);
    Ok(Verification {
        accepted_drafts: drafts.len(),
        consumed_inputs: target_logits.len(),
        tokens,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const GREEDY: SamplingConfig = SamplingConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
    };
    const RANDOM: SamplingConfig = SamplingConfig {
        temperature: 1.0,
        ..GREEDY
    };

    fn proposal(token: u32, probabilities: &[f64]) -> DraftProposal {
        DraftProposal {
            token,
            distribution: Distribution::from_probabilities(probabilities).unwrap(),
        }
    }

    #[test]
    fn greedy_rejection_at_each_position_commits_only_accepted_inputs() {
        for rejected in 0..3 {
            let drafts: Vec<_> = (0..3).map(|_| proposal(0, &[1.0, 0.0])).collect();
            let mut logits = vec![vec![2.0, 1.0]; 4];
            logits[rejected] = vec![1.0, 2.0];
            // A rejected suffix is neither validated nor sampled.
            for row in &mut logits[rejected + 1..] {
                *row = vec![f32::NAN];
            }
            let mut rng = Sampler::new(19);
            let result = verify(&drafts, &logits, GREEDY, &mut rng).unwrap();
            let mut expected = vec![0; rejected];
            expected.push(1);
            assert_eq!(result.tokens, expected);
            assert_eq!(result.accepted_drafts, rejected);
            assert_eq!(result.consumed_inputs, rejected + 1);
            assert_eq!(rng.uniform(), Sampler::new(19).uniform());
        }
    }

    #[test]
    fn greedy_full_acceptance_includes_bonus_and_matches_sampler_ties() {
        let drafts = vec![proposal(0, &[1.0, 0.0]), proposal(1, &[0.0, 1.0])];
        let logits = vec![vec![3.0, 3.0], vec![-0.0, 0.0], vec![4.0, 4.0]];
        let result = verify(&drafts, &logits, GREEDY, &mut Sampler::new(5)).unwrap();
        assert_eq!(result.tokens, vec![0, 1, 0]);
        assert_eq!(result.accepted_drafts, 2);
        assert_eq!(result.consumed_inputs, 3);
        for (row, &token) in logits.iter().zip(&result.tokens) {
            assert_eq!(Sampler::new(9).sample(row, 0.0, 1.0, 0).unwrap(), token);
        }
    }

    #[test]
    fn empty_draft_block_samples_anchor_logits() {
        let result = verify(&[], &[vec![0.0, 1.0]], GREEDY, &mut Sampler::new(0)).unwrap();
        assert_eq!(result.tokens, vec![1]);
        assert_eq!(result.accepted_drafts, 0);
        assert_eq!(result.consumed_inputs, 1);
    }

    #[test]
    fn distribution_matches_existing_sampler_top_k_then_top_p() {
        let logits = [0.5, 3.0, 3.0, -4.0, 1.5, -0.0, 0.0];
        for temperature in [0.01, 0.7, 1.0, 100.0] {
            for top_p in [0.01, 0.5, 0.95, 1.0] {
                for top_k in [0, 1, 2, 4, 100] {
                    let config = SamplingConfig {
                        temperature,
                        top_p,
                        top_k,
                    };
                    let distribution = Distribution::from_logits(&logits, config).unwrap();
                    let mass: f64 = (0..logits.len())
                        .map(|i| distribution.probability(i as u32))
                        .sum();
                    assert!((mass - 1.0).abs() < 1e-14);
                    for seed in 0..64 {
                        assert_eq!(
                            distribution.sample(&mut Sampler::new(seed)).unwrap(),
                            Sampler::new(seed)
                                .sample(&logits, temperature, top_p, top_k)
                                .unwrap(),
                            "temperature={temperature}, top_p={top_p}, top_k={top_k}, seed={seed}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn exact_residual_law_for_hand_chosen_distributions() {
        let p = Distribution::from_probabilities(&[0.2, 0.5, 0.3]).unwrap();
        let q = Distribution::from_probabilities(&[0.6, 0.1, 0.3]).unwrap();
        let residual = p.residual(&q).unwrap();
        assert_eq!(residual.probability(0), 0.0);
        assert_eq!(residual.probability(1), 1.0);
        assert_eq!(residual.probability(2), 0.0);
        let rejection_mass: f64 = (0..3)
            .map(|token| (q.probability(token) - p.probability(token)).max(0.0))
            .sum();
        for token in 0..3 {
            let output_mass = p.probability(token).min(q.probability(token))
                + rejection_mass * residual.probability(token);
            assert!((output_mass - p.probability(token)).abs() < 1e-14);
        }
    }

    #[test]
    fn stochastic_verification_recovers_target_distribution() {
        let logits: Vec<f32> = [0.2_f32, 0.5, 0.3].map(f32::ln).to_vec();
        let expected = Distribution::from_logits(&logits, RANDOM).unwrap();
        let draft = Distribution::from_probabilities(&[0.6, 0.1, 0.3]).unwrap();
        let target = vec![logits, vec![1.0, 0.0, -1.0]];
        let mut counts = [0_usize; 3];
        let mut rng = Sampler::new(0xabc);
        const TRIALS: usize = 80_000;
        for _ in 0..TRIALS {
            let token = draft.sample(&mut rng).unwrap();
            let proposal = DraftProposal {
                token,
                distribution: draft.clone(),
            };
            let result = verify(&[proposal], &target, RANDOM, &mut rng).unwrap();
            counts[result.tokens[0] as usize] += 1;
        }
        for (token, count) in counts.into_iter().enumerate() {
            assert!(
                (count as f64 / TRIALS as f64 - expected.probability(token as u32)).abs() < 0.006
            );
        }
    }

    #[test]
    fn acceptance_uses_probability_ratio_and_only_one_uniform_draw() {
        let target_logits = vec![0.25_f32.ln(), 0.75_f32.ln()];
        let target = Distribution::from_logits(&target_logits, RANDOM).unwrap();
        let draft = proposal(0, &[0.75, 0.25]);
        let ratio = target.probability(0) / draft.distribution.probability(0);
        for seed in 0..128 {
            let mut expected_rng = Sampler::new(seed);
            let accepted = expected_rng.uniform() < ratio;
            let mut actual_rng = Sampler::new(seed);
            let result = verify(
                std::slice::from_ref(&draft),
                &[target_logits.clone(), vec![0.0, -1000.0]],
                RANDOM,
                &mut actual_rng,
            )
            .unwrap();
            assert_eq!(result.accepted_drafts, usize::from(accepted));
            assert_eq!(result.consumed_inputs, 1 + usize::from(accepted));
            assert_eq!(result.tokens, if accepted { vec![0, 0] } else { vec![1] });
            assert_eq!(actual_rng.uniform(), expected_rng.uniform());
        }
    }

    #[test]
    fn one_hot_sampling_and_verification_do_not_consume_rng() {
        let one_hot = Distribution::from_logits(&[0.0, -1000.0], RANDOM).unwrap();
        let mut rng = Sampler::new(11);
        assert_eq!(one_hot.sample(&mut rng).unwrap(), 0);
        let result = verify(
            &[DraftProposal {
                token: 0,
                distribution: one_hot,
            }],
            &[vec![0.0, -1000.0], vec![-1000.0, 0.0]],
            RANDOM,
            &mut rng,
        )
        .unwrap();
        assert_eq!(result.tokens, vec![0, 1]);
        assert_eq!(rng.uniform(), Sampler::new(11).uniform());
    }

    #[test]
    fn rejects_invalid_inputs_without_masking_zero_residual() {
        for logits in [vec![], vec![f32::NAN], vec![f32::INFINITY]] {
            assert!(Distribution::from_logits(&logits, RANDOM).is_err());
        }
        for config in [
            SamplingConfig {
                temperature: -1.0,
                ..RANDOM
            },
            SamplingConfig {
                temperature: f32::NAN,
                ..RANDOM
            },
            SamplingConfig {
                temperature: f32::INFINITY,
                ..RANDOM
            },
            SamplingConfig {
                top_p: 0.0,
                ..RANDOM
            },
            SamplingConfig {
                top_p: 1.1,
                ..RANDOM
            },
            SamplingConfig {
                top_p: f32::NAN,
                ..RANDOM
            },
        ] {
            assert!(Distribution::from_logits(&[1.0], config).is_err());
            assert!(verify(&[], &[vec![1.0]], config, &mut Sampler::new(0)).is_err());
        }
        for probabilities in [
            vec![],
            vec![0.0],
            vec![-1.0, 2.0],
            vec![f64::NAN],
            vec![f64::INFINITY],
        ] {
            assert!(Distribution::from_probabilities(&probabilities).is_err());
        }
        let p = Distribution::from_probabilities(&[0.25, 0.75]).unwrap();
        assert!(p.residual(&p).is_err());
        assert_eq!(p.probability(2), 0.0);
        assert!(verify(&[], &[], RANDOM, &mut Sampler::new(0)).is_err());
        assert!(
            verify(
                &[proposal(0, &[0.0, 1.0])],
                &[vec![0.0, 1.0], vec![0.0, 1.0]],
                RANDOM,
                &mut Sampler::new(0)
            )
            .is_err()
        );
        assert!(
            verify(
                &[proposal(2, &[0.5, 0.5])],
                &[vec![0.0, 1.0], vec![0.0, 1.0]],
                RANDOM,
                &mut Sampler::new(0)
            )
            .is_err()
        );
        assert!(
            verify(
                &[proposal(0, &[1.0])],
                &[vec![0.0, 1.0], vec![0.0, 1.0]],
                RANDOM,
                &mut Sampler::new(0)
            )
            .is_err()
        );
        assert!(
            verify(
                &[proposal(0, &[1.0, 0.0]), proposal(0, &[1.0])],
                &[vec![1.0, 0.0], vec![1.0], vec![1.0, 0.0]],
                GREEDY,
                &mut Sampler::new(0),
            )
            .is_err()
        );
        assert!(
            verify(
                &[proposal(0, &[1.0, 0.0])],
                &[vec![1.0, 0.0], vec![1.0]],
                GREEDY,
                &mut Sampler::new(0),
            )
            .is_err()
        );
    }

    #[test]
    fn first_rejection_does_not_touch_later_proposals_or_rng() {
        let drafts = vec![proposal(0, &[1.0, 0.0]), proposal(99, &[1.0])];
        let result = verify(
            &drafts,
            &[vec![-1.0, 1.0], vec![f32::NAN], vec![]],
            GREEDY,
            &mut Sampler::new(12),
        )
        .unwrap();
        assert_eq!(result.tokens, vec![1]);
        assert_eq!(result.consumed_inputs, 1);
    }

    #[test]
    fn residual_normalization_handles_tiny_positive_mass() {
        let p = Distribution::from_probabilities(&[0.5, 0.5]).unwrap();
        let q = Distribution::from_probabilities(&[0.5 + 1e-14, 0.5 - 1e-14]).unwrap();
        let residual = p.residual(&q).unwrap();
        assert_eq!(residual.probability(0), 0.0);
        assert_eq!(residual.probability(1), 1.0);
    }

    #[test]
    fn probability_mass_constructor_scales_before_summing() {
        for scale in [f64::MIN_POSITIVE, f64::MAX] {
            let distribution = Distribution::from_probabilities(&[scale, scale]).unwrap();
            assert_eq!(distribution.probability(0), 0.5);
            assert_eq!(distribution.probability(1), 0.5);
        }
        let distribution = Distribution::from_logits(&[f32::MAX, -f32::MAX], RANDOM).unwrap();
        assert_eq!(distribution.probability(0), 1.0);
        assert_eq!(distribution.probability(1), 0.0);
    }
}
