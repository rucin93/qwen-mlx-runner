use std::{borrow::Cow, collections::BTreeMap};

use anyhow::{Context, Result, ensure};

/// OpenAI-style logit adjustments. Penalties count generated completion tokens
/// only, including reasoning/tool syntax; prompt and prior-message tokens are
/// deliberately excluded from each request's history.
#[derive(Clone, Debug, Default)]
pub struct SamplingOptions {
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub logit_bias: BTreeMap<u32, f32>,
}

impl SamplingOptions {
    pub fn validate(&self, vocab_size: usize) -> Result<()> {
        ensure!(
            vocab_size > 0 && vocab_size <= u32::MAX as usize,
            "sampling vocabulary must contain 1..=u32::MAX tokens"
        );
        for (name, value) in [
            ("frequency_penalty", self.frequency_penalty),
            ("presence_penalty", self.presence_penalty),
        ] {
            ensure!(
                value.is_finite() && (-2.0..=2.0).contains(&value),
                "{name} must be finite and in [-2, 2]"
            );
        }
        for (&token, &bias) in &self.logit_bias {
            ensure!(
                (token as usize) < vocab_size,
                "logit_bias token {token} is outside vocabulary {vocab_size}"
            );
            ensure!(
                bias.is_finite() && (-100.0..=100.0).contains(&bias),
                "logit_bias for token {token} must be finite and in [-100, 100]"
            );
        }
        Ok(())
    }
}

/// Request-local emitted-token history. Clone this state for speculative
/// proposals; committing output must update only the original state.
#[derive(Clone, Debug)]
pub struct LogitProcessor {
    options: SamplingOptions,
    vocab_size: usize,
    counts: BTreeMap<u32, usize>,
    has_penalties: bool,
}

impl LogitProcessor {
    pub fn new(mut options: SamplingOptions, vocab_size: usize) -> Result<Self> {
        options.validate(vocab_size)?;
        options.logit_bias.retain(|_, bias| *bias != 0.0);
        let has_penalties = options.frequency_penalty != 0.0 || options.presence_penalty != 0.0;
        Ok(Self {
            options,
            vocab_size,
            counts: BTreeMap::new(),
            has_penalties,
        })
    }

    pub fn is_active(&self) -> bool {
        self.has_penalties || !self.options.logit_bias.is_empty()
    }

    pub fn process<'a>(&self, logits: &'a [f32]) -> Result<Cow<'a, [f32]>> {
        self.validate_logits(logits)?;
        if self.counts.is_empty() && self.options.logit_bias.is_empty() {
            return Ok(Cow::Borrowed(logits));
        }
        let mut adjusted = logits.to_vec();
        self.process_in_place(&mut adjusted)?;
        Ok(Cow::Owned(adjusted))
    }

    pub fn process_in_place(&self, logits: &mut [f32]) -> Result<()> {
        self.validate_logits(logits)?;
        if self.counts.is_empty() && self.options.logit_bias.is_empty() {
            return Ok(());
        }
        for (&token, &bias) in &self.options.logit_bias {
            logits[token as usize] += bias;
        }
        for (&token, &count) in &self.counts {
            logits[token as usize] -= self.options.frequency_penalty * count as f32;
            logits[token as usize] -= self.options.presence_penalty;
        }
        Ok(())
    }

    pub fn record(&mut self, token: u32) -> Result<()> {
        ensure!(
            (token as usize) < self.vocab_size,
            "generated token {token} is outside vocabulary {}",
            self.vocab_size
        );
        if self.has_penalties {
            let count = self.counts.entry(token).or_default();
            *count = count
                .checked_add(1)
                .context("completion token count overflow")?;
        }
        Ok(())
    }

    /// Row zero uses the emitted prefix, including its pending anchor. Row i
    /// additionally sees drafts[..i], and the bonus row sees every draft.
    /// This transformation never commits any proposed token to `self`.
    pub fn process_block(&self, drafts: &[u32], logits: &mut [Vec<f32>]) -> Result<()> {
        ensure!(
            drafts.len().checked_add(1) == Some(logits.len()),
            "sampling block needs one logit row per draft plus one bonus row"
        );
        if !self.is_active() {
            return Ok(());
        }
        let mut hypothetical = self.clone();
        for (index, row) in logits.iter_mut().enumerate() {
            hypothetical.process_in_place(row)?;
            if let Some(&token) = drafts.get(index) {
                hypothetical.record(token)?;
            }
        }
        Ok(())
    }

    fn validate_logits(&self, logits: &[f32]) -> Result<()> {
        ensure!(
            logits.len() == self.vocab_size,
            "logit vocabulary changed: expected {}, got {}",
            self.vocab_size,
            logits.len()
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        chat::Sampler,
        speculative::{Distribution, DraftProposal, SamplingConfig, verify},
    };

    #[test]
    fn sampling_options_accept_endpoints_and_reject_invalid_values() {
        for value in [-2.0, 0.0, 2.0] {
            SamplingOptions {
                frequency_penalty: value,
                presence_penalty: value,
                logit_bias: BTreeMap::from([(0, -100.0), (3, 100.0)]),
            }
            .validate(4)
            .unwrap();
        }
        for invalid in [-2.01, 2.01, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(
                SamplingOptions {
                    frequency_penalty: invalid,
                    ..Default::default()
                }
                .validate(4)
                .is_err()
            );
            assert!(
                SamplingOptions {
                    presence_penalty: invalid,
                    ..Default::default()
                }
                .validate(4)
                .is_err()
            );
        }
        for invalid in [-100.01, 100.01, f32::NAN, f32::INFINITY] {
            assert!(
                SamplingOptions {
                    logit_bias: BTreeMap::from([(1, invalid)]),
                    ..Default::default()
                }
                .validate(4)
                .is_err()
            );
        }
        assert!(
            SamplingOptions {
                logit_bias: BTreeMap::from([(4, 1.0)]),
                ..Default::default()
            }
            .validate(4)
            .is_err()
        );
        assert!(SamplingOptions::default().validate(0).is_err());
    }

    #[test]
    fn sampling_bias_applies_to_first_token_then_penalties_count_completion_tokens() {
        let mut processor = LogitProcessor::new(
            SamplingOptions {
                frequency_penalty: 0.5,
                presence_penalty: 1.0,
                logit_bias: BTreeMap::from([(0, 2.0), (2, -1.0)]),
            },
            3,
        )
        .unwrap();
        let logits = [10.0, 8.0, 2.0];
        assert_eq!(&*processor.process(&logits).unwrap(), &[12.0, 8.0, 1.0]);
        processor.record(0).unwrap();
        processor.record(0).unwrap();
        processor.record(1).unwrap();
        assert_eq!(&*processor.process(&logits).unwrap(), &[10.0, 6.5, 1.0]);
        assert_eq!(logits, [10.0, 8.0, 2.0]);
        let mut owned = logits;
        processor.process_in_place(&mut owned).unwrap();
        assert_eq!(owned, [10.0, 6.5, 1.0]);
    }

    #[test]
    fn sampling_negative_penalties_reward_seen_tokens() {
        let mut processor = LogitProcessor::new(
            SamplingOptions {
                frequency_penalty: -0.5,
                presence_penalty: -1.0,
                ..Default::default()
            },
            2,
        )
        .unwrap();
        processor.record(1).unwrap();
        processor.record(1).unwrap();
        assert_eq!(&*processor.process(&[0.0, 0.0]).unwrap(), &[0.0, 2.0]);
    }

    #[test]
    fn sampling_neutral_options_borrow_logits_and_preserve_seeded_sampling() {
        for options in [
            SamplingOptions::default(),
            SamplingOptions {
                logit_bias: BTreeMap::from([(1, 0.0)]),
                ..Default::default()
            },
        ] {
            let mut processor = LogitProcessor::new(options, 3).unwrap();
            assert!(!processor.is_active());
            let mut original = Sampler::new(42);
            let mut processed = Sampler::new(42);
            let logits = [0.0, 2.0, 1.0];
            for _ in 0..32 {
                let adjusted = processor.process(&logits).unwrap();
                assert!(matches!(adjusted, Cow::Borrowed(_)));
                let token = processed.sample(&adjusted, 0.7, 0.9, 0).unwrap();
                assert_eq!(token, original.sample(&logits, 0.7, 0.9, 0).unwrap());
                processor.record(token).unwrap();
            }
        }
    }

    #[test]
    fn sampling_speculative_rows_use_hypothetical_prefix_without_committing() {
        let mut processor = LogitProcessor::new(
            SamplingOptions {
                frequency_penalty: 1.0,
                presence_penalty: 0.5,
                ..Default::default()
            },
            3,
        )
        .unwrap();
        processor.record(0).unwrap(); // The anchor is already emitted.
        let mut block = vec![vec![4.0; 3]; 3];
        processor.process_block(&[1, 1], &mut block).unwrap();
        assert_eq!(
            block,
            vec![
                vec![2.5, 4.0, 4.0],
                vec![2.5, 2.5, 4.0],
                vec![2.5, 1.5, 4.0],
            ]
        );
        assert_eq!(&*processor.process(&[4.0; 3]).unwrap(), &[2.5, 4.0, 4.0]);
    }

    #[test]
    fn sampling_rejected_draft_counts_never_enter_emitted_prefix() {
        let mut processor = LogitProcessor::new(
            SamplingOptions {
                frequency_penalty: 2.0,
                ..Default::default()
            },
            3,
        )
        .unwrap();
        processor.record(0).unwrap();
        let config = SamplingConfig {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
        };
        let mut hypothetical = processor.clone();
        let mut drafts = Vec::new();
        for _ in 0..2 {
            let distribution =
                Distribution::from_logits(&hypothetical.process(&[0.0, 4.0, 0.0]).unwrap(), config)
                    .unwrap();
            let token = distribution.sample(&mut Sampler::new(1)).unwrap();
            assert_eq!(token, 1);
            hypothetical.record(token).unwrap();
            drafts.push(DraftProposal {
                token,
                distribution,
            });
        }
        let mut target = vec![vec![0.0, 4.0, 0.0], vec![0.0, 1.0, 3.0], vec![0.0; 3]];
        processor.process_block(&[1, 1], &mut target).unwrap();
        let verified = verify(&drafts, &target, config, &mut Sampler::new(3)).unwrap();
        assert_eq!(verified.accepted_drafts, 1);
        assert_eq!(verified.tokens, vec![1, 2]);
        for token in verified.tokens {
            processor.record(token).unwrap();
        }
        assert_eq!(&*processor.process(&[4.0; 3]).unwrap(), &[2.0, 2.0, 2.0]);
    }

    #[test]
    fn sampling_rejects_mismatched_vocabulary_and_prefix_shapes() {
        let mut processor = LogitProcessor::new(
            SamplingOptions {
                presence_penalty: 1.0,
                ..Default::default()
            },
            3,
        )
        .unwrap();
        assert!(processor.record(3).is_err());
        assert!(processor.process(&[0.0; 2]).is_err());
        assert!(processor.process_in_place(&mut [0.0; 2]).is_err());
        assert!(processor.process_block(&[1], &mut [vec![0.0; 3]]).is_err());
    }
}
