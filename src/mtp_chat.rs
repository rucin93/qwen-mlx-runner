//! Optional native MTP chat generation with exact target verification.
//!
//! Both caches are reset for every request and on every return path. This path
//! deliberately does not reuse chat prefixes. Only verified target tokens reach
//! the output callback; acceptance statistics never stand in for output speed.

mod timing;
pub use timing::TargetDecodeTiming;

use std::{path::Path, time::Instant};

use anyhow::{Context, Result, bail, ensure};
use serde::Serialize;

use crate::{
    chat::{
        ChatTokenizer, GenerationOutput, GenerationRequest, Sampler, TextGenerator,
        sampling::LogitProcessor,
    },
    engine::{Engine, Mtp},
    speculative::{Distribution, DraftProposal, SamplingConfig, verify},
};

const DRAFT_SEED_XOR: u64 = 0xd1b5_4a32_d192_ed03;

#[derive(Clone, Debug, Default, Serialize)]
pub struct MtpGenerationStats {
    /// Actual verified completion IDs, including EOS, matching completion_tokens.
    pub token_ids: Vec<u32>,
    pub completion_tokens: usize,
    pub decode_seconds: f64,
    /// From prompt processing start until the first verified token is available.
    pub time_to_first_token_seconds: Option<f64>,
    pub proposed_drafts: usize,
    /// Proposals accepted by the verifier, including any unused tail after EOS.
    pub accepted_drafts: usize,
    pub rounds: usize,
    pub round_widths: Vec<usize>,
    /// Decode target execution, including GPU completion and readback.
    pub target_seconds: f64,
    /// Existing whole-command timestamps, excluding prefill and rollback.
    /// GPU execution overlaps the completion wait; these fields are not additive.
    pub target_decode_timing: TargetDecodeTiming,
    /// Decode proposal construction, MTP execution, cache repair and seeding.
    pub draft_seconds: f64,
    /// CPU target acceptance and residual sampling.
    pub verification_seconds: f64,
    /// Target prefix commit and MTP truncation; MTP replay is in draft_seconds.
    pub rollback_seconds: f64,
    pub prefill_target_seconds: f64,
    pub prefill_draft_seconds: f64,
}

pub struct MtpChatEngine {
    engine: Engine,
    mtp: Mtp,
    tokenizer: ChatTokenizer,
    model_id: String,
    block_size: usize,
}

impl MtpChatEngine {
    pub fn load(
        target_path: &Path,
        mtp_path: &Path,
        context: usize,
        block_size: usize,
    ) -> Result<Self> {
        ensure!(
            (1..=4).contains(&block_size),
            "MTP block size must be 1..=4"
        );
        let tokenizer = ChatTokenizer::load(target_path)?;
        let engine = Engine::load(target_path, context)?;
        let mtp = Mtp::load(&engine, mtp_path, context)?;
        let model_id = target_path
            .file_name()
            .context("model path needs a directory name")?
            .to_string_lossy()
            .into_owned();
        Ok(Self {
            engine,
            mtp,
            tokenizer,
            model_id,
            block_size,
        })
    }

    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    /// Select only the target verification schedule; the native MTP adapter is
    /// unchanged so a factorial benchmark can isolate the two target kernels.
    pub fn set_block_kernel_mode(&mut self, mode: crate::gpu::BlockKernelMode) -> Result<()> {
        self.engine.set_block_kernel_mode(mode)
    }

    pub fn clear_cache(&mut self) {
        self.engine.reset();
        self.mtp.reset();
    }

    pub fn generate_with_stats(
        &mut self,
        request: &GenerationRequest,
        on_text: &mut dyn FnMut(&str) -> bool,
    ) -> Result<(GenerationOutput, MtpGenerationStats)> {
        self.clear_cache();
        let result = self.generate_inner(request, on_text, true);
        // In particular, cancellation in an accepted block cannot leave either
        // model positioned past the output visible to the next request.
        self.clear_cache();
        result
    }

    /// Ordinary sequential target generation using the same loaded weights,
    /// tokenizer and output accounting, with no MTP or block execution.
    pub fn generate_reference_with_stats(
        &mut self,
        request: &GenerationRequest,
        on_text: &mut dyn FnMut(&str) -> bool,
    ) -> Result<(GenerationOutput, MtpGenerationStats)> {
        self.clear_cache();
        let result = self.generate_inner(request, on_text, false);
        self.clear_cache();
        result
    }

    fn generate_inner(
        &mut self,
        request: &GenerationRequest,
        on_text: &mut dyn FnMut(&str) -> bool,
        use_mtp: bool,
    ) -> Result<(GenerationOutput, MtpGenerationStats)> {
        ensure!(request.max_tokens > 0, "max_tokens must be positive");
        ensure!(
            request.temperature.is_finite() && request.temperature >= 0.0,
            "invalid temperature"
        );
        ensure!(
            request.top_p.is_finite() && request.top_p > 0.0 && request.top_p <= 1.0,
            "invalid top_p"
        );
        let mut processor =
            LogitProcessor::new(request.sampling.clone(), self.engine.config().vocab_size)?;
        let prompt = self
            .tokenizer
            .encode(&self.tokenizer.render_request(request)?)?;
        ensure!(!prompt.is_empty(), "empty prompt after tokenization");
        ensure!(
            prompt
                .len()
                .checked_add(request.max_tokens)
                .is_some_and(|end| end <= self.engine.context_capacity()),
            "prompt ({} tokens) plus max_tokens ({}) exceeds context {}",
            prompt.len(),
            request.max_tokens,
            self.engine.context_capacity()
        );
        let config = SamplingConfig {
            temperature: request.temperature,
            top_p: request.top_p,
            top_k: request.top_k,
        };
        let mut stats = MtpGenerationStats::default();
        let prefill_started = Instant::now();
        let mut previous_hidden = Vec::new();
        let mut logits = Vec::new();
        if use_mtp {
            for (chunk_index, chunk) in prompt.chunks(self.block_size).enumerate() {
                if !on_text("") {
                    bail!("request cancelled during prefill");
                }
                // Target weights are reused across prompt positions, while
                // stateful layers still execute the positions causally.
                let started = Instant::now();
                let final_chunk = chunk_index * self.block_size + chunk.len() == prompt.len();
                let mut block = self.engine.prefill_block(chunk, final_chunk)?;
                stats.prefill_target_seconds += started.elapsed().as_secs_f64();
                if final_chunk {
                    logits = block
                        .logits
                        .pop()
                        .context("final prefill block returned no logits")?;
                }
                for (relative, (&token, hidden)) in chunk.iter().zip(block.hidden).enumerate() {
                    if !on_text("") {
                        bail!("request cancelled during prefill");
                    }
                    if chunk_index * self.block_size + relative > 0 {
                        // MTP position i-1 combines x_i with target h_(i-1),
                        // including pairs spanning two target prompt blocks.
                        let started = Instant::now();
                        self.mtp.prefill_token(token, &previous_hidden)?;
                        stats.prefill_draft_seconds += started.elapsed().as_secs_f64();
                    }
                    previous_hidden = hidden;
                }
            }
        } else {
            for (index, &token) in prompt.iter().enumerate() {
                if !on_text("") {
                    bail!("request cancelled during prefill");
                }
                let started = Instant::now();
                logits = self
                    .engine
                    .prefill_token(token, index + 1 == prompt.len())?;
                stats.prefill_target_seconds += started.elapsed().as_secs_f64();
            }
        }
        let prefill_seconds = prefill_started.elapsed().as_secs_f64();
        let decode_started = Instant::now();
        let mut rng = Sampler::new(request.seed);
        let mut draft_rng = Sampler::new(request.seed ^ DRAFT_SEED_XOR);
        let mut output = OutputCollector::new(request.max_tokens);
        // The initial pending anchor is already a target sample. It has not yet
        // been consumed by the target, and so needs no speculative verification.
        let mut anchor =
            Distribution::from_logits(&processor.process(&logits)?, config)?.sample(&mut rng)?;
        let keep_going = output.push(
            anchor,
            &self.tokenizer,
            request,
            &self.engine.config().eos_token_ids,
            on_text,
        )?;
        if !output.token_ids.is_empty() {
            processor.record(anchor)?;
        }
        if keep_going && !use_mtp {
            while output.finish.is_none() {
                if !on_text("") {
                    output.finish = Some("cancelled");
                    break;
                }
                let started = Instant::now();
                let logits = self.engine.forward(anchor)?;
                stats.target_seconds += started.elapsed().as_secs_f64();
                stats
                    .target_decode_timing
                    .record(self.engine.last_frame_timing());
                anchor = Distribution::from_logits(&processor.process(&logits)?, config)?
                    .sample(&mut rng)?;
                let previous_count = output.token_ids.len();
                let keep_going = output.push(
                    anchor,
                    &self.tokenizer,
                    request,
                    &self.engine.config().eos_token_ids,
                    on_text,
                )?;
                if output.token_ids.len() > previous_count {
                    processor.record(anchor)?;
                }
                if !keep_going {
                    break;
                }
            }
        } else if keep_going {
            let started = Instant::now();
            let mut seed = self.mtp.forward(anchor, &previous_hidden, true)?;
            stats.draft_seconds += started.elapsed().as_secs_f64();
            while output.finish.is_none() {
                if !on_text("") {
                    output.finish = Some("cancelled");
                    break;
                }
                let base = self.engine.position();
                ensure!(
                    self.mtp.position() == base,
                    "target and MTP cache positions diverged before verification"
                );
                let remaining = request.max_tokens - output.token_ids.len();
                // A width-B round emits at most B new tokens, despite taking an
                // already-emitted anchor plus B-1 draft tokens as target inputs.
                let width = self
                    .block_size
                    .min(remaining)
                    .min(self.engine.context_capacity() - base);
                ensure!(width > 0, "no context remains for target verification");
                let started = Instant::now();
                let mut drafts = Vec::with_capacity(width - 1);
                let mut inputs = Vec::with_capacity(width);
                let mut hypothetical = processor.clone();
                inputs.push(anchor);
                for index in 0..width - 1 {
                    let distribution =
                        Distribution::from_logits(&hypothetical.process(&seed.logits)?, config)?;
                    let token = distribution.sample(&mut draft_rng)?;
                    drafts.push(DraftProposal {
                        token,
                        distribution,
                    });
                    inputs.push(token);
                    hypothetical.record(token)?;
                    if index + 1 < width - 1 {
                        seed = self.mtp.forward(token, &seed.hidden, true)?;
                    }
                }
                stats.draft_seconds += started.elapsed().as_secs_f64();
                stats.proposed_drafts += drafts.len();
                stats.rounds += 1;
                stats.round_widths.push(width);

                let started = Instant::now();
                let mut block = self.engine.verify_block(&inputs)?;
                stats.target_seconds += started.elapsed().as_secs_f64();
                stats
                    .target_decode_timing
                    .record(self.engine.last_frame_timing());
                ensure!(
                    block.base_position == base,
                    "target verification base changed"
                );
                let started = Instant::now();
                processor.process_block(&inputs[1..], &mut block.logits)?;
                let verified = verify(&drafts, &block.logits, config, &mut rng)?;
                stats.verification_seconds += started.elapsed().as_secs_f64();
                stats.accepted_drafts += verified.accepted_drafts;
                let started = Instant::now();
                self.engine.commit_block_prefix(verified.consumed_inputs)?;
                stats.rollback_seconds += started.elapsed().as_secs_f64();
                for &token in &verified.tokens {
                    let previous_count = output.token_ids.len();
                    let keep_going = output.push(
                        token,
                        &self.tokenizer,
                        request,
                        &self.engine.config().eos_token_ids,
                        on_text,
                    )?;
                    if output.token_ids.len() > previous_count {
                        processor.record(token)?;
                    }
                    if !keep_going {
                        break;
                    }
                }
                if output.finish.is_some() {
                    // generate_with_stats resets both caches, including any
                    // verified but un-emitted suffix following EOS/cancellation.
                    break;
                }

                let appended = drafts.len().saturating_sub(1);
                let kept = verified.accepted_drafts.min(appended);
                let started = Instant::now();
                self.mtp.truncate(base + kept)?;
                stats.rollback_seconds += started.elapsed().as_secs_f64();
                let started = Instant::now();
                // The final proposal was sampled without consuming it in MTP.
                // Replay missing accepted pairs using the verified target h for
                // the preceding input; never replay a target model operation.
                for (index, draft) in drafts
                    .iter()
                    .enumerate()
                    .take(verified.accepted_drafts)
                    .skip(kept)
                {
                    self.mtp.forward(draft.token, &block.hidden[index], false)?;
                }
                anchor = *verified
                    .tokens
                    .last()
                    .context("verifier returned no output token")?;
                seed = self
                    .mtp
                    .forward(anchor, &block.hidden[verified.accepted_drafts], true)?;
                stats.draft_seconds += started.elapsed().as_secs_f64();
                ensure!(
                    self.mtp.position() == self.engine.position(),
                    "target and MTP cache positions diverged after repair"
                );
            }
        }
        output.flush(&self.tokenizer, request.preserve_special_tokens(), on_text)?;
        stats.completion_tokens = output.token_ids.len();
        stats.decode_seconds = decode_started.elapsed().as_secs_f64();
        stats.time_to_first_token_seconds = output
            .first_token_at
            .map(|instant| instant.duration_since(prefill_started).as_secs_f64());
        stats.token_ids = output.token_ids;
        let result = GenerationOutput {
            text: output.text,
            prompt_tokens: prompt.len(),
            completion_tokens: stats.completion_tokens,
            finish_reason: output.finish.unwrap_or("length").into(),
            prefill_seconds,
            decode_seconds: stats.decode_seconds,
        };
        Ok((result, stats))
    }
}

impl TextGenerator for MtpChatEngine {
    fn generate(
        &mut self,
        request: &GenerationRequest,
        on_text: &mut dyn FnMut(&str) -> bool,
    ) -> Result<GenerationOutput> {
        self.generate_with_stats(request, on_text)
            .map(|(output, _)| output)
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn vocab_size(&self) -> Option<usize> {
        Some(self.engine.config().vocab_size)
    }

    fn prepare_request(&self, request: &mut GenerationRequest) -> Result<()> {
        request.sampling.validate(self.engine.config().vocab_size)?;
        self.tokenizer
            .prepare_request(request, self.engine.context_capacity())
    }
}

struct OutputCollector {
    token_ids: Vec<u32>,
    text_tokens: Vec<u32>,
    text: String,
    limit: usize,
    finish: Option<&'static str>,
    first_token_at: Option<Instant>,
}

impl OutputCollector {
    fn new(limit: usize) -> Self {
        Self {
            token_ids: Vec::with_capacity(limit),
            text_tokens: Vec::with_capacity(limit),
            text: String::new(),
            limit,
            finish: None,
            first_token_at: None,
        }
    }

    fn push(
        &mut self,
        token: u32,
        tokenizer: &ChatTokenizer,
        request: &GenerationRequest,
        eos: &[u32],
        on_text: &mut dyn FnMut(&str) -> bool,
    ) -> Result<bool> {
        if self.finish.is_some() || self.token_ids.len() >= self.limit {
            return Ok(false);
        }
        if !on_text("") {
            self.finish = Some("cancelled");
            return Ok(false);
        }
        self.token_ids.push(token);
        self.first_token_at.get_or_insert_with(Instant::now);
        if eos.contains(&token) {
            self.finish = Some("stop");
            return Ok(false);
        }
        self.text_tokens.push(token);
        let decoded = if request.preserve_special_tokens() {
            tokenizer.decode_with_special_tokens(&self.text_tokens)?
        } else {
            tokenizer.decode(&self.text_tokens)?
        };
        if !decoded.ends_with('\u{fffd}') {
            ensure!(
                decoded.starts_with(&self.text),
                "tokenizer decoder changed previously emitted text"
            );
            let chunk = &decoded[self.text.len()..];
            if !chunk.is_empty() && !on_text(chunk) {
                self.finish = Some("cancelled");
                return Ok(false);
            }
            self.text = decoded;
        }
        if self.token_ids.len() == self.limit {
            self.finish = Some("length");
            return Ok(false);
        }
        Ok(true)
    }

    fn flush(
        &mut self,
        tokenizer: &ChatTokenizer,
        preserve_special_tokens: bool,
        on_text: &mut dyn FnMut(&str) -> bool,
    ) -> Result<()> {
        if self.finish == Some("cancelled") {
            return Ok(());
        }
        let decoded = if preserve_special_tokens {
            tokenizer.decode_with_special_tokens(&self.text_tokens)?
        } else {
            tokenizer.decode(&self.text_tokens)?
        };
        ensure!(
            decoded.starts_with(&self.text),
            "unstable final tokenizer output"
        );
        if decoded.len() > self.text.len() && !on_text(&decoded[self.text.len()..]) {
            self.finish = Some("cancelled");
            return Ok(());
        }
        self.text = decoded;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::Message;

    fn request(max_tokens: usize) -> GenerationRequest {
        GenerationRequest {
            messages: vec![Message {
                role: "user".into(),
                content: "w6 w7".into(),
                ..Default::default()
            }],
            max_tokens,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            seed: 42,
            enable_thinking: false,
            ..Default::default()
        }
    }

    fn fixture(name: &str) -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name)
    }

    fn zero_head_fixture() -> tempfile::TempDir {
        let target = tempfile::tempdir().unwrap();
        for name in ["config.json", "tokenizer.json", "chat_template.jinja"] {
            std::fs::copy(fixture("tiny-q4").join(name), target.path().join(name)).unwrap();
        }
        let mut weights = std::fs::read(fixture("tiny-q4").join("model.safetensors")).unwrap();
        let header_size = u64::from_le_bytes(weights[..8].try_into().unwrap()) as usize;
        let header: serde_json::Value =
            serde_json::from_slice(&weights[8..8 + header_size]).unwrap();
        let mut changed = 0;
        for (name, tensor) in header.as_object().unwrap() {
            if name.starts_with("language_model.lm_head.") {
                let start = tensor["data_offsets"][0].as_u64().unwrap() as usize;
                let end = tensor["data_offsets"][1].as_u64().unwrap() as usize;
                weights[8 + header_size + start..8 + header_size + end].fill(0);
                changed += 1;
            }
        }
        assert_eq!(changed, 3);
        std::fs::write(target.path().join("model.safetensors"), weights).unwrap();
        target
    }

    #[test]
    #[ignore = "requires a real Apple Metal GPU"]
    fn sampling_penalties_match_target_for_every_mtp_width_and_first_anchor() {
        use crate::{chat::sampling::SamplingOptions, engine::ChatEngine};
        use std::collections::BTreeMap;

        let target = zero_head_fixture();
        let mut request = request(11);
        request.sampling = SamplingOptions {
            frequency_penalty: 2.0,
            presence_penalty: 1.0,
            logit_bias: BTreeMap::from([(6, 20.0), (7, 20.0), (8, 20.0)]),
        };
        // Prompt contains w6/w7; only completion counts may affect this cycle.
        let expected_ids = vec![6, 7, 8, 6, 7, 8, 6, 7, 8, 6, 7];
        let mut baseline = ChatEngine::load(target.path(), 128).unwrap();
        let expected = baseline.generate(&request, &mut |_| true).unwrap();
        for width in 1..=4 {
            let mut mtp =
                MtpChatEngine::load(target.path(), &fixture("tiny-mtp"), 128, width).unwrap();
            let (actual, stats) = mtp.generate_with_stats(&request, &mut |_| true).unwrap();
            assert_eq!(stats.token_ids, expected_ids, "width={width}");
            assert_eq!(actual.text, expected.text, "width={width}");
            assert_eq!(actual.completion_tokens, 11);
            assert_eq!(actual.finish_reason, "length");
            assert_eq!(stats.accepted_drafts, stats.proposed_drafts);
            let (_, reference) = mtp
                .generate_reference_with_stats(&request, &mut |_| true)
                .unwrap();
            assert_eq!(reference.token_ids, expected_ids, "reference width={width}");
            let (_, repeated) = mtp.generate_with_stats(&request, &mut |_| true).unwrap();
            assert_eq!(
                repeated.token_ids, expected_ids,
                "request-local counts width={width}"
            );
        }
    }

    #[test]
    fn output_limits_and_eos_stop_inside_verified_blocks() {
        let tokenizer = ChatTokenizer::load(&fixture("tiny-q4")).unwrap();
        let mut output = OutputCollector::new(4);
        assert!(
            output
                .push(6, &tokenizer, &request(4), &[1], &mut |_| true)
                .unwrap()
        );
        assert!(
            !output
                .push(1, &tokenizer, &request(4), &[1], &mut |_| true)
                .unwrap()
        );
        assert!(
            !output
                .push(7, &tokenizer, &request(4), &[1], &mut |_| panic!(
                    "must not emit after EOS"
                ))
                .unwrap()
        );
        assert_eq!(output.token_ids, vec![6, 1]);
        assert_eq!(output.text_tokens, vec![6]);
        assert_eq!(output.finish, Some("stop"));
        let mut output = OutputCollector::new(1);
        assert!(
            !output
                .push(6, &tokenizer, &request(1), &[1], &mut |_| true)
                .unwrap()
        );
        assert!(
            !output
                .push(7, &tokenizer, &request(1), &[1], &mut |_| panic!(
                    "must not exceed max_tokens"
                ))
                .unwrap()
        );
        assert_eq!(output.token_ids, vec![6]);
        assert_eq!(output.finish, Some("length"));
    }

    #[test]
    fn cancellation_counts_only_processed_verified_tokens() {
        let tokenizer = ChatTokenizer::load(&fixture("tiny-q4")).unwrap();
        let mut output = OutputCollector::new(4);
        assert!(
            !output
                .push(6, &tokenizer, &request(4), &[1], &mut |_| false)
                .unwrap()
        );
        assert!(output.token_ids.is_empty());
        assert!(output.first_token_at.is_none());
        let mut output = OutputCollector::new(4);
        assert!(
            !output
                .push(6, &tokenizer, &request(4), &[1], &mut |chunk| chunk
                    .is_empty())
                .unwrap()
        );
        assert_eq!(output.token_ids, vec![6]);
        assert!(output.text.is_empty());
        assert_eq!(output.finish, Some("cancelled"));
        output
            .flush(&tokenizer, false, &mut |_| {
                panic!("must not flush after cancellation")
            })
            .unwrap();
    }

    #[test]
    #[ignore = "requires a real Apple Metal GPU"]
    fn mtp_chat_greedy_matches_target_and_resets_after_cancellation() {
        use crate::engine::ChatEngine;
        let target = fixture("tiny-q4");
        let adapter = fixture("tiny-mtp");
        let mut baseline = ChatEngine::load(&target, 128).unwrap();
        for width in 1..=4 {
            let mut mtp = MtpChatEngine::load(&target, &adapter, 128, width).unwrap();
            for limit in [1, 2, 3, 4, 9, 20] {
                let request = request(limit);
                baseline.clear_cache();
                let expected = baseline.generate(&request, &mut |_| true).unwrap();
                let (actual, stats) = mtp.generate_with_stats(&request, &mut |_| true).unwrap();
                assert_eq!(actual.text, expected.text, "width={width}, limit={limit}");
                assert_eq!(actual.completion_tokens, expected.completion_tokens);
                assert_eq!(actual.finish_reason, expected.finish_reason);
                assert_eq!(stats.token_ids.len(), actual.completion_tokens);
                assert_eq!(stats.completion_tokens, actual.completion_tokens);
                assert_eq!(stats.decode_seconds, actual.decode_seconds);
                assert!(stats.accepted_drafts <= stats.proposed_drafts);
                assert_eq!(mtp.engine.position(), 0);
                assert_eq!(mtp.mtp.position(), 0);
                let (reference, reference_stats) = mtp
                    .generate_reference_with_stats(&request, &mut |_| true)
                    .unwrap();
                assert_eq!(reference.text, expected.text);
                assert_eq!(reference.completion_tokens, expected.completion_tokens);
                assert_eq!(
                    stats.token_ids, reference_stats.token_ids,
                    "width={width}, limit={limit}"
                );
                assert_eq!(reference_stats.proposed_drafts, 0);
                assert_eq!(reference_stats.rounds, 0);
                assert_eq!(reference_stats.draft_seconds, 0.0);
                assert_eq!(reference_stats.prefill_draft_seconds, 0.0);
            }
            let mut chunks = 0;
            let (cancelled, _) = mtp
                .generate_with_stats(&request(20), &mut |chunk| {
                    if !chunk.is_empty() {
                        chunks += 1;
                    }
                    chunks < 2
                })
                .unwrap();
            assert_eq!(cancelled.finish_reason, "cancelled");
            assert_eq!(mtp.engine.position(), 0);
            assert_eq!(mtp.mtp.position(), 0);
            assert!(
                mtp.generate_with_stats(&request(20), &mut |_| false)
                    .is_err()
            );
            assert_eq!(mtp.engine.position(), 0);
            assert_eq!(mtp.mtp.position(), 0);
            baseline.clear_cache();
            let expected = baseline.generate(&request(20), &mut |_| true).unwrap();
            let actual = mtp.generate(&request(20), &mut |_| true).unwrap();
            assert_eq!(actual.text, expected.text);
            assert_eq!(actual.completion_tokens, expected.completion_tokens);
        }
    }

    #[test]
    #[ignore = "requires a real Apple Metal GPU"]
    fn full_acceptance_repair_context_boundary_and_real_eos_match_reference() {
        // A zero shared head makes both greedy distributions one-hot at token
        // zero. The target's recurrent/attention layers and MTP layer remain
        // nontrivial. This exercises every all-accepted repair branch without
        // depending on the random fixture drafter's accidental acceptance rate.
        let target = zero_head_fixture();
        let tokenizer = ChatTokenizer::load(target.path()).unwrap();
        let request = request(20);
        let prompt_len = tokenizer
            .encode(&tokenizer.render(&request.messages, false).unwrap())
            .unwrap()
            .len();
        for width in 2..=4 {
            let mut engine = MtpChatEngine::load(
                target.path(),
                &fixture("tiny-mtp"),
                prompt_len + request.max_tokens,
                width,
            )
            .unwrap();
            let (output, stats) = engine.generate_with_stats(&request, &mut |_| true).unwrap();
            assert_eq!(output.completion_tokens, request.max_tokens);
            assert_eq!(output.finish_reason, "length");
            assert_eq!(stats.token_ids, vec![0; request.max_tokens]);
            assert!(stats.proposed_drafts > 0);
            assert_eq!(stats.accepted_drafts, stats.proposed_drafts);
            assert_eq!(
                stats.round_widths.iter().sum::<usize>(),
                request.max_tokens - 1
            );
            assert_eq!(engine.engine.position(), 0);
            assert_eq!(engine.mtp.position(), 0);
            let (reference, reference_stats) = engine
                .generate_reference_with_stats(&request, &mut |_| true)
                .unwrap();
            assert_eq!(output.text, reference.text);
            assert_eq!(stats.token_ids, reference_stats.token_ids);
            let mut chunks = 0;
            let (cancelled, cancelled_stats) = engine
                .generate_with_stats(&request, &mut |chunk| {
                    if !chunk.is_empty() {
                        chunks += 1;
                    }
                    chunks < 2
                })
                .unwrap();
            assert_eq!(cancelled.finish_reason, "cancelled");
            assert_eq!(cancelled.completion_tokens, 2);
            assert_eq!(cancelled_stats.rounds, 1);
            assert!(cancelled_stats.accepted_drafts > 0);
            let (_, after) = engine.generate_with_stats(&request, &mut |_| true).unwrap();
            assert_eq!(after.token_ids, reference_stats.token_ids);
        }

        let config_path = target.path().join("config.json");
        let mut config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&config_path).unwrap()).unwrap();
        config["eos_token_id"] = serde_json::json!([0]);
        std::fs::write(&config_path, serde_json::to_vec(&config).unwrap()).unwrap();
        let mut engine = MtpChatEngine::load(target.path(), &fixture("tiny-mtp"), 128, 3).unwrap();
        let (output, stats) = engine.generate_with_stats(&request, &mut |_| true).unwrap();
        assert_eq!(output.finish_reason, "stop");
        assert_eq!(output.completion_tokens, 1);
        assert_eq!(output.text, "");
        assert_eq!(stats.token_ids, vec![0]);
        assert_eq!(stats.rounds, 0);
        assert_eq!(stats.draft_seconds, 0.0);
        let (reference, reference_stats) = engine
            .generate_reference_with_stats(&request, &mut |_| true)
            .unwrap();
        assert_eq!(reference.text, output.text);
        assert_eq!(reference_stats.token_ids, stats.token_ids);
    }
}
