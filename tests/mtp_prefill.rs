//! Prompt-only shortcuts must preserve the state used by subsequent decoding.
use anyhow::Result;
use qwen_metal::{
    chat::{GenerationRequest, Message},
    engine::Engine,
    mtp_chat::MtpChatEngine,
};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn same_bits(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: width");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{label}[{index}]: {actual} != {expected}"
        );
    }
}

#[test]
#[ignore = "requires a real Apple Metal GPU"]
fn prefill_blocks_skip_intermediate_logits_and_preserve_every_width_and_tail() -> Result<()> {
    for name in ["tiny-q4", "tiny-bf16"] {
        let mut baseline = Engine::load(&fixture(name), 32)?;
        let mut optimized = Engine::load(&fixture(name), 32)?;
        for width in 1..=4 {
            for tail in 1..=width {
                baseline.reset();
                optimized.reset();
                let prompt: Vec<u32> = (0..2 * width + tail)
                    .map(|i| ((i * 13 + 5) % baseline.config().vocab_size) as u32)
                    .collect();
                let mut consumed = 0;
                for chunk in prompt.chunks(width) {
                    let final_chunk = consumed + chunk.len() == prompt.len();
                    let expected = baseline.verify_block(chunk)?;
                    baseline.commit_block_prefix(chunk.len())?;
                    let actual = optimized.prefill_block(chunk, final_chunk)?;
                    assert_eq!(actual.base_position, consumed);
                    assert_eq!(actual.hidden.len(), chunk.len());
                    for (actual, expected) in actual.hidden.iter().zip(&expected.hidden) {
                        same_bits(actual, expected, "prefill hidden");
                    }
                    if final_chunk {
                        assert_eq!(actual.logits.len(), chunk.len());
                        for (actual, expected) in actual.logits.iter().zip(&expected.logits) {
                            same_bits(actual, expected, "final prompt logits");
                        }
                    } else {
                        assert!(
                            actual.logits.is_empty(),
                            "nonfinal prompt block projected logits"
                        );
                    }
                    consumed += chunk.len();
                    assert_eq!(optimized.position(), consumed);
                    // The public prefill operation commits all its inputs itself.
                    assert!(optimized.commit_block_prefix(chunk.len()).is_err());
                }
                for token in [31, 11, 23] {
                    same_bits(
                        &optimized.forward(token)?,
                        &baseline.forward(token)?,
                        "continuation after prefill",
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires a real Apple Metal GPU"]
fn prefill_block_validation_and_pending_state_preserve_the_accepted_prefix() -> Result<()> {
    let mut baseline = Engine::load(&fixture("tiny-q4"), 4)?;
    let mut optimized = Engine::load(&fixture("tiny-q4"), 4)?;
    baseline.verify_block(&[5])?;
    baseline.commit_block_prefix(1)?;
    optimized.prefill_block(&[5], false)?;
    for invalid in [vec![], vec![1, 2, 3, 4, 5], vec![64], vec![1, 2, 3, 4]] {
        assert!(optimized.prefill_block(&invalid, false).is_err());
        assert_eq!(optimized.position(), 1);
    }
    optimized.verify_block(&[9])?;
    assert!(optimized.prefill_block(&[7], false).is_err());
    assert_eq!(optimized.position(), 2);
    optimized.commit_block_prefix(0)?;
    same_bits(
        &optimized.forward(21)?,
        &baseline.forward(21)?,
        "after rejected calls",
    );
    optimized.prefill_block(&[11, 23], true)?;
    assert_eq!(optimized.position(), 4);
    assert!(optimized.prefill_block(&[7], false).is_err());
    assert_eq!(optimized.position(), 4);
    optimized.reset();
    baseline.reset();
    same_bits(
        &optimized.prefill_block(&[9], true)?.logits[0],
        &baseline.verify_block(&[9])?.logits[0],
        "prefill reset",
    );
    Ok(())
}

#[test]
#[ignore = "requires a real Apple Metal GPU"]
fn nonfinal_prefill_omits_exactly_the_vocabulary_projection() -> Result<()> {
    let mut engine = Engine::load(&fixture("tiny-q4"), 8)?;
    engine.enable_command_profiling()?;
    for width in 1..=4 {
        let tokens = [5, 9, 21, 11];
        engine.reset();
        engine.prefill_block(&tokens[..width], true)?;
        let full: BTreeMap<_, _> = engine
            .profile_report()?
            .into_iter()
            .map(|row| ((row.kernel, row.parameters), row.dispatches))
            .collect();
        engine.reset();
        engine.prefill_block(&tokens[..width], false)?;
        let mut hidden: BTreeMap<_, _> = engine
            .profile_report()?
            .into_iter()
            .map(|row| ((row.kernel, row.parameters), row.dispatches))
            .collect();
        let mut removed = Vec::new();
        for ((kernel, parameters), full_count) in full {
            let hidden_count = hidden
                .remove(&(kernel.clone(), parameters.clone()))
                .unwrap_or(0);
            assert!(
                hidden_count <= full_count,
                "nonfinal prefill added {kernel}"
            );
            if full_count != hidden_count {
                removed.push((kernel, parameters, full_count - hidden_count));
            }
        }
        assert!(hidden.is_empty(), "nonfinal prefill added a dispatch");
        assert_eq!(removed.len(), 1, "only the head projection should differ");
        let (kernel, parameters, count) = &removed[0];
        assert!(kernel.starts_with("matmul_"));
        assert_eq!(
            parameters[..2],
            [
                engine.config().vocab_size as u32,
                engine.config().hidden_size as u32
            ]
        );
        assert_eq!(*count, 1);
    }
    Ok(())
}

#[test]
#[ignore = "requires a real Apple Metal GPU"]
fn cancellation_during_partial_prefill_leaves_next_request_unchanged() -> Result<()> {
    let request = GenerationRequest {
        messages: vec![Message {
            role: "user".into(),
            content: "w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 w16 w17".into(),
            ..Default::default()
        }],
        max_tokens: 6,
        temperature: 0.0,
        top_p: 1.0,
        seed: 42,
        ..Default::default()
    };
    for width in 1..=4 {
        let mut fresh = MtpChatEngine::load(&fixture("tiny-q4"), &fixture("tiny-mtp"), 128, width)?;
        let mut reused =
            MtpChatEngine::load(&fixture("tiny-q4"), &fixture("tiny-mtp"), 128, width)?;
        let (expected, expected_stats) = fresh.generate_with_stats(&request, &mut |_| true)?;
        for cancel_at in [2, 4, 7] {
            let mut checkpoints = 0;
            let result = reused.generate_with_stats(&request, &mut |chunk| {
                assert!(chunk.is_empty(), "cancellation must occur during prefill");
                checkpoints += 1;
                checkpoints < cancel_at
            });
            assert_eq!(checkpoints, cancel_at);
            assert!(
                result
                    .unwrap_err()
                    .to_string()
                    .contains("cancelled during prefill")
            );
            assert_eq!(reused.engine().position(), 0);
            let (actual, actual_stats) = reused.generate_with_stats(&request, &mut |_| true)?;
            assert_eq!(actual.text, expected.text);
            assert_eq!(actual.finish_reason, expected.finish_reason);
            assert_eq!(actual.completion_tokens, expected.completion_tokens);
            assert_eq!(actual_stats.token_ids, expected_stats.token_ids);
        }
    }
    Ok(())
}
