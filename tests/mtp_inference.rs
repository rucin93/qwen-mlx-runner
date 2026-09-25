//! Native adapter coverage against the independent scalar Python fixture.
use qwen_metal::engine::{Engine, Mtp, MtpOutput};
use serde::Deserialize;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

#[derive(Deserialize)]
struct Golden {
    tokens: Vec<u32>,
    positions: Vec<usize>,
    input_hidden: Vec<Vec<f32>>,
    hidden: Vec<Vec<f32>>,
    logits: Vec<Vec<f32>>,
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn golden() -> Golden {
    serde_json::from_slice(&std::fs::read(fixture("tiny-mtp").join("golden.json")).unwrap())
        .unwrap()
}

fn assert_close(got: &[f32], expected: &[f32], label: &str) {
    assert_eq!(got.len(), expected.len(), "{label}: output width");
    for (index, (&actual, &want)) in got.iter().zip(expected).enumerate() {
        assert!(
            actual.is_finite() && (actual - want).abs() <= 3e-4 + 3e-4 * want.abs(),
            "{label}[{index}]: {actual} differs from {want}"
        );
    }
}

fn assert_output(got: &MtpOutput, expected: &MtpOutput, label: &str) {
    assert_close(&got.hidden, &expected.hidden, &format!("{label} hidden"));
    assert_close(&got.logits, &expected.logits, &format!("{label} logits"));
}

#[test]
#[ignore = "requires a real Apple Metal GPU"]
fn native_mtp_matches_independent_hidden_logits_and_reset() {
    let gold = golden();
    let target = Engine::load(&fixture("tiny-q4"), 8).unwrap();
    let mut mtp = Mtp::load(&target, &fixture("tiny-mtp"), 8).unwrap();
    // The adapter retains the shared immutable embedding/head buffers. Their
    // lifetime cannot depend on keeping the target Engine value alive.
    drop(target);
    for repetition in 0..2 {
        mtp.reset();
        assert_eq!(mtp.position(), 0);
        for (i, &token) in gold.tokens.iter().enumerate() {
            assert_eq!(mtp.position(), gold.positions[i]);
            let got = mtp.forward(token, &gold.input_hidden[i], true).unwrap();
            assert_close(
                &got.hidden,
                &gold.hidden[i],
                &format!("reset {repetition}, token {i}, hidden"),
            );
            assert_close(
                &got.logits,
                &gold.logits[i],
                &format!("reset {repetition}, token {i}, logits"),
            );
            assert_eq!(mtp.position(), gold.positions[i] + 1);
        }
    }
}

#[test]
#[ignore = "requires a real Apple Metal GPU"]
fn hidden_only_mtp_advances_cache_and_leaves_target_state_independent() {
    let gold = golden();
    let mut target = Engine::load(&fixture("tiny-q4"), 8).unwrap();
    let mut mtp = Mtp::load(&target, &fixture("tiny-mtp"), 8).unwrap();
    let target_gold: Value =
        serde_json::from_slice(&std::fs::read(fixture("tiny-q4").join("golden.json")).unwrap())
            .unwrap();
    for (i, &token) in gold.tokens.iter().enumerate() {
        // Both engines use the same immutable embedding/head while maintaining
        // separate temporaries and attention caches.
        let target_logits = target.forward(token).unwrap();
        let expected_target: Vec<f32> = target_gold["logits"][i]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap() as f32)
            .collect();
        assert_close(
            &target_logits,
            &expected_target,
            "target interleaved with MTP",
        );
        let output_logits = i % 2 == 1;
        let got = mtp
            .forward(token, &gold.input_hidden[i], output_logits)
            .unwrap();
        assert_close(&got.hidden, &gold.hidden[i], "hidden-only MTP");
        if output_logits {
            assert_close(
                &got.logits,
                &gold.logits[i],
                "MTP logits after hidden-only step",
            );
        } else {
            assert!(
                got.logits.is_empty(),
                "hidden-only call returned stale logits"
            );
        }
        assert_eq!(target.position(), i + 1);
        assert_eq!(mtp.position(), i + 1);
    }
    mtp.reset();
    assert_eq!(target.position(), gold.tokens.len());
    target.reset();
    let got = mtp
        .forward(gold.tokens[0], &gold.input_hidden[0], true)
        .unwrap();
    assert_close(&got.logits, &gold.logits[0], "reset MTP with reused target");
    assert_eq!(target.position(), 0);
}

#[test]
#[ignore = "requires a real Apple Metal GPU"]
fn truncated_mtp_overwrites_rejected_suffix_and_matches_fresh_prefix() {
    let gold = golden();
    let target = Engine::load(&fixture("tiny-q4"), 8).unwrap();
    let mut reused = Mtp::load(&target, &fixture("tiny-mtp"), 8).unwrap();
    let mut fresh = Mtp::load(&target, &fixture("tiny-mtp"), 8).unwrap();
    for prefix in 0..=gold.tokens.len() {
        reused.reset();
        fresh.reset();
        for (i, &token) in gold.tokens.iter().enumerate() {
            reused.forward(token, &gold.input_hidden[i], false).unwrap();
            if i < prefix {
                fresh.forward(token, &gold.input_hidden[i], false).unwrap();
            }
        }
        reused.truncate(prefix).unwrap();
        assert_eq!(reused.position(), prefix);
        // Different tokens and hidden inputs ensure the old suffix cannot
        // accidentally satisfy the comparison through replaying identical data.
        for step in 0..2 {
            let source = (prefix + step + 2) % gold.tokens.len();
            let token = (gold.tokens[source] + 13) % target.config().vocab_size as u32;
            let alternate_hidden: Vec<f32> = gold.input_hidden[source]
                .iter()
                .enumerate()
                .map(|(i, value)| if i % 2 == 0 { -*value } else { *value })
                .collect();
            let actual = reused.forward(token, &alternate_hidden, true).unwrap();
            let expected = fresh.forward(token, &alternate_hidden, true).unwrap();
            assert_output(&actual, &expected, &format!("prefix {prefix}, step {step}"));
            assert_eq!(reused.position(), prefix + step + 1);
        }
    }
}

#[test]
#[ignore = "requires a real Apple Metal GPU"]
fn invalid_mtp_inputs_preserve_cache_and_capacity_boundary() {
    let gold = golden();
    let target = Engine::load(&fixture("tiny-q4"), 8).unwrap();
    let mut mtp = Mtp::load(&target, &fixture("tiny-mtp"), gold.tokens.len()).unwrap();
    mtp.forward(gold.tokens[0], &gold.input_hidden[0], true)
        .unwrap();
    assert!(
        mtp.forward(
            target.config().vocab_size as u32,
            &gold.input_hidden[1],
            true
        )
        .is_err()
    );
    assert_eq!(mtp.position(), 1);
    assert!(
        mtp.forward(gold.tokens[1], &gold.input_hidden[1][1..], true)
            .is_err()
    );
    assert_eq!(mtp.position(), 1);
    let mut too_long = gold.input_hidden[1].clone();
    too_long.push(0.0);
    assert!(mtp.forward(gold.tokens[1], &too_long, true).is_err());
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut invalid_hidden = gold.input_hidden[1].clone();
        invalid_hidden[7] = invalid;
        assert!(mtp.forward(gold.tokens[1], &invalid_hidden, true).is_err());
        assert_eq!(mtp.position(), 1);
    }
    assert!(mtp.truncate(2).is_err());
    assert_eq!(mtp.position(), 1);
    for i in 1..gold.tokens.len() {
        let got = mtp
            .forward(gold.tokens[i], &gold.input_hidden[i], true)
            .unwrap();
        assert_close(&got.hidden, &gold.hidden[i], "after invalid input hidden");
        assert_close(&got.logits, &gold.logits[i], "after invalid input logits");
    }
    assert!(
        mtp.forward(gold.tokens[0], &gold.input_hidden[0], true)
            .is_err()
    );
    assert_eq!(mtp.position(), gold.tokens.len());
    mtp.truncate(1).unwrap();
    let replay = mtp
        .forward(gold.tokens[1], &gold.input_hidden[1], true)
        .unwrap();
    assert_close(
        &replay.logits,
        &gold.logits[1],
        "replay after full capacity",
    );
    assert!(Mtp::load(&target, &fixture("tiny-mtp"), 0).is_err());
    assert!(Mtp::load(&target, &fixture("tiny-mtp"), 9).is_err());
}

#[test]
#[ignore = "requires a real Apple Metal GPU"]
fn mtp_loader_rejects_incompatible_adapter_configurations() {
    let target = Engine::load(&fixture("tiny-q4"), 8).unwrap();
    let original: Value =
        serde_json::from_slice(&std::fs::read(fixture("tiny-mtp").join("config.json")).unwrap())
            .unwrap();
    let temporary = tempfile::tempdir().unwrap();
    std::fs::copy(
        fixture("tiny-mtp").join("model.safetensors"),
        temporary.path().join("model.safetensors"),
    )
    .unwrap();
    for (pointer, bad_value) in [
        ("/model_type", json!("qwen3_5")),
        ("/block_size", json!(1)),
        ("/text_config/mtp_num_hidden_layers", json!(2)),
        ("/text_config/mtp_use_dedicated_embeddings", json!(true)),
        ("/text_config/hidden_size", json!(64)),
        ("/text_config/vocab_size", json!(128)),
        ("/text_config/head_dim", json!(16)),
        ("/text_config/rms_norm_eps", json!(0.00001)),
        ("/text_config/rope_parameters/rope_theta", json!(20000.0)),
    ] {
        let mut config = original.clone();
        *config.pointer_mut(pointer).unwrap() = bad_value;
        std::fs::write(
            temporary.path().join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        assert!(
            Mtp::load(&target, temporary.path(), 8).is_err(),
            "incompatible adapter {pointer} was accepted"
        );
        assert_eq!(target.position(), 0);
    }
    // A failed adapter load must not prevent loading the original sanitized
    // adapter. Its multiplicative norm convention is checked by the full oracle.
    let mut valid = Mtp::load(&target, &fixture("tiny-mtp"), 8).unwrap();
    let gold = golden();
    let got = valid
        .forward(gold.tokens[0], &gold.input_hidden[0], true)
        .unwrap();
    assert_close(
        &got.logits,
        &gold.logits[0],
        "sanitized adapter after rejected loads",
    );
}
