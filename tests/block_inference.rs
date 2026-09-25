//! A block must preserve causal target computation and transactional state.
use anyhow::Result;
use qwen_metal::{chat::Sampler, engine::Engine};
use std::path::{Path, PathBuf};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn near(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    for (i, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        assert!(
            got.is_finite() && (got - want).abs() < 3e-3 + 3e-3 * want.abs(),
            "{label}, element {i}: {got} != {want}"
        );
    }
}

#[test]
#[ignore = "requires a real Apple Metal GPU"]
fn wider_key_dimension_fallback_preserves_every_rollback_prefix() -> Result<()> {
    // Exercise the K>128 fallback with a real hybrid checkpoint, including
    // nonzero added channels. A dispatch-only test would miss bad engine views.
    let source = fixture("tiny");
    let temporary = tempfile::tempdir()?;
    let mut config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(source.join("config.json"))?)?;
    config["text_config"]["linear_key_head_dim"] = serde_json::json!(160);
    std::fs::write(
        temporary.path().join("config.json"),
        serde_json::to_vec(&config)?,
    )?;
    let original = std::fs::read(source.join("model.safetensors"))?;
    let header_size = u64::from_le_bytes(original[..8].try_into().unwrap()) as usize;
    let mut header: serde_json::Value = serde_json::from_slice(&original[8..8 + header_size])?;
    let original_data = &original[8 + header_size..];
    let mut data = Vec::new();
    for (name, tensor) in header.as_object_mut().unwrap() {
        if name == "__metadata__" {
            continue;
        }
        let start = data.len();
        if name.contains("linear_attn.in_proj_qkv") || name.contains("linear_attn.conv1d") {
            tensor["shape"][0] = serde_json::json!(352); // 2 * KH1 * K160 + VH2 * V16
            let elements: usize = tensor["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|n| n.as_u64().unwrap() as usize)
                .product();
            for i in 0..elements {
                let value = ((i * 17 % 41) as f32 - 20.) / 256.;
                data.extend_from_slice(&half::f16::from_f32(value).to_bits().to_le_bytes());
            }
        } else {
            let range = tensor["data_offsets"].as_array().unwrap();
            data.extend_from_slice(
                &original_data
                    [range[0].as_u64().unwrap() as usize..range[1].as_u64().unwrap() as usize],
            );
        }
        tensor["data_offsets"] = serde_json::json!([start, data.len()]);
    }
    let mut encoded_header = serde_json::to_vec(&header)?;
    encoded_header.resize(encoded_header.len().div_ceil(8) * 8, b' ');
    let mut checkpoint = (encoded_header.len() as u64).to_le_bytes().to_vec();
    checkpoint.extend(encoded_header);
    checkpoint.extend(data);
    std::fs::write(temporary.path().join("model.safetensors"), checkpoint)?;
    let mut sequential = Engine::load(temporary.path(), 32)?;
    let mut block = Engine::load(temporary.path(), 32)?;
    let history = [5, 9, 13];
    let candidates = [7, 21, 4, 30];
    for width in 1..=4 {
        for keep in 0..=width {
            sequential.reset();
            block.reset();
            for token in history {
                sequential.forward(token)?;
                block.forward(token)?;
            }
            let output = block.verify_block(&candidates[..width])?;
            for (i, &token) in candidates[..width].iter().enumerate() {
                near(
                    &output.logits[i],
                    &sequential.forward(token)?,
                    "wide-key block",
                );
            }
            block.commit_block_prefix(keep)?;
            sequential.reset();
            for token in history
                .into_iter()
                .chain(candidates[..keep].iter().copied())
            {
                sequential.forward(token)?;
            }
            for token in [31, 11, 19] {
                near(
                    &block.forward(token)?,
                    &sequential.forward(token)?,
                    "wide-key rollback",
                );
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires a real Apple Metal GPU"]
fn block_widths_match_independent_logits_and_sequential_normalized_hidden() -> Result<()> {
    for name in ["tiny", "tiny-q4", "tiny-bf16"] {
        let path = fixture(name);
        let gold: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path.join("golden.json"))?)?;
        let tokens: Vec<u32> = gold["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let expected: Vec<Vec<f32>> = gold["logits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| {
                row.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap() as f32)
                    .collect()
            })
            .collect();
        let mut sequential = Engine::load(&path, 32)?;
        let mut block = Engine::load(&path, 32)?;
        for width in 1..=4 {
            sequential.reset();
            block.reset();
            for (chunk_index, chunk) in tokens.chunks(width).enumerate() {
                let base = chunk_index * width;
                let verified = block.verify_block(chunk)?;
                assert_eq!(verified.base_position, base);
                assert_eq!(block.position(), base + chunk.len());
                assert_eq!(verified.logits.len(), chunk.len());
                assert_eq!(verified.hidden.len(), chunk.len());
                for (i, &token) in chunk.iter().enumerate() {
                    let (logits, hidden) = sequential.forward_with_hidden(token, true)?;
                    near(
                        &verified.logits[i],
                        &expected[base + i],
                        "independent logits",
                    );
                    near(&verified.logits[i], &logits, "sequential logits");
                    assert_eq!(
                        Sampler::new(0).sample(&verified.logits[i], 0., 1., 0)?,
                        Sampler::new(0).sample(&logits, 0., 1., 0)?,
                        "greedy target decision must match for every verified position"
                    );
                    near(&verified.hidden[i], &hidden, "normalized hidden");
                }
                block.commit_block_prefix(chunk.len())?;
            }
        }
        sequential.reset();
        let (_, with_logits) = sequential.forward_with_hidden(tokens[0], true)?;
        sequential.reset();
        let (without_logits, hidden_only) = sequential.forward_with_hidden(tokens[0], false)?;
        assert!(without_logits.is_empty());
        near(&hidden_only, &with_logits, "hidden-only final norm");
    }
    Ok(())
}

#[test]
#[ignore = "requires a real Apple Metal GPU"]
fn every_block_rollback_prefix_matches_fresh_causal_continuation() -> Result<()> {
    for name in ["tiny", "tiny-q4", "tiny-bf16"] {
        let path = fixture(name);
        let mut reference = Engine::load(&path, 64)?;
        let mut block = Engine::load(&path, 64)?;
        for width in 1..=4 {
            for keep in 0..=width {
                reference.reset();
                block.reset();
                for token in [5, 9, 13] {
                    reference.forward(token)?;
                    block.forward(token)?;
                }
                let candidates = [7, 21, 4, 30];
                let output = block.verify_block(&candidates[..width])?;
                assert_eq!(output.base_position, 3);
                assert!(
                    block.forward(2).is_err(),
                    "pending block must forbid normal forward"
                );
                assert!(
                    block.verify_block(&[2]).is_err(),
                    "pending block must forbid another verify"
                );
                assert!(block.commit_block_prefix(width + 1).is_err());
                assert_eq!(
                    block.position(),
                    3 + width,
                    "validation error must preserve transaction"
                );
                block.commit_block_prefix(keep)?;
                assert_eq!(block.position(), 3 + keep);
                assert!(
                    block.commit_block_prefix(0).is_err(),
                    "commit consumes transaction"
                );
                for &token in &candidates[..keep] {
                    reference.forward(token)?;
                }
                for token in [31, 11, 19] {
                    near(
                        &block.forward(token)?,
                        &reference.forward(token)?,
                        "rollback continuation",
                    );
                }
                // A second transaction catches accidental retention of the first one's snapshots.
                block.verify_block(&[8, 17, 22])?;
                block.commit_block_prefix(1)?;
                reference.forward(8)?;
                near(
                    &block.forward(14)?,
                    &reference.forward(14)?,
                    "second rollback",
                );
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires a real Apple Metal GPU"]
fn block_validation_and_reset_preserve_state_without_partial_mutation() -> Result<()> {
    let path = fixture("tiny-q4");
    let mut engine = Engine::load(&path, 8)?;
    let mut reference = Engine::load(&path, 8)?;
    engine.forward(5)?;
    reference.forward(5)?;
    for invalid in [vec![], vec![1, 2, 3, 4, 5], vec![6, 64], vec![u32::MAX]] {
        assert!(engine.verify_block(&invalid).is_err());
        assert_eq!(engine.position(), 1);
    }
    near(
        &engine.forward(9)?,
        &reference.forward(9)?,
        "validation preserves state",
    );
    engine.verify_block(&[2, 3, 4])?;
    assert!(engine.forward_with_hidden(7, true).is_err());
    assert!(engine.prefill_token(7, false).is_err());
    engine.reset();
    reference.reset();
    assert_eq!(engine.position(), 0);
    assert!(engine.commit_block_prefix(1).is_err());
    near(
        &engine.forward(11)?,
        &reference.forward(11)?,
        "reset pending block",
    );
    for _ in 0..5 {
        engine.forward(1)?;
    }
    assert_eq!(engine.position(), 6);
    assert!(engine.verify_block(&[2, 3, 4]).is_err());
    assert_eq!(engine.position(), 6);
    engine.verify_block(&[2, 3])?;
    engine.commit_block_prefix(2)?;
    assert_eq!(engine.position(), 8);
    assert!(engine.verify_block(&[4]).is_err());
    Ok(())
}

#[test]
#[ignore = "requires a real Apple Metal GPU"]
fn block_causality_and_rollback_cross_attention_history_threshold() -> Result<()> {
    let source = fixture("tiny-q4");
    let mut config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(source.join("config.json"))?)?;
    config["text_config"]["max_position_embeddings"] = serde_json::json!(160);
    let temporary = tempfile::tempdir()?;
    std::fs::write(
        temporary.path().join("config.json"),
        serde_json::to_vec(&config)?,
    )?;
    std::fs::copy(
        source.join("model.safetensors"),
        temporary.path().join("model.safetensors"),
    )?;
    let mut reference = Engine::load(temporary.path(), 160)?;
    let mut block = Engine::load(temporary.path(), 160)?;
    reference.set_parallel_attention(true);
    block.set_parallel_attention(true);
    for i in 0..126 {
        let token = (3 + i * 7 % 59) as u32;
        reference.forward(token)?;
        block.forward(token)?;
    }
    let tokens = [8, 27, 16, 41];
    let result = block.verify_block(&tokens)?;
    for (i, token) in tokens.iter().copied().enumerate() {
        near(
            &result.logits[i],
            &reference.forward(token)?,
            "history threshold",
        );
    }
    block.commit_block_prefix(2)?;
    reference.reset();
    for i in 0..126 {
        reference.forward((3 + i * 7 % 59) as u32)?;
    }
    for token in tokens[..2].iter().copied() {
        reference.forward(token)?;
    }
    for token in [4, 19, 32] {
        near(
            &block.forward(token)?,
            &reference.forward(token)?,
            "threshold rollback",
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires a real Apple Metal GPU"]
fn unsupported_block_quantization_keeps_existing_sequential_history() -> Result<()> {
    let source = fixture("tiny-q4");
    let temporary = tempfile::tempdir()?;
    let mut config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(source.join("config.json"))?)?;
    assert_eq!(config["quantization"]["group_size"], 32);
    config["quantization"]["group_size"] = serde_json::json!(16);
    std::fs::write(
        temporary.path().join("config.json"),
        serde_json::to_vec(&config)?,
    )?;
    // A group32 scale/bias duplicated into two group16 entries represents the
    // exact same packed weights, but exercises sequential-only quantization.
    let bytes = std::fs::read(source.join("model.safetensors"))?;
    let header_length = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let mut header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + header_length])?;
    let payload = &bytes[8 + header_length..];
    let mut rewritten = Vec::new();
    for (name, descriptor) in header.as_object_mut().unwrap() {
        if name == "__metadata__" {
            continue;
        }
        let start = descriptor["data_offsets"][0].as_u64().unwrap() as usize;
        let end = descriptor["data_offsets"][1].as_u64().unwrap() as usize;
        let output_start = rewritten.len();
        if name.ends_with(".scales") || name.ends_with(".biases") {
            let item_bytes = match descriptor["dtype"].as_str().unwrap() {
                "F32" => 4,
                "BF16" | "F16" => 2,
                dtype => panic!("unexpected affine metadata type {dtype}"),
            };
            for value in payload[start..end].chunks_exact(item_bytes) {
                rewritten.extend_from_slice(value);
                rewritten.extend_from_slice(value);
            }
            let shape = descriptor["shape"].as_array_mut().unwrap();
            let last = shape.last_mut().unwrap();
            *last = serde_json::json!(last.as_u64().unwrap() * 2);
        } else {
            rewritten.extend_from_slice(&payload[start..end]);
        }
        descriptor["data_offsets"] = serde_json::json!([output_start, rewritten.len()]);
    }
    let mut encoded = serde_json::to_vec(&header)?;
    while encoded.len() % 8 != 0 {
        encoded.push(b' ');
    }
    let mut transformed = (encoded.len() as u64).to_le_bytes().to_vec();
    transformed.extend_from_slice(&encoded);
    transformed.extend_from_slice(&rewritten);
    std::fs::write(temporary.path().join("model.safetensors"), transformed)?;

    let mut group16 = Engine::load(temporary.path(), 32)?;
    let mut reference = Engine::load(&source, 32)?;
    for token in [5, 8, 13] {
        near(
            &group16.forward(token)?,
            &reference.forward(token)?,
            "group16 equivalent history",
        );
    }
    let before_bytes = group16.allocated_bytes();
    let error = group16.verify_block(&[3, 7]).unwrap_err();
    assert!(error.to_string().contains("groups 32, 64 or 128"));
    assert_eq!(group16.position(), 3);
    assert_eq!(
        group16.allocated_bytes(),
        before_bytes,
        "unsupported block must not allocate scratch"
    );
    for token in [9, 17, 21] {
        near(
            &group16.forward(token)?,
            &reference.forward(token)?,
            "unsupported block preserves history",
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires a real Apple Metal GPU"]
fn reference_block_preflight_preserves_history() -> Result<()> {
    // A separate test process isolates the environment-based runtime mode from
    // other GPU tests; changing process environment in a test thread is unsafe.
    const CHILD: &str = "QWEN_BLOCK_REFERENCE_TEST_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let status = std::process::Command::new(std::env::current_exe()?)
            .args([
                "--exact",
                "reference_block_preflight_preserves_history",
                "--include-ignored",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(CHILD, "1")
            .env("QWEN_METAL_REFERENCE", "1")
            .status()?;
        assert!(
            status.success(),
            "isolated reference-mode regression failed"
        );
        return Ok(());
    }
    let mut engine = Engine::load(&fixture("tiny-q4"), 32)?;
    let mut reference = Engine::load(&fixture("tiny-q4"), 32)?;
    assert_eq!(engine.kernel_mode(), "reference");
    for token in [4, 7, 13] {
        engine.forward(token)?;
        reference.forward(token)?;
    }
    let before_bytes = engine.allocated_bytes();
    assert!(
        engine
            .verify_block(&[8, 9])
            .unwrap_err()
            .to_string()
            .contains("reference kernels")
    );
    assert_eq!(engine.position(), 3);
    assert_eq!(engine.allocated_bytes(), before_bytes);
    for token in [11, 16] {
        near(
            &engine.forward(token)?,
            &reference.forward(token)?,
            "reference-mode block rejection",
        );
    }
    Ok(())
}
