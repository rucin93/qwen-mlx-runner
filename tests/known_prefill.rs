//! The optional known-prompt path must preserve target/MTP continuation.
use anyhow::Result;
use qwen_metal::{
    chat::{GenerationRequest, Message},
    engine::Engine,
    gpu::{BlockKernelMode, BlockMatmulMode},
    mtp_chat::MtpChatEngine,
};
use std::path::{Path, PathBuf};
fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}
fn equal(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && (*a - *b).abs() <= 3e-4 + 3e-4 * b.abs(),
            "element {i}: {a} vs {b}"
        );
    }
}
#[test]
#[ignore = "requires a real Apple Metal GPU"]
fn known_prompt_all_widths_tails_history_and_decode_match_existing_blocks() -> Result<()> {
    for name in ["tiny", "tiny-q4", "tiny-bf16"] {
        let mut baseline = Engine::load(&fixture(name), 96)?;
        let mut candidate = Engine::load(&fixture(name), 96)?;
        for batched_delta in [false, true] {
            let mode = BlockKernelMode {
                matmul: BlockMatmulMode::Legacy,
                batched_delta,
            };
            baseline.set_block_kernel_mode(mode)?;
            candidate.set_block_kernel_mode(mode)?;
            for width in 1..=16 {
                for tail in 1..=width {
                    baseline.reset();
                    candidate.reset();
                    for token in [5, 9] {
                        baseline.forward(token)?;
                        candidate.forward(token)?;
                    }
                    let tokens: Vec<_> = (0..width + tail)
                        .map(|i| ((i * 13 + 7) % 64) as u32)
                        .collect();
                    let mut hidden = vec![];
                    let mut logits = vec![];
                    for chunk in tokens.chunks(3) {
                        let out = baseline.verify_block(chunk)?;
                        baseline.commit_block_prefix(chunk.len())?;
                        hidden.extend(out.hidden);
                        logits.extend(out.logits);
                    }
                    let mut offset = 0;
                    for chunk in tokens.chunks(width) {
                        let final_chunk = offset + chunk.len() == tokens.len();
                        let out = candidate.prefill_known_block(chunk, final_chunk)?;
                        assert_eq!(out.base_position, offset + 2);
                        for (i, row) in out.hidden.iter().enumerate() {
                            equal(row, &hidden[offset + i]);
                        }
                        if final_chunk {
                            for (i, row) in out.logits.iter().enumerate() {
                                equal(row, &logits[offset + i]);
                            }
                        } else {
                            assert!(out.logits.is_empty());
                        }
                        assert!(candidate.commit_block_prefix(chunk.len()).is_err());
                        offset += chunk.len();
                    }
                    for token in [31, 11, 23] {
                        equal(&candidate.forward(token)?, &baseline.forward(token)?);
                    }
                }
            }
            baseline.reset();
            candidate.reset();
        }
    }
    Ok(())
}
#[test]
#[ignore = "requires a real Apple Metal GPU"]
fn known_prompt_rejections_preserve_history_and_verification_limit() -> Result<()> {
    let mut baseline = Engine::load(&fixture("tiny-q4"), 20)?;
    let mut candidate = Engine::load(&fixture("tiny-q4"), 20)?;
    baseline.forward(5)?;
    candidate.forward(5)?;
    for tokens in [vec![], vec![1; 17], vec![64], vec![1; 20]] {
        assert!(candidate.prefill_known_block(&tokens, false).is_err());
        assert_eq!(candidate.position(), 1);
    }
    candidate.verify_block(&[9])?;
    assert!(candidate.prefill_known_block(&[7; 8], false).is_err());
    candidate.commit_block_prefix(0)?;
    assert!(candidate.verify_block(&[7; 8]).is_err());
    equal(&candidate.forward(21)?, &baseline.forward(21)?);
    candidate.reset();
    baseline.reset();
    equal(
        &candidate.prefill_known_block(&[9], true)?.logits[0],
        &baseline.forward(9)?,
    );
    Ok(())
}
#[test]
#[ignore = "requires a real Apple Metal GPU"]
fn wide_mtp_prefill_cancellation_reset_and_ids_match_default() -> Result<()> {
    let request = GenerationRequest {
        messages: vec![Message {
            role: "user".into(),
            content: (6..45)
                .map(|i| format!("w{i}"))
                .collect::<Vec<_>>()
                .join(" "),
            ..Default::default()
        }],
        max_tokens: 8,
        temperature: 0.,
        top_p: 1.,
        seed: 42,
        ..Default::default()
    };
    let mut engine = MtpChatEngine::load(&fixture("tiny-q4"), &fixture("tiny-mtp"), 128, 3)?;
    assert_eq!(engine.prefill_batch_size(), 3);
    let (baseline, stats) = engine.generate_with_stats(&request, &mut |_| true)?;
    for width in [8, 16, 3] {
        engine.set_prefill_batch_size(width)?;
        assert_eq!(engine.prefill_batch_size(), width);
        for invalid in [0, 1, 2, 4, 7, 9, 17] {
            assert!(engine.set_prefill_batch_size(invalid).is_err());
            assert_eq!(engine.prefill_batch_size(), width);
        }
        for cancel_at in [1, 2, 7, 18] {
            let mut count = 0;
            assert!(
                engine
                    .generate_with_stats(&request, &mut |s| {
                        assert!(s.is_empty());
                        count += 1;
                        count < cancel_at
                    })
                    .unwrap_err()
                    .to_string()
                    .contains("cancelled during prefill")
            );
            assert_eq!(engine.engine().position(), 0);
            let (actual, actual_stats) = engine.generate_with_stats(&request, &mut |_| true)?;
            assert_eq!(actual.text, baseline.text);
            assert_eq!(actual.finish_reason, baseline.finish_reason);
            assert_eq!(actual_stats.token_ids, stats.token_ids);
        }
    }
    Ok(())
}
#[test]
#[ignore = "requires a real Apple Metal GPU"]
fn known_prompt_profile_contains_no_snapshot_copies() -> Result<()> {
    let mut engine = Engine::load(&fixture("tiny-q4"), 32)?;
    engine.enable_command_profiling()?;
    for batched_delta in [false, true] {
        engine.reset();
        engine.set_block_kernel_mode(BlockKernelMode {
            matmul: BlockMatmulMode::Legacy,
            batched_delta,
        })?;
        engine.prefill_known_block(&[5; 16], false)?;
        let rows = engine.profile_report()?;
        assert!(
            rows.iter()
                .all(|r| r.kernel != "copy_f32" && r.kernel != "delta_step_block")
        );
        assert_eq!(
            rows.iter().any(|r| r.kernel == "delta_step_prefill"),
            batched_delta
        );
    }
    Ok(())
}

// A runtime fixture with 512-wide projections exercises wide kernels inside
// the complete causal graph, alongside dense DeltaNet output projections.
fn aligned_mixed_fixture() -> Result<tempfile::TempDir> {
    let source = fixture("tiny");
    let temporary = tempfile::tempdir()?;
    let mut config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(source.join("config.json"))?)?;
    config["text_config"]["hidden_size"] = serde_json::json!(512);
    config["text_config"]["intermediate_size"] = serde_json::json!(512);
    config["quantization"] = serde_json::json!({"bits":4,"group_size":64,"mode":"affine"});
    std::fs::write(
        temporary.path().join("config.json"),
        serde_json::to_vec(&config)?,
    )?;
    let original = std::fs::read(source.join("model.safetensors"))?;
    let header_length = u64::from_le_bytes(original[..8].try_into().unwrap()) as usize;
    let original_header: serde_json::Value =
        serde_json::from_slice(&original[8..8 + header_length])?;
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    let mut add = |name: String, shape: Vec<usize>, dtype: &str, bytes: Vec<u8>| {
        let start = data.len();
        data.extend(bytes);
        header.insert(
            name,
            serde_json::json!({"dtype":dtype,"shape":shape,"data_offsets":[start,data.len()]}),
        );
    };
    for (name, tensor) in original_header.as_object().unwrap() {
        if name == "__metadata__" {
            continue;
        }
        let mut shape: Vec<usize> = tensor["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        if shape.len() == 2 {
            if name.contains("mlp.") {
                shape = vec![512, 512];
            } else if name.contains("out_proj") || name.contains("o_proj") {
                shape[0] = 512;
            } else {
                shape[1] = 512;
            }
            let (rows, cols) = (shape[0], shape[1]);
            if cols % 64 == 0 {
                let offset = name.bytes().map(u32::from).sum::<u32>();
                let packed: Vec<u32> = (0..rows * cols / 8)
                    .map(|i| (i as u32).wrapping_mul(0x9e3779b9).wrapping_add(offset))
                    .collect();
                add(
                    name.clone(),
                    vec![rows, cols / 8],
                    "U32",
                    bytemuck::cast_slice(&packed).to_vec(),
                );
                let base = name.trim_end_matches(".weight");
                let scales = vec![(0.001953125f32.to_bits() >> 16) as u16; rows * cols / 64];
                let biases = vec![((-0.015625f32).to_bits() >> 16) as u16; scales.len()];
                add(
                    format!("{base}.scales"),
                    vec![rows, cols / 64],
                    "BF16",
                    bytemuck::cast_slice(&scales).to_vec(),
                );
                add(
                    format!("{base}.biases"),
                    vec![rows, cols / 64],
                    "BF16",
                    bytemuck::cast_slice(&biases).to_vec(),
                );
            } else {
                let dense: Vec<u16> = (0..rows * cols)
                    .map(|i| half::f16::from_f32(((i * 17 % 41) as f32 - 20.) / 4096.).to_bits())
                    .collect();
                add(
                    name.clone(),
                    shape,
                    "F16",
                    bytemuck::cast_slice(&dense).to_vec(),
                );
            }
        } else {
            if name.ends_with("model.norm.weight")
                || name.contains("input_layernorm")
                || name.contains("post_attention_layernorm")
            {
                shape[0] = 512;
            }
            let start = tensor["data_offsets"][0].as_u64().unwrap() as usize;
            let end = tensor["data_offsets"][1].as_u64().unwrap() as usize;
            let old = &original[8 + header_length + start..8 + header_length + end];
            let bytes: Vec<u8> = (0..shape.iter().product::<usize>() * 2)
                .map(|i| old[i % old.len()])
                .collect();
            add(name.clone(), shape, "F16", bytes);
        }
    }
    let mut encoded = serde_json::to_vec(&header)?;
    encoded.resize(encoded.len().div_ceil(8) * 8, b' ');
    let mut checkpoint = (encoded.len() as u64).to_le_bytes().to_vec();
    checkpoint.extend(encoded);
    checkpoint.extend(data);
    std::fs::write(temporary.path().join("model.safetensors"), checkpoint)?;
    Ok(temporary)
}
#[test]
#[ignore = "requires a real Apple Metal GPU"]
fn aligned_mixed_graph_preserves_hidden_logits_and_continuation() -> Result<()> {
    let fixture = aligned_mixed_fixture()?;
    let mut baseline = Engine::load(fixture.path(), 64)?;
    let mut candidate = Engine::load(fixture.path(), 64)?;
    candidate.enable_command_profiling()?;
    for width in [8, 16] {
        for tail in [1, 5, 7] {
            baseline.reset();
            candidate.reset();
            for token in [5, 9] {
                baseline.forward(token)?;
                candidate.forward(token)?;
            }
            let tokens: Vec<_> = (0..width + tail)
                .map(|i| ((i * 13 + 7) % 64) as u32)
                .collect();
            let mut expected_hidden = vec![];
            let mut expected_logits = vec![];
            for chunk in tokens.chunks(3) {
                let out = baseline.verify_block(chunk)?;
                baseline.commit_block_prefix(chunk.len())?;
                expected_hidden.extend(out.hidden);
                expected_logits.extend(out.logits);
            }
            let mut offset = 0;
            for chunk in tokens.chunks(width) {
                let out = candidate.prefill_known_block(chunk, true)?;
                for (index, row) in out.hidden.iter().enumerate() {
                    equal(row, &expected_hidden[offset + index]);
                }
                for (index, row) in out.logits.iter().enumerate() {
                    equal(row, &expected_logits[offset + index]);
                }
                offset += chunk.len();
            }
            for token in [31, 11, 23] {
                equal(&candidate.forward(token)?, &baseline.forward(token)?);
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires a real Apple Metal GPU; synthetic causal graph comparison"]
fn grouped_prompt_graph_microbenchmark() -> Result<()> {
    use std::time::Instant;
    let fixture = aligned_mixed_fixture()?;
    let mut engine = Engine::load(fixture.path(), 64)?;
    let prompt: Vec<_> = (0..48).map(|i| ((i * 13 + 7) % 64) as u32).collect();
    let measure = |engine: &mut Engine, width: usize| -> Result<f64> {
        engine.reset();
        let started = Instant::now();
        let mut consumed = 0;
        for chunk in prompt.chunks(width) {
            consumed += chunk.len();
            if width == 3 {
                engine.prefill_block(chunk, consumed == prompt.len())?;
            } else {
                engine.prefill_known_block(chunk, consumed == prompt.len())?;
            }
        }
        Ok(started.elapsed().as_secs_f64())
    };
    for _ in 0..3 {
        for width in [3, 8, 16] {
            measure(&mut engine, width)?;
        }
    }
    let mut samples: [Vec<f64>; 3] = std::array::from_fn(|_| vec![]);
    for iteration in 0..20 {
        for i in 0..3 {
            let slot = (i + iteration) % 3;
            samples[slot].push(measure(&mut engine, [3, 8, 16][slot])?);
        }
    }
    let median = |v: &[f64]| {
        let mut v = v.to_vec();
        v.sort_by(f64::total_cmp);
        (v[9] + v[10]) * 500.
    };
    println!(
        "{}",
        serde_json::json!({"kind":"grouped_known_prompt_graph_microbenchmark","device":engine.device_name(),"prompt_tokens":48,"layers":4,"hidden_size":512,"mixed_dense_q4_bf16":true,"matrix_batch":3,"samples_per_mode":20,"prompt_batch_sizes":[3,8,16],"wall_median_ms":[median(&samples[0]),median(&samples[1]),median(&samples[2])],"note":"Temporary synthetic untrained checkpoint with512-wide projections; existing transactionalB3 versus snapshot-free groupedknownprompts. Not fullmodel or M5 throughput."})
    );
    Ok(())
}
