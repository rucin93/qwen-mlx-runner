//! Independent FP64 attention-value sums exercise GQA indexing and reductions.
use anyhow::Result;
use qwen_metal::gpu::Gpu;

const GUARD: f32 = 1_234_567.25;
const GUARD_LEN: usize = 19;

fn check_case(
    gpus: &[Gpu],
    heads: usize,
    kv_heads: usize,
    dim: usize,
    history: usize,
    zero_values: bool,
    alias_gate: bool,
) -> Result<()> {
    // Each head has a different, nonuniform probability distribution. The
    // oracle uses the uploaded FP32 probabilities, without renormalizing them.
    let mut scores = Vec::with_capacity(heads * history);
    for h in 0..heads {
        let raw: Vec<f64> = (0..history)
            .map(|t| (1 + (t * 17 + h * 23 + t * h) % 97) as f64)
            .collect();
        let sum: f64 = raw.iter().sum();
        scores.extend(raw.into_iter().map(|v| (v / sum) as f32));
    }
    // Distinct token, KV head, and channel coordinates expose swapped strides.
    let values: Vec<f32> = (0..history)
        .flat_map(|t| {
            (0..kv_heads).flat_map(move |kh| {
                (0..dim).map(move |d| {
                    if zero_values {
                        0.
                    } else {
                        ((t * 29 + kh * 37 + d * 11) % 251) as f32 / 31. - 4. + kh as f32 * 0.037
                    }
                })
            })
        })
        .collect();
    let gate_cases = [-80., -20., -2.5, 0., 2.5, 20., 80.];
    let gates: Vec<f32> = (0..heads * dim)
        .map(|i| gate_cases[i % gate_cases.len()])
        .collect();
    let expected: Vec<f64> = (0..heads)
        .flat_map(|h| (0..dim).map(move |d| (h, d)))
        .map(|(h, d)| {
            let kh = h / (heads / kv_heads);
            let weighted: f64 = (0..history)
                .map(|t| {
                    f64::from(scores[h * history + t])
                        * f64::from(values[(t * kv_heads + kh) * dim + d])
                })
                .sum();
            weighted / (1. + (-f64::from(gates[h * dim + d])).exp())
        })
        .collect();

    for gpu in gpus {
        let score_buffer = gpu.upload_f32(&scores)?;
        let value_buffer = gpu.upload_f32(&values)?;
        let mut guarded_gates = gates.clone();
        guarded_gates.extend([GUARD; GUARD_LEN]);
        let gate_buffer = gpu.upload_f32(&guarded_gates)?;
        let output = if alias_gate {
            gate_buffer.clone()
        } else {
            gpu.upload_f32(&vec![GUARD; heads * dim + GUARD_LEN])?
        };
        let command = gpu.begin();
        let encoder = command.new_compute_command_encoder();
        gpu.encode(
            encoder,
            "attn_values",
            &[&score_buffer, &value_buffer, &gate_buffer, &output],
            &[heads as u32, kv_heads as u32, dim as u32, history as u32],
            heads * dim,
            128,
        )?;
        encoder.end_encoding();
        gpu.finish(command)?;
        let actual = gpu.read_f32(&output, heads * dim + GUARD_LEN)?;
        for (i, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
            let error = (f64::from(got) - want).abs();
            assert!(
                got.is_finite() && error < 3e-6 + want.abs() * 3e-5,
                "attention={} kernel={} H={heads} KVH={kv_heads} D={dim} L={history} \
                 zero={zero_values} gate_alias={alias_gate} i={i}: {got} != {want}",
                gpu.attention_mode(),
                gpu.kernel_mode(),
            );
        }
        assert!(
            actual[heads * dim..]
                .iter()
                .all(|v| v.to_bits() == GUARD.to_bits()),
            "attention output overwrote its guard: mode={} H={heads} D={dim} L={history}",
            gpu.attention_mode()
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires a real Apple Metal GPU"]
fn attention_values_match_fp64_across_gqa_tails_gates_and_alias_fallback() -> Result<()> {
    let mut serial = Gpu::new_with_reference(false)?;
    serial.set_parallel_attention(false);
    let mut parallel = Gpu::new_with_reference(false)?;
    parallel.set_parallel_attention(true);
    let mut reference = Gpu::new_with_reference(true)?;
    reference.set_parallel_attention(true);
    assert_eq!(serial.attention_mode(), "serial");
    assert_eq!(parallel.attention_mode(), "parallel");
    assert_eq!(reference.attention_mode(), "serial");
    let gpus = [serial, parallel, reference];

    for dim in [1, 3, 31, 32, 33, 128, 256] {
        for history in [1, 31, 127, 128, 129, 515] {
            check_case(&gpus, 6, 2, dim, history, false, false)?;
        }
    }
    for (heads, kv_heads, dim, history) in [
        (1, 1, 33, 129),
        (8, 8, 128, 515),
        (24, 4, 256, 515),
        (6, 2, 33, 8192),
        (6, 2, 256, 8192),
    ] {
        check_case(&gpus, heads, kv_heads, dim, history, false, false)?;
    }
    check_case(&gpus, 6, 2, 33, 129, true, false)?;
    check_case(&gpus, 6, 2, 256, 515, true, false)?;
    // Reading and writing the same gate element is safe in the serial kernel;
    // the optimized path must fall back when input and output buffers alias.
    check_case(&gpus, 6, 2, 33, 129, false, true)?;
    check_case(&gpus, 6, 2, 256, 515, false, true)?;
    Ok(())
}

#[test]
#[ignore = "requires a real Apple Metal GPU"]
fn parallel_attention_preserves_complete_hybrid_logits_across_history_threshold_and_reset()
-> Result<()> {
    use qwen_metal::engine::Engine;
    use std::path::Path;

    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-q4");
    let mut config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(source.join("config.json"))?)?;
    assert_eq!(config["text_config"]["head_dim"].as_u64(), Some(32));
    assert!(
        config["text_config"]["layer_types"]
            .as_array()
            .unwrap()
            .iter()
            .any(|kind| kind == "full_attention")
    );
    // The committed fixture permits only 128 positions. Extend only that
    // configuration limit in a temporary copy so positions 127, 128, and 129
    // exercise the parallel attention kernel at and beyond its L=128 threshold.
    // Weights, RoPE parameters, and the committed fixture remain unchanged.
    config["text_config"]["max_position_embeddings"] = serde_json::json!(160);
    let fixture = tempfile::tempdir()?;
    std::fs::write(
        fixture.path().join("config.json"),
        serde_json::to_vec(&config)?,
    )?;
    std::fs::copy(
        source.join("model.safetensors"),
        fixture.path().join("model.safetensors"),
    )?;
    let mut serial = Engine::load(fixture.path(), 160)?;
    let mut parallel = Engine::load(fixture.path(), 160)?;
    serial.set_parallel_attention(false);
    parallel.set_parallel_attention(true);
    assert_eq!(serial.attention_mode(), "serial");
    assert_eq!(parallel.attention_mode(), "parallel");

    let compare = |actual: &[f32], expected: &[f32], step: usize, phase: &str| {
        assert_eq!(actual.len(), 64, "complete fixture vocabulary required");
        assert_eq!(actual.len(), expected.len());
        for (index, (&got, &want)) in actual.iter().zip(expected).enumerate() {
            assert!(
                got.is_finite()
                    && want.is_finite()
                    && (got - want).abs() < 1e-5 + want.abs() * 2e-5,
                "{phase}: token {step}, logit {index}: {got} != {want}"
            );
        }
    };
    // Fixed tokens avoid turning tiny floating-point differences into a
    // different autoregressive input sequence. Every vocabulary logit is checked.
    let tokens: Vec<u32> = (0..130).map(|step| (step * 17 + 7) % 64).collect();
    let mut initial_prefix = Vec::new();
    for (step, &token) in tokens.iter().enumerate() {
        let expected = serial.forward(token)?;
        let actual = parallel.forward(token)?;
        compare(&actual, &expected, step, "serial versus parallel");
        if step < 5 {
            initial_prefix.push(expected);
        }
    }
    assert_eq!(serial.position(), 130);
    assert_eq!(parallel.position(), 130);
    serial.reset();
    parallel.reset();
    assert_eq!(serial.position(), 0);
    assert_eq!(parallel.position(), 0);
    for (step, (&token, expected)) in tokens.iter().zip(&initial_prefix).enumerate() {
        compare(
            &serial.forward(token)?,
            expected,
            step,
            "serial after reset",
        );
        compare(
            &parallel.forward(token)?,
            expected,
            step,
            "parallel after reset",
        );
    }
    assert_eq!(serial.position(), 5);
    assert_eq!(parallel.position(), 5);
    Ok(())
}
