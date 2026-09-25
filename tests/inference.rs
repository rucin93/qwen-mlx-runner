use qwen_metal::engine::Engine;
use std::path::Path;

#[test]
#[ignore = "requires a real Apple Metal GPU"]
fn profiling_preserves_logits_and_resets_samples_between_tokens() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-q4");
    let gold: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path.join("golden.json")).unwrap()).unwrap();
    let mut model = Engine::load(&path, 8).unwrap();
    assert!(model.profile_report().is_err());
    model.enable_command_profiling().unwrap();
    assert!(model.profile_report().is_err());
    let mut previous_count = None;
    for (i, token) in gold["tokens"].as_array().unwrap().iter().enumerate() {
        let got = model.forward(token.as_u64().unwrap() as u32).unwrap();
        for (got, want) in got.iter().zip(gold["logits"][i].as_array().unwrap()) {
            let want = want.as_f64().unwrap() as f32;
            assert!((got - want).abs() < 2e-3 + 2e-3 * want.abs());
        }
        let rows = model.profile_report().unwrap();
        assert_eq!(model.profile_backend(), Some("command_buffers"));
        assert!(rows.iter().all(|r| r.raw_gpu_ticks.is_none()));
        let count: usize = rows.iter().map(|r| r.dispatches).sum();
        assert!(count > 0 && rows.iter().any(|r| r.gpu_nanoseconds > 0));
        if let Some(previous) = previous_count {
            assert_eq!(count, previous);
        }
        previous_count = Some(count);
        assert!(rows.iter().any(|r| r.kernel.starts_with("matvec_affine")));
    }
}

#[test]
#[ignore = "requires a real Apple Metal GPU"]
fn chat_prefix_reuse_and_cancelled_state_match_fresh_generation() {
    use qwen_metal::{
        chat::{GenerationRequest, Message, TextGenerator},
        engine::ChatEngine,
    };
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny");
    let request = GenerationRequest {
        messages: vec![Message {
            role: "user".into(),
            content: "w6 w7".into(),
        }],
        max_tokens: 6,
        temperature: 0.,
        top_p: 1.,
        top_k: 0,
        seed: 42,
        enable_thinking: false,
    };
    let mut reused = ChatEngine::load(&path, 128).unwrap();
    let initial = reused.generate(&request, &mut |_| true).unwrap();
    let mut continuation = request.clone();
    continuation.messages.push(Message {
        role: "assistant".into(),
        content: initial.text,
    });
    continuation.messages.push(Message {
        role: "user".into(),
        content: "w8".into(),
    });
    let warm = reused.generate(&continuation, &mut |_| true).unwrap();
    let mut fresh = ChatEngine::load(&path, 128).unwrap();
    let cold = fresh.generate(&continuation, &mut |_| true).unwrap();
    assert_eq!(
        (warm.text, warm.completion_tokens),
        (cold.text, cold.completion_tokens)
    );
    // Cancel in the generation phase, then ensure the next request has no stale state.
    let mut chunks = 0;
    let cancelled = reused
        .generate(&request, &mut |part| {
            if !part.is_empty() {
                chunks += 1;
            }
            chunks < 2
        })
        .unwrap();
    assert_eq!(cancelled.finish_reason, "cancelled");
    let after = reused.generate(&request, &mut |_| true).unwrap();
    fresh.clear_cache();
    let expected = fresh.generate(&request, &mut |_| true).unwrap();
    assert_eq!(
        (after.text, after.completion_tokens),
        (expected.text, expected.completion_tokens)
    );
}

// Catches layout, norm-offset, RoPE, recurrent-state and layer-order bugs by
// comparing the complete GPU path to independently generated scalar logits.
#[test]
#[ignore = "requires a real Apple Metal GPU; run cargo test --test inference -- --ignored"]
fn hybrid_forward_matches_independent_scalar_oracle_and_reset() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny");
    let gold: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path.join("golden.json")).unwrap()).unwrap();
    let mut model = Engine::load(&path, 8).expect("real Metal engine");
    let tokens = gold["tokens"].as_array().unwrap();
    for repetition in 0..2 {
        model.reset();
        for (i, token) in tokens.iter().enumerate() {
            let logits = model.forward(token.as_u64().unwrap() as u32).unwrap();
            for (j, (got, want)) in logits
                .iter()
                .zip(gold["logits"][i].as_array().unwrap())
                .enumerate()
            {
                let want = want.as_f64().unwrap() as f32;
                assert!(
                    (got - want).abs() < 3e-4,
                    "reset {repetition}, token {i}, logit {j}: {got} vs {want}"
                );
            }
        }
    }
    model.reset();
    assert!(
        model.forward(64).is_err(),
        "out-of-vocabulary token must be rejected"
    );
    for _ in 0..8 {
        model.forward(3).unwrap();
    }
    assert!(
        model.forward(3).is_err(),
        "context overflow must fail before writing cache"
    );
}

// The packed fixture includes MLX-converted norms/convolutions and exercises
// embed_affine plus the optimized affine matvec kernels in every layer.
#[test]
#[ignore = "requires a real Apple Metal GPU; run cargo test --test inference -- --ignored"]
fn packed_q4_hybrid_forward_matches_independent_scalar_oracle_and_reset() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-q4");
    let gold: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path.join("golden.json")).unwrap()).unwrap();
    let mut model = Engine::load(&path, 8).expect("real Metal engine with packed affine Q4");
    let tokens = gold["tokens"].as_array().unwrap();
    assert_eq!(tokens.len(), 5);
    for repetition in 0..2 {
        model.reset();
        for (i, token) in tokens.iter().enumerate() {
            let logits = model.forward(token.as_u64().unwrap() as u32).unwrap();
            let expected = gold["logits"][i].as_array().unwrap();
            assert_eq!(logits.len(), expected.len());
            for (j, (got, want)) in logits.iter().zip(expected).enumerate() {
                let want = want.as_f64().unwrap() as f32;
                assert!(
                    got.is_finite() && (got - want).abs() < 2e-3 + 2e-3 * want.abs(),
                    "Q4 reset {repetition}, token {i}, logit {j}: {got} vs {want}"
                );
            }
        }
    }
}
