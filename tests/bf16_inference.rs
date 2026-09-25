use qwen_metal::engine::Engine;
use std::path::Path;

#[test]
#[ignore = "requires a real Apple Metal GPU"]
fn bf16_checkpoint_matches_scalar_logits_and_reports_exact_storage_savings() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-bf16");
    let gold: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path.join("golden.json")).unwrap()).unwrap();
    let mut engine = Engine::load(&path, 8).unwrap();
    let stats = engine.metadata_stats();
    if engine.metadata_mode() == "bf16" {
        assert_eq!(stats, (33, 3288 * 2));
    } else {
        assert_eq!(stats, (0, 0));
    }
    for _ in 0..2 {
        engine.reset();
        for (i, token) in gold["tokens"].as_array().unwrap().iter().enumerate() {
            let actual = engine.forward(token.as_u64().unwrap() as u32).unwrap();
            for (got, want) in actual.iter().zip(gold["logits"][i].as_array().unwrap()) {
                let want = want.as_f64().unwrap() as f32;
                assert!(
                    got.is_finite() && (got - want).abs() < 2e-3 + 2e-3 * want.abs(),
                    "step{i}: {got} != {want}"
                );
            }
        }
    }
}
