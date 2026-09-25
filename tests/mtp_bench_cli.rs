//! Exercise the public benchmark command against real Metal, including report accounting.
use serde_json::{Value, json};
use std::{path::Path, process::Command};

#[test]
#[ignore = "requires a real Apple Metal GPU"]
fn mixed_cli_compares_verified_ids_and_never_promotes_short_samples() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let work = tempfile::tempdir().unwrap();
    let prompts = work.path().join("prompts.json");
    std::fs::write(
        &prompts,
        json!([
            {"name":"first", "prompt":"a b"},
            {"name":"second", "prompt":"b c"}
        ])
        .to_string(),
    )
    .unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_qwen-metal"))
        .env("QWEN_METAL_REFERENCE", "0")
        .arg("mtp-bench")
        .arg("--model")
        .arg(root.join("tests/fixtures/tiny-q4"))
        .arg("--mtp")
        .arg(root.join("tests/fixtures/tiny-mtp"))
        .arg("--prompts")
        .arg(prompts)
        .args([
            "--context",
            "96",
            "--max-tokens",
            "8",
            "--runs",
            "1",
            "--compare",
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(report["kind"], "verified_mtp_chat_benchmark");
    assert_eq!(report["greedy_agreement"], true);
    assert_eq!(report["goal_32_tps_confirmed"], false);
    assert_eq!(report["aggregate_mtp"]["measured_runs"], 2);
    assert_eq!(report["aggregate_sequential_target"]["measured_runs"], 2);
    let captures = report["captures"].as_array().unwrap();
    assert_eq!(captures.len(), 8);
    let mut sum_tokens = 0;
    let mut sum_seconds = 0.0;
    for capture in captures {
        let count = capture["completion_tokens"].as_u64().unwrap();
        let sustained = capture["sustained_decode_tokens"].as_u64().unwrap();
        let seconds = capture["decode_seconds"].as_f64().unwrap();
        assert!(count > 0 && count <= 8);
        assert_eq!(
            count as usize,
            capture["stats"]["token_ids"].as_array().unwrap().len()
        );
        assert_eq!(sustained, count - 1);
        assert!(seconds.is_finite() && seconds > 0.0);
        let rate = capture["sustained_decode_tokens_per_second"]
            .as_f64()
            .unwrap();
        assert!((rate - sustained as f64 / seconds).abs() < 1e-8);
        if capture["mode"] == "mtp" && capture["warmup"] == false {
            sum_tokens += sustained;
            sum_seconds += seconds;
        }
    }
    assert_eq!(
        report["aggregate_mtp"]["sustained_decode_tokens"],
        sum_tokens
    );
    let aggregate = report["aggregate_mtp"]["weighted_sustained_decode_tokens_per_second"]
        .as_f64()
        .unwrap();
    assert!((aggregate - sum_tokens as f64 / sum_seconds).abs() < 1e-8);
}
