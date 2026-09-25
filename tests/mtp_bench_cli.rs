//! Exercise the public benchmark command against real Metal, including report accounting.
use serde_json::{Value, json};
use std::{path::Path, process::Command};

#[test]
fn factorial_cli_rejects_invalid_comparisons_before_loading_weights() {
    for (args, reference, kernel, expected) in [
        (
            vec!["--temperature", "0.7"],
            "0",
            "aligned",
            "requires --temperature 0",
        ),
        (vec!["--compare"], "0", "aligned", "cannot be used with"),
        (vec![], "1", "aligned", "requires QWEN_METAL_REFERENCE=0"),
        (vec![], "0", "stream", "requires QWEN_METAL_GEMV=aligned"),
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_qwen-metal"))
            .env("QWEN_METAL_REFERENCE", reference)
            .env("QWEN_METAL_GEMV", kernel)
            .args([
                "mtp-bench",
                "--model",
                "/missing-model",
                "--mtp",
                "/missing-adapter",
                "--compare-block-kernels",
            ])
            .args(args)
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(
            String::from_utf8_lossy(&output.stderr).contains(expected),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
#[ignore = "requires a real Apple Metal GPU"]
fn factorial_cli_separates_variants_and_compares_every_greedy_output() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let work = tempfile::tempdir().unwrap();
    let prompts = work.path().join("prompts.json");
    std::fs::write(
        &prompts,
        r#"[{"name":"first","prompt":"a b"},{"name":"second","prompt":"b c"}]"#,
    )
    .unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_qwen-metal"))
        .env("QWEN_METAL_REFERENCE", "0")
        .env("QWEN_METAL_GEMV", "aligned")
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
            "--compare-block-kernels",
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(report["kind"], "mtp_block_kernel_comparison");
    assert_eq!(report["variant_greedy_agreement"], true);
    assert_eq!(report["goal_32_tps_confirmed"], false);
    assert!(report.get("aggregate_mtp").is_none());
    let variants = report["variants"].as_array().unwrap();
    assert_eq!(variants.len(), 4);
    assert_eq!(report["variant_comparisons"].as_array().unwrap().len(), 12);
    let captures = report["captures"].as_array().unwrap();
    assert_eq!(captures.len(), 16);
    for capture in captures {
        let timing = &capture["stats"]["target_decode_timing"];
        assert_eq!(timing["command_buffers"], capture["stats"]["rounds"]);
        let timed = timing["timed_command_buffers"].as_u64().unwrap();
        let missing = timing["missing_command_buffers"].as_u64().unwrap();
        assert_eq!(timed + missing, timing["command_buffers"].as_u64().unwrap());
        if timed > 0 {
            assert!(timing["observed_gpu_seconds"].as_f64().unwrap() > 0.);
        }
    }
    for variant in variants {
        let id = &variant["variant"];
        let rows: Vec<_> = captures
            .iter()
            .filter(|row| row["variant"] == *id && row["warmup"] == false)
            .collect();
        assert_eq!(rows.len(), 2);
        assert_eq!(variant["aggregate_mtp"]["measured_runs"], 2);
        let tokens: u64 = rows
            .iter()
            .map(|row| row["sustained_decode_tokens"].as_u64().unwrap())
            .sum();
        let seconds: f64 = rows
            .iter()
            .map(|row| row["decode_seconds"].as_f64().unwrap())
            .sum();
        let actual = variant["aggregate_mtp"]["weighted_sustained_decode_tokens_per_second"]
            .as_f64()
            .unwrap();
        assert!((actual - tokens as f64 / seconds).abs() < 1e-8);
        for row in rows {
            assert_eq!(row["block_kernel_mode"], variant["block_kernel_mode"]);
            assert_eq!(
                row["stats"]["token_ids"].as_array().unwrap().len() as u64,
                row["completion_tokens"].as_u64().unwrap()
            );
        }
    }
}

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
        .env_remove("QWEN_METAL_BLOCK_MATMUL")
        .env_remove("QWEN_METAL_BLOCK_DELTA")
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
    let default_mode = json!({"shared_matmul": false, "batched_delta": false});
    assert_eq!(report["block_kernel_mode"], default_mode);
    assert_eq!(report["greedy_agreement"], true);
    assert_eq!(report["goal_32_tps_confirmed"], false);
    assert_eq!(report["aggregate_mtp"]["measured_runs"], 2);
    assert_eq!(report["aggregate_sequential_target"]["measured_runs"], 2);
    let captures = report["captures"].as_array().unwrap();
    assert_eq!(captures.len(), 8);
    let mut sum_tokens = 0;
    let mut sum_seconds = 0.0;
    for capture in captures {
        assert_eq!(capture["block_kernel_mode"], default_mode);
        let count = capture["completion_tokens"].as_u64().unwrap();
        let sustained = capture["sustained_decode_tokens"].as_u64().unwrap();
        let seconds = capture["decode_seconds"].as_f64().unwrap();
        assert!(count > 0 && count <= 8);
        assert_eq!(
            count as usize,
            capture["stats"]["token_ids"].as_array().unwrap().len()
        );
        assert_eq!(sustained, count - 1);
        let timing = &capture["stats"]["target_decode_timing"];
        let expected_commands = if capture["mode"] == "mtp" {
            capture["stats"]["rounds"].as_u64().unwrap()
        } else {
            sustained
        };
        assert_eq!(timing["command_buffers"], expected_commands);
        let timed = timing["timed_command_buffers"].as_u64().unwrap();
        let missing = timing["missing_command_buffers"].as_u64().unwrap();
        assert_eq!(timed + missing, expected_commands);
        if timed > 0 {
            assert!(timing["observed_gpu_seconds"].as_f64().unwrap() > 0.);
        }
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
