//! Exercise the public benchmark command against real Metal, including report accounting.
use serde_json::{Value, json};
use std::{path::Path, process::Command};

#[test]
fn mlp_r2_cli_rejects_inapplicable_candidates_before_loading_weights() {
    for (args, reference, kernel, metadata, expected) in [
        (
            vec!["--temperature", "0.7"],
            "0",
            "aligned",
            "bf16",
            "requires --temperature 0",
        ),
        (
            vec!["--block-size", "2"],
            "0",
            "aligned",
            "bf16",
            "requires --block-size 3",
        ),
        (
            vec![],
            "1",
            "aligned",
            "bf16",
            "requires QWEN_METAL_REFERENCE=0",
        ),
        (
            vec![],
            "0",
            "stream",
            "bf16",
            "requires QWEN_METAL_GEMV=aligned",
        ),
        (
            vec![],
            "0",
            "aligned",
            "f32",
            "requires QWEN_METAL_METADATA=bf16",
        ),
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_qwen-metal"))
            .env("QWEN_METAL_REFERENCE", reference)
            .env("QWEN_METAL_GEMV", kernel)
            .env("QWEN_METAL_METADATA", metadata)
            .args([
                "mtp-bench",
                "--model",
                "/missing-model",
                "--mtp",
                "/missing-adapter",
                "--compare-mlp-r2",
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
fn mlp_r2_cli_pairs_both_mtp_variants_with_ordinary_target_and_separates_accounting() {
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
        .env("QWEN_METAL_METADATA", "bf16")
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
            "3",
            "--compare-mlp-r2",
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(report["kind"], "verified_mtp_mlp_r2_comparison");
    assert_eq!(report["greedy_agreement"], true);
    assert_eq!(report["variant_greedy_agreement"], true);
    assert_eq!(
        report["independent_sequential_target_comparison_performed"],
        true
    );
    assert_eq!(report["goal_32_tps_confirmed"], false);
    assert_eq!(report["sustained_sample_sufficient"], false);
    assert_eq!(report["candidate_measured_schedule_complete"], true);
    assert_eq!(report["mlp_r2_eligible_target_matrices"], 0);
    assert_eq!(report["candidate_applicable"], false);
    assert!(report.get("aggregate_mtp").is_none());
    let captures = report["captures"].as_array().unwrap();
    assert_eq!(captures.len(), 24);
    assert_eq!(report["greedy_comparisons"].as_array().unwrap().len(), 16);
    assert_eq!(report["variant_comparisons"].as_array().unwrap().len(), 8);
    let groups = report["schedule"]["executed_groups"].as_array().unwrap();
    assert_eq!(groups.len(), 8);
    for (group, capture_group) in groups.iter().zip(captures.chunks_exact(3)) {
        assert_eq!(
            group["variants"],
            json!(
                capture_group
                    .iter()
                    .map(|c| &c["variant"])
                    .collect::<Vec<_>>()
            )
        );
        for capture in capture_group {
            assert_eq!(capture["run"], group["run"]);
            assert_eq!(capture["prompt_name"], group["prompt_name"]);
            assert_eq!(capture["warmup"], group["warmup"]);
        }
    }
    let variants = report["variants"].as_array().unwrap();
    assert_eq!(variants.len(), 3);
    for variant in variants {
        let rows: Vec<_> = captures
            .iter()
            .filter(|c| c["variant"] == variant["variant"] && c["warmup"] == false)
            .collect();
        assert_eq!(rows.len(), 6);
        assert_eq!(variant["aggregate"]["measured_runs"], 6);
        for prompt in ["first", "second"] {
            assert_eq!(
                captures
                    .iter()
                    .filter(|c| c["variant"] == variant["variant"]
                        && c["warmup"] == true
                        && c["prompt_name"] == prompt)
                    .count(),
                1
            );
            let positions: Vec<_> = (1..=3)
                .map(|run| {
                    captures
                        .iter()
                        .filter(|c| c["run"] == run && c["prompt_name"] == prompt)
                        .position(|c| c["variant"] == variant["variant"])
                        .unwrap()
                })
                .collect();
            for position in 0..3 {
                assert_eq!(positions.iter().filter(|&&p| p == position).count(), 1);
            }
        }
        let tokens: u64 = rows
            .iter()
            .map(|c| c["sustained_decode_tokens"].as_u64().unwrap())
            .sum();
        let seconds: f64 = rows
            .iter()
            .map(|c| c["decode_seconds"].as_f64().unwrap())
            .sum();
        let actual = variant["aggregate"]["weighted_sustained_decode_tokens_per_second"]
            .as_f64()
            .unwrap();
        assert!((actual - tokens as f64 / seconds).abs() < 1e-8);
        for capture in rows {
            assert_eq!(capture["block_kernel_mode"], variant["block_kernel_mode"]);
            assert_eq!(capture["mode"], variant["mode"]);
            assert_eq!(
                capture["completion_tokens"].as_u64().unwrap() as usize,
                capture["stats"]["token_ids"].as_array().unwrap().len()
            );
            let expected_commands = if capture["mode"] == "sequential_target" {
                capture["sustained_decode_tokens"].as_u64().unwrap()
            } else {
                capture["stats"]["rounds"].as_u64().unwrap()
            };
            assert_eq!(
                capture["stats"]["target_decode_timing"]["command_buffers"],
                expected_commands
            );
        }
    }
    for comparison in report["greedy_comparisons"].as_array().unwrap() {
        assert_eq!(comparison["baseline_variant"], "sequential_target");
        let candidate = captures
            .iter()
            .find(|c| {
                c["variant"] == comparison["variant"]
                    && c["prompt_name"] == comparison["prompt_name"]
                    && c["run"] == comparison["run"]
            })
            .unwrap();
        let reference = captures
            .iter()
            .find(|c| {
                c["variant"] == "sequential_target"
                    && c["prompt_name"] == comparison["prompt_name"]
                    && c["run"] == comparison["run"]
            })
            .unwrap();
        assert_eq!(
            candidate["stats"]["token_ids"],
            reference["stats"]["token_ids"]
        );
    }
    for comparison in report["variant_comparisons"].as_array().unwrap() {
        let matching = |id| {
            captures
                .iter()
                .find(|c| {
                    c["variant"] == id
                        && c["prompt_name"] == comparison["prompt_name"]
                        && c["run"] == comparison["run"]
                })
                .unwrap()
        };
        let candidate = matching("mlp_r2_batched")["sustained_decode_tokens_per_second"]
            .as_f64()
            .unwrap();
        let baseline = matching("legacy_batched")["sustained_decode_tokens_per_second"]
            .as_f64()
            .unwrap();
        assert!(
            (comparison["sustained_decode_speedup_vs_legacy_batched"]
                .as_f64()
                .unwrap()
                - candidate / baseline)
                .abs()
                < 1e-8
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
    let default_mode = json!({"matmul": "legacy", "batched_delta": true});
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
