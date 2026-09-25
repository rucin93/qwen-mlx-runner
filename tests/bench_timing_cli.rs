use std::path::Path;
use std::process::Command;

#[test]
#[ignore = "requires a real Apple Metal GPU"]
fn benchmark_timing_covers_each_step_and_keeps_warmup_out_of_median() {
    let model = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-q4");
    let output = Command::new(env!("CARGO_BIN_EXE_qwen-metal"))
        .args(["bench", "--timing", "--model"])
        .arg(&model)
        .args([
            "--context",
            "96",
            "--prompt-tokens",
            "33",
            "--generate-tokens",
            "33",
            "--runs",
            "1",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let runs = report["runs"].as_array().unwrap();
    assert_eq!(runs.len(), 1);
    assert_eq!(
        report["median_decode_tokens_per_second"],
        runs[0]["decode_tokens_per_second"]
    );
    let warmup = &report["warmup_run"];
    assert_eq!(warmup["warmup"], true);
    assert_eq!(warmup["run"], 0);
    for run in [warmup, &runs[0]] {
        for phase in ["prefill", "decode"] {
            let t = &run["timing"][phase];
            assert_eq!(t["sample_count"], 33);
            assert_eq!(t["valid_gpu_samples"], 33);
            assert_eq!(t["missing_gpu_samples"], 0);
            assert_eq!(t["blocks"].as_array().unwrap().len(), 2);
            assert!(t["forward"]["total_seconds"].as_f64().unwrap() > 0.);
            assert!(t["gpu"]["total_seconds"].as_f64().unwrap() > 0.);
            let measured = t["forward"]["total_seconds"].as_f64().unwrap()
                + t["sampling"]["total_seconds"].as_f64().unwrap();
            let elapsed = run[format!("{phase}_seconds")].as_f64().unwrap();
            assert!(
                elapsed + 1e-9 >= measured,
                "{phase}: intervals exceed phase wall time"
            );
            let start = if phase == "prefill" { 0 } else { 33 };
            assert_eq!(
                t["blocks"][0]["history_range"],
                serde_json::json!([start, start + 31])
            );
            assert_eq!(
                t["blocks"][1]["history_range"],
                serde_json::json!([start + 32, start + 32])
            );
            assert_eq!(t["blocks"][0]["sample_count"], 32);
            assert_eq!(t["blocks"][1]["sample_count"], 1);
        }
        assert_eq!(run["timing"]["prefill"]["sampling"]["total_seconds"], 0.);
        assert!(
            run["timing"]["decode"]["sampling"]["total_seconds"]
                .as_f64()
                .unwrap()
                > 0.
        );
    }
    let plain = Command::new(env!("CARGO_BIN_EXE_qwen-metal"))
        .args(["bench", "--model"])
        .arg(&model)
        .args([
            "--context",
            "8",
            "--prompt-tokens",
            "2",
            "--generate-tokens",
            "2",
            "--runs",
            "1",
        ])
        .output()
        .unwrap();
    assert!(
        plain.status.success(),
        "{}",
        String::from_utf8_lossy(&plain.stderr)
    );
    let plain: serde_json::Value = serde_json::from_slice(&plain.stdout).unwrap();
    assert!(plain.get("warmup_run").is_none());
    assert!(plain.get("timing_enabled").is_none());
    assert!(plain["runs"][0].get("timing").is_none());
    assert_eq!(plain["runs"].as_array().unwrap().len(), 1);
    assert_eq!(plain["runs"][0]["warmup"], false);
    assert_eq!(plain["kind"], report["kind"]);
}
