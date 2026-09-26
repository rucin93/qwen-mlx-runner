//! One-load comparison of real target + MTP prompt prefill, using synthetic text.
//! Run with --help for options. This does not evaluate coding quality.
use anyhow::{Result, ensure};
use clap::Parser;
use qwen_metal::{
    chat::{ChatTokenizer, GenerationRequest, Message},
    mtp_chat::MtpChatEngine,
};
use serde::Serialize;
use serde_json::{Value, json};
use std::{path::PathBuf, process::Command, time::Instant};

#[path = "../src/system_status.rs"]
mod system_status;

const DECODE_WIDTH: usize = 3;
const TEXT: &str = "The synthetic benchmark notebook records a quiet river, a numbered stone, and a clear morning. ";

#[derive(Debug, Parser)]
#[command(
    version,
    about = "One-load greedy MTP prefill comparison on synthetic fixed text"
)]
struct Options {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    mtp: PathBuf,
    #[arg(long, default_value_t = 8192)]
    context: usize,
    /// Requested total rendered prompt tokens, including the checkpoint template.
    #[arg(long, default_value_t = 512)]
    prompt_tokens: usize,
    #[arg(long, default_value_t = 16)]
    max_tokens: usize,
    #[arg(long, default_value_t = 3)]
    runs: usize,
    #[arg(long, default_value_t = 16, value_parser = candidate_width)]
    prefill_batch_size: usize,
}

fn candidate_width(value: &str) -> std::result::Result<usize, String> {
    match value {
        "8" => Ok(8),
        "16" => Ok(16),
        _ => Err("candidate prefill batch size must be 8 or 16".into()),
    }
}

impl Options {
    fn validate(&self) -> Result<()> {
        ensure!((1..=20).contains(&self.runs), "runs must be 1..=20");
        ensure!(self.prompt_tokens > 0, "prompt-tokens must be positive");
        ensure!(self.max_tokens > 0, "max-tokens must be positive");
        ensure!(
            self.prompt_tokens
                .checked_add(self.max_tokens)
                .is_some_and(|n| n <= self.context),
            "requested prompt-tokens plus max-tokens exceeds context"
        );
        ensure!(
            matches!(self.prefill_batch_size, 8 | 16),
            "candidate must be 8 or 16"
        );
        Ok(())
    }
}

fn request(repetitions: usize, max_tokens: usize) -> GenerationRequest {
    GenerationRequest {
        messages: vec![Message {
            role: "user".into(),
            content: format!(
                "Read this synthetic notebook and reply briefly. {}",
                TEXT.repeat(repetitions)
            ),
            ..Default::default()
        }],
        max_tokens,
        temperature: 0.,
        top_p: 1.,
        top_k: 0,
        seed: 42,
        ..Default::default()
    }
}

// Find the repetition count immediately below the token budget. Encoding the
// complete rendered request preserves the checkpoint's original chat template.
fn select_repetitions(mut count: impl FnMut(usize) -> Result<usize>, goal: usize) -> Result<usize> {
    ensure!(
        count(0)? <= goal,
        "prompt-tokens is smaller than the fixed instruction and template overhead"
    );
    let (mut low, mut high) = (0, 1);
    while count(high)? <= goal {
        low = high;
        high = high
            .checked_mul(2)
            .ok_or_else(|| anyhow::anyhow!("prompt sizing overflow"))?;
        ensure!(
            high <= goal.saturating_mul(4).max(4),
            "synthetic text does not grow under this tokenizer/template"
        );
    }
    while high - low > 1 {
        let mid = low + (high - low) / 2;
        if count(mid)? <= goal {
            low = mid;
        } else {
            high = mid;
        }
    }
    Ok(low)
}

#[derive(Debug, Serialize)]
struct Capture {
    variant: &'static str,
    pair: usize,
    sequence: usize,
    warmup: bool,
    comparison_matches: bool,
    prefill_batch_size: usize,
    prompt_tokens: usize,
    completion_tokens: usize,
    token_ids: Vec<u32>,
    text: String,
    finish_reason: String,
    prefill_seconds: f64,
    prefill_target_seconds: f64,
    prefill_draft_seconds: f64,
    time_to_first_token_seconds: Option<f64>,
    allocated_bytes_after: u64,
    conditions_before: Value,
    conditions_after: Value,
}

fn capture(
    engine: &mut MtpChatEngine,
    req: &GenerationRequest,
    width: usize,
    pair: usize,
    sequence: usize,
    warmup: bool,
) -> Result<Capture> {
    engine.set_prefill_batch_size(width)?;
    let conditions_before = system_status::snapshot();
    let (output, stats) = engine.generate_with_stats(req, &mut |_| true)?;
    Ok(Capture {
        variant: if width == DECODE_WIDTH {
            "baseline"
        } else {
            "candidate"
        },
        pair,
        sequence,
        warmup,
        comparison_matches: false,
        prefill_batch_size: engine.prefill_batch_size(),
        prompt_tokens: output.prompt_tokens,
        completion_tokens: output.completion_tokens,
        token_ids: stats.token_ids,
        text: output.text,
        finish_reason: output.finish_reason,
        prefill_seconds: output.prefill_seconds,
        prefill_target_seconds: stats.prefill_target_seconds,
        prefill_draft_seconds: stats.prefill_draft_seconds,
        time_to_first_token_seconds: stats.time_to_first_token_seconds,
        allocated_bytes_after: engine.engine().allocated_bytes(),
        conditions_before,
        conditions_after: system_status::snapshot(),
    })
}

fn equal(a: &Capture, b: &Capture) -> bool {
    a.token_ids == b.token_ids
        && a.text == b.text
        && a.finish_reason == b.finish_reason
        && a.completion_tokens == b.completion_tokens
        && a.prompt_tokens == b.prompt_tokens
        && a.token_ids.len() == a.completion_tokens
        && b.token_ids.len() == b.completion_tokens
}

fn median(mut values: Vec<f64>) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let n = values.len();
    Some(if n % 2 == 0 {
        (values[n / 2 - 1] + values[n / 2]) / 2.
    } else {
        values[n / 2]
    })
}

fn summary(captures: &[Capture], variant: &str) -> Value {
    let rows: Vec<_> = captures
        .iter()
        .filter(|c| c.variant == variant && !c.warmup && c.comparison_matches)
        .collect();
    let seconds: f64 = rows.iter().map(|c| c.prefill_seconds).sum();
    let tokens: usize = rows.iter().map(|c| c.prompt_tokens).sum();
    json!({
        "valid_measured_captures": rows.len(), "total_prompt_tokens": tokens,
        "total_prefill_seconds": seconds,
        "total_prefill_target_seconds": rows.iter().map(|c| c.prefill_target_seconds).sum::<f64>(),
        "total_prefill_draft_seconds": rows.iter().map(|c| c.prefill_draft_seconds).sum::<f64>(),
        "ttft_capture_count": rows.iter().filter(|c| c.time_to_first_token_seconds.is_some()).count(),
        "total_time_to_first_token_seconds": rows.iter().filter_map(|c| c.time_to_first_token_seconds).sum::<f64>(),
        "prompt_tokens_per_second": if seconds > 0. { Some(tokens as f64 / seconds) } else { None },
        "median_prefill_seconds": median(rows.iter().map(|c| c.prefill_seconds).collect()),
        "median_prefill_target_seconds": median(rows.iter().map(|c| c.prefill_target_seconds).collect()),
        "median_prefill_draft_seconds": median(rows.iter().map(|c| c.prefill_draft_seconds).collect()),
        "median_time_to_first_token_seconds": median(rows.iter().filter_map(|c| c.time_to_first_token_seconds).collect()),
        "excluded_warmups_and_mismatched_pairs": true
    })
}

fn pair_order(pair: usize, candidate: usize) -> [usize; 2] {
    if pair == 0 || pair % 2 == 1 {
        [DECODE_WIDTH, candidate]
    } else {
        [candidate, DECODE_WIDTH]
    }
}

fn version(command: &str, args: &[&str]) -> Option<String> {
    Command::new(command)
        .args(args)
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_owned())
}

fn main() -> Result<()> {
    let options = Options::parse();
    options.validate()?;
    let tokenizer = ChatTokenizer::load(&options.model)?;
    let repetitions = select_repetitions(
        |n| {
            let req = request(n, options.max_tokens);
            Ok(tokenizer.encode(&tokenizer.render_request(&req)?)?.len())
        },
        options.prompt_tokens,
    )?;
    let req = request(repetitions, options.max_tokens);
    let prompt_ids = tokenizer.encode(&tokenizer.render_request(&req)?)?;
    ensure!(
        !prompt_ids.is_empty()
            && prompt_ids
                .len()
                .checked_add(options.max_tokens)
                .is_some_and(|n| n <= options.context),
        "actual rendered prompt and completion budget must fit context"
    );
    drop(tokenizer);
    eprintln!(
        "Loading target and adapter once; {} actual synthetic prompt tokens, {} measured pairs.",
        prompt_ids.len(),
        options.runs
    );
    let start = Instant::now();
    let mut engine =
        MtpChatEngine::load(&options.model, &options.mtp, options.context, DECODE_WIDTH)?;
    let load_seconds = start.elapsed().as_secs_f64();
    let configuration = json!({"device": engine.engine().device_name(), "kernel_mode": engine.engine().kernel_mode(),
        "block_kernel_mode": engine.engine().block_kernel_mode(), "norm_mode": engine.engine().norm_mode(),
        "attention_mode": engine.engine().attention_mode(), "metadata_mode": engine.engine().metadata_mode(),
        "metadata_stats": engine.engine().metadata_stats(), "allocated_bytes_after_load": engine.engine().allocated_bytes()});
    let mut captures = Vec::new();
    let mut correct = true;
    for pair in 0..=options.runs {
        let order = pair_order(pair, options.prefill_batch_size);
        let a = capture(&mut engine, &req, order[0], pair, captures.len(), pair == 0)?;
        let b = capture(
            &mut engine,
            &req,
            order[1],
            pair,
            captures.len() + 1,
            pair == 0,
        )?;
        let matches = equal(&a, &b) && a.prompt_tokens == prompt_ids.len();
        correct &= matches;
        captures.extend([
            Capture {
                comparison_matches: matches,
                ..a
            },
            Capture {
                comparison_matches: matches,
                ..b
            },
        ]);
        eprintln!(
            "{} pair {pair}: outputs {}",
            if pair == 0 { "warmup" } else { "measured" },
            if matches { "match" } else { "MISMATCH" }
        );
    }
    let baseline = summary(&captures, "baseline");
    let candidate = summary(&captures, "candidate");
    let ratio = |field: &str| -> Option<f64> {
        let a = baseline[field].as_f64()?;
        let b = candidate[field].as_f64()?;
        if correct && b > 0. { Some(a / b) } else { None }
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "kind": "one_load_mtp_prefill_comparison", "report_version": 1,
            "versions": {"qwen_metal": env!("CARGO_PKG_VERSION"), "rustc": version("rustc", &["--version"]), "macos": version("sw_vers", &["-productVersion"])},
            "model": options.model, "mtp": options.mtp, "context": options.context,
            "decode_block_size": DECODE_WIDTH, "baseline_prefill_batch_size": DECODE_WIDTH,
            "candidate_prefill_batch_size": options.prefill_batch_size,
            "requested_prompt_tokens": options.prompt_tokens, "actual_prompt_tokens": prompt_ids.len(),
            "prompt_token_ids": prompt_ids, "synthetic_text_repetitions": repetitions,
            "max_tokens": options.max_tokens, "runs": options.runs, "temperature": 0,
            "target_and_adapter_loads": 1, "load_seconds": load_seconds, "configuration": configuration,
            "correct_comparison": correct, "baseline": baseline, "candidate": candidate,
        "baseline_over_candidate_median_prefill_ratio": ratio("median_prefill_seconds"),
        "baseline_over_candidate_total_prefill_ratio": ratio("total_prefill_seconds"),
        "baseline_over_candidate_median_target_prefill_ratio": ratio("median_prefill_target_seconds"),
        "baseline_over_candidate_median_draft_prefill_ratio": ratio("median_prefill_draft_seconds"),
            "baseline_over_candidate_median_ttft_ratio": ratio("median_time_to_first_token_seconds"),
            "captures": captures,
            "notes": ["Real checkpoint and adapter weights are loaded once; the prompt is synthetic fixed text rendered with the checkpoint's unchanged chat template, not a real OpenCode request or a coding-quality test.",
                "Each variant has one excluded warmup. Measured pairs alternate AB/BA on the same engine with per-request cache resets. Warmups are also checked for output equality.",
                "Ratios above one mean lower candidate latency; ratios are withheld on any mismatch. No output-token-rate goal or automatic candidate promotion.",
            "The default 512-token probe does not establish latency for an 81k context or an 11k-token prompt. Run the short probe before a longer 8065-token run on the same host.",
            "This example makes no M5 performance claim; every result applies only to the recorded host, runtime configuration and prompt length.",
                "Prompt selection uses whole text repetitions closest below the requested rendered token budget; actual token count and IDs are reported."]
        }))?
    );
    ensure!(
        correct,
        "greedy output mismatch: candidate is not validated; inspect JSON captures"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options(extra: &[&str]) -> Options {
        Options::try_parse_from(
            ["bench", "--model", "target", "--mtp", "adapter"]
                .into_iter()
                .chain(extra.iter().copied()),
        )
        .unwrap()
    }

    #[test]
    fn cli_validation() {
        assert!(options(&[]).validate().is_ok());
        for args in [
            vec!["--runs", "0"],
            vec!["--runs", "21"],
            vec!["--prompt-tokens", "0"],
            vec!["--max-tokens", "0"],
            vec!["--context", "512"],
            vec!["--prompt-tokens", "18446744073709551615"],
        ] {
            assert!(options(&args).validate().is_err());
        }
        assert!(
            Options::try_parse_from([
                "bench",
                "--model",
                "x",
                "--mtp",
                "y",
                "--prefill-batch-size",
                "3"
            ])
            .is_err()
        );
    }

    #[test]
    fn prompt_selection_accounts_for_template_overhead() {
        assert_eq!(select_repetitions(|n| Ok(17 + 11 * n), 512).unwrap(), 45);
        assert_eq!(select_repetitions(|n| Ok(17 + 11 * n), 17).unwrap(), 0);
        assert!(select_repetitions(|n| Ok(17 + 11 * n), 16).is_err());
    }

    #[test]
    fn median_handles_odd_even_and_empty() {
        assert_eq!(median(vec![3., 1., 2.]), Some(2.));
        assert_eq!(median(vec![4., 1., 3., 2.]), Some(2.5));
        assert_eq!(median(vec![]), None);
    }

    #[test]
    fn measured_pairs_alternate_ab_ba() {
        assert_eq!(pair_order(0, 16), [3, 16]);
        for pair in 1..=20 {
            assert_eq!(
                pair_order(pair, 16),
                if pair % 2 == 1 { [3, 16] } else { [16, 3] }
            );
        }
    }

    fn row(warmup: bool, matches: bool, seconds: f64) -> Capture {
        Capture {
            variant: "baseline",
            pair: 0,
            sequence: 0,
            warmup,
            comparison_matches: matches,
            prefill_batch_size: 3,
            prompt_tokens: 100,
            completion_tokens: 1,
            token_ids: vec![7],
            text: "sample".into(),
            finish_reason: "length".into(),
            prefill_seconds: seconds,
            prefill_target_seconds: seconds * 0.75,
            prefill_draft_seconds: seconds * 0.25,
            time_to_first_token_seconds: Some(seconds + 1.),
            allocated_bytes_after: 0,
            conditions_before: Value::Null,
            conditions_after: Value::Null,
        }
    }

    #[test]
    fn summaries_exclude_warmup_and_mismatch_and_use_total_time() {
        let captures = [
            row(true, true, 100.),
            row(false, false, 50.),
            row(false, true, 2.),
            row(false, true, 4.),
        ];
        let summary = summary(&captures, "baseline");
        assert_eq!(summary["valid_measured_captures"], 2);
        assert_eq!(summary["total_prompt_tokens"], 200);
        assert_eq!(summary["total_prefill_seconds"], 6.);
        assert_eq!(summary["median_prefill_seconds"], 3.);
        assert_eq!(summary["total_prefill_target_seconds"], 4.5);
        assert_eq!(summary["total_prefill_draft_seconds"], 1.5);
        assert_eq!(summary["total_time_to_first_token_seconds"], 8.);
        assert_eq!(summary["prompt_tokens_per_second"], 200. / 6.);
    }

    #[test]
    fn equality_checks_actual_ids_text_finish_and_count() {
        let a = row(false, true, 1.);
        let mut b = row(false, true, 2.);
        assert!(equal(&a, &b));
        b.token_ids = vec![8];
        assert!(!equal(&a, &b));
        b.token_ids = vec![7];
        b.text = "different".into();
        assert!(!equal(&a, &b));
        b.text = a.text.clone();
        b.finish_reason = "stop".into();
        assert!(!equal(&a, &b));
        b.finish_reason = a.finish_reason.clone();
        b.completion_tokens = 2;
        assert!(!equal(&a, &b));
    }
}
