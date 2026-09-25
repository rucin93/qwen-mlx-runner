//! Actual verified-token MTP benchmark, with the sequential target resident once.
use anyhow::{Context, Result, ensure};
use qwen_metal::{
    chat::{ChatTokenizer, GenerationOutput, GenerationRequest, Message},
    mtp_chat::{MtpChatEngine, MtpGenerationStats},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{collections::HashSet, path::PathBuf, time::Instant};

pub struct Options {
    pub model: PathBuf,
    pub mtp: PathBuf,
    pub context: usize,
    pub max_tokens: usize,
    pub runs: usize,
    pub block_size: usize,
    pub prompts: Option<PathBuf>,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub seed: u64,
    pub thinking: bool,
    pub compare: bool,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
struct Prompt {
    name: String,
    prompt: String,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Mode {
    Mtp,
    SequentialTarget,
}
#[derive(Serialize)]
struct Capture {
    prompt_name: String,
    run: usize,
    warmup: bool,
    mode: Mode,
    prompt_tokens: usize,
    completion_tokens: usize,
    sustained_decode_tokens: usize,
    prefill_seconds: f64,
    decode_seconds: f64,
    generation_wall_seconds: f64,
    completion_tokens_per_decode_second: Option<f64>,
    sustained_decode_tokens_per_second: Option<f64>,
    end_to_end_tokens_per_second: Option<f64>,
    finish_reason: String,
    output_text: String,
    stats: MtpGenerationStats,
    conditions_before: Value,
    conditions_after: Value,
}
fn rate(tokens: usize, seconds: f64) -> Option<f64> {
    if !seconds.is_finite() || seconds <= 0. {
        return None;
    }
    let value = tokens as f64 / seconds;
    value.is_finite().then_some(value)
}
fn median(mut values: Vec<f64>) -> Option<f64> {
    if values.is_empty() || values.iter().any(|v| !v.is_finite()) {
        return None;
    }
    values.sort_by(f64::total_cmp);
    Some((values[(values.len() - 1) / 2] + values[values.len() / 2]) / 2.)
}
fn validate_prompts(prompts: &[Prompt]) -> Result<()> {
    ensure!(
        !prompts.is_empty() && prompts.len() <= 32,
        "prompts must contain 1..32 items"
    );
    let mut names = HashSet::new();
    for prompt in prompts {
        ensure!(
            !prompt.name.trim().is_empty() && prompt.name.len() <= 128,
            "prompt names must contain 1..128 UTF-8 bytes"
        );
        ensure!(
            names.insert(&prompt.name),
            "duplicate prompt name {}",
            prompt.name
        );
        ensure!(
            !prompt.prompt.trim().is_empty() && prompt.prompt.len() <= 16384,
            "prompt {} must contain 1..16384 UTF-8 bytes",
            prompt.name
        );
    }
    Ok(())
}
fn load_prompts(path: &Option<PathBuf>) -> Result<Vec<Prompt>> {
    let prompts: Vec<Prompt> = if let Some(path) = path {
        ensure!(
            std::fs::metadata(path)?.len() <= 1024 * 1024,
            "prompt file exceeds 1 MiB"
        );
        serde_json::from_slice(&std::fs::read(path)?)
            .with_context(|| format!("invalid prompt list in {}", path.display()))?
    } else {
        serde_json::from_str(include_str!("../docs/mixed-prompts.json"))?
    };
    validate_prompts(&prompts)?;
    Ok(prompts)
}
fn request(options: &Options, prompt: &Prompt) -> GenerationRequest {
    GenerationRequest {
        messages: vec![Message {
            role: "user".into(),
            content: prompt.prompt.clone(),
        }],
        max_tokens: options.max_tokens,
        temperature: options.temperature,
        top_p: options.top_p,
        top_k: options.top_k,
        seed: options.seed,
        enable_thinking: options.thinking,
    }
}
fn record(
    prompt: &Prompt,
    run: usize,
    mode: Mode,
    output: GenerationOutput,
    stats: MtpGenerationStats,
    wall: f64,
    before: Value,
    after: Value,
) -> Result<Capture> {
    ensure!(
        output.completion_tokens == stats.token_ids.len()
            && output.completion_tokens == stats.completion_tokens,
        "generation completion count disagrees with verified token IDs"
    );
    ensure!(
        output.decode_seconds.is_finite()
            && output.decode_seconds > 0.
            && output.prefill_seconds.is_finite()
            && output.prefill_seconds >= 0.
            && wall.is_finite()
            && wall > 0.,
        "generation timing is invalid"
    );
    ensure!(
        output.decode_seconds == stats.decode_seconds,
        "generation decode timing disagrees with stats"
    );
    let sustained = output.completion_tokens.saturating_sub(1);
    Ok(Capture {
        prompt_name: prompt.name.clone(),
        run,
        warmup: run == 0,
        mode,
        prompt_tokens: output.prompt_tokens,
        completion_tokens: output.completion_tokens,
        sustained_decode_tokens: sustained,
        prefill_seconds: output.prefill_seconds,
        decode_seconds: output.decode_seconds,
        generation_wall_seconds: wall,
        completion_tokens_per_decode_second: rate(output.completion_tokens, output.decode_seconds),
        sustained_decode_tokens_per_second: rate(sustained, output.decode_seconds),
        end_to_end_tokens_per_second: rate(output.completion_tokens, wall),
        finish_reason: output.finish_reason,
        output_text: output.text,
        stats,
        conditions_before: before,
        conditions_after: after,
    })
}
fn summarize(captures: &[Capture], prompt: Option<&str>, mode: Mode) -> Value {
    let rows: Vec<_> = captures
        .iter()
        .filter(|c| !c.warmup && c.mode == mode && prompt.is_none_or(|name| c.prompt_name == name))
        .collect();
    let completion: usize = rows.iter().map(|c| c.completion_tokens).sum();
    let sustained: usize = rows.iter().map(|c| c.sustained_decode_tokens).sum();
    let decode: f64 = rows.iter().map(|c| c.decode_seconds).sum();
    let wall: f64 = rows.iter().map(|c| c.generation_wall_seconds).sum();
    let proposed: usize = rows.iter().map(|c| c.stats.proposed_drafts).sum();
    let accepted: usize = rows.iter().map(|c| c.stats.accepted_drafts).sum();
    let rounds: usize = rows.iter().map(|c| c.stats.rounds).sum();
    json!({
        "mode": mode,
        "measured_runs": rows.len(),
        "completion_tokens": completion,
        "sustained_decode_tokens": sustained,
        "total_decode_seconds": decode,
        "total_generation_wall_seconds": wall,
        "weighted_completion_tokens_per_decode_second": rate(completion, decode),
        "weighted_sustained_decode_tokens_per_second": rate(sustained, decode),
        "weighted_end_to_end_tokens_per_second": rate(completion, wall),
        "median_completion_tokens_per_decode_second": median(rows.iter().filter_map(|r| r.completion_tokens_per_decode_second).collect()),
        "median_sustained_decode_tokens_per_second": median(rows.iter().filter_map(|r| r.sustained_decode_tokens_per_second).collect()),
        "proposed_drafts": proposed, "accepted_drafts": accepted, "verification_rounds": rounds,
        "draft_acceptance_ratio": (proposed > 0).then(|| accepted as f64 / proposed as f64),
        "note": "Only measured runs; actual verified output IDs determine throughput. The initial completion comes from prefill logits, so sustained throughput excludes one token per generation. Accepted draft counts may include an unused verified suffix after EOS and never determine completion throughput."
    })
}
fn compare_greedy(first: &Capture, second: &Capture) -> Value {
    let (mtp, reference) = if first.mode == Mode::Mtp {
        (first, second)
    } else {
        (second, first)
    };
    let first_difference = mtp
        .stats
        .token_ids
        .iter()
        .zip(&reference.stats.token_ids)
        .position(|(a, b)| a != b)
        .or_else(|| {
            (mtp.stats.token_ids.len() != reference.stats.token_ids.len()).then_some(
                mtp.stats
                    .token_ids
                    .len()
                    .min(reference.stats.token_ids.len()),
            )
        });
    json!({
        "prompt_name": mtp.prompt_name, "run": mtp.run, "warmup": mtp.warmup,
        "token_ids_equal": first_difference.is_none(),
        "output_text_equal": mtp.output_text == reference.output_text,
        "finish_reason_equal": mtp.finish_reason == reference.finish_reason,
        "first_differing_token_index": first_difference,
        "mtp_token_at_difference": first_difference.and_then(|i| mtp.stats.token_ids.get(i)),
        "target_token_at_difference": first_difference.and_then(|i| reference.stats.token_ids.get(i)),
        "sustained_decode_speedup": reference.sustained_decode_tokens_per_second
            .zip(mtp.sustained_decode_tokens_per_second)
            .and_then(|(r, m)| (r > 0.).then_some(m / r)),
    })
}

pub fn run(options: Options) -> Result<()> {
    ensure!(
        options.runs > 0 && options.runs <= 100,
        "runs must be 1..100"
    );
    ensure!(
        options.max_tokens > 0 && options.max_tokens <= 8192,
        "max-tokens must be 1..8192"
    );
    ensure!(
        (1..=4).contains(&options.block_size),
        "block-size must be 1..4"
    );
    ensure!(
        options.temperature.is_finite() && options.temperature >= 0.,
        "temperature must be finite and nonnegative"
    );
    ensure!(
        options.top_p.is_finite() && options.top_p > 0. && options.top_p <= 1.,
        "top-p must be in (0,1]"
    );
    let prompts = load_prompts(&options.prompts)?;
    let tokenizer = ChatTokenizer::load(&options.model)?;
    let mut workload = Vec::with_capacity(prompts.len());
    for prompt in &prompts {
        let request = request(&options, prompt);
        let rendered = tokenizer.render(&request.messages, request.enable_thinking)?;
        let token_ids = tokenizer.encode(&rendered)?;
        ensure!(
            !token_ids.is_empty()
                && token_ids
                    .len()
                    .checked_add(options.max_tokens)
                    .is_some_and(|n| n <= options.context),
            "prompt {} plus max-tokens exceeds context",
            prompt.name
        );
        workload.push(json!({"name": prompt.name, "prompt": prompt.prompt,
            "rendered_prompt": rendered, "prompt_token_ids": token_ids, "prompt_tokens": token_ids.len()}));
    }
    drop(tokenizer);
    eprintln!(
        "Loading target and MTP adapter once; {} prompts, {} measured runs per mode.",
        prompts.len(),
        options.runs
    );
    let loaded = Instant::now();
    let mut engine = MtpChatEngine::load(
        &options.model,
        &options.mtp,
        options.context,
        options.block_size,
    )?;
    let load_seconds = loaded.elapsed().as_secs_f64();
    let mut captures = Vec::new();
    let mut comparisons = Vec::new();
    for prompt in &prompts {
        let request = request(&options, prompt);
        for run in 0..=options.runs {
            let modes: &[Mode] = if !options.compare {
                &[Mode::Mtp]
            } else if run % 2 == 0 {
                &[Mode::SequentialTarget, Mode::Mtp]
            } else {
                &[Mode::Mtp, Mode::SequentialTarget]
            };
            let pair_start = captures.len();
            for &mode in modes {
                eprintln!(
                    "{} {:?} {} {}",
                    prompt.name,
                    mode,
                    if run == 0 { "warmup" } else { "run" },
                    run
                );
                let before = crate::system_status::snapshot();
                let started = Instant::now();
                let (output, stats) = match mode {
                    Mode::Mtp => engine.generate_with_stats(&request, &mut |_| true)?,
                    Mode::SequentialTarget => {
                        engine.generate_reference_with_stats(&request, &mut |_| true)?
                    }
                };
                let wall = started.elapsed().as_secs_f64();
                let after = crate::system_status::snapshot();
                let row = record(prompt, run, mode, output, stats, wall, before, after)?;
                eprintln!(
                    "{}",
                    json!({"prompt": prompt.name, "mode": mode, "run": run,
                    "warmup": run == 0, "completion_tokens": row.completion_tokens,
                    "sustained_decode_tokens_per_second": row.sustained_decode_tokens_per_second,
                    "decode_seconds": row.decode_seconds})
                );
                captures.push(row);
            }
            if options.compare && options.temperature == 0. {
                comparisons.push(compare_greedy(
                    &captures[pair_start],
                    &captures[pair_start + 1],
                ));
            }
        }
    }
    let mut per_prompt = Vec::new();
    let mut all_prompt_medians_at_least_32 = true;
    for prompt in &prompts {
        let mtp = summarize(&captures, Some(&prompt.name), Mode::Mtp);
        all_prompt_medians_at_least_32 &= mtp["median_sustained_decode_tokens_per_second"]
            .as_f64()
            .is_some_and(|v| v >= 32.);
        per_prompt.push(json!({"prompt_name": prompt.name, "mtp": mtp,
            "sequential_target": options.compare.then(|| summarize(&captures, Some(&prompt.name), Mode::SequentialTarget))}));
    }
    let greedy_agreement = (!comparisons.is_empty()).then(|| {
        comparisons.iter().all(|c| {
            c["token_ids_equal"] == true
                && c["output_text_equal"] == true
                && c["finish_reason_equal"] == true
        })
    });
    let minimum_sustained_decode_tokens_per_run = captures
        .iter()
        .filter(|c| !c.warmup && c.mode == Mode::Mtp)
        .map(|c| c.sustained_decode_tokens)
        .min();
    let sustained_sample_sufficient =
        minimum_sustained_decode_tokens_per_run.is_some_and(|n| n >= 64);
    let target = engine.engine();
    let report = json!({
        "kind": "verified_mtp_chat_benchmark", "engine_version": env!("CARGO_PKG_VERSION"),
        "device": target.device_name(), "model_path": options.model, "mtp_path": options.mtp,
        "context_capacity": options.context, "block_size": options.block_size,
        "max_tokens": options.max_tokens, "measured_runs_per_prompt_per_mode": options.runs,
        "load_seconds": load_seconds, "allocated_bytes": target.allocated_bytes(),
        "allocation_note": "Metal device-wide current allocation, including target, MTP adapter and scratch buffers.",
        "kernel_mode": target.kernel_mode(), "norm_mode": target.norm_mode(),
        "attention_mode": target.attention_mode(), "metadata_mode": target.metadata_mode(),
        "compacted_matrices": target.metadata_stats().0, "metadata_saved_bytes": target.metadata_stats().1,
        "sampling": {"temperature": options.temperature, "top_p": options.top_p,
            "top_k": options.top_k, "seed": options.seed, "thinking": options.thinking},
        "comparison_enabled": options.compare, "greedy_agreement": greedy_agreement,
        "all_mtp_prompt_medians_at_least_32_sustained_tps": all_prompt_medians_at_least_32,
        "minimum_sustained_decode_tokens_per_run": minimum_sustained_decode_tokens_per_run,
        "sustained_sample_sufficient": sustained_sample_sufficient,
        "goal_32_tps_confirmed": all_prompt_medians_at_least_32 && greedy_agreement == Some(true) && options.runs >= 3 && sustained_sample_sufficient,
        "workload": workload, "per_prompt": per_prompt,
        "aggregate_mtp": summarize(&captures, None, Mode::Mtp),
        "aggregate_sequential_target": options.compare.then(|| summarize(&captures, None, Mode::SequentialTarget)),
        "greedy_comparisons": comparisons, "captures": captures,
        "notes": "Real prompt/chat tokenization and verified completion IDs, EOS honored, no prompt cache reuse. One separately recorded warmup per prompt/mode is excluded from all summaries. Paired modes alternate AB/BA and share one loaded target; MTP adapter remains resident for both. Sustained decode rate is (completion_tokens-1)/decode_seconds; completion_tokens/decode_seconds is also provided. Both use the entire post-prefill decode interval. System conditions are snapshots outside generation wall timing. MTP proposal acceptance is diagnostic, never output throughput. Greedy IDs/text/finish reason must agree and each prompt's median must reach 32 sustained tok/s over >=3 measured runs, with >=64 sustained decoded tokens in every measured MTP run, for goal_32_tps_confirmed. This confirmation applies only to the recorded workload and conditions. Stochastic paired runs need not have identical output."
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn row(mode: Mode, run: usize, tokens: usize, seconds: f64) -> Capture {
        let stats = MtpGenerationStats {
            token_ids: (0..tokens as u32).collect(),
            completion_tokens: tokens,
            decode_seconds: seconds,
            ..Default::default()
        };
        record(
            &Prompt {
                name: "one".into(),
                prompt: "Example".into(),
            },
            run,
            mode,
            GenerationOutput {
                text: "answer".into(),
                prompt_tokens: 7,
                completion_tokens: tokens,
                finish_reason: "length".into(),
                prefill_seconds: 2.,
                decode_seconds: seconds,
            },
            stats,
            seconds + 2.,
            Value::Null,
            Value::Null,
        )
        .unwrap()
    }
    #[test]
    fn summary_excludes_warmup_counts_actual_tokens_and_weights_elapsed_time() {
        let rows = vec![
            row(Mode::Mtp, 0, 1000, 0.1),
            row(Mode::Mtp, 1, 5, 1.),
            row(Mode::Mtp, 2, 13, 3.),
            row(Mode::SequentialTarget, 1, 900, 0.1),
        ];
        let summary = summarize(&rows, Some("one"), Mode::Mtp);
        assert_eq!(summary["measured_runs"], 2);
        assert_eq!(summary["completion_tokens"], 18);
        assert_eq!(summary["sustained_decode_tokens"], 16);
        assert_eq!(summary["weighted_sustained_decode_tokens_per_second"], 4.);
        assert_eq!(summary["weighted_completion_tokens_per_decode_second"], 4.5);
        assert_eq!(summary["median_sustained_decode_tokens_per_second"], 4.);
    }
    #[test]
    fn first_token_only_is_zero_sustained_decode_and_invalid_times_are_not_rates() {
        let capture = row(Mode::Mtp, 1, 1, 0.1);
        assert_eq!(capture.completion_tokens_per_decode_second, Some(10.));
        assert_eq!(capture.sustained_decode_tokens_per_second, Some(0.));
        for seconds in [0., -1., f64::NAN, f64::INFINITY, f64::MIN_POSITIVE / 1000.] {
            assert_eq!(rate(1, seconds), None);
        }
        assert_eq!(median(vec![]), None);
        assert_eq!(median(vec![2., 8., 4., 6.]), Some(5.));
    }
    #[test]
    fn greedy_comparison_detects_token_and_length_divergence() {
        let mtp = row(Mode::Mtp, 1, 3, 1.);
        let mut target = row(Mode::SequentialTarget, 1, 3, 2.);
        assert_eq!(compare_greedy(&mtp, &target)["token_ids_equal"], true);
        target.stats.token_ids[1] = 99;
        assert_eq!(
            compare_greedy(&mtp, &target)["first_differing_token_index"],
            1
        );
        let target = row(Mode::SequentialTarget, 1, 4, 2.);
        assert_eq!(
            compare_greedy(&target, &mtp)["first_differing_token_index"],
            3
        );
    }
    #[test]
    fn prompt_validation_rejects_empty_duplicate_and_oversized_workloads() {
        assert!(validate_prompts(&[]).is_err());
        let prompt = Prompt {
            name: "x".into(),
            prompt: "hello".into(),
        };
        assert!(validate_prompts(&[prompt.clone(), prompt]).is_err());
        let builtin = load_prompts(&None).unwrap();
        assert_eq!(builtin.len(), 5);
        assert!(builtin.iter().any(|p| p.name.starts_with("pl-")));
        assert!(builtin.iter().any(|p| p.name.starts_with("en-")));
    }
}
