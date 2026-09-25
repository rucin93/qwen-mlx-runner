//! Actual verified-token MTP benchmark, with the sequential target resident once.
use anyhow::{Context, Result, ensure};
use qwen_metal::{
    chat::{ChatTokenizer, GenerationOutput, GenerationRequest, Message},
    gpu::{BlockKernelMode, BlockMatmulMode},
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
    pub compare_block_kernels: bool,
    pub compare_mlp_r2: bool,
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum BlockVariant {
    SequentialTarget,
    LegacySequential,
    SharedSequential,
    LegacyBatched,
    SharedBatched,
    MlpR2Batched,
}
const BLOCK_VARIANTS: [BlockVariant; 4] = [
    BlockVariant::LegacySequential,
    BlockVariant::SharedSequential,
    BlockVariant::LegacyBatched,
    BlockVariant::SharedBatched,
];
impl BlockVariant {
    fn generation_mode(self) -> Mode {
        if self == Self::SequentialTarget {
            Mode::SequentialTarget
        } else {
            Mode::Mtp
        }
    }
    fn kernel_mode(self) -> BlockKernelMode {
        BlockKernelMode {
            matmul: match self {
                Self::SharedSequential | Self::SharedBatched => BlockMatmulMode::Shared,
                Self::MlpR2Batched => BlockMatmulMode::MlpR2,
                _ => BlockMatmulMode::Legacy,
            },
            batched_delta: !matches!(self, Self::LegacySequential | Self::SharedSequential),
        }
    }
}
const MLP_R2_VARIANTS: [BlockVariant; 3] = [
    BlockVariant::SequentialTarget,
    BlockVariant::LegacyBatched,
    BlockVariant::MlpR2Batched,
];
// Latin rotation balances timing positions every three measured runs per prompt.
// It does not balance immediate predecessors.
fn mlp_r2_variant_order(prompt_index: usize, run: usize) -> [BlockVariant; 3] {
    let start = (prompt_index + run.saturating_sub(1)) % MLP_R2_VARIANTS.len();
    std::array::from_fn(|position| MLP_R2_VARIANTS[(start + position) % MLP_R2_VARIANTS.len()])
}
// Each four-run cycle places each configuration once in every timing position.
// Williams rows balance immediate predecessors; alternate prompts/cycles reverse them.
fn variant_order(prompt_index: usize, run: usize) -> [BlockVariant; 4] {
    const ROWS: [[usize; 4]; 4] = [[0, 1, 3, 2], [1, 2, 0, 3], [2, 3, 1, 0], [3, 0, 2, 1]];
    let measured_index = run.saturating_sub(1);
    let row = ROWS[(prompt_index + measured_index) % 4];
    let mut order = row.map(|index| BLOCK_VARIANTS[index]);
    if (prompt_index + measured_index / 4) % 2 == 1 {
        order.reverse();
    }
    order
}
#[derive(Serialize)]
struct Capture {
    prompt_name: String,
    run: usize,
    warmup: bool,
    mode: Mode,
    #[serde(skip_serializing_if = "Option::is_none")]
    variant: Option<BlockVariant>,
    block_kernel_mode: BlockKernelMode,
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
            ..Default::default()
        }],
        max_tokens: options.max_tokens,
        temperature: options.temperature,
        top_p: options.top_p,
        top_k: options.top_k,
        seed: options.seed,
        enable_thinking: options.thinking,
        ..Default::default()
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
    block_kernel_mode: BlockKernelMode,
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
        variant: None,
        block_kernel_mode,
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
    summarize_variant(captures, prompt, mode, None)
}
fn summarize_variant(
    captures: &[Capture],
    prompt: Option<&str>,
    mode: Mode,
    variant: Option<BlockVariant>,
) -> Value {
    let rows: Vec<_> = captures
        .iter()
        .filter(|c| {
            !c.warmup
                && c.mode == mode
                && c.variant == variant
                && prompt.is_none_or(|name| c.prompt_name == name)
        })
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
        "total_target_seconds": rows.iter().map(|c| c.stats.target_seconds).sum::<f64>(),
        "total_draft_seconds": rows.iter().map(|c| c.stats.draft_seconds).sum::<f64>(),
        "total_verification_seconds": rows.iter().map(|c| c.stats.verification_seconds).sum::<f64>(),
        "total_rollback_seconds": rows.iter().map(|c| c.stats.rollback_seconds).sum::<f64>(),
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

fn condition_summary(captures: &[Capture], variant: Option<BlockVariant>) -> Value {
    let snapshots: Vec<_> = captures
        .iter()
        .filter(|capture| !capture.warmup && variant.is_none_or(|v| capture.variant == Some(v)))
        .flat_map(|capture| [&capture.conditions_before, &capture.conditions_after])
        .collect();
    let low_power_true = snapshots
        .iter()
        .filter(|s| s["low_power_mode"] == true)
        .count();
    let low_power_false = snapshots
        .iter()
        .filter(|s| s["low_power_mode"] == false)
        .count();
    let mut thermal_states: Vec<_> = snapshots
        .iter()
        .filter_map(|s| s["thermal_state"].as_str())
        .collect();
    let known_thermal = thermal_states
        .iter()
        .filter(|&&state| matches!(state, "nominal" | "fair" | "serious" | "critical"))
        .count();
    let thermal_unknown = snapshots.len() - known_thermal;
    thermal_states.sort_unstable();
    thermal_states.dedup();
    let low_power_unknown = snapshots.len() - low_power_true - low_power_false;
    json!({
        "measured_snapshot_count": snapshots.len(),
        "low_power_mode_true_snapshots": low_power_true,
        "low_power_mode_false_snapshots": low_power_false,
        "low_power_mode_unknown_snapshots": low_power_unknown,
        "low_power_mode_disabled_for_all_measured_snapshots": !snapshots.is_empty() && low_power_false == snapshots.len(),
        "power_mode_constant_and_known": !snapshots.is_empty() && low_power_unknown == 0 && (low_power_true == 0 || low_power_false == 0),
        "observed_thermal_states": thermal_states,
        "thermal_state_unknown_snapshots": thermal_unknown,
        "thermal_state_constant_and_known": thermal_unknown == 0 && thermal_states.len() == 1,
        "note": "Measured captures only, before/after snapshots. Constant conditions do not establish equal clocks or rule out changes between snapshots. Low Power Mode being off does not prove AC power."
    })
}

fn run_block_comparison(
    options: &Options,
    prompts: &[Prompt],
    workload: Vec<Value>,
    engine: &mut MtpChatEngine,
    load_seconds: f64,
) -> Result<()> {
    ensure!(
        engine.engine().kernel_mode() == "aligned",
        "--compare-block-kernels requires the aligned nonreference GPU path"
    );
    let mut captures = Vec::new();
    let mut comparisons = Vec::new();
    for run in 0..=options.runs {
        for (prompt_index, prompt) in prompts.iter().enumerate() {
            let request = request(options, prompt);
            let group_start = captures.len();
            for variant in variant_order(prompt_index, run) {
                engine.clear_cache();
                engine.set_block_kernel_mode(variant.kernel_mode())?;
                let actual_mode = engine.engine().block_kernel_mode();
                ensure!(
                    actual_mode == variant.kernel_mode(),
                    "target block kernel selection differs from requested configuration"
                );
                eprintln!(
                    "{} {:?} {} {}",
                    prompt.name,
                    variant,
                    if run == 0 { "warmup" } else { "run" },
                    run
                );
                let before = crate::system_status::snapshot();
                let started = Instant::now();
                let (output, stats) = engine.generate_with_stats(&request, &mut |_| true)?;
                let wall = started.elapsed().as_secs_f64();
                let after = crate::system_status::snapshot();
                let mut row = record(
                    prompt,
                    run,
                    Mode::Mtp,
                    output,
                    stats,
                    wall,
                    before,
                    after,
                    actual_mode,
                )?;
                row.variant = Some(variant);
                eprintln!(
                    "{}",
                    json!({"prompt": prompt.name, "variant": variant, "run": run,
                    "warmup": run == 0, "completion_tokens": row.completion_tokens,
                    "sustained_decode_tokens_per_second": row.sustained_decode_tokens_per_second,
                    "decode_seconds": row.decode_seconds})
                );
                captures.push(row);
            }
            let group = &captures[group_start..];
            let baseline = group
                .iter()
                .find(|c| c.variant == Some(BlockVariant::LegacySequential))
                .expect("schedule includes baseline");
            for candidate in group.iter().filter(|c| c.variant != baseline.variant) {
                let mut comparison = compare_greedy(candidate, baseline);
                let object = comparison.as_object_mut().unwrap();
                let candidate_token = object.remove("mtp_token_at_difference").unwrap();
                let baseline_token = object.remove("target_token_at_difference").unwrap();
                let speedup = object.remove("sustained_decode_speedup").unwrap();
                object.insert("variant".into(), json!(candidate.variant));
                object.insert("baseline_variant".into(), json!(baseline.variant));
                object.insert("variant_token_at_difference".into(), candidate_token);
                object.insert("baseline_token_at_difference".into(), baseline_token);
                object.insert(
                    "sustained_decode_speedup_vs_legacy_sequential".into(),
                    speedup,
                );
                comparisons.push(comparison);
            }
        }
    }
    let agreement = !comparisons.is_empty()
        && comparisons.iter().all(|c| {
            c["token_ids_equal"] == true
                && c["output_text_equal"] == true
                && c["finish_reason_equal"] == true
        });
    let variants: Vec<_> = BLOCK_VARIANTS.iter().map(|&variant| {
        let per_prompt: Vec<_> = prompts.iter().map(|prompt| json!({
            "prompt_name": prompt.name,
            "mtp": summarize_variant(&captures, Some(&prompt.name), Mode::Mtp, Some(variant)),
        })).collect();
        json!({
            "variant": variant, "block_kernel_mode": variant.kernel_mode(),
            "aggregate_mtp": summarize_variant(&captures, None, Mode::Mtp, Some(variant)),
            "per_prompt": per_prompt, "system_conditions": condition_summary(&captures, Some(variant)),
        })
    }).collect();
    let target = engine.engine();
    let report = json!({
        "kind": "mtp_block_kernel_comparison", "engine_version": env!("CARGO_PKG_VERSION"),
        "device": target.device_name(), "model_path": options.model, "mtp_path": options.mtp,
        "context_capacity": options.context, "block_size": options.block_size,
        "max_tokens": options.max_tokens, "measured_runs_per_prompt_per_variant": options.runs,
        "load_seconds": load_seconds, "allocated_bytes": target.allocated_bytes(),
        "allocation_note": "Metal device-wide current allocation, including target, MTP adapter and scratch buffers, after all variants.",
        "kernel_mode": target.kernel_mode(), "norm_mode": target.norm_mode(),
        "attention_mode": target.attention_mode(), "metadata_mode": target.metadata_mode(),
        "compacted_matrices": target.metadata_stats().0, "metadata_saved_bytes": target.metadata_stats().1,
        "sampling": {"temperature": options.temperature, "top_p": options.top_p,
            "top_k": options.top_k, "seed": options.seed, "thinking": options.thinking},
        "baseline_variant": BlockVariant::LegacySequential,
        "variant_greedy_agreement": agreement,
        "independent_sequential_target_comparison_performed": false,
        "goal_32_tps_confirmed": false,
        "goal_confirmation_note": "Kernel diagnostic only. Variant equality compares MTP outputs against legacy-matmul/sequential-Delta MTP; it is not an independent sequential-target correctness comparison. No configuration is automatically promoted.",
        "schedule": {
            "iteration_order": "run, prompt, variant",
            "variant_ids": BLOCK_VARIANTS,
            "williams_rows": [[0,1,3,2],[1,2,0,3],[2,3,1,0],[3,0,2,1]],
            "rule": "Run 0 is excluded warmup for every prompt and variant, before measured runs. For measured run r>=1 and zero-based prompt p, choose row (p+r-1)%4; reverse if (p+floor((r-1)/4))%2==1. Warmup uses the run-1 row. Four measured runs place each variant in every position once per prompt. Fewer runs are only partially balanced. Captures are in execution order."
        },
        "workload": workload, "variants": variants, "variant_comparisons": comparisons,
        "system_conditions": condition_summary(&captures, None), "captures": captures,
        "notes": "One loaded target and adapter shared by all configurations. Caches reset and target kernel selection switched outside timing before every generation; adapter kernels are unchanged. Each prompt/configuration has a separately recorded warmup excluded from its summary. Summaries never pool configurations. Verified output IDs determine throughput, EOS is honored, sustained decode excludes the first prefill-produced token. All post-prefill work is timed. Every candidate is compared with legacy/sequential MTP for identical prompt/run including warmup. The word sequential in a variant refers only to DeltaNet recurrence, not to ordinary target generation."
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn comparisons_agree(comparisons: &[Value]) -> bool {
    !comparisons.is_empty()
        && comparisons.iter().all(|c| {
            c["token_ids_equal"] == true
                && c["output_text_equal"] == true
                && c["finish_reason_equal"] == true
        })
}

fn mlp_r2_goal_status(
    captures: &[Capture],
    prompts: &[Prompt],
    runs: usize,
    ordinary_comparisons: &[Value],
    eligible_matrices: usize,
) -> Value {
    let candidate = BlockVariant::MlpR2Batched;
    let measured: Vec<_> = captures
        .iter()
        .filter(|c| !c.warmup && c.mode == Mode::Mtp && c.variant == Some(candidate))
        .collect();
    let minimum = measured.iter().map(|c| c.sustained_decode_tokens).min();
    let sufficient = minimum.is_some_and(|n| n >= 64);
    let complete = !prompts.is_empty()
        && measured.len() == prompts.len() * runs
        && prompts.iter().all(|p| {
            (1..=runs).all(|run| {
                measured
                    .iter()
                    .filter(|c| c.prompt_name == p.name && c.run == run)
                    .count()
                    == 1
            })
        });
    let all_medians = !prompts.is_empty()
        && prompts.iter().all(|prompt| {
            summarize_variant(captures, Some(&prompt.name), Mode::Mtp, Some(candidate))
            ["median_sustained_decode_tokens_per_second"].as_f64().is_some_and(|rate| rate >= 32.)
        });
    let agreement = ordinary_comparisons.len() == prompts.len() * (runs + 1) * 2
        && comparisons_agree(ordinary_comparisons);
    let candidate_applicable = eligible_matrices > 0;
    json!({
        "candidate_variant": candidate,
        "candidate_applicable": candidate_applicable,
        "all_candidate_prompt_medians_at_least_32_sustained_tps": all_medians,
        "minimum_sustained_decode_tokens_per_candidate_run": minimum,
        "sustained_sample_sufficient": sufficient,
        "candidate_measured_schedule_complete": complete,
        "greedy_agreement": agreement,
        "goal_32_tps_confirmed": candidate_applicable && runs >= 3 && complete && all_medians && sufficient && agreement,
        "goal_confirmation_note": "Applies only to the selective MLP R2 candidate on the recorded workload and conditions. The target must contain at least one eligible selective R2 matrix. Every candidate prompt median must reach 32 sustained tok/s over at least three measured runs, each measured candidate run must contain at least 64 sustained tokens, and both MTP variants must match ordinary target IDs, text and finish reason for every prompt/run including warmups."
    })
}

fn run_mlp_r2_comparison(
    options: &Options,
    prompts: &[Prompt],
    workload: Vec<Value>,
    engine: &mut MtpChatEngine,
    load_seconds: f64,
) -> Result<()> {
    ensure!(
        engine.engine().kernel_mode() == "aligned",
        "--compare-mlp-r2 requires the aligned nonreference GPU path"
    );
    ensure!(
        engine.engine().metadata_mode() == "bf16",
        "--compare-mlp-r2 requires QWEN_METAL_METADATA=bf16"
    );
    let mut captures = Vec::new();
    let mut ordinary_comparisons = Vec::new();
    let mut variant_comparisons = Vec::new();
    let mut executed_schedule = Vec::new();
    for run in 0..=options.runs {
        for (prompt_index, prompt) in prompts.iter().enumerate() {
            let request = request(options, prompt);
            let order = mlp_r2_variant_order(prompt_index, run);
            executed_schedule.push(json!({"prompt_name": prompt.name, "run": run, "warmup": run == 0, "variants": order}));
            let start = captures.len();
            for variant in order {
                engine.clear_cache();
                engine.set_block_kernel_mode(variant.kernel_mode())?;
                let actual_mode = engine.engine().block_kernel_mode();
                ensure!(
                    actual_mode == variant.kernel_mode(),
                    "target block kernel selection differs from requested configuration"
                );
                eprintln!(
                    "{} {:?} {} {}",
                    prompt.name,
                    variant,
                    if run == 0 { "warmup" } else { "run" },
                    run
                );
                let before = crate::system_status::snapshot();
                let started = Instant::now();
                let mode = variant.generation_mode();
                let (output, stats) = match mode {
                    Mode::Mtp => engine.generate_with_stats(&request, &mut |_| true)?,
                    Mode::SequentialTarget => {
                        engine.generate_reference_with_stats(&request, &mut |_| true)?
                    }
                };
                let wall = started.elapsed().as_secs_f64();
                let after = crate::system_status::snapshot();
                let mut row = record(
                    prompt,
                    run,
                    mode,
                    output,
                    stats,
                    wall,
                    before,
                    after,
                    actual_mode,
                )?;
                row.variant = Some(variant);
                eprintln!(
                    "{}",
                    json!({"prompt": prompt.name, "variant": variant, "run": run,
                    "warmup": run == 0, "completion_tokens": row.completion_tokens,
                    "sustained_decode_tokens_per_second": row.sustained_decode_tokens_per_second,
                    "decode_seconds": row.decode_seconds})
                );
                captures.push(row);
            }
            let group = &captures[start..];
            let reference = group
                .iter()
                .find(|c| c.variant == Some(BlockVariant::SequentialTarget))
                .expect("schedule includes ordinary target");
            for candidate in group.iter().filter(|c| c.mode == Mode::Mtp) {
                let mut comparison = compare_greedy(candidate, reference);
                comparison["variant"] = json!(candidate.variant);
                comparison["baseline_variant"] = json!(BlockVariant::SequentialTarget);
                ordinary_comparisons.push(comparison);
            }
            let legacy = group
                .iter()
                .find(|c| c.variant == Some(BlockVariant::LegacyBatched))
                .expect("schedule includes legacy MTP");
            let candidate = group
                .iter()
                .find(|c| c.variant == Some(BlockVariant::MlpR2Batched))
                .expect("schedule includes candidate MTP");
            let mut comparison = compare_greedy(candidate, legacy);
            let object = comparison.as_object_mut().unwrap();
            let candidate_token = object.remove("mtp_token_at_difference").unwrap();
            let baseline_token = object.remove("target_token_at_difference").unwrap();
            let speedup = object.remove("sustained_decode_speedup").unwrap();
            object.insert("variant".into(), json!(BlockVariant::MlpR2Batched));
            object.insert(
                "baseline_variant".into(),
                json!(BlockVariant::LegacyBatched),
            );
            object.insert("variant_token_at_difference".into(), candidate_token);
            object.insert("baseline_token_at_difference".into(), baseline_token);
            object.insert("sustained_decode_speedup_vs_legacy_batched".into(), speedup);
            variant_comparisons.push(comparison);
        }
    }
    let variants: Vec<_> = MLP_R2_VARIANTS.iter().map(|&variant| {
        let mode = variant.generation_mode();
        let per_prompt: Vec<_> = prompts.iter().map(|prompt| json!({
            "prompt_name": prompt.name,
            "aggregate": summarize_variant(&captures, Some(&prompt.name), mode, Some(variant)),
        })).collect();
        json!({
            "variant": variant, "mode": mode, "block_kernel_mode": variant.kernel_mode(),
            "aggregate": summarize_variant(&captures, None, mode, Some(variant)),
            "per_prompt": per_prompt, "system_conditions": condition_summary(&captures, Some(variant)),
        })
    }).collect();
    let goal = mlp_r2_goal_status(
        &captures,
        prompts,
        options.runs,
        &ordinary_comparisons,
        engine.engine().mlp_r2_eligible_matrices(),
    );
    let target = engine.engine();
    let mut report = json!({
        "kind": "verified_mtp_mlp_r2_comparison", "engine_version": env!("CARGO_PKG_VERSION"),
        "device": target.device_name(), "model_path": options.model, "mtp_path": options.mtp,
        "context_capacity": options.context, "block_size": options.block_size,
        "max_tokens": options.max_tokens, "measured_runs_per_prompt_per_variant": options.runs,
        "load_seconds": load_seconds, "allocated_bytes": target.allocated_bytes(),
        "allocation_note": "Metal device-wide current allocation, including the one target, resident MTP adapter and scratch buffers, after all variants.",
        "kernel_mode": target.kernel_mode(), "norm_mode": target.norm_mode(),
        "attention_mode": target.attention_mode(), "metadata_mode": target.metadata_mode(),
        "compacted_matrices": target.metadata_stats().0, "metadata_saved_bytes": target.metadata_stats().1,
        "sampling": {"temperature": options.temperature, "top_p": options.top_p,
            "top_k": options.top_k, "seed": options.seed, "thinking": options.thinking},
        "baseline_variant": BlockVariant::LegacyBatched,
        "independent_sequential_target_comparison_performed": true,
        "mlp_r2_eligible_target_matrices": target.mlp_r2_eligible_matrices(),
        "variant_greedy_agreement": comparisons_agree(&variant_comparisons),
        "candidate_scope": "Selective R2 is used only for eligible aligned B3 Q4/group64 BF16 MLP matrices: 17408x5120 and 5120x17408. All other matrices and shorter blocks retain legacy dispatch. Adapter kernels remain unchanged. Small fixtures can exercise accounting without eligible matrices.",
        "schedule": {
            "iteration_order": "run, prompt, variant",
            "variant_ids": MLP_R2_VARIANTS,
            "rule": "Run 0 is one excluded warmup per prompt and variant, completed before measured runs. For measured run r>=1 and zero-based prompt p, rotate variant_ids left by (p+r-1)%3; warmup uses the run-1 order. Three measured runs place each variant once in each position per prompt. This rotation does not balance immediate predecessors. Captures and executed_groups are in execution order.",
            "executed_groups": executed_schedule,
        },
        "workload": workload, "variants": variants,
        "greedy_comparisons": ordinary_comparisons, "variant_comparisons": variant_comparisons,
        "system_conditions": condition_summary(&captures, None), "captures": captures,
        "notes": "One loaded target and adapter shared by all variants. The explicit pre-capture cache reset and kernel selection happen before generation wall timing. Additional entry/exit cache resets inside the generation API are included in generation_wall_seconds and excluded from decode_seconds. Ordinary sequential_target calls the ordinary autoregressive target path; legacy_batched and mlp_r2_batched use verified MTP with batched DeltaNet. Each prompt/variant has one separately recorded warmup excluded from all aggregates. Summaries never pool variants. Actual verified output IDs determine throughput; EOS is honored. Sustained decode excludes the first prefill-produced token and includes the entire post-prefill interval. Both MTP variants are compared to ordinary target outputs for identical prompt/run, including warmups; selective R2 is also compared directly to legacy MTP. No kernel selection is automatically promoted."
    });
    report
        .as_object_mut()
        .unwrap()
        .extend(goal.as_object().unwrap().clone());
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
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
    if options.compare_mlp_r2 {
        ensure!(
            !options.compare && !options.compare_block_kernels,
            "--compare-mlp-r2 cannot be used with --compare or --compare-block-kernels"
        );
        ensure!(
            options.temperature == 0.,
            "--compare-mlp-r2 requires --temperature 0"
        );
        ensure!(
            options.block_size == 3,
            "--compare-mlp-r2 requires --block-size 3"
        );
        ensure!(
            !std::env::var("QWEN_METAL_REFERENCE").is_ok_and(|value| value == "1"),
            "--compare-mlp-r2 requires QWEN_METAL_REFERENCE=0"
        );
        ensure!(
            std::env::var("QWEN_METAL_GEMV").unwrap_or_else(|_| "aligned".into()) == "aligned",
            "--compare-mlp-r2 requires QWEN_METAL_GEMV=aligned"
        );
        ensure!(
            std::env::var("QWEN_METAL_METADATA").is_ok_and(|value| value == "bf16"),
            "--compare-mlp-r2 requires QWEN_METAL_METADATA=bf16"
        );
    }
    if options.compare_block_kernels {
        ensure!(
            !options.compare,
            "--compare-block-kernels cannot be used with --compare"
        );
        ensure!(
            options.temperature == 0.,
            "--compare-block-kernels requires --temperature 0"
        );
        ensure!(
            !std::env::var("QWEN_METAL_REFERENCE").is_ok_and(|value| value == "1"),
            "--compare-block-kernels requires QWEN_METAL_REFERENCE=0"
        );
        ensure!(
            std::env::var("QWEN_METAL_GEMV").unwrap_or_else(|_| "aligned".into()) == "aligned",
            "--compare-block-kernels requires QWEN_METAL_GEMV=aligned"
        );
    }
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
    if crate::system_status::snapshot()["low_power_mode"] == true {
        eprintln!(
            "WARNING: macOS Low Power Mode is enabled. For peak-throughput comparisons, disable it before running this benchmark; connecting power alone does not establish that it is off. Power mode is recorded with each capture."
        );
    }
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
    if options.compare_mlp_r2 {
        return run_mlp_r2_comparison(&options, &prompts, workload, &mut engine, load_seconds);
    }
    if options.compare_block_kernels {
        return run_block_comparison(&options, &prompts, workload, &mut engine, load_seconds);
    }
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
                let row = record(
                    prompt,
                    run,
                    mode,
                    output,
                    stats,
                    wall,
                    before,
                    after,
                    engine.engine().block_kernel_mode(),
                )?;
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
        "block_kernel_mode": target.block_kernel_mode(),
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
            BlockKernelMode::default(),
        )
        .unwrap()
    }
    #[test]
    fn mlp_r2_rotation_balances_positions_without_pooling_variants_or_warmups() {
        for prompt in 0..5 {
            for position in 0..3 {
                let seen: Vec<_> = (1..=3)
                    .map(|run| mlp_r2_variant_order(prompt, run)[position])
                    .collect();
                for variant in MLP_R2_VARIANTS {
                    assert_eq!(seen.iter().filter(|&&v| v == variant).count(), 1);
                }
            }
        }
        assert_eq!(mlp_r2_variant_order(0, 0), mlp_r2_variant_order(0, 1));
        let mut captures = Vec::new();
        for (index, variant) in MLP_R2_VARIANTS.into_iter().enumerate() {
            for (run, tokens, seconds) in [(0, 1000, 0.001), (1, 9, 1.), (2, 17, 3.)] {
                let mut capture = row(
                    variant.generation_mode(),
                    run,
                    tokens,
                    seconds * (index + 1) as f64,
                );
                capture.variant = Some(variant);
                captures.push(capture);
            }
        }
        for (index, variant) in MLP_R2_VARIANTS.into_iter().enumerate() {
            let summary =
                summarize_variant(&captures, None, variant.generation_mode(), Some(variant));
            assert_eq!(summary["measured_runs"], 2);
            assert_eq!(summary["sustained_decode_tokens"], 24);
            assert_eq!(
                summary["weighted_sustained_decode_tokens_per_second"],
                6. / (index + 1) as f64
            );
        }
    }

    #[test]
    fn mlp_r2_goal_requires_every_prompt_sufficient_samples_and_all_ordinary_comparisons() {
        let prompts = vec![
            Prompt {
                name: "one".into(),
                prompt: "Example".into(),
            },
            Prompt {
                name: "two".into(),
                prompt: "Other".into(),
            },
        ];
        let mut captures = Vec::new();
        let mut comparisons = Vec::new();
        for prompt in &prompts {
            for run in 0..=3 {
                let mut reference = row(Mode::SequentialTarget, run, 65, 4.);
                reference.prompt_name = prompt.name.clone();
                reference.variant = Some(BlockVariant::SequentialTarget);
                for variant in [BlockVariant::LegacyBatched, BlockVariant::MlpR2Batched] {
                    let mut capture = row(Mode::Mtp, run, 65, if run == 0 { 100. } else { 2. });
                    capture.prompt_name = prompt.name.clone();
                    capture.variant = Some(variant);
                    comparisons.push(compare_greedy(&capture, &reference));
                    captures.push(capture);
                }
                captures.push(reference);
            }
        }
        let status = mlp_r2_goal_status(&captures, &prompts, 3, &comparisons, 1);
        assert_eq!(status["goal_32_tps_confirmed"], true);
        assert_eq!(status["candidate_applicable"], true);
        let inapplicable = mlp_r2_goal_status(&captures, &prompts, 3, &comparisons, 0);
        assert_eq!(inapplicable["candidate_applicable"], false);
        assert_eq!(inapplicable["goal_32_tps_confirmed"], false);
        assert_eq!(
            inapplicable["all_candidate_prompt_medians_at_least_32_sustained_tps"],
            true
        );
        assert_eq!(
            status["minimum_sustained_decode_tokens_per_candidate_run"],
            64
        );
        assert_eq!(
            mlp_r2_goal_status(&captures, &prompts, 2, &comparisons, 1)["goal_32_tps_confirmed"],
            false
        );
        // Warmup mismatches count even though warmup timings never enter medians.
        comparisons[0]["token_ids_equal"] = json!(false);
        assert_eq!(
            mlp_r2_goal_status(&captures, &prompts, 3, &comparisons, 1)["goal_32_tps_confirmed"],
            false
        );
        comparisons[0]["token_ids_equal"] = json!(true);
        comparisons[0]["finish_reason_equal"] = json!(false);
        assert_eq!(
            mlp_r2_goal_status(&captures, &prompts, 3, &comparisons, 1)["goal_32_tps_confirmed"],
            false
        );
        comparisons[0]["finish_reason_equal"] = json!(true);
        comparisons[0]["output_text_equal"] = json!(false);
        assert_eq!(
            mlp_r2_goal_status(&captures, &prompts, 3, &comparisons, 1)["goal_32_tps_confirmed"],
            false
        );
        comparisons[0]["output_text_equal"] = json!(true);
        assert_eq!(
            mlp_r2_goal_status(
                &captures,
                &prompts,
                3,
                &comparisons[..comparisons.len() - 1],
                1
            )["goal_32_tps_confirmed"],
            false
        );
        let candidate = captures
            .iter_mut()
            .find(|c| c.variant == Some(BlockVariant::MlpR2Batched) && c.run == 1)
            .unwrap();
        candidate.sustained_decode_tokens = 63;
        assert_eq!(
            mlp_r2_goal_status(&captures, &prompts, 3, &comparisons, 1)["goal_32_tps_confirmed"],
            false
        );
        let candidate = captures
            .iter_mut()
            .find(|c| c.variant == Some(BlockVariant::MlpR2Batched) && c.run == 1)
            .unwrap();
        candidate.sustained_decode_tokens = 64;
        for capture in captures
            .iter_mut()
            .filter(|c| c.variant == Some(BlockVariant::MlpR2Batched) && c.prompt_name == "two")
        {
            capture.sustained_decode_tokens_per_second = Some(31.9);
        }
        assert_eq!(
            mlp_r2_goal_status(&captures, &prompts, 3, &comparisons, 1)["goal_32_tps_confirmed"],
            false
        );
        assert_eq!(
            mlp_r2_goal_status(&captures, &[], 3, &[], 1)["goal_32_tps_confirmed"],
            false
        );
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
    fn factorial_schedule_balances_positions_and_separates_configuration_summaries() {
        for prompt_index in 0..5 {
            for cycle in 0..2 {
                for position in 0..4 {
                    let mut seen = Vec::new();
                    for run in cycle * 4 + 1..=cycle * 4 + 4 {
                        let order = variant_order(prompt_index, run);
                        for &variant in &BLOCK_VARIANTS {
                            assert_eq!(order.iter().filter(|&&v| v == variant).count(), 1);
                        }
                        seen.push(order[position]);
                    }
                    for &variant in &BLOCK_VARIANTS {
                        assert_eq!(seen.iter().filter(|&&v| v == variant).count(), 1);
                    }
                }
            }
        }
        let mut rows = Vec::new();
        for (index, variant) in BLOCK_VARIANTS.iter().enumerate() {
            for (run, tokens, seconds) in [(0, 1000, 0.001), (1, 9, 1.), (2, 17, 3.)] {
                let mut capture = row(Mode::Mtp, run, tokens, seconds * (index + 1) as f64);
                capture.variant = Some(*variant);
                capture.block_kernel_mode = variant.kernel_mode();
                rows.push(capture);
            }
        }
        rows.push(row(Mode::Mtp, 1, 1000, 0.001));
        for (index, &variant) in BLOCK_VARIANTS.iter().enumerate() {
            let summary = summarize_variant(&rows, Some("one"), Mode::Mtp, Some(variant));
            assert_eq!(summary["measured_runs"], 2);
            assert_eq!(summary["sustained_decode_tokens"], 24);
            assert_eq!(
                summary["weighted_sustained_decode_tokens_per_second"],
                6. / (index + 1) as f64
            );
        }
        assert_eq!(summarize(&rows, None, Mode::Mtp)["measured_runs"], 1);
    }
    #[test]
    fn condition_summary_reports_power_and_thermal_changes_without_inventing_ac_state() {
        let mut measured = row(Mode::Mtp, 1, 9, 1.);
        measured.conditions_before = json!({"low_power_mode": false, "thermal_state": "nominal"});
        measured.conditions_after = json!({"low_power_mode": false, "thermal_state": "fair"});
        let mut warmup = row(Mode::Mtp, 0, 9, 1.);
        warmup.conditions_before = json!({"low_power_mode": true, "thermal_state": "serious"});
        let summary = condition_summary(&[warmup, measured], None);
        assert_eq!(summary["measured_snapshot_count"], 2);
        assert_eq!(
            summary["low_power_mode_disabled_for_all_measured_snapshots"],
            true
        );
        assert_eq!(summary["power_mode_constant_and_known"], true);
        assert_eq!(summary["thermal_state_constant_and_known"], false);
        assert_eq!(
            summary["observed_thermal_states"],
            json!(["fair", "nominal"])
        );
        let unknown = condition_summary(&[row(Mode::Mtp, 1, 9, 1.)], None);
        assert_eq!(unknown["power_mode_constant_and_known"], false);
        assert_eq!(unknown["thermal_state_constant_and_known"], false);
        let mut literal_unknown = row(Mode::Mtp, 1, 9, 1.);
        literal_unknown.conditions_before =
            json!({"low_power_mode": false, "thermal_state": "unknown"});
        literal_unknown.conditions_after = literal_unknown.conditions_before.clone();
        let unknown = condition_summary(&[literal_unknown], None);
        assert_eq!(unknown["thermal_state_constant_and_known"], false);
        assert_eq!(unknown["thermal_state_unknown_snapshots"], 2);
        assert_eq!(unknown["observed_thermal_states"], json!(["unknown"]));
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
