use anyhow::{Context, Result, ensure};
use clap::{Parser, Subcommand, ValueEnum};
use qwen_metal::{
    benchmark_timing::{PhaseTimings, StepTiming},
    chat::{GenerationRequest, Message, Sampler, TextGenerator},
    engine::{ChatEngine, Engine},
    gpu::Gpu,
    mtp_chat::MtpChatEngine,
    weights::Checkpoint,
};
mod mtp_benchmark;
mod system_status;
use serde_json::json;
use std::{
    io::{self, Write},
    net::SocketAddr,
    path::PathBuf,
    time::Instant,
};

#[derive(Parser)]
#[command(
    version,
    about = "Independent Rust/Metal Qwen text inference. No external inference engine."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Clone, Copy, ValueEnum)]
enum ProfileBackend {
    Auto,
    Commands,
}

fn parse_mtp_prefill_batch_size(value: &str) -> std::result::Result<usize, String> {
    match value {
        "8" => Ok(8),
        "16" => Ok(16),
        _ => Err("MTP prompt batch size must be 8 or 16".into()),
    }
}

#[derive(Subcommand)]
enum Command {
    /// Validate checkpoint metadata without loading weights onto the GPU.
    Inspect {
        #[arg(long)]
        model: PathBuf,
    },
    /// Start the local OpenAI-style chat API (single active generation).
    Serve {
        #[arg(long)]
        model: PathBuf,
        /// Optional native MTP adapter; only target-verified tokens are emitted.
        #[arg(long)]
        mtp: Option<PathBuf>,
        #[arg(long, default_value_t = 3)]
        mtp_block_size: usize,
        /// Opt-in larger prompt batches; MTP decode block size is unchanged.
        #[arg(long, requires = "mtp", value_parser = parse_mtp_prefill_batch_size)]
        mtp_prefill_batch_size: Option<usize>,
        #[arg(long, default_value_t = 8192)]
        context: usize,
        #[arg(long, default_value = "127.0.0.1:8080")]
        listen: SocketAddr,
    },
    /// Generate a response to one text prompt, streaming to stdout.
    Generate {
        #[arg(long)]
        model: PathBuf,
        /// Optional native MTP adapter; only target-verified tokens are emitted.
        #[arg(long)]
        mtp: Option<PathBuf>,
        #[arg(long, default_value_t = 3)]
        mtp_block_size: usize,
        /// Opt-in larger prompt batches; MTP decode block size is unchanged.
        #[arg(long, requires = "mtp", value_parser = parse_mtp_prefill_batch_size)]
        mtp_prefill_batch_size: Option<usize>,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value_t = 8192)]
        context: usize,
        #[arg(long, default_value_t = 256)]
        max_tokens: usize,
        #[arg(long, default_value_t = 0.7)]
        temperature: f32,
        #[arg(long, default_value_t = 0.8)]
        top_p: f32,
        #[arg(long, default_value_t = 20)]
        top_k: usize,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long)]
        thinking: bool,
    },
    /// Real verified-token mixed-workload benchmark, optionally paired with the target.
    MtpBench {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        mtp: PathBuf,
        #[arg(long, default_value_t = 8192)]
        context: usize,
        #[arg(long, default_value_t = 128)]
        max_tokens: usize,
        #[arg(long, default_value_t = 3)]
        runs: usize,
        #[arg(long, default_value_t = 3)]
        block_size: usize,
        /// JSON list of {name,prompt}; omit for the five built-in Polish/English prompts.
        #[arg(long)]
        prompts: Option<PathBuf>,
        #[arg(long, default_value_t = 0.)]
        temperature: f32,
        #[arg(long, default_value_t = 0.8)]
        top_p: f32,
        #[arg(long, default_value_t = 20)]
        top_k: usize,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long)]
        thinking: bool,
        /// Compare against sequential target generation on the same loaded model.
        #[arg(long)]
        compare: bool,
        /// Compare all four target block kernel configurations on one loaded model (greedy only).
        #[arg(long, conflicts_with_all = ["compare", "compare_mlp_r2"])]
        compare_block_kernels: bool,
        /// Compare selective MLP R2 with legacy MTP and ordinary target generation (greedy B3/BF16).
        #[arg(long, conflicts_with_all = ["compare", "compare_block_kernels"])]
        compare_mlp_r2: bool,
    },
    /// Fixed-token autoregressive benchmark; EOS is deliberately ignored.
    Bench {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value_t = 512)]
        prompt_tokens: usize,
        #[arg(long, default_value_t = 128)]
        generate_tokens: usize,
        #[arg(long, default_value_t = 3)]
        runs: usize,
        #[arg(long, default_value_t = 8192)]
        context: usize,
        /// Record sustained CPU/GPU timing without changing the command graph.
        #[arg(long)]
        timing: bool,
    },
    /// Diagnostic ONLY: real 27B shapes/layer schedule but reused synthetic weights.
    SyntheticBench {
        #[arg(long, default_value_t = 2048)]
        context: usize,
        /// Historical KV slots are ZERO; this is not a real prompt prefill.
        #[arg(long, default_value_t = 512)]
        history: usize,
        #[arg(long, default_value_t = 3)]
        steps: usize,
    },
    /// Diagnostic GPU timestamps for one token; extra encoders affect scheduling.
    Profile {
        /// Omit to use the synthetic reused-weight graph with zero history.
        #[arg(long)]
        model: Option<PathBuf>,
        #[arg(long, default_value_t = 2048)]
        context: usize,
        /// Real fixed-token prefill with --model; zeroed historical KV otherwise.
        #[arg(long, default_value_t = 512)]
        history: usize,
        /// Compare serial and parallel RMS using one load and one prefill.
        #[arg(long)]
        compare_norm: bool,
        /// Commands avoids hardware counter sampling on drivers returning zero samples.
        #[arg(long, value_enum, default_value_t = ProfileBackend::Auto)]
        profile_backend: ProfileBackend,
    },
    /// Measure a packed matrix-vector kernel, not LLM tokens/s.
    KernelBench {
        #[arg(long, default_value_t = 17408)]
        rows: usize,
        #[arg(long, default_value_t = 5120)]
        cols: usize,
        #[arg(long, default_value_t = 50)]
        iterations: usize,
        #[arg(long, default_value_t = 4)]
        bits: u32,
        #[arg(long, default_value_t = 64)]
        group: usize,
        /// Use the original unspecialized kernel for A/B comparison.
        #[arg(long)]
        reference: bool,
        /// Store exactly representable scale/bias values in BF16 (not rounded weights).
        #[arg(long)]
        bf16_metadata: bool,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::MtpBench {
            model,
            mtp,
            context,
            max_tokens,
            runs,
            block_size,
            prompts,
            temperature,
            top_p,
            top_k,
            seed,
            thinking,
            compare,
            compare_block_kernels,
            compare_mlp_r2,
        } => mtp_benchmark::run(mtp_benchmark::Options {
            model,
            mtp,
            context,
            max_tokens,
            runs,
            block_size,
            prompts,
            temperature,
            top_p,
            top_k,
            seed,
            thinking,
            compare,
            compare_block_kernels,
            compare_mlp_r2,
        })?,
        Command::Profile {
            model,
            context,
            history,
            compare_norm,
            profile_backend,
        } => {
            ensure!(
                history
                    .checked_add(if compare_norm { 8 } else { 4 })
                    .is_some_and(|n| n <= context),
                "history plus warmup and two diagnostic steps must fit context"
            );
            let mut engine = if let Some(path) = &model {
                let mut engine = Engine::load(path, context)?;
                for i in 0..history {
                    let token = ((i * 17 + 3) % engine.config().vocab_size) as u32;
                    engine.prefill_token(token, false)?;
                }
                engine
            } else {
                Engine::synthetic_qwen27b(context, history)?
            };
            let mut captures = Vec::new();
            if compare_norm {
                ensure!(
                    engine.kernel_mode() != "reference",
                    "--compare-norm requires QWEN_METAL_REFERENCE=0"
                );
                for parallel in [false, true] {
                    engine.set_parallel_norm(parallel);
                    let capture = profile_step(&mut engine, profile_backend)?;
                    let aborted = capture["profile_execution_failed"].as_bool() == Some(true);
                    captures.push(capture);
                    if aborted {
                        break;
                    }
                }
            } else {
                captures.push(profile_step(&mut engine, profile_backend)?);
            }
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "kind":"gpu_operation_profile", "device":engine.device_name(),
                    "engine_version":env!("CARGO_PKG_VERSION"), "model_path":model,
                    "metadata_mode":engine.metadata_mode(),
                    "attention_mode":engine.attention_mode(),
                    "compacted_matrices":engine.metadata_stats().0, "metadata_saved_bytes":engine.metadata_stats().1,
                    "context_capacity":context, "compare_norm":compare_norm,
                    "synthetic_reused_weights":model.is_none(),
                    "captures":captures,
                    "note":"Diagnostic only: counters use one encoder per operation; commands use one synchronous command buffer per operation. Both change scheduling and are not normal model throughput. Without --model, reused synthetic weights and zeroed historical KV."
                }))?
            );
        }
        Command::SyntheticBench {
            context,
            history,
            steps,
        } => {
            ensure!(
                steps > 0
                    && steps <= 100
                    && history.checked_add(steps + 1).is_some_and(|n| n <= context),
                "history + steps + warmup must fit context; steps in 1..100"
            );
            let start = Instant::now();
            let mut engine = Engine::synthetic_qwen27b(context, history)?;
            let load_seconds = start.elapsed().as_secs_f64();
            let mut sampler = Sampler::new(42);
            let mut logits = engine.forward(3)?;
            let start = Instant::now();
            for _ in 0..steps {
                let token = sampler.sample(&logits, 0., 1., 0)?;
                logits = engine.forward(token)?;
            }
            let seconds = start.elapsed().as_secs_f64();
            println!(
                "{}",
                serde_json::to_string_pretty(
                    &json!({"kind":"synthetic_reused_weights_zero_history",
                "device":engine.device_name(),"allocated_bytes":engine.allocated_bytes(),"load_seconds":load_seconds,
                "context":context,"zero_history":history,"steps":steps,"seconds":seconds,"milliseconds_per_step":seconds*1000./steps as f64,
                "reference":std::env::var("QWEN_METAL_REFERENCE").is_ok_and(|v|v=="1"),
                "kernel_mode":engine.kernel_mode(), "norm_mode":engine.norm_mode(), "attention_mode":engine.attention_mode(),
                "note":"NOT REAL QWEN THROUGHPUT. Shares immutable weights across 64 layers; zero historical KV, one warmup step; lower residency than real checkpoint."})
                )?
            );
        }
        Command::Inspect { model } => {
            let cp = Checkpoint::open(&model)?;
            let c = &cp.config;
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "path":model,"layers":c.num_hidden_layers,"hidden_size":c.hidden_size,
                    "vocab_size":c.vocab_size,"full_attention_layers":c.layer_types.iter().filter(|s|*s=="full_attention").count(),
                    "delta_layers":c.layer_types.iter().filter(|s|*s=="linear_attention").count(),
                    "rotary_dim":c.rotary_dim(),"max_position_embeddings":c.max_position_embeddings,
                    "eos_token_ids":c.eos_token_ids,
                    "validation":"metadata and safetensor structure only; does not validate generation"
                }))?
            );
        }
        Command::Serve {
            model,
            mtp,
            mtp_block_size,
            mtp_prefill_batch_size,
            context,
            listen,
        } => {
            ensure!(
                listen.ip().is_loopback(),
                "--listen must be a loopback address"
            );
            ensure!(
                mtp.is_some() || mtp_block_size == 3,
                "--mtp-block-size requires --mtp"
            );
            let engine: Box<dyn TextGenerator> = if let Some(path) = mtp {
                let mut engine = MtpChatEngine::load(&model, &path, context, mtp_block_size)?;
                if let Some(size) = mtp_prefill_batch_size {
                    engine.set_prefill_batch_size(size)?;
                }
                eprintln!(
                    "Loaded {} with native MTP block size {}, prompt batch size {} on {}; {:.2} GiB device allocated. Listening on http://{listen}",
                    engine.model_id(),
                    mtp_block_size,
                    engine.prefill_batch_size(),
                    engine.engine().device_name(),
                    engine.engine().allocated_bytes() as f64 / 1073741824.
                );
                Box::new(engine)
            } else {
                let engine = ChatEngine::load(&model, context)?;
                eprintln!(
                    "Loaded {} on {}; {:.2} GiB allocated. Listening on http://{listen}",
                    engine.model_id(),
                    engine.engine().device_name(),
                    engine.engine().allocated_bytes() as f64 / 1073741824.
                );
                Box::new(engine)
            };
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?
                .block_on(qwen_metal::server::serve(engine, listen))?;
        }
        Command::Generate {
            model,
            mtp,
            mtp_block_size,
            mtp_prefill_batch_size,
            prompt,
            context,
            max_tokens,
            temperature,
            top_p,
            top_k,
            seed,
            thinking,
        } => {
            ensure!(
                mtp.is_some() || mtp_block_size == 3,
                "--mtp-block-size requires --mtp"
            );
            let request = GenerationRequest {
                messages: vec![Message {
                    role: "user".into(),
                    content: prompt,
                    ..Default::default()
                }],
                max_tokens,
                temperature,
                top_p,
                top_k,
                seed,
                enable_thinking: thinking,
                ..Default::default()
            };
            let mut stream = |text: &str| {
                if text.is_empty() {
                    return true;
                }
                let mut out = io::stdout().lock();
                out.write_all(text.as_bytes())
                    .and_then(|_| out.flush())
                    .is_ok()
            };
            let (output, mtp_stats) = if let Some(path) = mtp {
                let mut engine = MtpChatEngine::load(&model, &path, context, mtp_block_size)?;
                if let Some(size) = mtp_prefill_batch_size {
                    engine.set_prefill_batch_size(size)?;
                }
                if thinking {
                    print!("<think>\n");
                }
                let (output, stats) = engine.generate_with_stats(&request, &mut stream)?;
                (output, Some(stats))
            } else {
                let mut engine = ChatEngine::load(&model, context)?;
                if thinking {
                    print!("<think>\n");
                }
                (engine.generate(&request, &mut stream)?, None)
            };
            println!();
            let mut report = json!({"prompt_tokens":output.prompt_tokens,"completion_tokens":output.completion_tokens,
                "prefill_seconds":output.prefill_seconds,"decode_seconds":output.decode_seconds,"finish_reason":output.finish_reason});
            if let Some(stats) = mtp_stats {
                report["mtp_block_size"] = json!(mtp_block_size);
                report["mtp_prefill_batch_size"] =
                    json!(mtp_prefill_batch_size.unwrap_or(mtp_block_size));
                report["mtp_stats"] = serde_json::to_value(stats)?;
                report["sustained_decode_tokens_per_second"] = json!(
                    (output.decode_seconds > 0.)
                        .then(|| output.completion_tokens.saturating_sub(1) as f64
                            / output.decode_seconds)
                );
            }
            eprintln!("{report}");
        }
        Command::Bench {
            model,
            prompt_tokens,
            generate_tokens,
            runs,
            context,
            timing,
        } => {
            ensure!(
                prompt_tokens > 0 && generate_tokens > 0 && runs > 0 && runs <= 100,
                "token counts must be positive and runs in 1..=100"
            );
            ensure!(
                prompt_tokens
                    .checked_add(generate_tokens)
                    .is_some_and(|n| n <= context),
                "benchmark tokens exceed --context"
            );
            let load = Instant::now();
            let mut engine = Engine::load(&model, context)?;
            let load_seconds = load.elapsed().as_secs_f64();
            let mut records = Vec::new();
            let mut warmup_run = None;
            for run in 0..=runs {
                engine.reset();
                // Preallocate and sample system conditions outside timed phases.
                let mut prefill_timing = timing.then(|| PhaseTimings::with_capacity(prompt_tokens));
                let mut decode_timing =
                    timing.then(|| PhaseTimings::with_capacity(generate_tokens));
                let before_prefill = timing.then(system_status::snapshot);
                let start = Instant::now();
                let mut logits = Vec::new();
                // Synthetic fixed token sequence avoids tokenizer/prompt variability.
                for i in 0..prompt_tokens {
                    let token = ((i * 17 + 3) % engine.config().vocab_size) as u32;
                    let history = engine.position();
                    let step_start = timing.then(Instant::now);
                    logits = engine.prefill_token(token, i + 1 == prompt_tokens)?;
                    if let Some(samples) = &mut prefill_timing {
                        let forward_seconds = step_start.unwrap().elapsed().as_secs_f64();
                        samples.push(StepTiming {
                            history,
                            sampling_seconds: 0.,
                            forward_seconds,
                            frame: engine.last_frame_timing(),
                        })?;
                    }
                }
                let prefill_seconds = start.elapsed().as_secs_f64();
                let before_decode = timing.then(system_status::snapshot);
                let start = Instant::now();
                let mut sampler = Sampler::new(42);
                for _ in 0..generate_tokens {
                    let sampling_start = timing.then(Instant::now);
                    let token = sampler.sample(&logits, 0., 1., 0)?;
                    let sampling_seconds = sampling_start.map(|t| t.elapsed().as_secs_f64());
                    let history = engine.position();
                    let step_start = timing.then(Instant::now);
                    logits = engine.forward(token)?;
                    if let Some(samples) = &mut decode_timing {
                        let forward_seconds = step_start.unwrap().elapsed().as_secs_f64();
                        samples.push(StepTiming {
                            history,
                            sampling_seconds: sampling_seconds.unwrap(),
                            forward_seconds,
                            frame: engine.last_frame_timing(),
                        })?;
                    }
                }
                let decode_seconds = start.elapsed().as_secs_f64();
                let after_decode = timing.then(system_status::snapshot);
                let mut row = json!({"run":run,"warmup":run==0,"prefill_seconds":prefill_seconds,
                    "prefill_tokens_per_second":prompt_tokens as f64/prefill_seconds,
                    "decode_seconds":decode_seconds,"decode_tokens_per_second":generate_tokens as f64/decode_seconds});
                eprintln!("{row}");
                if let (Some(prefill), Some(decode)) = (prefill_timing, decode_timing) {
                    row["timing"] = json!({
                        "prefill":prefill.report(), "decode":decode.report(),
                        "conditions_before_prefill":before_prefill,
                        "conditions_before_decode":before_decode,
                        "conditions_after_decode":after_decode,
                        "prefill_final_token_has_logits":true,
                        "note":"Normal one-command-buffer execution. Sampling and forward are separate. GPU overlaps completion wait. Only the last prefill token computes/reads logits. System conditions are snapshots outside timed phases, not GPU frequency or temperature measurements."
                    });
                }
                if run > 0 {
                    records.push(row);
                } else if timing {
                    warmup_run = Some(row);
                }
            }
            let mut speeds: Vec<f64> = records
                .iter()
                .map(|r| r["decode_tokens_per_second"].as_f64().unwrap())
                .collect();
            speeds.sort_by(f64::total_cmp);
            let median = if speeds.len() % 2 == 1 {
                speeds[speeds.len() / 2]
            } else {
                (speeds[speeds.len() / 2 - 1] + speeds[speeds.len() / 2]) / 2.
            };
            let mut report = json!({"kind":"model_fixed_token_benchmark",
                "engine_version":env!("CARGO_PKG_VERSION"),"model_path":model,"device":engine.device_name(),"kernel_mode":engine.kernel_mode(),"norm_mode":engine.norm_mode(),
                "attention_mode":engine.attention_mode(),
                "metadata_mode":engine.metadata_mode(),"compacted_matrices":engine.metadata_stats().0,"metadata_saved_bytes":engine.metadata_stats().1,
                "allocated_bytes":engine.allocated_bytes(),"context_capacity":context,"prompt_tokens":prompt_tokens,
                "generated_steps":generate_tokens,"load_seconds":load_seconds,"median_decode_tokens_per_second":median,
                "notes":"one unreported warmup; no prompt cache reuse; greedy; fixed tokens; EOS ignored; not a chat quality evaluation",
                "runs":records});
            if let Some(warmup) = warmup_run {
                report["warmup_run"] = warmup;
                report["timing_enabled"] = json!(true);
                report["notes"] = json!(
                    "one separately reported warmup excluded from median; no prompt cache reuse; greedy; fixed tokens; EOS ignored; not a chat quality evaluation; timing adds host bookkeeping, no extra GPU encoders or waits"
                );
            }
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::KernelBench {
            rows,
            cols,
            iterations,
            bits,
            group,
            reference,
            bf16_metadata,
        } => {
            ensure!(
                !reference || !bf16_metadata,
                "--reference requires FP32 metadata"
            );
            ensure!(
                matches!(bits, 4 | 8) && matches!(group, 32 | 64 | 128),
                "bits must be 4 or 8, group 32/64/128"
            );
            ensure!(
                rows > 0 && rows <= 262144 && cols > 0 && cols <= 32768 && cols % group == 0,
                "rows must be 1..262144; cols a multiple of group up to 32768"
            );
            ensure!(
                iterations > 0 && iterations <= 10000,
                "iterations must be 1..10000"
            );
            let elements = rows.checked_mul(cols).context("shape overflow")?;
            ensure!(
                elements <= u32::MAX as usize,
                "shape exceeds GPU index range"
            );
            objc::rc::autoreleasepool(|| -> Result<()> {
                let g = Gpu::new_with_reference(reference)?;
                let packed = vec![0x76543210u32; elements / (32 / bits as usize)];
                let w = g.upload_bytes(bytemuck::cast_slice(&packed))?;
                // Identical, exactly BF16-representable values in both storage modes.
                let scale_values = vec![0.015625f32; elements / group];
                let bias_values = vec![-0.0625f32; elements / group];
                let (scale, bias) = if bf16_metadata {
                    let s = qwen_metal::gpu::pack_bf16_exact(&scale_values).unwrap();
                    let b = qwen_metal::gpu::pack_bf16_exact(&bias_values).unwrap();
                    (
                        g.upload_bytes(bytemuck::cast_slice(&s))?,
                        g.upload_bytes(bytemuck::cast_slice(&b))?,
                    )
                } else {
                    (g.upload_f32(&scale_values)?, g.upload_f32(&bias_values)?)
                };
                let x = g.upload_f32(&vec![0.1; cols])?;
                let y = g.alloc_f32(rows)?;
                let dispatch = |count: usize| -> Result<f64> {
                    let start = Instant::now();
                    let cmd = g.begin();
                    let e = cmd.new_compute_command_encoder();
                    for _ in 0..count {
                        g.encode(
                            e,
                            if bf16_metadata {
                                "matvec_affine_bf16"
                            } else {
                                "matvec_affine"
                            },
                            &[&w, &scale, &bias, &x, &y],
                            &[rows as u32, cols as u32, bits, group as u32],
                            rows * 32,
                            128,
                        )?;
                    }
                    e.end_encoding();
                    g.finish(cmd)?;
                    Ok(start.elapsed().as_secs_f64())
                };
                dispatch(3)?;
                let seconds = dispatch(iterations)?;
                let output = g.read_f32(&y, rows)?;
                ensure!(
                    output.iter().all(|v| v.is_finite()),
                    "non-finite kernel output"
                );
                println!(
                    "{}",
                    serde_json::to_string_pretty(
                        &json!({"kind":"packed_matvec_microbenchmark", "bits":bits,"group_size":group,"kernel_mode":g.kernel_mode(),
                    "metadata_mode":if bf16_metadata {"bf16"} else {"f32"},
                    "synthetic_metadata_values":{"scale":0.015625,"bias":-0.0625},
                    "device":g.device.name(),"rows":rows,"cols":cols,"iterations":iterations,"seconds":seconds,
                    "milliseconds_per_matvec":seconds*1000./iterations as f64,
                    "effective_weight_gb_per_second":(w.length()+scale.length()+bias.length()) as f64*iterations as f64/seconds/1e9,
                    "note":"Repeated synthetic weights may benefit from cache. This is NOT model tokens per second."})
                    )?
                );
                Ok(())
            })?;
        }
    }
    Ok(())
}

fn profile_step(engine: &mut Engine, backend: ProfileBackend) -> Result<serde_json::Value> {
    engine.disable_profiling();
    let token = (3 % engine.config().vocab_size) as u32;
    engine.forward(token)?;
    let normal_history = engine.position();
    let normal_start = Instant::now();
    engine.forward(token)?;
    let normal_wall_seconds = normal_start.elapsed().as_secs_f64();
    let normal_timing = engine.last_frame_timing();
    let mut counter_error = None;
    match backend {
        ProfileBackend::Commands => engine.enable_command_profiling()?,
        ProfileBackend::Auto => {
            if let Err(error) = engine.enable_profiling() {
                counter_error = Some(error.to_string());
                engine.enable_command_profiling()?;
            }
        }
    }
    let mut profile_history = engine.position();
    let mut start = Instant::now();
    let mut execution_failed = false;
    let mut report = match engine.forward(token) {
        Ok(_) => engine.profile_report(),
        Err(error) => {
            execution_failed = true;
            Err(error)
        }
    };
    let mut seconds = start.elapsed().as_secs_f64();
    if !execution_failed && report.is_err() && engine.profile_backend() != Some("command_buffers") {
        counter_error = report.as_ref().err().map(ToString::to_string);
        engine.enable_command_profiling()?;
        profile_history = engine.position();
        start = Instant::now();
        report = match engine.forward(token) {
            Ok(_) => engine.profile_report(),
            Err(error) => {
                execution_failed = true;
                Err(error)
            }
        };
        seconds = start.elapsed().as_secs_f64();
    }
    // Preserve normal-command timings even if the diagnostic backend fails.
    // Missing samples are never converted to zero-duration operations.
    let profile_error = report.as_ref().err().map(ToString::to_string);
    let rows = report.ok();
    Ok(json!({
        "kernel_mode":engine.kernel_mode(), "norm_mode":engine.norm_mode(),
        "attention_mode":engine.attention_mode(),
        "profile_backend":engine.profile_backend(), "counter_error":counter_error, "profile_error":profile_error,
        "profile_execution_failed":execution_failed,
        "normal_step_history":normal_history, "profiled_step_history":profile_history,
        "wall_seconds":seconds, "normal_wall_seconds":normal_wall_seconds,
        "normal_timing":normal_timing,
        "normal_timing_error":if normal_timing.is_none() {Some("Metal command timestamps unavailable")} else {None},
        "profiled_timing":if engine.profile_backend()==Some("command_buffers") {None} else {engine.last_frame_timing()},
        "summed_kernel_seconds":rows.as_ref().map(|rows| rows.iter().map(|r| r.gpu_seconds).sum::<f64>()),
        "operations":rows
    }))
}

#[cfg(test)]
mod cli_tests {
    use super::*;

    #[test]
    fn existing_generate_defaults_keep_mtp_opt_in() {
        let cli = Cli::try_parse_from([
            "qwen-metal",
            "generate",
            "--model",
            "target",
            "--prompt",
            "hello",
        ])
        .unwrap();
        match cli.command {
            Command::Generate {
                mtp,
                mtp_block_size,
                context,
                temperature,
                max_tokens,
                ..
            } => {
                assert!(mtp.is_none());
                assert_eq!(mtp_block_size, 3);
                assert_eq!(context, 8192);
                assert_eq!(temperature, 0.7);
                assert_eq!(max_tokens, 256);
            }
            _ => panic!("wrong command"),
        }
    }

    #[test]
    fn mlp_r2_benchmark_cli_is_available_and_comparison_modes_are_exclusive() {
        let base = [
            "qwen-metal",
            "mtp-bench",
            "--model",
            "target",
            "--mtp",
            "adapter",
        ];
        let mut args = base.to_vec();
        args.push("--compare-mlp-r2");
        assert!(Cli::try_parse_from(&args).is_ok());
        for conflict in ["--compare", "--compare-block-kernels"] {
            let mut conflicting = args.clone();
            conflicting.push(conflict);
            assert!(Cli::try_parse_from(conflicting).is_err());
        }
    }

    #[test]
    fn mtp_benchmark_defaults_are_greedy_with_explicit_paired_option() {
        let cli = Cli::try_parse_from([
            "qwen-metal",
            "mtp-bench",
            "--model",
            "target",
            "--mtp",
            "adapter",
            "--compare",
        ])
        .unwrap();
        match cli.command {
            Command::MtpBench {
                temperature,
                block_size,
                runs,
                max_tokens,
                compare,
                prompts,
                ..
            } => {
                assert_eq!(temperature, 0.);
                assert_eq!(block_size, 3);
                assert_eq!(runs, 3);
                assert_eq!(max_tokens, 128);
                assert!(compare);
                assert!(prompts.is_none());
            }
            _ => panic!("wrong command"),
        }
        assert!(Cli::try_parse_from(["qwen-metal", "mtp-bench", "--model", "target"]).is_err());
    }

    #[test]
    fn server_accepts_mtp_adapter_and_block_size_without_changing_listen_default() {
        let cli = Cli::try_parse_from([
            "qwen-metal",
            "serve",
            "--model",
            "target",
            "--mtp",
            "adapter",
            "--mtp-block-size",
            "4",
        ])
        .unwrap();
        match cli.command {
            Command::Serve {
                mtp,
                mtp_block_size,
                listen,
                ..
            } => {
                assert_eq!(mtp, Some(PathBuf::from("adapter")));
                assert_eq!(mtp_block_size, 4);
                assert_eq!(listen, "127.0.0.1:8080".parse::<SocketAddr>().unwrap());
            }
            _ => panic!("wrong command"),
        }
    }

    #[test]
    fn mtp_prompt_batch_is_opt_in_and_rejects_invalid_cli_requests() {
        for size in ["8", "16"] {
            for command in ["serve", "generate"] {
                let mut args = vec![
                    "qwen-metal",
                    command,
                    "--model",
                    "target",
                    "--mtp",
                    "adapter",
                    "--mtp-prefill-batch-size",
                    size,
                ];
                if command == "generate" {
                    args.extend(["--prompt", "Hej"]);
                }
                assert!(Cli::try_parse_from(args).is_ok(), "{command} batch {size}");
            }
        }
        for size in ["0", "7", "9", "32", "invalid"] {
            assert!(
                Cli::try_parse_from([
                    "qwen-metal",
                    "serve",
                    "--model",
                    "target",
                    "--mtp",
                    "adapter",
                    "--mtp-prefill-batch-size",
                    size,
                ])
                .is_err()
            );
        }
        assert!(
            Cli::try_parse_from([
                "qwen-metal",
                "serve",
                "--model",
                "target",
                "--mtp-prefill-batch-size",
                "8",
            ])
            .is_err()
        );
    }
}
