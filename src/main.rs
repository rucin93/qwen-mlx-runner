use anyhow::{Context, Result, ensure};
use clap::{Parser, Subcommand};
use qwen_metal::{
    chat::{GenerationRequest, Message, Sampler, TextGenerator},
    engine::{ChatEngine, Engine},
    gpu::Gpu,
    weights::Checkpoint,
};
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
        #[arg(long, default_value_t = 8192)]
        context: usize,
        #[arg(long, default_value = "127.0.0.1:8080")]
        listen: SocketAddr,
    },
    /// Generate a response to one text prompt, streaming to stdout.
    Generate {
        #[arg(long)]
        model: PathBuf,
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
    },
    /// Measure only the custom 4-bit matrix-vector kernel, not LLM tokens/s.
    KernelBench {
        #[arg(long, default_value_t = 17408)]
        rows: usize,
        #[arg(long, default_value_t = 5120)]
        cols: usize,
        #[arg(long, default_value_t = 50)]
        iterations: usize,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
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
            context,
            listen,
        } => {
            ensure!(
                listen.ip().is_loopback(),
                "--listen must be a loopback address"
            );
            let engine = ChatEngine::load(&model, context)?;
            eprintln!(
                "Loaded {} on {}; {:.2} GiB allocated. Listening on http://{listen}",
                engine.model_id(),
                engine.engine().device_name(),
                engine.engine().allocated_bytes() as f64 / 1073741824.
            );
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?
                .block_on(qwen_metal::server::serve(Box::new(engine), listen))?;
        }
        Command::Generate {
            model,
            prompt,
            context,
            max_tokens,
            temperature,
            top_p,
            top_k,
            seed,
            thinking,
        } => {
            let mut engine = ChatEngine::load(&model, context)?;
            let request = GenerationRequest {
                messages: vec![Message {
                    role: "user".into(),
                    content: prompt,
                }],
                max_tokens,
                temperature,
                top_p,
                top_k,
                seed,
                enable_thinking: thinking,
            };
            if thinking {
                print!("<think>\n");
            }
            let output = engine.generate(&request, &mut |text| {
                if text.is_empty() {
                    return true;
                }
                let mut out = io::stdout().lock();
                out.write_all(text.as_bytes())
                    .and_then(|_| out.flush())
                    .is_ok()
            })?;
            println!();
            eprintln!(
                "{}",
                json!({"prompt_tokens":output.prompt_tokens,"completion_tokens":output.completion_tokens,
                "prefill_seconds":output.prefill_seconds,"decode_seconds":output.decode_seconds,"finish_reason":output.finish_reason})
            );
        }
        Command::Bench {
            model,
            prompt_tokens,
            generate_tokens,
            runs,
            context,
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
            for run in 0..=runs {
                engine.reset();
                let start = Instant::now();
                let mut logits = Vec::new();
                // Synthetic fixed token sequence avoids tokenizer/prompt variability.
                for i in 0..prompt_tokens {
                    let token = ((i * 17 + 3) % engine.config().vocab_size) as u32;
                    logits = engine.prefill_token(token, i + 1 == prompt_tokens)?;
                }
                let prefill_seconds = start.elapsed().as_secs_f64();
                let start = Instant::now();
                let mut sampler = Sampler::new(42);
                for _ in 0..generate_tokens {
                    let token = sampler.sample(&logits, 0., 1., 0)?;
                    logits = engine.forward(token)?;
                }
                let decode_seconds = start.elapsed().as_secs_f64();
                let row = json!({"run":run,"warmup":run==0,"prefill_seconds":prefill_seconds,
                    "prefill_tokens_per_second":prompt_tokens as f64/prefill_seconds,
                    "decode_seconds":decode_seconds,"decode_tokens_per_second":generate_tokens as f64/decode_seconds});
                eprintln!("{row}");
                if run > 0 {
                    records.push(row);
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
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({"kind":"model_fixed_token_benchmark",
                "engine_version":env!("CARGO_PKG_VERSION"),"model_path":model,"device":engine.device_name(),
                "allocated_bytes":engine.allocated_bytes(),"context_capacity":context,"prompt_tokens":prompt_tokens,
                "generated_steps":generate_tokens,"load_seconds":load_seconds,"median_decode_tokens_per_second":median,
                "notes":"one unreported warmup; no prompt cache reuse; greedy; fixed tokens; EOS ignored; not a chat quality evaluation",
                "runs":records}))?
            );
        }
        Command::KernelBench {
            rows,
            cols,
            iterations,
        } => {
            ensure!(
                rows > 0 && rows <= 262144 && cols > 0 && cols <= 32768 && cols % 64 == 0,
                "rows must be 1..262144; cols a multiple of 64 up to 32768"
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
                let g = Gpu::new()?;
                let packed = vec![0x76543210u32; elements / 8];
                let w = g.upload_bytes(bytemuck::cast_slice(&packed))?;
                let scale = g.upload_f32(&vec![0.01; elements / 64])?;
                let bias = g.upload_f32(&vec![-0.07; elements / 64])?;
                let x = g.upload_f32(&vec![0.1; cols])?;
                let y = g.alloc_f32(rows)?;
                let dispatch = |count: usize| -> Result<f64> {
                    let start = Instant::now();
                    let cmd = g.begin();
                    let e = cmd.new_compute_command_encoder();
                    for _ in 0..count {
                        g.encode(
                            e,
                            "matvec_affine",
                            &[&w, &scale, &bias, &x, &y],
                            &[rows as u32, cols as u32, 4, 64],
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
                    serde_json::to_string_pretty(&json!({"kind":"q4_matvec_microbenchmark",
                    "device":g.device.name(),"rows":rows,"cols":cols,"iterations":iterations,"seconds":seconds,
                    "milliseconds_per_matvec":seconds*1000./iterations as f64,
                    "effective_weight_gb_per_second":(w.length()+scale.length()+bias.length()) as f64*iterations as f64/seconds/1e9,
                    "note":"Repeated synthetic weights may benefit from cache. This is NOT model tokens per second."}))?
                );
                Ok(())
            })?;
        }
    }
    Ok(())
}
