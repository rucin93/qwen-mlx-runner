//! Reused-buffer attention microbenchmark, not model generation throughput.
//!
//! Run: cargo run --release --locked --example attention_bench
use anyhow::{Context, Result, ensure};
use metal::BufferRef;
use qwen_metal::gpu::Gpu;
use serde_json::json;
use std::time::Instant;

fn dispatch(gpu: &Gpu, buffers: &[&BufferRef], params: &[u32]) -> Result<(f64, f64)> {
    let started = Instant::now();
    let command = gpu.begin();
    let encoder = command.new_compute_command_encoder();
    gpu.encode(
        encoder,
        "attn_values",
        buffers,
        params,
        params[0] as usize * params[2] as usize,
        128,
    )?;
    encoder.end_encoding();
    gpu.finish(command)?;
    let wall_seconds = started.elapsed().as_secs_f64();
    let timing = gpu
        .last_frame_timing()
        .context("Metal command completed without valid GPU timestamps")?;
    Ok((timing.gpu_seconds, wall_seconds))
}

fn summary(seconds: &[f64]) -> serde_json::Value {
    let total: f64 = seconds.iter().sum();
    let mut sorted = seconds.to_vec();
    sorted.sort_by(f64::total_cmp);
    let midpoint = sorted.len() / 2;
    let median = if sorted.len() % 2 == 0 {
        (sorted[midpoint - 1] + sorted[midpoint]) / 2.
    } else {
        sorted[midpoint]
    };
    json!({
        "total_seconds": total,
        "mean_milliseconds": total * 1000. / seconds.len() as f64,
        "median_milliseconds": median * 1000.,
        "minimum_milliseconds": sorted[0] * 1000.,
        "maximum_milliseconds": sorted[sorted.len() - 1] * 1000.,
    })
}

fn main() -> Result<()> {
    let mut gpu = Gpu::new_with_reference(false)?;
    let heads = 24usize;
    let kv_heads = 4usize;
    let dim = 256usize;
    let warmup_iterations = 5usize;
    let mut captures = Vec::new();

    for history in [128usize, 515, 2048, 8192] {
        let iterations = if history == 8192 { 10 } else { 50 };
        let mut scores = Vec::with_capacity(heads * history);
        for head in 0..heads {
            let raw: Vec<f64> = (0..history)
                .map(|token| (1 + (token * 17 + head * 23) % 97) as f64)
                .collect();
            let sum: f64 = raw.iter().sum();
            scores.extend(raw.into_iter().map(|v| (v / sum) as f32));
        }
        let values: Vec<f32> = (0..history * kv_heads * dim)
            .map(|i| ((i * 29 % 251) as f32 - 125.) / 31.)
            .collect();
        let gates: Vec<f32> = (0..heads * dim)
            .map(|i| ((i % 17) as f32 - 8.) / 4.)
            .collect();
        let score_buffer = gpu.upload_f32(&scores)?;
        let value_buffer = gpu.upload_f32(&values)?;
        let gate_buffer = gpu.upload_f32(&gates)?;
        let output = gpu.alloc_f32(heads * dim)?;
        drop((scores, values, gates));
        let buffers = [&*score_buffer, &*value_buffer, &*gate_buffer, &*output];
        let params = [heads as u32, kv_heads as u32, dim as u32, history as u32];
        let mut serial_output: Option<Vec<f32>> = None;

        // One device and identical buffers for both modes; all GPU work is
        // sequential. Warmup and validation readback are outside timed samples.
        for parallel in [false, true] {
            gpu.set_parallel_attention(parallel);
            for _ in 0..warmup_iterations {
                dispatch(&gpu, &buffers, &params)?;
            }
            let mut gpu_seconds = Vec::with_capacity(iterations);
            let mut wall_seconds = Vec::with_capacity(iterations);
            for _ in 0..iterations {
                let (gpu_elapsed, wall_elapsed) = dispatch(&gpu, &buffers, &params)?;
                gpu_seconds.push(gpu_elapsed);
                wall_seconds.push(wall_elapsed);
            }
            let actual = gpu.read_f32(&output, heads * dim)?;
            ensure!(
                actual.iter().all(|v| v.is_finite()),
                "Non-finite attention output"
            );
            let max_absolute_difference_from_serial = if let Some(serial) = &serial_output {
                let mut maximum = 0f64;
                for (&got, &want) in actual.iter().zip(serial) {
                    let difference = (f64::from(got) - f64::from(want)).abs();
                    ensure!(
                        difference < 3e-6 + f64::from(want).abs() * 3e-5,
                        "Attention modes disagree at history {history}: {got} != {want}"
                    );
                    maximum = maximum.max(difference);
                }
                Some(maximum)
            } else {
                serial_output = Some(actual);
                None
            };
            captures.push(json!({
                "attention_mode": gpu.attention_mode(),
                "heads": heads,
                "kv_heads": kv_heads,
                "head_dim": dim,
                "history": history,
                "warmup_iterations": warmup_iterations,
                "measured_iterations": iterations,
                "gpu_interval": summary(&gpu_seconds),
                "wall_interval": summary(&wall_seconds),
                "max_absolute_difference_from_serial": max_absolute_difference_from_serial,
            }));
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "kind": "attention_values_microbenchmark",
            "engine_version": env!("CARGO_PKG_VERSION"),
            "device": gpu.device.name(),
            "logical_kernel": "attn_values",
            "note": "One production attention-values dispatch per command buffer; reused inputs may benefit from cache. GPU interval and wall interval overlap and must not be added. This is NOT model tokens per second.",
            "captures": captures,
        }))?
    );
    Ok(())
}
