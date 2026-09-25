//! Recurrent state traffic only; this is not model generation throughput.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use metal::Buffer;
use qwen_metal::gpu::Gpu;
use serde_json::json;
use std::time::Instant;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 48)]
    layers: usize,
    #[arg(long, default_value_t = 3)]
    batch: usize,
    #[arg(long, default_value_t = 30)]
    iterations: usize,
}

const KH: usize = 16;
const VH: usize = 48;
const K: usize = 128;
const V: usize = 128;
const ROWS: usize = VH * V;
const STATE: usize = ROWS * K;
const QKV: usize = 2 * KH * K + ROWS;

struct Data {
    q: Buffer,
    a: Buffer,
    b: Buffer,
    alog: Buffer,
    dt: Buffer,
    initial: Buffer,
    state: Buffer,
    y: Buffer,
    snapshots: Buffer,
}

fn values(n: usize, seed: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 71 + seed) % 1021) as f32 / 1021. - 0.5) * scale)
        .collect()
}

fn measure(gpu: &Gpu, d: &Data, args: &Args, fused: bool) -> Result<(f64, f64)> {
    objc::rc::autoreleasepool(|| {
        // Identical starting trajectory for every observation, outside timing.
        let reset = gpu.begin();
        let encoder = gpu.begin_encoding(reset);
        encoder.encode(
            "copy_f32",
            &[&d.initial, &d.state],
            &[(args.layers * STATE) as u32],
            args.layers * STATE,
            128,
        )?;
        encoder.end_encoding()?;
        gpu.finish(reset)?;
        let started = Instant::now();
        let command = gpu.begin();
        let encoder = gpu.begin_encoding(command);
        for layer in 0..args.layers {
            let state_offset = layer * STATE * 4;
            let snapshot_offset = layer * (args.batch + 1) * STATE * 4;
            let output_offset = layer * args.batch * ROWS * 4;
            if fused {
                encoder.encode_offsets(
                    "delta_step_block",
                    &[
                        &d.q,
                        &d.a,
                        &d.b,
                        &d.alog,
                        &d.dt,
                        &d.state,
                        &d.y,
                        &d.snapshots,
                    ],
                    &[0, 0, 0, 0, 0, state_offset, output_offset, snapshot_offset],
                    &[KH as u32, VH as u32, K as u32, V as u32, args.batch as u32],
                    ROWS * 32,
                    128,
                )?;
            } else {
                encoder.encode_offsets(
                    "copy_f32",
                    &[&d.state, &d.snapshots],
                    &[state_offset, snapshot_offset],
                    &[STATE as u32],
                    STATE,
                    128,
                )?;
                for token in 0..args.batch {
                    encoder.encode_offsets(
                        "delta_step",
                        &[&d.q, &d.a, &d.b, &d.alog, &d.dt, &d.state, &d.y],
                        &[
                            token * QKV * 4,
                            token * VH * 4,
                            token * VH * 4,
                            0,
                            0,
                            state_offset,
                            output_offset + token * ROWS * 4,
                        ],
                        &[KH as u32, VH as u32, K as u32, V as u32],
                        ROWS * 32,
                        128,
                    )?;
                    encoder.encode_offsets(
                        "copy_f32",
                        &[&d.state, &d.snapshots],
                        &[state_offset, snapshot_offset + (token + 1) * STATE * 4],
                        &[STATE as u32],
                        STATE,
                        128,
                    )?;
                }
            }
        }
        encoder.end_encoding()?;
        gpu.finish(command)?;
        let wall = started.elapsed().as_secs_f64();
        Ok((
            gpu.last_frame_timing()
                .context("GPU timestamps unavailable")?
                .gpu_seconds,
            wall,
        ))
    })
}

fn median(samples: &[f64]) -> f64 {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    (sorted[(sorted.len() - 1) / 2] + sorted[sorted.len() / 2]) / 2.
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!((1..=4).contains(&args.batch), "batch must be 1..4");
    ensure!((1..=64).contains(&args.layers), "layers must be 1..64");
    ensure!(
        (1..=1000).contains(&args.iterations),
        "iterations must be 1..1000"
    );
    let gpu = Gpu::new_with_reference(false)?;
    let mut data = Data {
        q: gpu.upload_f32(&values(args.batch * QKV, 73, 0.25))?,
        a: gpu.upload_f32(&values(args.batch * VH, 12, 1.5))?,
        b: gpu.upload_f32(&values(args.batch * VH, 43, 2.))?,
        alog: gpu.upload_f32(&values(VH, 21, 0.8))?,
        dt: gpu.upload_f32(&values(VH, 97, 0.5))?,
        initial: gpu.upload_f32(&values(args.layers * STATE, 68, 0.16))?,
        state: gpu.alloc_f32(args.layers * STATE)?,
        y: gpu.alloc_f32(args.layers * args.batch * ROWS)?,
        snapshots: gpu.alloc_f32(args.layers * (args.batch + 1) * STATE)?,
    };
    for i in 0..4 {
        for fused in [i % 2 == 0, i % 2 != 0] {
            measure(&gpu, &data, &args, fused)?;
        }
    }
    let mut baseline_gpu = Vec::new();
    let mut baseline_wall = Vec::new();
    let mut fused_gpu = Vec::new();
    let mut fused_wall = Vec::new();
    let mut observations = Vec::new();
    for iteration in 0..args.iterations {
        for fused in [iteration % 2 != 0, iteration % 2 == 0] {
            let (gpu_time, wall) = measure(&gpu, &data, &args, fused)?;
            observations.push(json!({"pair": iteration, "fused": fused,
                "gpu_seconds": gpu_time, "wall_seconds": wall}));
            if fused {
                fused_gpu.push(gpu_time);
                fused_wall.push(wall);
            } else {
                baseline_gpu.push(gpu_time);
                baseline_wall.push(wall);
            }
        }
    }
    measure(&gpu, &data, &args, false)?;
    let old_state = gpu.read_f32(&data.state, args.layers * STATE)?;
    let old_y = gpu.read_f32(&data.y, args.layers * args.batch * ROWS)?;
    let old_snapshots = gpu.read_f32(&data.snapshots, args.layers * (args.batch + 1) * STATE)?;
    // A fresh zeroed destination catches a missing snapshot write even when
    // the baseline already stored an identical prefix in the original buffer.
    data.snapshots = gpu.alloc_f32(old_snapshots.len())?;
    measure(&gpu, &data, &args, true)?;
    let mut max_difference = 0f32;
    for (buffer, old) in [(&data.state, &old_state), (&data.y, &old_y)] {
        let new = gpu.read_f32(buffer, old.len())?;
        for (&new, &old) in new.iter().zip(old) {
            ensure!(
                new.is_finite() && new.to_bits() == old.to_bits(),
                "Recurrent trajectory mismatch: {new} vs {old}"
            );
            max_difference = max_difference.max((new - old).abs());
        }
    }
    let new_snapshots = gpu.read_f32(&data.snapshots, old_snapshots.len())?;
    for (i, (&new, &old)) in new_snapshots.iter().zip(&old_snapshots).enumerate() {
        if (i / STATE) % (args.batch + 1) == args.batch {
            ensure!(new == 0., "Fused kernel wrote the unused final snapshot");
        } else {
            ensure!(
                new.is_finite() && new.to_bits() == old.to_bits(),
                "Snapshot mismatch at {i}: {new} vs {old}"
            );
            max_difference = max_difference.max((new - old).abs());
        }
    }
    let live_bytes = args.layers * STATE * 4;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "device": gpu.device.name(), "batch": args.batch, "layers": args.layers,
            "dimensions": {"key_heads": KH, "value_heads": VH, "key_dim": K, "value_dim": V},
            "live_state_bytes": live_bytes, "iterations_per_mode": args.iterations,
            "warmup_pairs_excluded": 4, "max_output_state_snapshot_difference": max_difference,
            "baseline": {"dispatches": args.layers * (2 * args.batch + 1),
                "snapshot_copy_payload_bytes": live_bytes * (args.batch + 1),
                "snapshot_copy_read_write_bytes": live_bytes * (args.batch + 1) * 2,
                "gpu_median_seconds": median(&baseline_gpu), "wall_median_seconds": median(&baseline_wall)},
            "fused": {"dispatches": args.layers, "snapshot_write_bytes": live_bytes * args.batch,
                "live_state_read_write_bytes": live_bytes * 2,
                "gpu_median_seconds": median(&fused_gpu), "wall_median_seconds": median(&fused_wall)},
            "gpu_speedup": median(&baseline_gpu) / median(&fused_gpu),
            "gpu_saved_milliseconds": (median(&baseline_gpu) - median(&fused_gpu)) * 1000.,
            "notes": "Pooled state across all layers, original base+every-prefix copies vs register-retained batched updates. AB/BA paired, deterministic nonzero state reset outside every timed observation. Snapshot prefix B is deliberately left unchanged by fused mode and is ignored by the engine. Timings exclude convolution and all model matrices; no model or M5 throughput claim.",
            "observations": observations
        }))?
    );
    Ok(())
}
