//! Shared-weight block vs separate GEMVs, never model generation throughput.
//! cargo run --release --locked --example block_matrix_bench -- --batch 3
use anyhow::{Context, Result, ensure};
use clap::{Parser, ValueEnum};
use metal::BufferRef;
use qwen_metal::gpu::Gpu;
use serde_json::json;
use std::time::Instant;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Metadata {
    F32,
    Bf16,
}
#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 17408)]
    rows: usize,
    #[arg(long, default_value_t = 5120)]
    cols: usize,
    #[arg(long, default_value_t = 3)]
    batch: usize,
    #[arg(long, default_value_t = 30)]
    iterations: usize,
    #[arg(long, value_enum, default_value_t = Metadata::Bf16)]
    metadata: Metadata,
}
fn measure(gpu: &Gpu, buffers: &[&BufferRef], args: &Args, block: bool) -> Result<(f64, f64)> {
    objc::rc::autoreleasepool(|| {
        let started = Instant::now();
        let command = gpu.begin();
        let encoder = gpu.begin_encoding(command);
        let bf16 = matches!(args.metadata, Metadata::Bf16);
        if block {
            encoder.encode(
                if bf16 {
                    "matmul_affine_bf16"
                } else {
                    "matmul_affine"
                },
                buffers,
                &[args.rows as u32, args.cols as u32, 4, 64, args.batch as u32],
                args.rows * 32,
                128,
            )?;
        } else {
            for token in 0..args.batch {
                encoder.encode_offsets(
                    if bf16 {
                        "matvec_affine_bf16"
                    } else {
                        "matvec_affine"
                    },
                    buffers,
                    &[0, 0, 0, token * args.cols * 4, token * args.rows * 4],
                    &[args.rows as u32, args.cols as u32, 4, 64],
                    args.rows * 32,
                    128,
                )?;
            }
        }
        encoder.end_encoding()?;
        gpu.finish(command)?;
        let wall = started.elapsed().as_secs_f64();
        let timing = gpu
            .last_frame_timing()
            .context("GPU timestamps unavailable")?;
        Ok((timing.gpu_seconds, wall))
    })
}
fn median(samples: &[f64]) -> f64 {
    let mut values = samples.to_vec();
    values.sort_by(f64::total_cmp);
    (values[(values.len() - 1) / 2] + values[values.len() / 2]) * 0.5
}
fn main() -> Result<()> {
    let args = Args::parse();
    ensure!((1..=4).contains(&args.batch), "batch must be 1..4");
    ensure!(
        args.iterations > 0 && args.iterations <= 10000,
        "iterations must be 1..10000"
    );
    let total = args
        .rows
        .checked_mul(args.cols)
        .context("matrix size overflow")?;
    ensure!(
        args.rows > 0 && args.cols > 0 && args.cols % 64 == 0 && total <= i32::MAX as usize,
        "matrix requires positive rows and 64-aligned columns with <= INT_MAX elements"
    );
    let gpu = Gpu::new_with_variant(false, "aligned")?;
    let weights: Vec<u32> = (0..total / 8)
        .map(|i| (i as u32).wrapping_mul(0x9e3779b9).wrapping_add(0x1478abcf))
        .collect();
    let scales: Vec<f32> = (0..total / 64)
        .map(|i| (1 + i % 11) as f32 / 512.)
        .collect();
    let biases: Vec<f32> = (0..total / 64)
        .map(|i| (i % 9) as f32 / 256. - 0.0625)
        .collect();
    let inputs: Vec<f32> = (0..args.batch * args.cols)
        .map(|i| ((i * 13 % 101) as f32 - 50.) / 50.)
        .collect();
    let upload_metadata = |values: &[f32]| {
        if matches!(args.metadata, Metadata::Bf16) {
            let packed: Vec<u16> = values.iter().map(|x| (x.to_bits() >> 16) as u16).collect();
            gpu.upload_bytes(bytemuck::cast_slice(&packed))
        } else {
            gpu.upload_f32(values)
        }
    };
    let w = gpu.upload_bytes(bytemuck::cast_slice(&weights))?;
    let s = upload_metadata(&scales)?;
    let b = upload_metadata(&biases)?;
    let x = gpu.upload_f32(&inputs)?;
    let y = gpu.alloc_f32(args.batch * args.rows)?;
    drop((weights, scales, biases, inputs));
    let buffers = [&*w, &*s, &*b, &*x, &*y];
    for _ in 0..5 {
        measure(&gpu, &buffers, &args, false)?;
        measure(&gpu, &buffers, &args, true)?;
    }
    let mut sequential_gpu = Vec::new();
    let mut sequential_wall = Vec::new();
    let mut block_gpu = Vec::new();
    let mut block_wall = Vec::new();
    for iteration in 0..args.iterations {
        // Alternate AB/BA so ordering and a warming GPU do not always favor one mode.
        for block in [iteration % 2 == 0, iteration % 2 != 0] {
            let (gpu_time, wall_time) = measure(&gpu, &buffers, &args, block)?;
            if block {
                block_gpu.push(gpu_time);
                block_wall.push(wall_time);
            } else {
                sequential_gpu.push(gpu_time);
                sequential_wall.push(wall_time);
            }
        }
    }
    measure(&gpu, &buffers, &args, false)?;
    let expected = gpu.read_f32(&y, args.batch * args.rows)?;
    measure(&gpu, &buffers, &args, true)?;
    let actual = gpu.read_f32(&y, args.batch * args.rows)?;
    let mut max_difference = 0f32;
    for (&got, &want) in actual.iter().zip(&expected) {
        let diff = (got - want).abs();
        max_difference = max_difference.max(diff);
        ensure!(
            got.is_finite() && diff <= 1e-3 + want.abs() * 5e-5,
            "block/GEMV mismatch {got} vs {want}"
        );
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "kind": "shared_weight_matrix_block_microbenchmark",
            "device": gpu.device.name(), "engine_version": env!("CARGO_PKG_VERSION"),
            "rows": args.rows, "cols": args.cols, "batch": args.batch,
            "bits": 4, "group_size": 64,
            "metadata_mode": if matches!(args.metadata, Metadata::Bf16) { "bf16" } else { "f32" },
            "iterations_per_mode": args.iterations,
            "weight_and_metadata_bytes": w.length() + s.length() + b.length(),
            "sequential_gpu_median_ms": median(&sequential_gpu) * 1000.,
            "block_gpu_median_ms": median(&block_gpu) * 1000.,
            "gpu_speedup": median(&sequential_gpu) / median(&block_gpu),
            "sequential_wall_median_ms": median(&sequential_wall) * 1000.,
            "block_wall_median_ms": median(&block_wall) * 1000.,
            "max_absolute_difference": max_difference,
            "note": "Same weights, inputs and output buffer; batch separate GEMVs in one command vs one actual shared-weight block dispatch. Five warmups per mode, alternating AB/BA order. Reused weights may benefit from cache. This is NOT model tokens per second."
        }))?
    );
    Ok(())
}
