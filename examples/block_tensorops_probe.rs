//! Isolated FP32 TensorOps Q4 probe. Not selected by the inference engine.
//! cargo run --release --locked --example block_tensorops_probe -- --self-test
//! cargo run --release --locked --example block_tensorops_probe -- --sweep
use anyhow::{Context, Result, anyhow, ensure};
use clap::Parser;
use metal::{Buffer, ComputePipelineState, Device, MTLResourceOptions, MTLSize};
use objc::{msg_send, sel, sel_impl};
use serde_json::{Value, json};
use std::io::Write;
use std::process::{Command, Stdio};

#[path = "../src/system_status.rs"]
mod system_status;

const SCALAR: &str = include_str!("../kernels/matmul_block.metal");
const TENSOR: &str = include_str!("shaders/block_tensorops.metal");
const GUARD: f32 = 12345.;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 17408)]
    rows: usize,
    #[arg(long, default_value_t = 5120)]
    cols: usize,
    #[arg(long, default_value_t = 3)]
    batch: usize,
    #[arg(long, default_value_t = 24)]
    iterations: usize,
    #[arg(long, conflicts_with = "sweep")]
    self_test: bool,
    #[arg(long)]
    sweep: bool,
}

struct Fixture {
    rows: usize,
    cols: usize,
    batch: usize,
    w: Vec<u32>,
    s: Vec<f32>,
    bias: Vec<f32>,
    x: Vec<f32>,
    buffers: Vec<Buffer>,
}

fn random(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

impl Fixture {
    fn new(device: &Device, rows: usize, cols: usize, batch: usize, seed: u32) -> Result<Self> {
        let count = rows.checked_mul(cols).context("matrix overflow")?;
        ensure!(
            rows > 0 && cols > 0 && cols.is_multiple_of(64),
            "invalid shape"
        );
        ensure!(
            count <= i32::MAX as usize && (1..=4).contains(&batch),
            "shape exceeds probe limits"
        );
        let padded_cols = cols.div_ceil(128) * 128;
        ensure!(
            count as u64 * 19 / 16 + (cols * batch + padded_cols * 8 + rows * batch) as u64 * 8
                < 3 * 1024u64.pow(3),
            "synthetic fixture exceeds the 3 GiB host+GPU probe budget"
        );
        let mut rng = seed;
        let w: Vec<u32> = (0..count / 8).map(|_| random(&mut rng)).collect();
        // Non-power-of-two BF16 scales and signed offsets, exactly representable in FP32.
        let s: Vec<f32> = (0..count / 64)
            .map(|_| f32::from_bits((0x3b00 + random(&mut rng) % 512) << 16))
            .collect();
        // Construct metadata through BF16 bits (no lossy host packing).
        let bias: Vec<f32> = (0..count / 64)
            .map(|_| f32::from_bits((0xbb80 + random(&mut rng) % 512) << 16))
            .collect();
        let x: Vec<f32> = (0..cols * batch)
            .map(|i| {
                let value = (random(&mut rng) as f64 / u32::MAX as f64 * 2. - 1.) as f32;
                if seed == 19 {
                    value * 2f32.powi((i % 13) as i32 - 6)
                } else {
                    value
                }
            })
            .collect();
        let upload = |data: &[u8]| {
            device.new_buffer_with_data(
                data.as_ptr().cast(),
                data.len() as u64,
                MTLResourceOptions::StorageModeShared,
            )
        };
        let metadata = |values: &[f32]| {
            let bits: Vec<u16> = values.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
            upload(bytemuck::cast_slice(&bits))
        };
        let mut guarded = vec![GUARD; 4];
        guarded.extend_from_slice(&x);
        guarded.extend_from_slice(&[GUARD; 4]);
        let buffers = vec![
            upload(bytemuck::cast_slice(&w)),
            metadata(&s),
            metadata(&bias),
            upload(bytemuck::cast_slice(&guarded)),
            upload(bytemuck::cast_slice(&vec![GUARD; rows * batch + 8])),
            upload(bytemuck::cast_slice(&vec![GUARD; padded_cols * 8 + 8])),
        ];
        Ok(Self {
            rows,
            cols,
            batch,
            w,
            s,
            bias,
            x,
            buffers,
        })
    }

    fn poison(&self) {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.buffers[4].contents().cast::<f32>().add(4),
                self.rows * self.batch,
            )
        }
        .fill(f32::NAN);
        unsafe {
            std::slice::from_raw_parts_mut(
                self.buffers[5].contents().cast::<f32>().add(4),
                self.padded_cols() * 8,
            )
        }
        .fill(f32::NAN);
    }

    fn padded_cols(&self) -> usize {
        self.cols.div_ceil(128) * 128
    }

    fn check_padding(&self) -> Result<()> {
        let data = unsafe {
            std::slice::from_raw_parts(
                self.buffers[5].contents().cast::<f32>().add(4),
                self.padded_cols() * 8,
            )
        };
        for b in 0..8 {
            for c in 0..self.padded_cols() {
                let want = if b < self.batch && c < self.cols {
                    self.x[b * self.cols + c]
                } else {
                    0.
                };
                ensure!(
                    data[b * self.padded_cols() + c].to_bits() == want.to_bits(),
                    "padded activation mismatch b={b} c={c}"
                );
            }
        }
        Ok(())
    }

    fn scalar_rows(&self) -> usize {
        if self.batch == 3 && [(17408, 5120), (5120, 17408)].contains(&(self.rows, self.cols)) {
            2
        } else {
            4
        }
    }

    fn output(&self) -> Result<Vec<f32>> {
        let read = |i: usize, n: usize| unsafe {
            std::slice::from_raw_parts(self.buffers[i].contents().cast::<f32>(), n + 8)
        };
        for (i, n) in [
            (3, self.x.len()),
            (4, self.rows * self.batch),
            (5, self.padded_cols() * 8),
        ] {
            let data = read(i, n);
            ensure!(
                data[..4].iter().chain(&data[n + 4..]).all(|&v| v == GUARD),
                "buffer {i} guard overwritten"
            );
        }
        ensure!(
            read(3, self.x.len())[4..self.x.len() + 4]
                .iter()
                .zip(&self.x)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "input overwritten"
        );
        let out = read(4, self.rows * self.batch)[4..self.rows * self.batch + 4].to_vec();
        ensure!(
            out.iter().all(|v| v.is_finite()),
            "nonfinite or unwritten output"
        );
        Ok(out)
    }

    fn oracle(&self, actual: &[f32], all_rows: bool) -> Result<Value> {
        let tested = if all_rows {
            self.rows
        } else {
            self.rows.min(32)
        };
        let mut max_abs = 0f64;
        let mut max_normalized = 0f64;
        for b in 0..self.batch {
            for sample in 0..tested {
                let row = if tested == 1 {
                    0
                } else {
                    sample * (self.rows - 1) / (tested - 1)
                };
                let (mut want, mut l1) = (0f64, 0f64);
                for c in 0..self.cols {
                    let q = (self.w[row * self.cols / 8 + c / 8] >> ((c % 8) * 4)) & 15;
                    let g = row * self.cols / 64 + c / 64;
                    let term = (q as f64 * self.s[g] as f64 + self.bias[g] as f64)
                        * self.x[b * self.cols + c] as f64;
                    want += term;
                    l1 += term.abs();
                }
                let error = (actual[b * self.rows + row] as f64 - want).abs();
                // Cancellation-safe FP32 error test; neither reduced-precision nor bitwise-equivalence claim.
                ensure!(
                    error <= 1e-6 + l1 * 1e-5,
                    "FP64 oracle mismatch shape={}x{} batch={} row={row} b={b}: got={} want={want} error={error} l1={l1}",
                    self.rows,
                    self.cols,
                    self.batch,
                    actual[b * self.rows + row]
                );
                max_abs = max_abs.max(error);
                max_normalized = max_normalized.max(error / l1.max(1e-30));
            }
        }
        Ok(
            json!({"checked_outputs":tested * self.batch,"max_absolute_error":max_abs,
            "max_error_over_sum_abs_terms":max_normalized,"tolerance":"1e-6 + 1e-5 * sum(abs(FP64 products))"}),
        )
    }
}

fn pipeline(
    device: &Device,
    library: &metal::LibraryRef,
    name: &str,
    threads: usize,
) -> Result<ComputePipelineState> {
    let f = library
        .get_function(name, None)
        .map_err(|e| anyhow!("{name}: {e}"))?;
    let descriptor = metal::ComputePipelineDescriptor::new();
    descriptor.set_compute_function(Some(&f));
    descriptor.set_thread_group_size_is_multiple_of_thread_execution_width(true);
    descriptor.set_max_total_threads_per_threadgroup(threads as u64);
    let p = device
        .new_compute_pipeline_state(&descriptor)
        .map_err(|e| anyhow!("pipeline {name}: {e}"))?;
    ensure!(
        p.thread_execution_width() == 32 && p.max_total_threads_per_threadgroup() >= threads as u64,
        "unsupported pipeline geometry"
    );
    Ok(p)
}

fn run(
    queue: &metal::CommandQueueRef,
    p: &ComputePipelineState,
    f: &Fixture,
    tile: Option<(&ComputePipelineState, usize)>,
) -> Result<f64> {
    objc::rc::autoreleasepool(|| {
        let command = queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        for (i, buffer) in f.buffers.iter().enumerate() {
            encoder.set_buffer(i as u64, Some(buffer), if i >= 3 { 16 } else { 0 });
        }
        // Specialized scalar kernels only read rows/cols; tensor additionally reads batch.
        let params = [
            f.rows as u32,
            f.cols as u32,
            f.batch as u32,
            f.padded_cols() as u32,
        ];
        encoder.set_bytes(
            15,
            std::mem::size_of_val(&params) as u64,
            params.as_ptr().cast(),
        );
        if let Some((pad, _)) = tile {
            encoder.set_compute_pipeline_state(pad);
            encoder.dispatch_threads(
                MTLSize::new((8 * f.padded_cols()) as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );
        }
        encoder.set_compute_pipeline_state(p);
        if let Some((_, rows)) = tile {
            encoder.dispatch_thread_groups(
                MTLSize::new(f.rows.div_ceil(rows) as u64, 1, 1),
                MTLSize::new(32, 1, 1),
            );
        } else {
            encoder.dispatch_threads(
                MTLSize::new((f.rows.div_ceil(f.scalar_rows()) * 32) as u64, 1, 1),
                MTLSize::new(64, 1, 1),
            );
        }
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        ensure!(
            command.status() == metal::MTLCommandBufferStatus::Completed,
            "GPU command failed: {:?}",
            command.status()
        );
        let start: f64 = unsafe { msg_send![command, GPUStartTime] };
        let end: f64 = unsafe { msg_send![command, GPUEndTime] };
        ensure!(
            start.is_finite() && end.is_finite() && start > 0. && end > start,
            "invalid GPU timestamps {start}..{end}"
        );
        Ok(end - start)
    })
}

fn median(values: &[f64]) -> f64 {
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    (v[(v.len() - 1) / 2] + v[v.len() / 2]) / 2.
}

fn sha256(bytes: &[u8]) -> Result<String> {
    let mut child = Command::new("/usr/bin/shasum")
        .args(["-a", "256"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    child.stdin.take().context("hash stdin")?.write_all(bytes)?;
    let output = child.wait_with_output()?;
    ensure!(output.status.success(), "hash failed");
    Ok(String::from_utf8(output.stdout)?
        .split_whitespace()
        .next()
        .context("hash missing")?
        .into())
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        (1..=4).contains(&args.batch) && (4..=1000).contains(&args.iterations),
        "batch 1..4, iterations 4..1000 required"
    );
    let os = Command::new("/usr/bin/sw_vers")
        .arg("-productVersion")
        .output()?;
    let os = String::from_utf8(os.stdout)?.trim().to_string();
    let version: Vec<u32> = os.split('.').filter_map(|s| s.parse().ok()).collect();
    ensure!(
        version.first().copied().unwrap_or(0) > 26
            || (version.first() == Some(&26) && version.get(1).copied().unwrap_or(0) >= 3),
        "FP32 cooperative-input TensorOps probe requires macOS 26.3+, found {os}"
    );
    let device = Device::system_default().context("no Metal device")?;
    let options = metal::CompileOptions::new();
    options.set_fast_math_enabled(false);
    // metal-rs 0.32 predates this enum value; send NSUInteger without constructing an invalid Rust enum.
    unsafe {
        let _: () = msg_send![&*options, setLanguageVersion: 0x40000u64];
    }
    let tensor = device
        .new_library_with_source(TENSOR, &options)
        .map_err(|e| {
            anyhow!(
                "FP32 TensorOps unavailable on {} / macOS {os}: {e}",
                device.name()
            )
        })?;
    let scalar_options = metal::CompileOptions::new();
    scalar_options.set_fast_math_enabled(true);
    let scalar = device
        .new_library_with_source(SCALAR, &scalar_options)
        .map_err(|e| anyhow!("scalar compile: {e}"))?;
    let queue = device.new_command_queue();
    let pad = pipeline(&device, &tensor, "tensor_pad_input", 256)?;
    let configs = [(16, 64), (16, 128), (32, 64), (32, 128)];
    let mut captures = Vec::new();
    for (m, k) in configs {
        let name = format!("tensor_q4_m{m}_k{k}");
        let candidate = pipeline(&device, &tensor, &name, 32)?;
        if args.self_test {
            for rows in [1, 17, 36] {
                for cols in [64, 192, 512, 5120, 17408] {
                    for batch in [1, 2, 3, 4] {
                        for seed in [7, 19] {
                            let fixture = Fixture::new(&device, rows, cols, batch, seed)?;
                            fixture.poison();
                            run(&queue, &candidate, &fixture, Some((&pad, m)))?;
                            fixture.check_padding()?;
                            let actual = fixture.output().with_context(|| {
                                format!("{name} rows={rows} cols={cols} batch={batch} seed={seed}")
                            })?;
                            captures.push(json!({"kernel":name,"rows":rows,"cols":cols,"batch":batch,"seed":seed,"oracle":fixture.oracle(&actual,true).with_context(|| format!("{name} seed={seed}, outputs={:?}", &actual[..actual.len().min(8)]))?}));
                        }
                    }
                }
            }
            continue;
        }
        let shapes = if args.sweep {
            vec![(17408, 5120), (5120, 17408), (248320, 5120)]
        } else {
            vec![(args.rows, args.cols)]
        };
        for (rows, cols) in shapes {
            ensure!(
                rows.is_multiple_of(4) && cols.is_multiple_of(512),
                "timing scalar baseline requires rows%4=0 cols%512=0"
            );
            let fixture = Fixture::new(&device, rows, cols, args.batch, 7)?;
            let baseline_name =
                if args.batch == 3 && [(17408, 5120), (5120, 17408)].contains(&(rows, cols)) {
                    "matmul_q4_g64_b3_mlp_r2_aligned_bf16".to_string()
                } else {
                    format!("matmul_q4_g64_b{}_aligned_bf16", args.batch)
                };
            let baseline = pipeline(&device, &scalar, &baseline_name, 64)?;
            fixture.poison();
            run(&queue, &baseline, &fixture, None)?;
            let expected = fixture.output()?;
            let scalar_oracle = fixture.oracle(&expected, false)?;
            fixture.poison();
            run(&queue, &candidate, &fixture, Some((&pad, m)))?;
            fixture.check_padding()?;
            let actual = fixture.output()?;
            let oracle_before = fixture.oracle(&actual, false)?;
            let bitwise_unequal = actual
                .iter()
                .zip(&expected)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            let max_scalar_difference = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            for _ in 0..10 {
                run(&queue, &baseline, &fixture, None)?;
                run(&queue, &candidate, &fixture, Some((&pad, m)))?;
            }
            let conditions_before = system_status::snapshot();
            let (mut a, mut b) = (Vec::new(), Vec::new());
            for i in 0..args.iterations {
                if i % 2 == 0 {
                    a.push(run(&queue, &baseline, &fixture, None)?);
                    b.push(run(&queue, &candidate, &fixture, Some((&pad, m)))?);
                } else {
                    b.push(run(&queue, &candidate, &fixture, Some((&pad, m)))?);
                    a.push(run(&queue, &baseline, &fixture, None)?);
                }
            }
            let conditions_after = system_status::snapshot();
            fixture.poison();
            run(&queue, &candidate, &fixture, Some((&pad, m)))?;
            fixture.check_padding()?;
            let oracle_after = fixture.oracle(&fixture.output()?, false)?;
            captures.push(json!({"kernel":name,"rows":rows,"cols":cols,"batch":args.batch,
                "baseline_kernel":baseline_name,"baseline_rows_per_simd":fixture.scalar_rows(),"baseline_threads":64,
                "candidate_rows_per_simd":m,"candidate_k_tile":k,"candidate_threads":32,"includes_activation_padding":true,
                "excluded_warmup_pairs":10,"pairs":args.iterations,"order":"AB/BA alternating",
                "baseline_gpu_seconds":a,"candidate_gpu_seconds":b,
                "baseline_median_ms":median(&a)*1000.,"candidate_median_ms":median(&b)*1000.,
                "median_gpu_speedup":median(&a)/median(&b),
                "candidate_faster_pairs":a.iter().zip(&b).filter(|(a,b)|b<a).count(),
                "scalar_oracle":scalar_oracle,"oracle_before":oracle_before,"oracle_after":oracle_after,
                "baseline_bitwise_unequal":bitwise_unequal,"max_absolute_scalar_difference":max_scalar_difference,
                "conditions_before":conditions_before,"conditions_after":conditions_after}));
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"kind":if args.self_test {"tensorops_fp32_oracle_check"} else {"tensorops_fp32_probe"},
        "device":device.name(),"os_version":os,"engine_version":env!("CARGO_PKG_VERSION"),
        "input_precision":"FP32","accumulator_precision":"FP32","relaxed_precision":false,
        "production_selected":false,"captures":captures,"tensor_shader_sha256":sha256(TENSOR.as_bytes())?,
        "scalar_shader_sha256":sha256(SCALAR.as_bytes())?,"harness_sha256":sha256(include_bytes!("block_tensorops_probe.rs"))?,
        "note":"Isolated synthetic Q4/g64 BF16-metadata probe; changed reduction/dequantization order, not a bitwise-equivalence claim. Repeated weights may benefit from cache. No trained-model quality or model tokens/s claim; runtime compilation does not establish M5 Neural Accelerator utilization."})
        )?
    );
    Ok(())
}
