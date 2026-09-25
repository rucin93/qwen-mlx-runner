//! Isolated legacy-schedule row-tile probe; never selected by the engine.
//! The candidate keeps one RHS input live, just like the frozen legacy baseline.
//! This differs from the rejected R2 shared-unpack experiment, which retained all RHS.
//! cargo run --release --locked --example block_legacy_tune -- --self-test
//! cargo run --release --locked --example block_legacy_tune -- --rows 17408 --cols 5120
//! Timings are repeated synthetic-weight GPU microbenchmarks, not model tok/s.
use anyhow::{Context, Result, anyhow, ensure};
use clap::{Parser, ValueEnum};
use metal::{Buffer, CommandQueueRef, ComputePipelineState, Device, MTLResourceOptions, MTLSize};
use objc::{msg_send, sel, sel_impl};
use serde_json::json;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

#[path = "../src/system_status.rs"]
mod system_status;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Metadata {
    F32,
    Bf16,
}
#[derive(Parser, Clone)]
struct Args {
    #[arg(long, default_value_t = 17408)]
    rows: usize,
    #[arg(long, default_value_t = 5120)]
    cols: usize,
    #[arg(long, default_value_t = 3)]
    batch: usize,
    #[arg(long, default_value_t = 100)]
    iterations: usize,
    #[arg(long, value_enum, default_value_t = Metadata::Bf16)]
    metadata: Metadata,
    #[arg(long, default_value_t = 2)]
    candidate_rows: usize,
    #[arg(long, default_value_t = 64)]
    threads: usize,
    #[arg(long)]
    self_test: bool,
    /// Run B3 BF16 up/down/vocabulary shapes at R1/R2/R4/R8 and T32/T64.
    #[arg(long, conflicts_with = "self_test")]
    sweep: bool,
}

// Frozen v0.6.0 aligned baseline from ef38979; independent of live production.
// Original full matmul_block.metal SHA256: 67e593a242ec5aa3fc5a390e28cb4c30da305184d4bbc2b002f8357e07e8abee
const BASELINE_SOURCE: &str = r#"
inline float baseline_block_metadata(float value) { return value; }
inline float baseline_block_metadata(ushort bits) { return as_type<float>(uint(bits) << 16); }
inline float baseline_block_u16_dot4(ushort q, float4 x) {
    return dot(float4(ushort4(q & ushort(0x000f), q & ushort(0x00f0),
                             q & ushort(0x0f00), q & ushort(0xf000))), x);
}
inline float baseline_block_u16_dot16(ushort4 q, float4 a, float4 b, float4 c, float4 d) {
    return baseline_block_u16_dot4(q.x,a) + baseline_block_u16_dot4(q.y,b)
         + baseline_block_u16_dot4(q.z,c) + baseline_block_u16_dot4(q.w,d);
}
template<uint B, typename M>
inline void baseline_block_aligned_q4(device const ushort *w, device const M *scales,
                             device const M *bias, device const float *x,
                             device float *y, constant uint *p, uint gid, ushort lane) {
    const int row = (int(gid) / 32) * 4;
    if (row >= int(p[0])) return;
    const int rows = int(p[0]), cols = int(p[1]);
    const int packed_stride = cols / 4, group_stride = cols / 64;
    const int tiles = cols / 512;
    float totals[B][4];
    #pragma unroll
    for (uint b = 0; b < B; ++b) {
        #pragma unroll
        for (uint r = 0; r < 4; ++r) totals[b][r] = 0.0f;
    }
    for (int tile = 0; tile < tiles; ++tile) {
        const int col = tile * 512 + int(lane) * 16;
        ushort4 packed[4];
        float scale[4], offset[4];
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            packed[r] = *reinterpret_cast<device const ushort4 *>(w + (row + r) * packed_stride + col / 4);
            const int gi = (row + r) * group_stride + col / 64;
            scale[r] = baseline_block_metadata(scales[gi]);
            offset[r] = baseline_block_metadata(bias[gi]);
        }
        #pragma unroll
        for (uint b = 0; b < B; ++b) {
            device const float *xb = x + int(b) * cols + col;
            const float4 a = *reinterpret_cast<device const float4 *>(xb);
            const float4 c1 = *reinterpret_cast<device const float4 *>(xb + 4);
            const float4 c2 = *reinterpret_cast<device const float4 *>(xb + 8);
            const float4 d = *reinterpret_cast<device const float4 *>(xb + 12);
            const float4 combined = a + c1 + c2 + d;
            const float xsum = combined.x + combined.y + combined.z + combined.w;
            const float4 factors = float4(1.0f,0x1p-4f,0x1p-8f,0x1p-12f);
            const float4 xa = a * factors, xc1 = c1 * factors, xc2 = c2 * factors, xd = d * factors;
            #pragma unroll
            for (uint r = 0; r < 4; ++r) {
                totals[b][r] += fma(scale[r], baseline_block_u16_dot16(packed[r],xa,xc1,xc2,xd), offset[r] * xsum);
            }
        }
    }
    #pragma unroll
    for (uint b = 0; b < B; ++b) {
        #pragma unroll
        for (uint r = 0; r < 4; ++r) {
            const float total = simd_sum(totals[b][r]);
            if (lane == 0) y[b * uint(rows) + uint(row) + r] = total;
        }
    }
}

#define TUNE_BASE(B,M,SUFFIX) \
kernel void baseline_q4_b##B##SUFFIX(device const ushort *w [[buffer(0)]], device const M *s [[buffer(1)]], \
    device const M *bias [[buffer(2)]], device const float *x [[buffer(3)]], device float *y [[buffer(4)]], \
    constant uint *p [[buffer(15)]], uint gid [[thread_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) { \
    baseline_block_aligned_q4<B,M>(w,s,bias,x,y,p,gid,lane); \
}
#define TUNE_BASE_BATCH(B) TUNE_BASE(B,float,) TUNE_BASE(B,ushort,_bf16)
TUNE_BASE_BATCH(1) TUNE_BASE_BATCH(2) TUNE_BASE_BATCH(3) TUNE_BASE_BATCH(4)
#undef TUNE_BASE_BATCH
#undef TUNE_BASE
"#;

// Same arithmetic and loop nesting; only rows per SIMD is a template parameter.
const CANDIDATE_SOURCE: &str = r#"
inline float legacy_block_metadata(float value) { return value; }
inline float legacy_block_metadata(ushort bits) { return as_type<float>(uint(bits) << 16); }
inline float legacy_block_u16_dot4(ushort q, float4 x) {
    return dot(float4(ushort4(q & ushort(0x000f), q & ushort(0x00f0),
                             q & ushort(0x0f00), q & ushort(0xf000))), x);
}
inline float legacy_block_u16_dot16(ushort4 q, float4 a, float4 b, float4 c, float4 d) {
    return legacy_block_u16_dot4(q.x,a) + legacy_block_u16_dot4(q.y,b)
         + legacy_block_u16_dot4(q.z,c) + legacy_block_u16_dot4(q.w,d);
}
template<uint B, uint R, typename M>
inline void legacy_block_aligned_q4(device const ushort *w, device const M *scales,
                             device const M *bias, device const float *x,
                             device float *y, constant uint *p, uint gid, ushort lane) {
    const int row = (int(gid) / 32) * int(R);
    if (row >= int(p[0])) return;
    const int rows = int(p[0]), cols = int(p[1]);
    const int packed_stride = cols / 4, group_stride = cols / 64;
    const int tiles = cols / 512;
    float totals[B][R];
    #pragma unroll
    for (uint b = 0; b < B; ++b) {
        #pragma unroll
        for (uint r = 0; r < R; ++r) totals[b][r] = 0.0f;
    }
    for (int tile = 0; tile < tiles; ++tile) {
        const int col = tile * 512 + int(lane) * 16;
        ushort4 packed[R];
        float scale[R], offset[R];
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            packed[r] = *reinterpret_cast<device const ushort4 *>(w + (row + r) * packed_stride + col / 4);
            const int gi = (row + r) * group_stride + col / 64;
            scale[r] = legacy_block_metadata(scales[gi]);
            offset[r] = legacy_block_metadata(bias[gi]);
        }
        #pragma unroll
        for (uint b = 0; b < B; ++b) {
            device const float *xb = x + int(b) * cols + col;
            const float4 a = *reinterpret_cast<device const float4 *>(xb);
            const float4 c1 = *reinterpret_cast<device const float4 *>(xb + 4);
            const float4 c2 = *reinterpret_cast<device const float4 *>(xb + 8);
            const float4 d = *reinterpret_cast<device const float4 *>(xb + 12);
            const float4 combined = a + c1 + c2 + d;
            const float xsum = combined.x + combined.y + combined.z + combined.w;
            const float4 factors = float4(1.0f,0x1p-4f,0x1p-8f,0x1p-12f);
            const float4 xa = a * factors, xc1 = c1 * factors, xc2 = c2 * factors, xd = d * factors;
            #pragma unroll
            for (uint r = 0; r < R; ++r) {
                totals[b][r] += fma(scale[r], legacy_block_u16_dot16(packed[r],xa,xc1,xc2,xd), offset[r] * xsum);
            }
        }
    }
    #pragma unroll
    for (uint b = 0; b < B; ++b) {
        #pragma unroll
        for (uint r = 0; r < R; ++r) {
            const float total = simd_sum(totals[b][r]);
            if (lane == 0) y[b * uint(rows) + uint(row) + r] = total;
        }
    }
}

#define LEGACY(B,R,M,SUFFIX) \
kernel void legacy_q4_b##B##_r##R##SUFFIX(device const ushort *w [[buffer(0)]], device const M *s [[buffer(1)]], \
    device const M *bias [[buffer(2)]], device const float *x [[buffer(3)]], device float *y [[buffer(4)]], \
    constant uint *p [[buffer(15)]], uint gid [[thread_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) { \
    legacy_block_aligned_q4<B,R,M>(w,s,bias,x,y,p,gid,lane); \
}
#define LEGACY_ROW(B,R) LEGACY(B,R,float,) LEGACY(B,R,ushort,_bf16)
#define LEGACY_BATCH(B) LEGACY_ROW(B,1) LEGACY_ROW(B,2) LEGACY_ROW(B,4) LEGACY_ROW(B,8)
LEGACY_BATCH(1) LEGACY_BATCH(2) LEGACY_BATCH(3) LEGACY_BATCH(4)
#undef LEGACY_BATCH
#undef LEGACY_ROW
#undef LEGACY
"#;

struct Fixture {
    rows: usize,
    cols: usize,
    batch: usize,
    weights: Vec<u32>,
    scales: Vec<f32>,
    biases: Vec<f32>,
    inputs: Vec<f32>,
    buffers: Vec<Buffer>,
}
impl Fixture {
    fn new(device: &Device, rows: usize, cols: usize, batch: usize, bf16: bool) -> Result<Self> {
        let count = rows.checked_mul(cols).context("matrix overflow")?;
        ensure!(
            rows > 0 && rows % 4 == 0 && cols > 0 && cols % 512 == 0 && count <= i32::MAX as usize,
            "requires rows % 4 = 0, cols % 512 = 0, and <= INT_MAX elements"
        );
        let weights: Vec<u32> = (0..count / 8)
            .map(|i| (i as u32).wrapping_mul(0x9e3779b9).wrapping_add(0x1478abcf))
            .collect();
        let scales: Vec<f32> = (0..count / 64)
            .map(|i| (1 + i % 11) as f32 / 512.)
            .collect();
        let biases: Vec<f32> = (0..count / 64)
            .map(|i| (i % 9) as f32 / 256. - 0.0625)
            .collect();
        let inputs: Vec<f32> = (0..batch * cols)
            .map(|i| ((i * 13 % 101) as f32 - 50.) / 50.)
            .collect();
        let upload = |bytes: &[u8]| {
            device.new_buffer_with_data(
                bytes.as_ptr().cast(),
                bytes.len() as u64,
                MTLResourceOptions::StorageModeShared,
            )
        };
        let metadata = |values: &[f32]| {
            if bf16 {
                let packed: Vec<u16> = values
                    .iter()
                    .map(|x| {
                        assert_eq!(x.to_bits() & 65535, 0);
                        (x.to_bits() >> 16) as u16
                    })
                    .collect();
                upload(bytemuck::cast_slice(&packed))
            } else {
                upload(bytemuck::cast_slice(values))
            }
        };
        let mut guarded_x = vec![12345.; 4];
        guarded_x.extend_from_slice(&inputs);
        guarded_x.extend_from_slice(&[12345.; 4]);
        let buffers = vec![
            upload(bytemuck::cast_slice(&weights)),
            metadata(&scales),
            metadata(&biases),
            upload(bytemuck::cast_slice(&guarded_x)),
            upload(bytemuck::cast_slice(&vec![12345f32; batch * rows + 8])),
        ];
        Ok(Self {
            rows,
            cols,
            batch,
            weights,
            scales,
            biases,
            inputs,
            buffers,
        })
    }
    fn poison_output(&self) {
        let values = unsafe {
            std::slice::from_raw_parts_mut(
                self.buffers[4].contents().cast::<f32>().add(4),
                self.batch * self.rows,
            )
        };
        values.fill(f32::NAN);
    }
    fn output(&self) -> Result<Vec<f32>> {
        let inputs = unsafe {
            std::slice::from_raw_parts(
                self.buffers[3].contents().cast::<f32>(),
                self.inputs.len() + 8,
            )
        };
        ensure!(
            inputs[..4]
                .iter()
                .chain(&inputs[inputs.len() - 4..])
                .all(|&v| v == 12345.),
            "input guard overwritten"
        );
        ensure!(
            inputs[4..inputs.len() - 4]
                .iter()
                .zip(&self.inputs)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "input values overwritten"
        );
        let values = unsafe {
            std::slice::from_raw_parts(
                self.buffers[4].contents().cast::<f32>(),
                self.batch * self.rows + 8,
            )
        };
        ensure!(
            values[..4]
                .iter()
                .chain(&values[values.len() - 4..])
                .all(|&v| v == 12345.),
            "output guard overwritten"
        );
        Ok(values[4..values.len() - 4].to_vec())
    }
    fn check_oracle(&self, actual: &[f32]) -> Result<f64> {
        let mut max = 0f64;
        // Check every output for small fixtures, deterministic spread for large timing matrices.
        let tested = self.rows.min(32);
        for b in 0..self.batch {
            for sample in 0..tested {
                let r = sample * (self.rows - 1) / (tested - 1);
                let expected: f64 = (0..self.cols)
                    .map(|c| {
                        let q = (self.weights[r * self.cols / 8 + c / 8] >> ((c % 8) * 4)) & 15;
                        let g = r * self.cols / 64 + c / 64;
                        (q as f64 * self.scales[g] as f64 + self.biases[g] as f64)
                            * self.inputs[b * self.cols + c] as f64
                    })
                    .sum();
                let got = actual[b * self.rows + r] as f64;
                let error = (got - expected).abs();
                ensure!(
                    got.is_finite() && error <= 4e-4 + expected.abs() * 3e-5,
                    "independent FP64 mismatch b={b} r={r}: {got} vs {expected}, error {error}"
                );
                max = max.max(error);
            }
        }
        Ok(max)
    }
}
fn pipeline(
    device: &Device,
    library: &metal::LibraryRef,
    name: &str,
    threads: usize,
) -> Result<ComputePipelineState> {
    let function = library
        .get_function(name, None)
        .map_err(|e| anyhow!("missing candidate {name}: {e}"))?;
    let descriptor = metal::ComputePipelineDescriptor::new();
    descriptor.set_compute_function(Some(&function));
    descriptor.set_thread_group_size_is_multiple_of_thread_execution_width(true);
    descriptor.set_max_total_threads_per_threadgroup(threads as u64);
    let pipeline = device
        .new_compute_pipeline_state(&descriptor)
        .map_err(|e| anyhow!("pipeline {name}: {e}"))?;
    ensure!(
        pipeline.thread_execution_width() == 32,
        "{name} requires a 32-lane SIMD group"
    );
    ensure!(
        threads <= pipeline.max_total_threads_per_threadgroup() as usize,
        "{name} does not support requested threadgroup size {threads}"
    );
    Ok(pipeline)
}
fn measure(
    queue: &CommandQueueRef,
    pipeline: &metal::ComputePipelineStateRef,
    fixture: &Fixture,
    rows_per_simd: usize,
    threads: usize,
) -> Result<(f64, f64)> {
    ensure!(
        fixture.rows % rows_per_simd == 0,
        "rows must be divisible by rows_per_simd"
    );
    objc::rc::autoreleasepool(|| {
        let started = Instant::now();
        let command = queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        for (i, buffer) in fixture.buffers.iter().enumerate() {
            encoder.set_buffer(i as u64, Some(buffer), if i >= 3 { 16 } else { 0 });
        }
        let params = [fixture.rows as u32, fixture.cols as u32];
        encoder.set_bytes(15, 8, params.as_ptr().cast());
        encoder.dispatch_threads(
            MTLSize::new((fixture.rows / rows_per_simd * 32) as u64, 1, 1),
            MTLSize::new(threads as u64, 1, 1),
        );
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        ensure!(
            command.status() == metal::MTLCommandBufferStatus::Completed,
            "Metal command failed: {:?}",
            command.status()
        );
        let start: f64 = unsafe { msg_send![command, GPUStartTime] };
        let end: f64 = unsafe { msg_send![command, GPUEndTime] };
        ensure!(
            start.is_finite() && end.is_finite() && end > start && start > 0.,
            "GPU timestamps unavailable or invalid: start={start}, end={end}"
        );
        Ok((end - start, started.elapsed().as_secs_f64()))
    })
}
fn compare(a: &[f32], b: &[f32]) -> Result<(usize, f32)> {
    ensure!(a.len() == b.len(), "baseline output length mismatch");
    let mut unequal = 0;
    let mut max = 0f32;
    for (&got, &want) in a.iter().zip(b) {
        let delta = (got - want).abs();
        ensure!(
            got.is_finite() && want.is_finite() && delta <= 1e-5 + want.abs() * 3e-6,
            "baseline mismatch {got} vs {want}"
        );
        unequal += usize::from(got.to_bits() != want.to_bits());
        max = max.max(delta);
    }
    ensure!(
        unequal == 0,
        "candidate changed {unequal} output bit patterns; max absolute difference {max}"
    );
    Ok((unequal, max))
}
fn median(samples: &[f64]) -> f64 {
    let mut v = samples.to_vec();
    v.sort_by(f64::total_cmp);
    (v[(v.len() - 1) / 2] + v[v.len() / 2]) * 0.5
}
fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        (1..=4).contains(&args.batch) && args.iterations > 0 && args.iterations <= 10000,
        "invalid batch or iterations"
    );
    ensure!(
        [1, 2, 4, 8].contains(&args.candidate_rows) && [32, 64].contains(&args.threads),
        "invalid rows/SIMD or threads"
    );
    let device = Device::system_default().context("No Metal GPU available")?;
    let options = metal::CompileOptions::new();
    options.set_fast_math_enabled(true);
    let source = format!(
        "#include <metal_stdlib>\nusing namespace metal;\n{}\n{}",
        BASELINE_SOURCE, CANDIDATE_SOURCE
    );
    let library = device
        .new_library_with_source(&source, &options)
        .map_err(|e| anyhow!("compile: {e}"))?;
    let queue = device.new_command_queue();
    let make_pipelines =
        |batch: usize, bf16: bool, rows_per_simd: usize, threads: usize| -> Result<_> {
            let suffix = if bf16 { "_bf16" } else { "" };
            Ok((
                pipeline(
                    &device,
                    &library,
                    &format!("baseline_q4_b{batch}{suffix}"),
                    64,
                )?,
                pipeline(
                    &device,
                    &library,
                    &format!("legacy_q4_b{batch}_r{rows_per_simd}{suffix}"),
                    threads,
                )?,
            ))
        };
    if args.self_test {
        let mut cases = 0;
        let mut unequal = 0;
        let configurations: Vec<_> = [1, 2, 4, 8]
            .into_iter()
            .flat_map(|rows_per_simd| {
                [32, 64]
                    .into_iter()
                    .map(move |threads| (rows_per_simd, threads))
            })
            .collect();
        for &(rows_per_simd, threads) in &configurations {
            for cols in [512, 1536, 5120, 17408] {
                for batch in 1..=4 {
                    for bf16 in [false, true] {
                        let fixture = Fixture::new(&device, 24, cols, batch, bf16)?;
                        let (baseline, candidate) =
                            make_pipelines(batch, bf16, rows_per_simd, threads)?;
                        fixture.poison_output();
                        measure(&queue, &baseline, &fixture, 4, 64)?;
                        let expected = fixture.output()?;
                        fixture.check_oracle(&expected)?;
                        fixture.poison_output();
                        measure(&queue, &candidate, &fixture, rows_per_simd, threads)?;
                        let actual = fixture.output()?;
                        fixture.check_oracle(&actual)?;
                        unequal += compare(&actual, &expected)?.0;
                        cases += 1;
                    }
                }
            }
        }
        println!(
            "{}",
            json!({
                "kind":"legacy_row_tile_oracle_check", "device":device.name(),
                "cases":cases, "baseline_bitwise_unequal":unequal,
                "configurations": configurations.iter().map(|&(rows_per_simd,threads)|
                    json!({"rows_per_simd":rows_per_simd,"threads":threads})).collect::<Vec<_>>(),
                "rows":24,"batches":[1,2,3,4], "metadata":["F32","Bf16"], "cols":[512,1536,5120,17408],
                "shader_source_sha256":sha256(source.as_bytes())?,
                "harness_source_sha256":sha256(include_bytes!("block_legacy_tune.rs"))?
            })
        );
        return Ok(());
    }
    let shader_source_sha256 = sha256(source.as_bytes())?;
    let harness_source_sha256 = sha256(include_bytes!("block_legacy_tune.rs"))?;
    let output = if args.sweep {
        ensure!(
            args.batch == 3 && matches!(args.metadata, Metadata::Bf16),
            "--sweep is the fixed B3 BF16 workload; omit --batch and --metadata overrides"
        );
        let mut captures = Vec::new();
        // A shape is released before loading the next. The largest fixture retains
        // about 1.6 GB of CPU + Metal data, never all three shape weights at once.
        for (rows, cols) in [(17408, 5120), (5120, 17408), (248320, 5120)] {
            let fixture = Fixture::new(&device, rows, cols, 3, true)?;
            for candidate_rows in [1, 2, 4, 8] {
                for threads in [32, 64] {
                    let pair_args = Args {
                        rows,
                        cols,
                        candidate_rows,
                        threads,
                        ..args.clone()
                    };
                    captures.push(run_pair(&device, &queue, &library, &fixture, &pair_args)?);
                }
            }
        }
        json!({
            "kind":"legacy_row_tile_sweep", "device":device.name(),
            "engine_version":env!("CARGO_PKG_VERSION"), "captures":captures,
            "shader_source_sha256":shader_source_sha256,
            "harness_source_sha256":harness_source_sha256,
            "production_selected":false,
            "note":"Isolated legacy R1/R2/R4/R8 with one RHS live, each paired against frozen R4/T64. These repeated synthetic weights may benefit from cache. Pairwise ABBA timing is primary; adjacent candidates are not simultaneous controls. No full-model throughput claim."
        })
    } else {
        let fixture = Fixture::new(
            &device,
            args.rows,
            args.cols,
            args.batch,
            matches!(args.metadata, Metadata::Bf16),
        )?;
        let mut output = run_pair(&device, &queue, &library, &fixture, &args)?;
        output["shader_source_sha256"] = json!(shader_source_sha256);
        output["harness_source_sha256"] = json!(harness_source_sha256);
        output
    };
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

// Hash the exact embedded sources through the macOS utility, outside GPU timing.
// No shell interpolation, network access, or inference runtime is involved.
fn sha256(bytes: &[u8]) -> Result<String> {
    let mut child = Command::new("/usr/bin/shasum")
        .args(["-a", "256"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("start macOS shasum")?;
    child
        .stdin
        .take()
        .context("open shasum stdin")?
        .write_all(bytes)?;
    let output = child.wait_with_output()?;
    ensure!(
        output.status.success(),
        "shasum failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let digest = String::from_utf8(output.stdout)?
        .split_whitespace()
        .next()
        .context("missing sha256 digest")?
        .to_owned();
    ensure!(
        digest.len() == 64 && digest.bytes().all(|b| b.is_ascii_hexdigit()),
        "invalid sha256 digest"
    );
    Ok(digest)
}

fn run_pair(
    device: &Device,
    queue: &CommandQueueRef,
    library: &metal::LibraryRef,
    fixture: &Fixture,
    args: &Args,
) -> Result<serde_json::Value> {
    let suffix = if matches!(args.metadata, Metadata::Bf16) {
        "_bf16"
    } else {
        ""
    };
    let baseline = pipeline(
        device,
        library,
        &format!("baseline_q4_b{}{suffix}", fixture.batch),
        64,
    )?;
    let candidate = pipeline(
        device,
        library,
        &format!(
            "legacy_q4_b{}_r{}{suffix}",
            fixture.batch, args.candidate_rows
        ),
        args.threads,
    )?;
    // Fail before collecting timings if this actual shape changes arithmetic or writes outside its view.
    fixture.poison_output();
    measure(queue, &baseline, fixture, 4, 64)?;
    let expected_before = fixture.output()?;
    fixture.check_oracle(&expected_before)?;
    fixture.poison_output();
    measure(
        queue,
        &candidate,
        fixture,
        args.candidate_rows,
        args.threads,
    )?;
    let actual_before = fixture.output()?;
    fixture.check_oracle(&actual_before)?;
    compare(&actual_before, &expected_before)?;
    let conditions_before = system_status::snapshot();
    for _ in 0..10 {
        measure(queue, &baseline, fixture, 4, 64)?;
        measure(
            queue,
            &candidate,
            fixture,
            args.candidate_rows,
            args.threads,
        )?;
    }
    let mut base_gpu = Vec::new();
    let mut candidate_gpu = Vec::new();
    let mut base_wall = Vec::new();
    let mut candidate_wall = Vec::new();
    for iteration in 0..args.iterations {
        for is_candidate in [iteration % 2 != 0, iteration % 2 == 0] {
            let (gpu, wall) = if is_candidate {
                measure(
                    queue,
                    &candidate,
                    fixture,
                    args.candidate_rows,
                    args.threads,
                )?
            } else {
                measure(queue, &baseline, fixture, 4, 64)?
            };
            if is_candidate {
                candidate_gpu.push(gpu);
                candidate_wall.push(wall);
            } else {
                base_gpu.push(gpu);
                base_wall.push(wall);
            }
        }
    }
    let conditions_after = system_status::snapshot();
    fixture.poison_output();
    measure(queue, &baseline, fixture, 4, 64)?;
    let expected = fixture.output()?;
    let baseline_oracle_error = fixture.check_oracle(&expected)?;
    fixture.poison_output();
    measure(
        queue,
        &candidate,
        fixture,
        args.candidate_rows,
        args.threads,
    )?;
    let actual = fixture.output()?;
    let candidate_oracle_error = fixture.check_oracle(&actual)?;
    let (unequal, max) = compare(&actual, &expected)?;
    Ok(json!({
        "kind":"legacy_row_tile_microbenchmark", "device":device.name(), "engine_version":env!("CARGO_PKG_VERSION"),
        "rows":fixture.rows, "cols":fixture.cols, "batch":fixture.batch, "metadata":format!("{:?}",args.metadata),
        "candidate_rows_per_simd":args.candidate_rows, "candidate_threads":args.threads,
        "candidate_source":"isolated_legacy_row_tile",
        "production_selected":false,
        "conditions_before":conditions_before,"conditions_after":conditions_after,
        "baseline_source":"frozen_ef38979",
        "baseline_commit":"ef38979",
        "baseline_original_source_sha256":"67e593a242ec5aa3fc5a390e28cb4c30da305184d4bbc2b002f8357e07e8abee",
        "iterations_per_mode":args.iterations, "warmups_per_mode":10,
        "baseline_gpu_ms":median(&base_gpu)*1000., "candidate_gpu_ms":median(&candidate_gpu)*1000.,
        "gpu_speedup":median(&base_gpu)/median(&candidate_gpu),
        "baseline_wall_ms":median(&base_wall)*1000., "candidate_wall_ms":median(&candidate_wall)*1000.,
        "baseline_bitwise_unequal":unequal, "baseline_max_absolute_difference":max,
        "baseline_fp64_max_absolute_error":baseline_oracle_error, "candidate_fp64_max_absolute_error":candidate_oracle_error,
        "baseline_gpu_samples_seconds":base_gpu, "candidate_gpu_samples_seconds":candidate_gpu,
        "note":"One command/dispatch per sample, alternating AB/BA order, same fixture, inputs/outputs use guarded aligned 16-byte views. Independent FP64 covers 32 evenly spread rows per RHS; baseline comparison covers every output. No model throughput claim."
    }))
}
