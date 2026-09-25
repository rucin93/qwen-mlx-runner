//! Standalone centered-Q4 / normalized-FP16 experiment; never selected by inference.
//! Both modes accumulate in FP32; compensated mode adds a second FP16 residual dot.
//! Five M16 configurations: full self-test has 500 cases / 18,000 outputs;
//! the B3+B4 production-shape sweep has 30 configurations. M32 is excluded
//! after a reproducible nonfinite/unwritten-output failure in isolated validation.
use anyhow::{Context, Result, anyhow, ensure};
use clap::Parser;
use half::f16;
use metal::{Buffer, ComputePipelineState, Device, MTLResourceOptions, MTLSize};
use objc::{msg_send, sel, sel_impl};
use serde_json::{Value, json};
use std::ffi::CStr;
use std::io::Write;
use std::process::{Command, Stdio};

#[path = "../src/system_status.rs"]
mod system_status;

const SCALAR: &str = include_str!("../kernels/matmul_block.metal");
const AFFINE: &str = include_str!("shaders/block_affine_fp16.metal");
const PACKED: &str = include_str!("shaders/block_packed_fp16.metal");
const NATIVE: &str = include_str!("shaders/block_native_q4_fp16.metal");
const GUARD_BYTES: usize = 16;
const GUARD: u8 = 0xa5;
const RESIDUAL_SCALE: f32 = 2048.0;
const ZERO_WEIGHT_ABSOLUTE_LIMIT: f64 = 1e-6;

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
    /// Restrict execution to one exact candidate entry-point name.
    #[arg(long)]
    kernel: Option<String>,
    #[arg(long, conflicts_with = "sweep")]
    self_test: bool,
    /// Fixed B3 and B4 sweep of all three production shapes; leave --batch at 3.
    #[arg(long)]
    sweep: bool,
}

#[derive(Clone, Copy, Debug)]
enum Mode {
    Single,
    Compensated,
}

impl Mode {
    fn name(self) -> &'static str {
        match self {
            Self::Single => "single",
            Self::Compensated => "compensated",
        }
    }
    fn quality_limit(self) -> f64 {
        match self {
            Self::Single => 1e-3,
            Self::Compensated => 1e-5,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Pattern {
    Normal,
    Wide,
    ZeroInput,
    ExactAffineCancellation,
    ZeroWeight,
}

impl Pattern {
    fn name(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::Wide => "wide_exponents",
            Self::ZeroInput => "zero_input",
            Self::ExactAffineCancellation => "q8_bias_negative_8scale",
            Self::ZeroWeight => "q0_bias_zero",
        }
    }
    fn seed(self) -> u32 {
        match self {
            Self::Normal => 7,
            Self::Wide => 19,
            Self::ZeroInput => 23,
            Self::ExactAffineCancellation => 31,
            Self::ZeroWeight => 29,
        }
    }
}

fn random(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

fn normalization_scale(max_abs: f32) -> f32 {
    if max_abs == 0.0 {
        return 1.0;
    }
    f32::from_bits((max_abs.to_bits() & 0x7f80_0000).max(0x0080_0000))
}

fn half_pair(x: f32, alpha: f32) -> (u16, u16) {
    let normalized = x / alpha;
    let hi = f16::from_f32(normalized);
    let lo = f16::from_f32((normalized - hi.to_f32()) * RESIDUAL_SCALE);
    (hi.to_bits(), lo.to_bits())
}

fn reconstructed(alpha: f32, hi: u16, lo: u16, mode: Mode) -> f64 {
    let residual = match mode {
        Mode::Single => 0.0,
        Mode::Compensated => f16::from_bits(lo).to_f64() / RESIDUAL_SCALE as f64,
    };
    alpha as f64 * (f16::from_bits(hi).to_f64() + residual)
}

fn check_original_sum(actual: f32, original_sum: f64, original_l1: f64) -> Result<()> {
    // Six SIMD reduction levels plus the initial two-value lane sum, rounded up.
    let limit = 1e-30 + 8.0 * f32::EPSILON as f64 * original_l1;
    ensure!(
        actual.is_finite() && (actual as f64 - original_sum).abs() <= limit,
        "original-X group sum mismatch: actual={actual}, FP64={original_sum}, limit={limit}"
    );
    Ok(())
}

fn check_algorithm_error(error: f64, affine_l1: f64) -> Result<()> {
    let limit = 1e-6 + 1e-5 * affine_l1;
    ensure!(
        error.is_finite() && error <= limit,
        "algorithm FP64 oracle mismatch: error={error}, affine_l1={affine_l1}, limit={limit}"
    );
    Ok(())
}

#[derive(Default)]
struct Reference {
    original: f64,
    algorithm: f64,
    original_l1: f64,
    algorithm_l1: f64,
    input_bound: f64,
    sum_bound: f64,
    offset_bound: f64,
}

impl Reference {
    fn approximation_bound(&self) -> f64 {
        self.input_bound + self.sum_bound + self.offset_bound
    }
    fn add(&mut self, other: Self) {
        self.original += other.original;
        self.algorithm += other.algorithm;
        self.original_l1 += other.original_l1;
        self.algorithm_l1 += other.algorithm_l1;
        self.input_bound += other.input_bound;
        self.sum_bound += other.sum_bound;
        self.offset_bound += other.offset_bound;
    }
}

#[allow(clippy::too_many_arguments)]
fn group_reference(
    centered_q: &[i8],
    scale: f32,
    bias: f32,
    x: &[f32],
    alpha: f32,
    hi: &[u16],
    lo: &[u16],
    gpu_original_sum: f32,
    mode: Mode,
) -> Reference {
    let mut r = Reference::default();
    let exact_offset = 8.0 * scale as f64 + bias as f64;
    let fp32_offset = (8.0 * scale + bias) as f64;
    let mut original_sum = 0.0;
    for i in 0..x.len() {
        let centered_weight = scale as f64 * centered_q[i] as f64;
        let original_x = x[i] as f64;
        let approximate_x = reconstructed(alpha, hi[i], lo[i], mode);
        // Direct original affine Q4 reference, independently of centered regrouping.
        r.original += ((centered_q[i] as f64 + 8.0) * scale as f64 + bias as f64) * original_x;
        r.algorithm += centered_weight * approximate_x;
        // Separate component magnitudes make the cancellation bound well-defined.
        r.original_l1 += (centered_weight * original_x).abs() + (exact_offset * original_x).abs();
        r.algorithm_l1 += (centered_weight * approximate_x).abs();
        r.input_bound += centered_weight.abs() * (approximate_x - original_x).abs();
        original_sum += original_x;
    }
    r.algorithm += fp32_offset * gpu_original_sum as f64;
    r.algorithm_l1 += (fp32_offset * gpu_original_sum as f64).abs();
    r.sum_bound = fp32_offset.abs() * (gpu_original_sum as f64 - original_sum).abs();
    r.offset_bound = (fp32_offset - exact_offset).abs() * original_sum.abs();
    r
}

struct GuardedBuffer {
    buffer: Buffer,
    payload_bytes: usize,
    prefix_bytes: usize,
}

impl GuardedBuffer {
    fn new(device: &Device, payload: &[u8]) -> Self {
        let mut guarded = vec![GUARD; payload.len() + 2 * GUARD_BYTES];
        guarded[GUARD_BYTES..GUARD_BYTES + payload.len()].copy_from_slice(payload);
        Self {
            buffer: device.new_buffer_with_data(
                guarded.as_ptr().cast(),
                guarded.len() as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            payload_bytes: payload.len(),
            prefix_bytes: GUARD_BYTES,
        }
    }
    fn zeroed_aligned(device: &Device, payload_bytes: usize, alignment: usize) -> Self {
        let buffer = device.new_buffer(
            (payload_bytes + 2 * alignment) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let all = unsafe {
            std::slice::from_raw_parts_mut(
                buffer.contents().cast::<u8>(),
                payload_bytes + 2 * alignment,
            )
        };
        all.fill(GUARD);
        all[alignment..alignment + payload_bytes].fill(0);
        Self {
            buffer,
            payload_bytes,
            prefix_bytes: alignment,
        }
    }
    fn bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self.buffer.contents().cast::<u8>().add(self.prefix_bytes),
                self.payload_bytes,
            )
        }
    }
    fn f32s(&self) -> &[f32] {
        bytemuck::cast_slice(self.bytes())
    }
    fn u16s(&self) -> &[u16] {
        bytemuck::cast_slice(self.bytes())
    }
    fn check_guard(&self) -> Result<()> {
        let all = unsafe {
            std::slice::from_raw_parts(
                self.buffer.contents().cast::<u8>(),
                self.payload_bytes + 2 * self.prefix_bytes,
            )
        };
        ensure!(
            all[..self.prefix_bytes]
                .iter()
                .chain(&all[self.prefix_bytes + self.payload_bytes..])
                .all(|&v| v == GUARD),
            "buffer guard overwritten"
        );
        Ok(())
    }
    fn poison_f32(&self) {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.buffer
                    .contents()
                    .cast::<u8>()
                    .add(self.prefix_bytes)
                    .cast::<f32>(),
                self.payload_bytes / 4,
            )
        }
        .fill(f32::NAN);
    }
    fn poison_half(&self) {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.buffer
                    .contents()
                    .cast::<u8>()
                    .add(self.prefix_bytes)
                    .cast::<u16>(),
                self.payload_bytes / 2,
            )
        }
        .fill(f16::NAN.to_bits());
    }
}

struct Fixture {
    rows: usize,
    cols: usize,
    batch: usize,
    padded_cols: usize,
    signed_cols: usize,
    signed_rows: usize,
    pattern: Pattern,
    w: Vec<u32>,
    s: Vec<f32>,
    bias: Vec<f32>,
    x: Vec<f32>,
    buffers: Vec<GuardedBuffer>,
}

impl Fixture {
    fn new(
        device: &Device,
        rows: usize,
        cols: usize,
        batch: usize,
        pattern: Pattern,
    ) -> Result<Self> {
        let count = rows.checked_mul(cols).context("matrix overflow")?;
        ensure!(
            rows > 0 && cols > 0 && cols.is_multiple_of(64),
            "positive rows and group64 columns required"
        );
        ensure!(
            count <= i32::MAX as usize && (1..=4).contains(&batch),
            "shape exceeds probe limits"
        );
        let padded_cols = cols.div_ceil(128) * 128;
        let signed_cols = cols.div_ceil(256) * 256;
        let signed_rows = rows.div_ceil(16) * 16;
        // Includes host Q4/FP32 metadata, all GPU allocations, a guarded upload copy,
        // readback/oracle vectors and conservative scratch allowance.
        let peak_bytes = count as u64 * 19 / 16
            + (signed_cols as u64 * signed_rows as u64 / 2).max(count as u64 / 2)
            + (cols * batch + padded_cols * 8 + rows * batch) as u64 * 24
            + 4096;
        ensure!(
            peak_bytes < 3 * 1024u64.pow(3),
            "fixture exceeds conservative 3 GiB host+GPU budget"
        );
        let mut rng = pattern.seed();
        let w = (0..count / 8)
            .map(|_| match pattern {
                Pattern::ExactAffineCancellation => 0x8888_8888,
                Pattern::ZeroWeight => 0,
                _ => random(&mut rng),
            })
            .collect::<Vec<u32>>();
        let s = (0..count / 64)
            .map(|_| f32::from_bits((0x3b00 + random(&mut rng) % 512) << 16))
            .collect::<Vec<_>>();
        let bias = s
            .iter()
            .map(|&scale| match pattern {
                Pattern::ExactAffineCancellation => -8.0 * scale,
                Pattern::ZeroWeight => 0.0,
                _ => f32::from_bits((0xbb80 + random(&mut rng) % 512) << 16),
            })
            .collect::<Vec<_>>();
        let x = (0..cols * batch)
            .map(|i| {
                let value = (random(&mut rng) as f64 / u32::MAX as f64 * 2.0 - 1.0) as f32;
                match pattern {
                    Pattern::ZeroInput => 0.0,
                    Pattern::Wide | Pattern::ExactAffineCancellation => {
                        let magnitude = match (i / 64) % 3 {
                            0 => 100_000.0,
                            1 => 1e-8,
                            _ => [1e-8, 1e-6, 0.03125, 1.0, 128.0, 1e5, 1e6, 0.0][i % 8],
                        };
                        value.signum() * (0.5 + value.abs()) * magnitude
                    }
                    _ => value,
                }
            })
            .collect::<Vec<_>>();
        ensure!(
            s.iter().chain(&bias).all(|v| v.to_bits() & 0xffff == 0),
            "metadata is not lossless BF16"
        );
        let metadata = |values: &[f32]| {
            let bits = values
                .iter()
                .map(|v| (v.to_bits() >> 16) as u16)
                .collect::<Vec<_>>();
            GuardedBuffer::new(device, bytemuck::cast_slice(&bits))
        };
        let mut buffers = vec![
            GuardedBuffer::new(device, bytemuck::cast_slice(&w)),
            metadata(&s),
            metadata(&bias),
            GuardedBuffer::new(device, bytemuck::cast_slice(&x)),
            GuardedBuffer::new(device, bytemuck::cast_slice(&vec![0f32; rows * batch])),
            GuardedBuffer::new(device, bytemuck::cast_slice(&vec![0u16; padded_cols * 8])),
            GuardedBuffer::new(device, bytemuck::cast_slice(&vec![0u16; padded_cols * 8])),
            GuardedBuffer::new(
                device,
                bytemuck::cast_slice(&vec![0f32; 8 * (padded_cols / 64) * 2]),
            ),
        ];
        // Repack once into shared GPU memory: signed nibble q-8 = original nibble XOR 8.
        // This preserves every Q4 code; no second host-sized weight copy is allocated.
        let native = GuardedBuffer::zeroed_aligned(device, signed_rows * signed_cols / 2, 128);
        let signed_words = unsafe {
            std::slice::from_raw_parts_mut(
                native.buffer.contents().cast::<u8>().add(128).cast::<u32>(),
                signed_rows * signed_cols / 8,
            )
        };
        for row in 0..rows {
            for c in 0..cols / 8 {
                signed_words[row * (signed_cols / 8) + c] = w[row * (cols / 8) + c] ^ 0x8888_8888;
            }
        }
        buffers.push(native);
        buffers.push(GuardedBuffer::new(
            device,
            bytemuck::cast_slice(&vec![0u16; padded_cols * 8]),
        ));
        Ok(Self {
            rows,
            cols,
            batch,
            padded_cols,
            signed_cols,
            signed_rows,
            pattern,
            w,
            s,
            bias,
            x,
            buffers,
        })
    }

    fn poison(&self) {
        self.buffers[4].poison_f32();
        self.buffers[5].poison_half();
        self.buffers[6].poison_half();
        self.buffers[7].poison_f32();
        self.buffers[9].poison_half();
    }

    fn scalar_rows(&self) -> usize {
        if self.batch == 3 && [(17408, 5120), (5120, 17408)].contains(&(self.rows, self.cols)) {
            2
        } else {
            4
        }
    }

    fn output(&self) -> Result<Vec<f32>> {
        for (i, buffer) in self.buffers.iter().enumerate() {
            buffer
                .check_guard()
                .with_context(|| format!("buffer {i}"))?;
        }
        ensure!(
            self.buffers[0].bytes() == bytemuck::cast_slice::<u32, u8>(&self.w),
            "packed weights overwritten"
        );
        for (i, expected) in [(1, &self.s), (2, &self.bias)] {
            ensure!(
                self.buffers[i]
                    .u16s()
                    .iter()
                    .zip(expected)
                    .all(|(&a, b)| a == (b.to_bits() >> 16) as u16),
                "metadata buffer {i} overwritten"
            );
        }
        ensure!(
            self.buffers[3].bytes() == bytemuck::cast_slice::<f32, u8>(&self.x),
            "original input overwritten"
        );
        let native_words: &[u32] = bytemuck::cast_slice(self.buffers[8].bytes());
        for row in 0..self.signed_rows {
            for c in 0..self.signed_cols / 8 {
                let want = if row < self.rows && c < self.cols / 8 {
                    self.w[row * (self.cols / 8) + c] ^ 0x8888_8888
                } else {
                    0
                };
                ensure!(
                    native_words[row * (self.signed_cols / 8) + c] == want,
                    "signed Q4 input or padding overwritten row={row} word={c}"
                );
            }
        }
        let out = self.buffers[4].f32s().to_vec();
        ensure!(
            out.iter().all(|v| v.is_finite()),
            "nonfinite or unwritten output"
        );
        if matches!(
            self.pattern,
            Pattern::ExactAffineCancellation | Pattern::ZeroInput
        ) {
            ensure!(
                out.iter().all(|&v| v == 0.0),
                "exact-zero fixture produced a nonzero output"
            );
        }
        Ok(out)
    }

    fn check_preparation(&self) -> Result<Value> {
        let groups = self.padded_cols / 64;
        let hi = self.buffers[5].u16s();
        let lo = self.buffers[6].u16s();
        let meta = self.buffers[7].f32s();
        let packed = self.buffers[9].u16s();
        let (mut max_sum_error, mut max_sum_normalized) = (0f64, 0f64);
        for b in 0..8 {
            for g in 0..groups {
                let mut original = [0f32; 64];
                if b < self.batch && g * 64 < self.cols {
                    original.copy_from_slice(
                        &self.x[b * self.cols + g * 64..b * self.cols + (g + 1) * 64],
                    );
                }
                let expected_alpha =
                    normalization_scale(original.iter().map(|v| v.abs()).fold(0f32, f32::max));
                let alpha = meta[(b * groups + g) * 2];
                let sum = meta[(b * groups + g) * 2 + 1];
                ensure!(
                    alpha.is_finite()
                        && alpha > 0.0
                        && alpha.to_bits() & 0x007f_ffff == 0
                        && alpha.to_bits() == expected_alpha.to_bits(),
                    "normalization mismatch b={b} group={g}: {alpha} vs {expected_alpha}"
                );
                let original_sum = original.iter().map(|&v| v as f64).sum::<f64>();
                let original_l1 = original.iter().map(|&v| (v as f64).abs()).sum::<f64>();
                check_original_sum(sum, original_sum, original_l1)?;
                max_sum_error = max_sum_error.max((sum as f64 - original_sum).abs());
                max_sum_normalized = max_sum_normalized
                    .max((sum as f64 - original_sum).abs() / original_l1.max(1e-30));
                for (c, &x) in original.iter().enumerate() {
                    let i = b * self.padded_cols + g * 64 + c;
                    let want = half_pair(x, expected_alpha);
                    ensure!(
                        hi[i] == want.0 && lo[i] == want.1,
                        "FP16 preparation mismatch b={b} group={g} c={c}: GPU={:04x}/{:04x}, CPU={:04x}/{:04x}; FP16 flushing is not silently accepted",
                        hi[i],
                        lo[i],
                        want.0,
                        want.1
                    );
                    ensure!(
                        f16::from_bits(hi[i]).is_finite() && f16::from_bits(lo[i]).is_finite(),
                        "nonfinite FP16 preparation"
                    );
                }
            }
        }
        for channel in 0..8 {
            for c in 0..self.padded_cols {
                let want = if channel < 4 {
                    hi[channel * self.padded_cols + c]
                } else {
                    lo[(channel - 4) * self.padded_cols + c]
                };
                ensure!(
                    packed[channel * self.padded_cols + c] == want,
                    "packed FP16 activation mismatch channel={channel} col={c}"
                );
            }
        }
        Ok(
            json!({"checked_groups":8*groups,"checked_half_values":3*8*self.padded_cols,
            "exact_cpu_half_bits":true,"power_of_two_alpha_valid":true,
            "max_original_sum_absolute_error":max_sum_error,
            "max_original_sum_error_over_l1":max_sum_normalized,
            "sum_tolerance":"1e-30 + 8 * FP32_EPSILON * sum(abs(original X))"}),
        )
    }

    fn row_reference(&self, b: usize, row: usize, mode: Mode) -> Reference {
        let mut total = Reference::default();
        let padded_groups = self.padded_cols / 64;
        for g in 0..self.cols / 64 {
            let mut q = [0i8; 64];
            for (c, v) in q.iter_mut().enumerate() {
                let col = g * 64 + c;
                *v = ((self.w[row * self.cols / 8 + col / 8] >> ((col % 8) * 4)) & 15) as i8 - 8;
            }
            let gi = row * (self.cols / 64) + g;
            let xi = b * self.cols + g * 64;
            let pi = b * self.padded_cols + g * 64;
            let mi = (b * padded_groups + g) * 2;
            total.add(group_reference(
                &q,
                self.s[gi],
                self.bias[gi],
                &self.x[xi..xi + 64],
                self.buffers[7].f32s()[mi],
                &self.buffers[5].u16s()[pi..pi + 64],
                &self.buffers[6].u16s()[pi..pi + 64],
                self.buffers[7].f32s()[mi + 1],
                mode,
            ));
        }
        total
    }

    fn oracle(&self, actual: &[f32], all_rows: bool, mode: Mode) -> Result<Value> {
        let tested = if all_rows {
            self.rows
        } else {
            self.rows.min(32)
        };
        let (mut max_original_error, mut max_algorithm_error) = (0f64, 0f64);
        let (mut max_original_normalized, mut max_algorithm_normalized) = (0f64, 0f64);
        let (mut max_input_bound, mut max_sum_bound, mut max_offset_bound) = (0f64, 0f64, 0f64);
        let mut max_reference_difference = 0f64;
        for b in 0..self.batch {
            for sample in 0..tested {
                let row = if tested == 1 {
                    0
                } else {
                    sample * (self.rows - 1) / (tested - 1)
                };
                let r = self.row_reference(b, row, mode);
                let got = actual[b * self.rows + row] as f64;
                let algorithm_error = (got - r.algorithm).abs();
                check_algorithm_error(algorithm_error, r.algorithm_l1).with_context(|| {
                    format!(
                        "{}x{} batch={} row={row} b={b} mode={}",
                        self.rows,
                        self.cols,
                        self.batch,
                        mode.name()
                    )
                })?;
                let reference_difference = (r.algorithm - r.original).abs();
                // This small allowance is FP64 oracle accumulation error, not an FP16 quality tolerance.
                let fp64_slack = 1e-12 + 1e-12 * (r.original_l1 + r.algorithm_l1);
                ensure!(
                    reference_difference <= r.approximation_bound() + fp64_slack,
                    "analytic input/sum/offset bound violated: difference={reference_difference}, bound={}",
                    r.approximation_bound()
                );
                let original_error = (got - r.original).abs();
                max_original_error = max_original_error.max(original_error);
                max_algorithm_error = max_algorithm_error.max(algorithm_error);
                max_original_normalized =
                    max_original_normalized.max(original_error / r.original_l1.max(1e-30));
                max_algorithm_normalized =
                    max_algorithm_normalized.max(algorithm_error / r.algorithm_l1.max(1e-30));
                max_input_bound = max_input_bound.max(r.input_bound);
                max_sum_bound = max_sum_bound.max(r.sum_bound);
                max_offset_bound = max_offset_bound.max(r.offset_bound);
                max_reference_difference = max_reference_difference.max(reference_difference);
            }
        }
        let normalized_passed = max_original_normalized <= mode.quality_limit();
        let zero_weight_passed = !matches!(self.pattern, Pattern::ZeroWeight)
            || max_original_error <= ZERO_WEIGHT_ABSOLUTE_LIMIT;
        Ok(
            json!({"checked_outputs":tested*self.batch,"mode":mode.name(),
            "algorithm_rounding_gate_passed":true,
            "algorithm_rounding_tolerance":"1e-6 + 1e-5 * sum(abs(centered affine components))",
            "max_algorithm_absolute_error":max_algorithm_error,
            "max_algorithm_error_over_affine_l1":max_algorithm_normalized,
            "max_original_absolute_error":max_original_error,
            "max_original_error_over_affine_l1":max_original_normalized,
            "original_normalized_error_limit":mode.quality_limit(),
            "original_normalized_error_gate_passed":normalized_passed,
            "zero_weight_absolute_error_limit":ZERO_WEIGHT_ABSOLUTE_LIMIT,
            "zero_weight_absolute_gate_applicable":matches!(self.pattern, Pattern::ZeroWeight),
            "zero_weight_absolute_gate_passed":zero_weight_passed,
            "all_original_error_gates_passed":normalized_passed&&zero_weight_passed,
            "max_algorithm_minus_original_absolute":max_reference_difference,
            "max_analytic_input_approximation_bound":max_input_bound,
            "max_original_sum_rounding_bound":max_sum_bound,
            "max_offset_rounding_bound":max_offset_bound,
            "analytic_bound_passed":true,
            "note":"Original reference uses untouched FP32 X and original affine Q4. Algorithm reference uses GPU hi/lo, alpha and original-X sum. Affine-component L1 is cancellation-safe and is not relative error in the final output. Gates do not validate trained-model quality or unchanged greedy choices."}),
        )
    }

    fn scalar_oracle(&self, actual: &[f32]) -> Result<Value> {
        let tested = self.rows.min(32);
        let (mut max_abs, mut max_normalized) = (0f64, 0f64);
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
                check_algorithm_error(error, l1)?;
                max_abs = max_abs.max(error);
                max_normalized = max_normalized.max(error / l1.max(1e-30));
            }
        }
        Ok(
            json!({"checked_outputs":tested*self.batch,"max_absolute_error":max_abs,
            "max_error_over_sum_abs_terms":max_normalized,
            "tolerance":"1e-6 + 1e-5 * sum(abs(original affine FP64 products))"}),
        )
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
        .map_err(|e| anyhow!("{name}: {e}"))?;
    let descriptor = metal::ComputePipelineDescriptor::new();
    descriptor.set_compute_function(Some(&function));
    descriptor.set_thread_group_size_is_multiple_of_thread_execution_width(true);
    descriptor.set_max_total_threads_per_threadgroup(threads as u64);
    let pipeline = device
        .new_compute_pipeline_state(&descriptor)
        .map_err(|e| anyhow!("pipeline {name}: {e}"))?;
    ensure!(
        pipeline.thread_execution_width() == 32
            && pipeline.max_total_threads_per_threadgroup() >= threads as u64,
        "unsupported pipeline geometry for {name}"
    );
    Ok(pipeline)
}

fn command_error_description(command: &metal::CommandBufferRef) -> String {
    fn nsstring(value: *mut objc::runtime::Object) -> String {
        if value.is_null() {
            return "<unavailable>".into();
        }
        let bytes: *const std::ffi::c_char = unsafe { msg_send![value, UTF8String] };
        if bytes.is_null() {
            "<unavailable>".into()
        } else {
            unsafe { CStr::from_ptr(bytes) }
                .to_string_lossy()
                .into_owned()
        }
    }
    let error: *mut objc::runtime::Object = unsafe { msg_send![command, error] };
    if error.is_null() {
        return "NSError unavailable".into();
    }
    let code: isize = unsafe { msg_send![error, code] };
    let domain: *mut objc::runtime::Object = unsafe { msg_send![error, domain] };
    let description: *mut objc::runtime::Object = unsafe { msg_send![error, localizedDescription] };
    let reason: *mut objc::runtime::Object = unsafe { msg_send![error, localizedFailureReason] };
    format!(
        "NSError domain={} code={code}: {}; failure_reason={}",
        nsstring(domain),
        nsstring(description),
        nsstring(reason)
    )
}

fn run(
    queue: &metal::CommandQueueRef,
    pipeline: &ComputePipelineState,
    fixture: &Fixture,
    candidate: Option<(&ComputePipelineState, usize)>,
) -> Result<f64> {
    objc::rc::autoreleasepool(|| {
        let command = queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        for (i, buffer) in fixture.buffers.iter().enumerate() {
            encoder.set_buffer(i as u64, Some(&buffer.buffer), buffer.prefix_bytes as u64);
        }
        let params = [
            fixture.rows as u32,
            fixture.cols as u32,
            fixture.batch as u32,
            fixture.padded_cols as u32,
            fixture.signed_cols as u32,
        ];
        encoder.set_bytes(
            15,
            std::mem::size_of_val(&params) as u64,
            params.as_ptr().cast(),
        );
        if let Some((prepare, _)) = candidate {
            encoder.set_compute_pipeline_state(prepare);
            encoder.dispatch_thread_groups(
                MTLSize::new((8 * fixture.padded_cols / 64) as u64, 1, 1),
                MTLSize::new(32, 1, 1),
            );
        }
        encoder.set_compute_pipeline_state(pipeline);
        if let Some((_, tile_rows)) = candidate {
            encoder.dispatch_thread_groups(
                MTLSize::new(fixture.rows.div_ceil(tile_rows) as u64, 1, 1),
                MTLSize::new(32, 1, 1),
            );
        } else {
            encoder.dispatch_threads(
                MTLSize::new(
                    (fixture.rows.div_ceil(fixture.scalar_rows()) * 32) as u64,
                    1,
                    1,
                ),
                MTLSize::new(64, 1, 1),
            );
        }
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        ensure!(
            command.status() == metal::MTLCommandBufferStatus::Completed,
            "GPU command failed: {:?}; {}",
            command.status(),
            command_error_description(command)
        );
        let start: f64 = unsafe { msg_send![command, GPUStartTime] };
        let end: f64 = unsafe { msg_send![command, GPUEndTime] };
        ensure!(
            start.is_finite() && end.is_finite() && start > 0.0 && end > start,
            "invalid GPU timestamps {start}..{end}"
        );
        Ok(end - start)
    })
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    (sorted[(sorted.len() - 1) / 2] + sorted[sorted.len() / 2]) / 2.0
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

struct Configuration {
    name: &'static str,
    rows: usize,
    mode: Mode,
    native: bool,
    packed: bool,
}

fn configurations() -> [Configuration; 5] {
    [
        Configuration {
            name: "affine_fp16_m16_single",
            rows: 16,
            mode: Mode::Single,
            native: false,
            packed: false,
        },
        Configuration {
            name: "affine_fp16_m16_compensated",
            rows: 16,
            mode: Mode::Compensated,
            native: false,
            packed: false,
        },
        Configuration {
            name: "affine_fp16_m16_packed",
            rows: 16,
            mode: Mode::Compensated,
            native: false,
            packed: true,
        },
        Configuration {
            name: "affine_native_q4_m16_single",
            rows: 16,
            mode: Mode::Single,
            native: true,
            packed: false,
        },
        Configuration {
            name: "affine_native_q4_m16_packed",
            rows: 16,
            mode: Mode::Compensated,
            native: true,
            packed: true,
        },
    ]
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        args.kernel
            .as_deref()
            .is_none_or(|name| configurations().iter().any(|config| config.name == name)),
        "unknown --kernel; expected one of {:?}",
        configurations()
            .iter()
            .map(|config| config.name)
            .collect::<Vec<_>>()
    );
    ensure!(
        (1..=4).contains(&args.batch) && (4..=1000).contains(&args.iterations),
        "batch 1..4 and iterations 4..1000 required"
    );
    ensure!(
        !args.sweep || args.batch == 3,
        "--sweep fixes batches to [3,4]; leave --batch at its default 3"
    );
    let os = Command::new("/usr/bin/sw_vers")
        .arg("-productVersion")
        .output()?;
    ensure!(os.status.success(), "sw_vers failed");
    let os = String::from_utf8(os.stdout)?.trim().to_string();
    let version: Vec<u32> = os.split('.').filter_map(|s| s.parse().ok()).collect();
    ensure!(
        version.first().copied().unwrap_or(0) > 26
            || (version.first() == Some(&26) && version.get(1).copied().unwrap_or(0) >= 4),
        "native signed-Q4 TensorOps probe requires macOS 26.4+, found {os}"
    );
    let device = Device::system_default().context("no Metal device")?;
    let options = metal::CompileOptions::new();
    options.set_fast_math_enabled(false);
    unsafe {
        let _: () = msg_send![&*options, setLanguageVersion: 0x40000u64];
    }
    let candidate_source = format!("{AFFINE}\n{PACKED}\n{NATIVE}");
    let library = device
        .new_library_with_source(&candidate_source, &options)
        .map_err(|e| {
            anyhow!(
                "affine FP16/native Q4 TensorOps unavailable on {} / macOS {os}: {e}",
                device.name()
            )
        })?;
    let scalar_options = metal::CompileOptions::new();
    scalar_options.set_fast_math_enabled(true);
    let scalar = device
        .new_library_with_source(SCALAR, &scalar_options)
        .map_err(|e| anyhow!("scalar compile: {e}"))?;
    let queue = device.new_command_queue();
    let prepare = pipeline(&device, &library, "affine_fp16_prepare", 32)?;
    let mut captures = Vec::new();
    for config in configurations().into_iter().filter(|config| {
        args.kernel
            .as_deref()
            .is_none_or(|name| config.name == name)
    }) {
        eprintln!("checking {}", config.name);
        let candidate = pipeline(&device, &library, config.name, 32)?;
        if args.self_test {
            // Bounded cross-coverage of all required dimensions, row tails, batches and patterns.
            for (rows, cols) in [(1, 64), (17, 192), (36, 512), (17, 5120), (1, 17408)] {
                for batch in 1..=4 {
                    for pattern in [
                        Pattern::Normal,
                        Pattern::Wide,
                        Pattern::ZeroInput,
                        Pattern::ExactAffineCancellation,
                        Pattern::ZeroWeight,
                    ] {
                        let fixture = Fixture::new(&device, rows, cols, batch, pattern)?;
                        fixture.poison();
                        let case = || {
                            format!(
                                "self-test {} {rows}x{cols} B{batch} {}",
                                config.name,
                                pattern.name()
                            )
                        };
                        run(&queue, &candidate, &fixture, Some((&prepare, config.rows)))
                            .with_context(|| {
                                format!("{}: preparation + candidate execution", case())
                            })?;
                        let actual = fixture.output().with_context(|| {
                            format!("{} {rows}x{cols} B{batch} {}", config.name, pattern.name())
                        })?;
                        let preparation = fixture
                            .check_preparation()
                            .with_context(|| format!("{}: preparation validation", case()))?;
                        let oracle = fixture
                            .oracle(&actual, true, config.mode)
                            .with_context(|| format!("{}: numerical oracle", case()))?;
                        captures.push(json!({"kernel":config.name,"mode":config.mode.name(),
                            "native_signed_q4_operand":config.native,"packed_hi_lo_channels":config.packed,
                            "rows":rows,"cols":cols,"batch":batch,"pattern":pattern.name(),"seed":pattern.seed(),
                            "preparation":preparation,"oracle":oracle}));
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
        let batches = if args.sweep {
            vec![3, 4]
        } else {
            vec![args.batch]
        };
        for batch in batches {
            for &(rows, cols) in &shapes {
                ensure!(
                    rows.is_multiple_of(4) && cols.is_multiple_of(512),
                    "timing baseline requires rows%4=0 and cols%512=0"
                );
                let fixture = Fixture::new(&device, rows, cols, batch, Pattern::Normal)?;
                let baseline_name = if fixture.scalar_rows() == 2 {
                    "matmul_q4_g64_b3_mlp_r2_aligned_bf16".to_string()
                } else {
                    format!("matmul_q4_g64_b{batch}_aligned_bf16")
                };
                let baseline = pipeline(&device, &scalar, &baseline_name, 64)?;
                let case = || format!("benchmark {} {rows}x{cols} B{batch} normal", config.name);
                let execute_baseline = || {
                    run(&queue, &baseline, &fixture, None)
                        .with_context(|| format!("{}: baseline {baseline_name}", case()))
                };
                let execute_candidate = || {
                    run(&queue, &candidate, &fixture, Some((&prepare, config.rows)))
                        .with_context(|| format!("{}: preparation + candidate execution", case()))
                };
                fixture.poison();
                execute_baseline()?;
                let expected = fixture
                    .output()
                    .with_context(|| format!("{}: baseline output validation", case()))?;
                let scalar_oracle = fixture
                    .scalar_oracle(&expected)
                    .with_context(|| format!("{}: baseline oracle", case()))?;
                fixture.poison();
                execute_candidate()?;
                let actual = fixture
                    .output()
                    .with_context(|| format!("{}: initial output validation", case()))?;
                let preparation_before = fixture
                    .check_preparation()
                    .with_context(|| format!("{}: initial preparation validation", case()))?;
                let oracle_before = fixture
                    .oracle(&actual, false, config.mode)
                    .with_context(|| format!("{}: initial numerical oracle", case()))?;
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
                    execute_baseline()?;
                    execute_candidate()?;
                }
                let conditions_before = system_status::snapshot();
                let (mut a, mut b) = (Vec::new(), Vec::new());
                for i in 0..args.iterations {
                    if i % 2 == 0 {
                        a.push(
                            execute_baseline().with_context(|| format!("measured pair {i} AB"))?,
                        );
                        b.push(
                            execute_candidate().with_context(|| format!("measured pair {i} AB"))?,
                        );
                    } else {
                        b.push(
                            execute_candidate().with_context(|| format!("measured pair {i} BA"))?,
                        );
                        a.push(
                            execute_baseline().with_context(|| format!("measured pair {i} BA"))?,
                        );
                    }
                }
                let conditions_after = system_status::snapshot();
                fixture.poison();
                execute_candidate()?;
                let actual_after = fixture
                    .output()
                    .with_context(|| format!("{}: final output validation", case()))?;
                let preparation_after = fixture
                    .check_preparation()
                    .with_context(|| format!("{}: final preparation validation", case()))?;
                let oracle_after = fixture
                    .oracle(&actual_after, false, config.mode)
                    .with_context(|| format!("{}: final numerical oracle", case()))?;
                let paired_ratios = a.iter().zip(&b).map(|(a, b)| a / b).collect::<Vec<_>>();
                captures.push(json!({"kernel":config.name,"mode":config.mode.name(),
                    "native_signed_q4_operand":config.native,"packed_hi_lo_channels":config.packed,
                    "activation_operand_precision":"FP16","accumulator_precision":"FP32",
                    "rows":rows,"cols":cols,"batch":batch,"pattern":"normal","seed":7,
                    "baseline_kernel":baseline_name,"baseline_rows_per_simd":fixture.scalar_rows(),"baseline_threads":64,
                    "candidate_rows_per_simd":config.rows,"candidate_k_tile":64,"candidate_threads":32,
                    "includes_activation_preparation":true,"includes_native_weight_repacking":false,
                    "native_weight_repacking_note":"Exact q XOR 8 plus zero stride/row padding is uploaded once before timing; no runtime weight quantization.",
                    "excluded_warmup_pairs":10,"pairs":args.iterations,"order":"AB/BA alternating",
                    "baseline_gpu_seconds":a,"candidate_gpu_seconds":b,
                    "baseline_median_ms":median(&a)*1000.0,"candidate_median_ms":median(&b)*1000.0,
                    "median_gpu_speedup":median(&a)/median(&b),"median_paired_gpu_speedup":median(&paired_ratios),
                    "candidate_faster_pairs":a.iter().zip(&b).filter(|(a,b)| b<a).count(),
                    "scalar_oracle":scalar_oracle,"oracle_before":oracle_before,"oracle_after":oracle_after,
                    "preparation_before":preparation_before,"preparation_after":preparation_after,
                    "baseline_bitwise_unequal":bitwise_unequal,"max_absolute_scalar_difference":max_scalar_difference,
                    "conditions_before":conditions_before,"conditions_after":conditions_after}));
            }
        }
    }
    let all_original_gates_passed = captures.iter().all(|capture| {
        ["oracle", "oracle_before", "oracle_after"]
            .iter()
            .filter_map(|k| capture.get(k))
            .all(|oracle| oracle["all_original_error_gates_passed"] == true)
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "kind":if args.self_test {"affine_fp16_oracle_check"} else {"affine_fp16_probe"},
            "device":device.name(),"os_version":os,"engine_version":env!("CARGO_PKG_VERSION"),
            "input_precision":"Original FP32; group-normalized FP16 hi and residual FP16 lo",
            "weight_precision":"Unchanged affine Q4/BF16; centered q-8 represented exactly as FP16 or native signed int4",
            "accumulator_precision":"FP32","candidate_fast_math":false,
            "production_selected":false,"model_quality_validated":false,
            "single_normalized_error_limit":Mode::Single.quality_limit(),
            "compensated_normalized_error_limit":Mode::Compensated.quality_limit(),
            "zero_weight_absolute_error_limit":ZERO_WEIGHT_ABSOLUTE_LIMIT,
            "all_original_error_gates_passed":all_original_gates_passed,
            "sweep_batches":if args.sweep {Some(vec![3,4])} else {None},
            "requested_kernel":args.kernel,
            "captures":captures,"affine_shader_sha256":sha256(AFFINE.as_bytes())?,
            "packed_shader_sha256":sha256(PACKED.as_bytes())?,"native_shader_sha256":sha256(NATIVE.as_bytes())?,
            "compiled_candidate_sources_sha256":sha256(candidate_source.as_bytes())?,
            "scalar_shader_sha256":sha256(SCALAR.as_bytes())?,
            "harness_sha256":sha256(include_bytes!("block_affine_fp16_probe.rs"))?,
            "note":"Isolated synthetic lower-precision experiment. Original-X bias correction is preserved, but centered input rounding can create nonzero error even for q=0,bias=0. Algorithm correctness, approximation error and trained-model quality are distinct. No bitwise-equivalence, greedy-equivalence, model tokens/s or Neural Accelerator utilization claim. GPU intervals include activation preparation. Native signed-Q4 repacking is exact and load-only. Repeated weights may benefit from cache. Production R2 remains unchanged."
        }))?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalization_handles_zero_large_tiny_and_subnormal_values() {
        assert_eq!(normalization_scale(0.0), 1.0);
        assert_eq!(normalization_scale(100_000.0), 65_536.0);
        assert_eq!(normalization_scale(1e-8), 2f32.powi(-27));
        assert_eq!(normalization_scale(f32::from_bits(1)), f32::MIN_POSITIVE);
    }

    #[test]
    fn compensated_reconstruction_reduces_half_input_error() {
        let x = 1.234_567_f32;
        let (hi, lo) = half_pair(x, 1.0);
        let single = reconstructed(1.0, hi, lo, Mode::Single);
        let compensated = reconstructed(1.0, hi, lo, Mode::Compensated);
        assert!((compensated - x as f64).abs() < (single - x as f64).abs() / 100.0);
    }

    #[test]
    fn centered_zero_weights_can_have_input_approximation_error() {
        let x = vec![1.234_567_f32; 64];
        let q = vec![-8_i8; 64]; // q=0 with bias=0 is exactly zero originally.
        let pairs: Vec<_> = x.iter().map(|&v| half_pair(v, 1.0)).collect();
        let hi: Vec<_> = pairs.iter().map(|p| p.0).collect();
        let lo: Vec<_> = pairs.iter().map(|p| p.1).collect();
        let sum = (x[0] as f64 * 64.0) as f32;
        let single = group_reference(&q, 0.125, 0.0, &x, 1.0, &hi, &lo, sum, Mode::Single);
        let compensated =
            group_reference(&q, 0.125, 0.0, &x, 1.0, &hi, &lo, sum, Mode::Compensated);
        assert_eq!(single.original, 0.0);
        assert!(single.algorithm.abs() > 0.0);
        assert!(single.original_l1 > 0.0);
        assert!((single.algorithm - single.original).abs() <= single.approximation_bound());
        assert!(compensated.algorithm.abs() < single.algorithm.abs() / 100.0);
    }

    #[test]
    fn q8_negative_eight_scale_cancels_before_input_approximation() {
        let x = vec![100_000.125_f32; 64];
        let q = vec![0_i8; 64];
        let alpha = normalization_scale(x[0]);
        let pairs: Vec<_> = x.iter().map(|&v| half_pair(v, alpha)).collect();
        let hi: Vec<_> = pairs.iter().map(|p| p.0).collect();
        let lo: Vec<_> = pairs.iter().map(|p| p.1).collect();
        for mode in [Mode::Single, Mode::Compensated] {
            let r = group_reference(&q, 0.125, -1.0, &x, alpha, &hi, &lo, x[0] * 64.0, mode);
            assert_eq!(r.original, 0.0);
            assert_eq!(r.algorithm, 0.0);
            assert_eq!(r.approximation_bound(), 0.0);
        }
    }

    #[test]
    fn sum_check_uses_original_values_and_rejects_wrong_reduction() {
        check_original_sum(0.0, 0.0, 2.0).unwrap();
        assert!(check_original_sum(1.0, 0.0, 2.0).is_err());
        assert!(check_original_sum(f32::NAN, 0.0, 2.0).is_err());
    }

    #[test]
    fn implementation_gate_and_approximation_quality_are_separate() {
        assert!(check_algorithm_error(1e-7, 1.0).is_ok());
        assert!(check_algorithm_error(0.1, 1.0).is_err());
        assert!(3e-5 < Mode::Single.quality_limit());
        assert!(3e-5 > Mode::Compensated.quality_limit());
    }
}
