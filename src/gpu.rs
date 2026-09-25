//! Original Metal compute backend; see kernels/qwen.metal for the dispatch ABI.
mod metadata;
mod profile;
mod timing;
use anyhow::{Context, Result, anyhow, bail, ensure};
pub use metadata::pack_bf16_exact;
use metadata::supported_bf16_quantization;
use metal::{
    Buffer, BufferRef, CommandBufferRef, CommandQueue, ComputeCommandEncoderRef,
    ComputePipelineState, Device, MTLCommandBufferStatus, MTLResourceOptions, MTLSize, ResourceRef,
};
pub use profile::ProfileRow;
use profile::StageProfiler;
use std::collections::HashMap;
use std::{cell::Cell, time::Instant};
pub use timing::FrameTiming;

/// Independently selectable target-block schedules for controlled GPU A/B runs.
/// M5 comparison selects original matmul plus batched recurrence by default.
/// Both alternatives remain explicit options. Single-token inference is unaffected.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct BlockKernelMode {
    pub shared_matmul: bool,
    pub batched_delta: bool,
}

impl Default for BlockKernelMode {
    fn default() -> Self {
        Self {
            shared_matmul: false,
            batched_delta: true,
        }
    }
}

impl BlockKernelMode {
    fn parse(matmul: &str, delta: &str) -> Result<Self> {
        ensure!(
            matches!(matmul, "legacy" | "shared"),
            "QWEN_METAL_BLOCK_MATMUL must be legacy or shared"
        );
        ensure!(
            matches!(delta, "sequential" | "batched"),
            "QWEN_METAL_BLOCK_DELTA must be sequential or batched"
        );
        Ok(Self {
            shared_matmul: matmul == "shared",
            batched_delta: delta == "batched",
        })
    }

    fn from_env() -> Result<Self> {
        let read = |name: &str, default: &str| -> Result<String> {
            match std::env::var(name) {
                Ok(value) => Ok(value),
                Err(std::env::VarError::NotPresent) => Ok(default.into()),
                Err(error) => Err(error).with_context(|| format!("invalid {name}")),
            }
        };
        Self::parse(
            &read("QWEN_METAL_BLOCK_MATMUL", "legacy")?,
            &read("QWEN_METAL_BLOCK_DELTA", "batched")?,
        )
    }
}

pub struct Gpu {
    pub device: Device,
    pub queue: CommandQueue,
    pipelines: HashMap<&'static str, ComputePipelineState>,
    reference_kernels: bool,
    matvec_variant: String,
    block_kernel_mode: BlockKernelMode,
    parallel_norm: bool,
    parallel_attention: bool,
    bf16_metadata: bool,
    compacted_matrices: Cell<usize>,
    metadata_saved_bytes: Cell<u64>,
    profiler: Option<StageProfiler>,
    frame_started: Cell<Option<Instant>>,
    frame_timing: Cell<Option<FrameTiming>>,
}

/// Normal inference has one serial encoder per token. Diagnostic profiling
/// instead samples one encoder per operation within the same command buffer.
pub struct DispatchEncoder<'a> {
    gpu: &'a Gpu,
    command: &'a CommandBufferRef,
    encoder: Option<&'a ComputeCommandEncoderRef>,
}
impl DispatchEncoder<'_> {
    pub fn encode(
        &self,
        name: &str,
        buffers: &[&BufferRef],
        params: &[u32],
        threads: usize,
        group_size: usize,
    ) -> Result<()> {
        self.encode_inner(name, buffers, None, params, threads, group_size)
    }
    /// Byte offsets are checked before any buffer binding or dispatch.
    pub fn encode_offsets(
        &self,
        name: &str,
        buffers: &[&BufferRef],
        offsets: &[usize],
        params: &[u32],
        threads: usize,
        group_size: usize,
    ) -> Result<()> {
        self.encode_inner(name, buffers, Some(offsets), params, threads, group_size)
    }
    fn encode_inner(
        &self,
        name: &str,
        buffers: &[&BufferRef],
        offsets: Option<&[usize]>,
        params: &[u32],
        threads: usize,
        group_size: usize,
    ) -> Result<()> {
        if let Some(profiler) = &self.gpu.profiler {
            let e = profiler.encoder(self.command, name, params)?;
            let result = self
                .gpu
                .encode_inner(e, name, buffers, offsets, params, threads, group_size);
            e.end_encoding();
            if result.is_ok() {
                profiler.finish_dispatch()?;
            } else {
                profiler.abort_dispatch();
            }
            result
        } else {
            self.gpu.encode_inner(
                self.encoder.unwrap(),
                name,
                buffers,
                offsets,
                params,
                threads,
                group_size,
            )
        }
    }
    pub fn end_encoding(self) -> Result<()> {
        if let Some(e) = self.encoder {
            e.end_encoding();
        }
        if let Some(profiler) = &self.gpu.profiler {
            profiler.resolve(self.command)?;
        }
        Ok(())
    }
}
const KERNELS: &[&str] = &[
    "matvec_f16",
    "matvec_affine",
    "embed_f16",
    "embed_affine",
    "embed_affine_bf16",
    "rms_norm",
    "rms_norm_parallel",
    "add",
    "swiglu",
    "conv_silu",
    "delta_norm",
    "delta_step",
    "delta_step_block",
    "gated_rms",
    "split_q_gate",
    "head_rms",
    "rope",
    "kv_append",
    "attn_scores",
    "softmax",
    "attn_values",
    "attn_values_parallel",
    "matvec_q4_g32",
    "matvec_q4_g64",
    "matvec_q4_g128",
    "matvec_q8_g32",
    "matvec_q8_g64",
    "matvec_q8_g128",
    "matvec_q4_g32_aligned",
    "matvec_q4_g64_aligned",
    "matvec_q4_g128_aligned",
    "matvec_q4_g32_stream",
    "matvec_q4_g64_stream",
    "matvec_q4_g128_stream",
    "matvec_q4_g32_bf16",
    "matvec_q4_g64_bf16",
    "matvec_q4_g128_bf16",
    "matvec_q8_g32_bf16",
    "matvec_q8_g64_bf16",
    "matvec_q8_g128_bf16",
    "matvec_q4_g64_aligned_bf16",
    "copy_f32",
    "matmul_q4_g32_b1",
    "matmul_q4_g32_b1_bf16",
    "matmul_q4_g32_b2",
    "matmul_q4_g32_b2_bf16",
    "matmul_q4_g32_b3",
    "matmul_q4_g32_b3_bf16",
    "matmul_q4_g32_b4",
    "matmul_q4_g32_b4_bf16",
    "matmul_q4_g64_b1",
    "matmul_q4_g64_b1_bf16",
    "matmul_q4_g64_b2",
    "matmul_q4_g64_b2_bf16",
    "matmul_q4_g64_b3",
    "matmul_q4_g64_b3_bf16",
    "matmul_q4_g64_b4",
    "matmul_q4_g64_b4_bf16",
    "matmul_q4_g128_b1",
    "matmul_q4_g128_b1_bf16",
    "matmul_q4_g128_b2",
    "matmul_q4_g128_b2_bf16",
    "matmul_q4_g128_b3",
    "matmul_q4_g128_b3_bf16",
    "matmul_q4_g128_b4",
    "matmul_q4_g128_b4_bf16",
    "matmul_q8_g32_b1",
    "matmul_q8_g32_b1_bf16",
    "matmul_q8_g32_b2",
    "matmul_q8_g32_b2_bf16",
    "matmul_q8_g32_b3",
    "matmul_q8_g32_b3_bf16",
    "matmul_q8_g32_b4",
    "matmul_q8_g32_b4_bf16",
    "matmul_q8_g64_b1",
    "matmul_q8_g64_b1_bf16",
    "matmul_q8_g64_b2",
    "matmul_q8_g64_b2_bf16",
    "matmul_q8_g64_b3",
    "matmul_q8_g64_b3_bf16",
    "matmul_q8_g64_b4",
    "matmul_q8_g64_b4_bf16",
    "matmul_q8_g128_b1",
    "matmul_q8_g128_b1_bf16",
    "matmul_q8_g128_b2",
    "matmul_q8_g128_b2_bf16",
    "matmul_q8_g128_b3",
    "matmul_q8_g128_b3_bf16",
    "matmul_q8_g128_b4",
    "matmul_q8_g128_b4_bf16",
    "matmul_f16_b1",
    "matmul_f16_b2",
    "matmul_f16_b3",
    "matmul_f16_b4",
    "matmul_q4_g64_b1_aligned",
    "matmul_q4_g64_b1_aligned_bf16",
    "matmul_q4_g64_b2_aligned",
    "matmul_q4_g64_b2_aligned_bf16",
    "matmul_q4_g64_b3_aligned",
    "matmul_q4_g64_b3_aligned_bf16",
    "matmul_q4_g64_b4_aligned",
    "matmul_q4_g64_b4_aligned_bf16",
    "matmul_q4_g64_b2_shared_aligned",
    "matmul_q4_g64_b2_shared_aligned_bf16",
    "matmul_q4_g64_b3_shared_aligned",
    "matmul_q4_g64_b3_shared_aligned_bf16",
];
impl Gpu {
    pub fn new() -> Result<Self> {
        Self::new_with_reference(std::env::var("QWEN_METAL_REFERENCE").is_ok_and(|v| v == "1"))
    }
    /// Keep the original arithmetic path available for same-binary A/B measurements.
    pub fn new_with_reference(reference_kernels: bool) -> Result<Self> {
        Self::new_with_variant(
            reference_kernels,
            &std::env::var("QWEN_METAL_GEMV").unwrap_or_else(|_| "aligned".into()),
        )
    }
    pub fn new_with_variant(reference_kernels: bool, variant: &str) -> Result<Self> {
        let block_kernel_mode = BlockKernelMode::from_env()?;
        ensure!(
            matches!(variant, "packed4" | "aligned" | "stream"),
            "QWEN_METAL_GEMV must be packed4, aligned or stream"
        );
        let norm_mode = std::env::var("QWEN_METAL_NORM").unwrap_or_else(|_| "serial".into());
        ensure!(
            matches!(norm_mode.as_str(), "parallel" | "serial"),
            "QWEN_METAL_NORM must be parallel or serial"
        );
        let attention_mode =
            std::env::var("QWEN_METAL_ATTN_VALUES").unwrap_or_else(|_| "serial".into());
        ensure!(
            matches!(attention_mode.as_str(), "parallel" | "serial"),
            "QWEN_METAL_ATTN_VALUES must be parallel or serial"
        );
        let metadata_mode = std::env::var("QWEN_METAL_METADATA").unwrap_or_else(|_| "f32".into());
        ensure!(
            matches!(metadata_mode.as_str(), "f32" | "bf16"),
            "QWEN_METAL_METADATA must be f32 or bf16"
        );
        let device = Device::system_default()
            .context("No Metal GPU available; this engine requires Apple Silicon")?;
        ensure!(
            device.has_unified_memory(),
            "Only unified-memory Apple Silicon GPUs are supported"
        );
        ensure!(
            device.supports_family(metal::MTLGPUFamily::Apple7),
            "GPU must support Apple7 or newer"
        );
        let options = metal::CompileOptions::new();
        // Numerical baseline: avoid the relaxed approximations of fast math.
        options.set_fast_math_enabled(false);
        let library = device
            .new_library_with_source(
                concat!(
                    include_str!("../kernels/qwen.metal"),
                    "\n",
                    include_str!("../kernels/norm_fast.metal"),
                    "\n",
                    include_str!("../kernels/attention_values.metal"),
                    "\n",
                    include_str!("../kernels/affine_bf16.metal")
                ),
                &options,
            )
            .map_err(|e| anyhow!("Metal shader compilation failed: {e}"))?;
        let mat_options = metal::CompileOptions::new();
        mat_options.set_fast_math_enabled(true);
        let mat_library = device
            .new_library_with_source(
                concat!(
                    include_str!("../kernels/matvec_fast.metal"),
                    "\n",
                    include_str!("../kernels/matvec_aligned.metal"),
                    "\n",
                    include_str!("../kernels/matvec_stream.metal"),
                    "\n",
                    include_str!("../kernels/affine_bf16.metal"),
                    "\n",
                    include_str!("../kernels/matmul_block.metal")
                ),
                &mat_options,
            )
            .map_err(|e| anyhow!("Metal matvec shader compilation failed: {e}"))?;
        let mut pipelines = HashMap::new();
        for &name in KERNELS {
            let function = (if name.starts_with("matvec_q")
                || name.starts_with("matmul_")
                || name == "copy_f32"
            {
                &mat_library
            } else {
                &library
            })
            .get_function(name, None)
            .map_err(|e| anyhow!("Metal function {name}: {e}"))?;
            let descriptor = metal::ComputePipelineDescriptor::new();
            descriptor.set_compute_function(Some(&function));
            descriptor.set_thread_group_size_is_multiple_of_thread_execution_width(true);
            if name.starts_with("matvec_q") {
                descriptor.set_max_total_threads_per_threadgroup(
                    if name.ends_with("_aligned")
                        || name.ends_with("_aligned_bf16")
                        || name.ends_with("_stream")
                    {
                        64
                    } else {
                        128
                    },
                );
            }
            if name.starts_with("matmul_") {
                descriptor.set_max_total_threads_per_threadgroup(
                    if name.ends_with("_aligned") || name.ends_with("_aligned_bf16") {
                        64
                    } else {
                        128
                    },
                );
            }
            if matches!(name, "rms_norm_parallel" | "attn_values_parallel") {
                descriptor.set_max_total_threads_per_threadgroup(256);
            }
            let pipeline = device
                .new_compute_pipeline_state(&descriptor)
                .map_err(|e| anyhow!("Metal pipeline {name}: {e}"))?;
            ensure!(
                pipeline.thread_execution_width() == 32,
                "{name} requires 32-lane SIMD groups"
            );
            pipelines.insert(name, pipeline);
        }
        let queue = device.new_command_queue();
        Ok(Self {
            device,
            queue,
            pipelines,
            reference_kernels,
            matvec_variant: variant.into(),
            block_kernel_mode,
            parallel_norm: norm_mode == "parallel",
            parallel_attention: attention_mode == "parallel",
            bf16_metadata: metadata_mode == "bf16" && !reference_kernels,
            compacted_matrices: Cell::new(0),
            metadata_saved_bytes: Cell::new(0),
            profiler: None,
            frame_started: Cell::new(None),
            frame_timing: Cell::new(None),
        })
    }
    pub fn kernel_mode(&self) -> &str {
        if self.reference_kernels {
            "reference"
        } else {
            &self.matvec_variant
        }
    }
    pub fn block_kernel_mode(&self) -> BlockKernelMode {
        self.block_kernel_mode
    }
    /// Call only between completed command buffers; Engine additionally requires
    /// empty model state before changing an inference schedule.
    pub fn set_block_kernel_mode(&mut self, mode: BlockKernelMode) {
        self.block_kernel_mode = mode;
    }
    pub fn norm_mode(&self) -> &str {
        if self.parallel_norm && !self.reference_kernels {
            "parallel"
        } else {
            "serial"
        }
    }
    pub fn attention_mode(&self) -> &str {
        if self.parallel_attention && !self.reference_kernels {
            "parallel"
        } else {
            "serial"
        }
    }
    /// Requested storage policy; individual inexact or unsupported matrices
    /// retain FP32 metadata. The reference path always reports and stores FP32.
    pub fn metadata_mode(&self) -> &str {
        if self.bf16_metadata { "bf16" } else { "f32" }
    }
    /// Successfully loaded compacted matrices and their actual saved bytes.
    pub fn metadata_stats(&self) -> (usize, u64) {
        (
            self.compacted_matrices.get(),
            self.metadata_saved_bytes.get(),
        )
    }
    pub(crate) fn compact_metadata(
        &self,
        bits: u32,
        group: usize,
        scales: &[f32],
        biases: &[f32],
    ) -> Option<(Vec<u16>, Vec<u16>)> {
        if !self.bf16_metadata || !supported_bf16_quantization(bits, group) {
            return None;
        }
        Some((pack_bf16_exact(scales)?, pack_bf16_exact(biases)?))
    }
    pub(crate) fn record_compacted_metadata(&self, saved_bytes: u64) {
        self.compacted_matrices
            .set(self.compacted_matrices.get() + 1);
        self.metadata_saved_bytes
            .set(self.metadata_saved_bytes.get() + saved_bytes);
    }
    pub fn enable_profiling(&mut self) -> Result<()> {
        self.profiler = Some(StageProfiler::new(&self.device)?);
        Ok(())
    }
    pub fn enable_command_profiling(&mut self) -> Result<()> {
        self.profiler = Some(StageProfiler::new_command_timing(&self.device)?);
        Ok(())
    }
    pub fn profile_backend(&self) -> Option<&str> {
        self.profiler.as_ref().map(|p| p.backend_name())
    }
    pub fn disable_profiling(&mut self) {
        self.profiler = None;
    }
    pub fn set_parallel_norm(&mut self, enabled: bool) {
        self.parallel_norm = enabled;
    }
    pub fn set_parallel_attention(&mut self, enabled: bool) {
        self.parallel_attention = enabled;
    }
    pub fn last_frame_timing(&self) -> Option<FrameTiming> {
        self.frame_timing.get()
    }
    pub fn profile_report(&self) -> Result<Vec<ProfileRow>> {
        self.profiler
            .as_ref()
            .context("GPU profiling is not enabled")?
            .report()
    }
    pub fn begin_encoding<'a>(&'a self, command: &'a CommandBufferRef) -> DispatchEncoder<'a> {
        if let Some(profiler) = &self.profiler {
            profiler.reset();
        }
        DispatchEncoder {
            gpu: self,
            command,
            encoder: if self.profiler.is_none() {
                Some(command.new_compute_command_encoder())
            } else {
                None
            },
        }
    }
    pub fn alloc_f32(&self, len: usize) -> Result<Buffer> {
        let bytes = len
            .checked_mul(4)
            .context("GPU allocation size overflow")?
            .max(4);
        self.check_allocation(bytes)?;
        let buffer = self
            .device
            .new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared);
        // Metal does not promise zero initialization; recurrent state and padding need it.
        unsafe {
            std::ptr::write_bytes(buffer.contents().cast::<u8>(), 0, bytes);
        }
        Ok(buffer)
    }
    fn check_allocation(&self, bytes: usize) -> Result<()> {
        ensure!(
            bytes > 0 && bytes <= self.device.max_buffer_length() as usize,
            "GPU allocation {bytes} exceeds device buffer limit {}",
            self.device.max_buffer_length()
        );
        Ok(())
    }
    pub fn upload_f32(&self, values: &[f32]) -> Result<Buffer> {
        self.upload_bytes(bytemuck::cast_slice(values))
    }
    pub fn upload_bytes(&self, values: &[u8]) -> Result<Buffer> {
        if values.is_empty() {
            return self.alloc_f32(0);
        }
        self.check_allocation(values.len())?;
        Ok(self.device.new_buffer_with_data(
            values.as_ptr().cast(),
            values.len() as u64,
            MTLResourceOptions::StorageModeShared,
        ))
    }
    /// Call only after finish() for the command that last wrote this buffer.
    pub fn read_f32(&self, buffer: &BufferRef, len: usize) -> Result<Vec<f32>> {
        ensure!(
            len.checked_mul(4)
                .is_some_and(|n| n <= buffer.length() as usize),
            "Read exceeds GPU buffer"
        );
        ensure!(
            buffer.storage_mode() == metal::MTLStorageMode::Shared,
            "Read requires shared storage"
        );
        Ok(unsafe { std::slice::from_raw_parts(buffer.contents().cast::<f32>(), len) }.to_vec())
    }
    /// The caller should wrap each token in objc::rc::autoreleasepool.
    pub fn begin(&self) -> &CommandBufferRef {
        self.frame_started.set(Some(Instant::now()));
        self.frame_timing.set(None);
        self.queue.new_command_buffer()
    }
    pub fn encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        name: &str,
        buffers: &[&BufferRef],
        params: &[u32],
        threads: usize,
        group_size: usize,
    ) -> Result<()> {
        self.encode_inner(encoder, name, buffers, None, params, threads, group_size)
    }
    /// Bind checked byte-offset views without allocating a list for ordinary dispatches.
    pub fn encode_offsets(
        &self,
        encoder: &ComputeCommandEncoderRef,
        name: &str,
        buffers: &[&BufferRef],
        offsets: &[usize],
        params: &[u32],
        threads: usize,
        group_size: usize,
    ) -> Result<()> {
        self.encode_inner(
            encoder,
            name,
            buffers,
            Some(offsets),
            params,
            threads,
            group_size,
        )
    }
    fn encode_inner(
        &self,
        encoder: &ComputeCommandEncoderRef,
        name: &str,
        buffers: &[&BufferRef],
        offsets: Option<&[usize]>,
        params: &[u32],
        threads: usize,
        group_size: usize,
    ) -> Result<()> {
        let (sizes, expected_threads) = dispatch_layout(name, params)?;
        ensure!(
            !self.reference_kernels || !name.starts_with("matmul_"),
            "Block matrix dispatch is unavailable in reference mode"
        );
        ensure!(
            sizes.len() == buffers.len(),
            "{name}: expected {} buffers, got {}",
            sizes.len(),
            buffers.len()
        );
        ensure!(
            offsets.is_none_or(|o| o.len() == buffers.len()),
            "{name}: offset count must match buffers"
        );
        let offset = |i: usize| offsets.map_or(0, |o| o[i]);
        for (i, (buffer, &minimum)) in buffers.iter().zip(&sizes).enumerate() {
            checked_buffer_range(offset(i), minimum, buffer.length() as usize)
                .with_context(|| format!("{name}: buffer {i}"))?;
        }
        // Matrix output is never an input; parallel copies require disjoint regions
        // unless the source and destination views are identical.
        if name.starts_with("matmul_") || name == "copy_f32" {
            let out = buffers.len() - 1;
            for i in 0..out {
                if std::ptr::eq(buffers[i], buffers[out]) {
                    let identical_copy = name == "copy_f32" && offset(i) == offset(out);
                    ensure!(
                        identical_copy
                            || !ranges_overlap(offset(i), sizes[i], offset(out), sizes[out]),
                        "{name}: output overlaps input buffer {i}"
                    );
                }
            }
        }
        if name == "delta_step_block" {
            // The three outputs may share an allocation only as disjoint views.
            // In particular, a snapshot must not overwrite the live state or a
            // later token's inputs while other SIMD groups still consume them.
            for out in [5, 6, 7] {
                for i in 0..buffers.len() {
                    if i != out && std::ptr::eq(buffers[i], buffers[out]) {
                        ensure!(
                            !ranges_overlap(offset(i), sizes[i], offset(out), sizes[out]),
                            "{name}: output {out} overlaps buffer {i}"
                        );
                    }
                }
            }
        }
        ensure!(
            !self.reference_kernels || !name.ends_with("_bf16"),
            "BF16 metadata dispatch is unavailable in reference mode"
        );
        let specialized = if name.starts_with("matmul_") {
            Some(
                specialized_matmul(
                    name,
                    params,
                    self.matvec_variant == "aligned"
                        && offset(0) % 8 == 0
                        && offset(if name == "matmul_f16" { 1 } else { 3 }) % 16 == 0,
                    self.block_kernel_mode.shared_matmul,
                )
                .context("Unsupported matrix block specialization")?,
            )
        } else if name == "matvec_affine_bf16" {
            ensure!(
                offset(3) % 16 == 0,
                "BF16 matvec input requires a 16-byte aligned offset"
            );
            // There is no BF16 stream kernel. Explicitly select the packed
            // variant so compressed metadata is never read as FP32.
            Some(
                specialized_matvec_bf16(
                    params,
                    self.matvec_variant == "aligned" && offset(0) % 8 == 0,
                )
                .context("Unsupported BF16 affine quantization")?,
            )
        } else if !self.reference_kernels && name == "matvec_affine" && offset(3) % 16 == 0 {
            let stream = if self.matvec_variant == "stream"
                && offset(0) % 8 == 0
                && params[2] == 4
                && u64::from(params[0]) * u64::from(params[1]) <= i32::MAX as u64
            {
                match params[3] {
                    32 => Some("matvec_q4_g32_stream"),
                    64 => Some("matvec_q4_g64_stream"),
                    128 => Some("matvec_q4_g128_stream"),
                    _ => None,
                }
            } else {
                None
            };
            stream.or_else(|| {
                specialized_matvec(
                    params,
                    self.matvec_variant == "aligned" && offset(0) % 8 == 0,
                )
            })
        } else if !self.reference_kernels
            && self.parallel_norm
            && name == "rms_norm"
            && params[0] >= 1024
            && buffers.len() == 3
            && (0..3).all(|i| offset(i) % 16 == 0)
            && !std::ptr::eq(buffers[0], buffers[2])
            && !std::ptr::eq(buffers[1], buffers[2])
        {
            Some("rms_norm_parallel")
        } else if !self.reference_kernels
            && self.parallel_attention
            && name == "attn_values"
            && params[2] >= 32
            && params[3] >= 128
            && buffers.len() == 4
            && buffers[..3].iter().all(|b| !std::ptr::eq(*b, buffers[3]))
        {
            Some("attn_values_parallel")
        } else {
            None
        };
        let pipeline = self
            .pipelines
            .get(specialized.unwrap_or(name))
            .with_context(|| format!("Unknown GPU kernel {name}"))?;
        ensure!(
            sizes.len() == buffers.len(),
            "{name}: expected {} buffers, got {}",
            sizes.len(),
            buffers.len()
        );
        ensure!(
            threads == expected_threads,
            "{name}: expected {expected_threads} threads, got {threads}"
        );
        let actual_group_size = if matches!(
            specialized,
            Some("rms_norm_parallel" | "attn_values_parallel")
        ) {
            256
        } else if specialized.is_some_and(|n| {
            n.ends_with("_aligned") || n.ends_with("_aligned_bf16") || n.ends_with("_stream")
        }) {
            64
        } else {
            group_size
        };
        ensure!(
            group_size >= 32
                && group_size % 32 == 0
                && actual_group_size <= pipeline.max_total_threads_per_threadgroup() as usize,
            "{name}: invalid threadgroup size {group_size}"
        );
        for (i, buffer) in buffers.iter().enumerate() {
            encoder.set_buffer(i as u64, Some(buffer), offset(i) as u64);
        }
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_bytes(
            15,
            std::mem::size_of_val(params) as u64,
            params.as_ptr().cast(),
        );
        let actual_threads = if specialized == Some("rms_norm_parallel") {
            256
        } else if specialized == Some("attn_values_parallel") {
            params[0] as usize * (params[2] as usize).div_ceil(32) * 256
        } else if specialized.is_some() {
            (params[0] as usize).div_ceil(4) * 32
        } else {
            threads
        };
        encoder.dispatch_thread_groups(
            MTLSize::new(actual_threads.div_ceil(actual_group_size) as u64, 1, 1),
            MTLSize::new(actual_group_size as u64, 1, 1),
        );
        // Explicit dispatch-to-dispatch ordering, including in-place state updates.
        if self.reference_kernels {
            let resources: Vec<&ResourceRef> = buffers.iter().map(|b| &***b).collect();
            encoder.memory_barrier_with_resources(&resources);
        } else {
            // Declare actual shader writes without a heap-allocated resource list.
            // The engine uses a serial encoder; Metal ignores these barriers there.
            // Keep the correct side-effect set explicit for the dispatch ABI.
            let outputs: &[usize] = match name {
                "conv_silu" => &[2, 3],
                "delta_step" => &[5, 6],
                "delta_step_block" => &[5, 6, 7],
                "split_q_gate" => &[1, 2],
                "kv_append" => &[2, 3],
                "delta_norm" | "head_rms" | "rope" | "softmax" => &[0],
                "matvec_affine" | "matvec_affine_bf16" | "matmul_affine" | "matmul_affine_bf16" => {
                    &[4]
                }
                "embed_affine" | "embed_affine_bf16" | "gated_rms" | "attn_values" => &[3],
                "embed_f16" | "copy_f32" => &[1],
                _ => &[2],
            };
            let mut written: [&ResourceRef; 3] = [&***buffers.first().unwrap(); 3];
            for (out, &index) in written.iter_mut().zip(outputs) {
                *out = &**buffers[index];
            }
            encoder.memory_barrier_with_resources(&written[..outputs.len()]);
        }
        Ok(())
    }
    pub fn finish(&self, command: &CommandBufferRef) -> Result<()> {
        let cpu_encode_seconds = self
            .frame_started
            .get()
            .map(|t| t.elapsed().as_secs_f64())
            .unwrap_or(0.);
        let commit_started = Instant::now();
        command.commit();
        let cpu_commit_seconds = commit_started.elapsed().as_secs_f64();
        let wait_started = Instant::now();
        command.wait_until_completed();
        let completion_wait_seconds = wait_started.elapsed().as_secs_f64();
        ensure!(
            command.status() == MTLCommandBufferStatus::Completed,
            "Metal command failed with status {:?}",
            command.status()
        );
        self.frame_timing.set(
            timing::command_timing(
                command,
                cpu_encode_seconds,
                cpu_commit_seconds,
                completion_wait_seconds,
            )
            .ok(),
        );
        Ok(())
    }
}

fn checked_buffer_range(offset: usize, minimum: usize, length: usize) -> Result<()> {
    ensure!(offset % 4 == 0, "GPU buffer offsets must be 4-byte aligned");
    let end = offset
        .checked_add(minimum)
        .context("GPU buffer view byte range overflow")?;
    ensure!(
        end <= length,
        "GPU buffer view ends at {end} bytes but buffer has {length}"
    );
    Ok(())
}

fn ranges_overlap(a: usize, a_len: usize, b: usize, b_len: usize) -> bool {
    // All views were checked for overflow before alias validation.
    a < b + b_len && b < a + a_len
}

fn specialized_matmul(name: &str, p: &[u32], aligned: bool, shared: bool) -> Option<&'static str> {
    if name == "matmul_f16" {
        return [
            "matmul_f16_b1",
            "matmul_f16_b2",
            "matmul_f16_b3",
            "matmul_f16_b4",
        ]
        .get(p[2] as usize - 1)
        .copied();
    }
    if aligned
        && p[0] % 4 == 0
        && p[1] % 512 == 0
        && p[2] == 4
        && p[3] == 64
        && u64::from(p[0]) * u64::from(p[1]) <= i32::MAX as u64
        && u64::from(p[1]) * u64::from(p[4]) <= i32::MAX as u64
    {
        if shared {
            match (p[4], name.ends_with("_bf16")) {
                (2, false) => return Some("matmul_q4_g64_b2_shared_aligned"),
                (2, true) => return Some("matmul_q4_g64_b2_shared_aligned_bf16"),
                (3, false) => return Some("matmul_q4_g64_b3_shared_aligned"),
                (3, true) => return Some("matmul_q4_g64_b3_shared_aligned_bf16"),
                _ => (),
            }
        }
        return match (p[4], name.ends_with("_bf16")) {
            (1, false) => Some("matmul_q4_g64_b1_aligned"),
            (1, true) => Some("matmul_q4_g64_b1_aligned_bf16"),
            (2, false) => Some("matmul_q4_g64_b2_aligned"),
            (2, true) => Some("matmul_q4_g64_b2_aligned_bf16"),
            (3, false) => Some("matmul_q4_g64_b3_aligned"),
            (3, true) => Some("matmul_q4_g64_b3_aligned_bf16"),
            (4, false) => Some("matmul_q4_g64_b4_aligned"),
            (4, true) => Some("matmul_q4_g64_b4_aligned_bf16"),
            _ => None,
        };
    }
    match (p[2], p[3], p[4], name.ends_with("_bf16")) {
        (4, 32, 1, false) => Some("matmul_q4_g32_b1"),
        (4, 32, 1, true) => Some("matmul_q4_g32_b1_bf16"),
        (4, 32, 2, false) => Some("matmul_q4_g32_b2"),
        (4, 32, 2, true) => Some("matmul_q4_g32_b2_bf16"),
        (4, 32, 3, false) => Some("matmul_q4_g32_b3"),
        (4, 32, 3, true) => Some("matmul_q4_g32_b3_bf16"),
        (4, 32, 4, false) => Some("matmul_q4_g32_b4"),
        (4, 32, 4, true) => Some("matmul_q4_g32_b4_bf16"),
        (4, 64, 1, false) => Some("matmul_q4_g64_b1"),
        (4, 64, 1, true) => Some("matmul_q4_g64_b1_bf16"),
        (4, 64, 2, false) => Some("matmul_q4_g64_b2"),
        (4, 64, 2, true) => Some("matmul_q4_g64_b2_bf16"),
        (4, 64, 3, false) => Some("matmul_q4_g64_b3"),
        (4, 64, 3, true) => Some("matmul_q4_g64_b3_bf16"),
        (4, 64, 4, false) => Some("matmul_q4_g64_b4"),
        (4, 64, 4, true) => Some("matmul_q4_g64_b4_bf16"),
        (4, 128, 1, false) => Some("matmul_q4_g128_b1"),
        (4, 128, 1, true) => Some("matmul_q4_g128_b1_bf16"),
        (4, 128, 2, false) => Some("matmul_q4_g128_b2"),
        (4, 128, 2, true) => Some("matmul_q4_g128_b2_bf16"),
        (4, 128, 3, false) => Some("matmul_q4_g128_b3"),
        (4, 128, 3, true) => Some("matmul_q4_g128_b3_bf16"),
        (4, 128, 4, false) => Some("matmul_q4_g128_b4"),
        (4, 128, 4, true) => Some("matmul_q4_g128_b4_bf16"),
        (8, 32, 1, false) => Some("matmul_q8_g32_b1"),
        (8, 32, 1, true) => Some("matmul_q8_g32_b1_bf16"),
        (8, 32, 2, false) => Some("matmul_q8_g32_b2"),
        (8, 32, 2, true) => Some("matmul_q8_g32_b2_bf16"),
        (8, 32, 3, false) => Some("matmul_q8_g32_b3"),
        (8, 32, 3, true) => Some("matmul_q8_g32_b3_bf16"),
        (8, 32, 4, false) => Some("matmul_q8_g32_b4"),
        (8, 32, 4, true) => Some("matmul_q8_g32_b4_bf16"),
        (8, 64, 1, false) => Some("matmul_q8_g64_b1"),
        (8, 64, 1, true) => Some("matmul_q8_g64_b1_bf16"),
        (8, 64, 2, false) => Some("matmul_q8_g64_b2"),
        (8, 64, 2, true) => Some("matmul_q8_g64_b2_bf16"),
        (8, 64, 3, false) => Some("matmul_q8_g64_b3"),
        (8, 64, 3, true) => Some("matmul_q8_g64_b3_bf16"),
        (8, 64, 4, false) => Some("matmul_q8_g64_b4"),
        (8, 64, 4, true) => Some("matmul_q8_g64_b4_bf16"),
        (8, 128, 1, false) => Some("matmul_q8_g128_b1"),
        (8, 128, 1, true) => Some("matmul_q8_g128_b1_bf16"),
        (8, 128, 2, false) => Some("matmul_q8_g128_b2"),
        (8, 128, 2, true) => Some("matmul_q8_g128_b2_bf16"),
        (8, 128, 3, false) => Some("matmul_q8_g128_b3"),
        (8, 128, 3, true) => Some("matmul_q8_g128_b3_bf16"),
        (8, 128, 4, false) => Some("matmul_q8_g128_b4"),
        (8, 128, 4, true) => Some("matmul_q8_g128_b4_bf16"),
        _ => None,
    }
}

// Called after dispatch_layout validates the four-word matrix ABI. Static names
// avoid allocating a String for every projection in the token hot path.
fn specialized_matvec(p: &[u32], aligned: bool) -> Option<&'static str> {
    let aligned = aligned
        && p[0] % 4 == 0
        && p[1] % 512 == 0
        && u64::from(p[0]) * u64::from(p[1]) <= i32::MAX as u64;
    match (p[2], p[3], aligned) {
        (4, 32, true) => Some("matvec_q4_g32_aligned"),
        (4, 64, true) => Some("matvec_q4_g64_aligned"),
        (4, 128, true) => Some("matvec_q4_g128_aligned"),
        (4, 32, _) => Some("matvec_q4_g32"),
        (4, 64, _) => Some("matvec_q4_g64"),
        (4, 128, _) => Some("matvec_q4_g128"),
        (8, 32, _) => Some("matvec_q8_g32"),
        (8, 64, _) => Some("matvec_q8_g64"),
        (8, 128, _) => Some("matvec_q8_g128"),
        _ => None,
    }
}

fn specialized_matvec_bf16(p: &[u32], aligned: bool) -> Option<&'static str> {
    let aligned = aligned
        && p[0] % 4 == 0
        && p[1] % 512 == 0
        && u64::from(p[0]) * u64::from(p[1]) <= i32::MAX as u64;
    match (p[2], p[3], aligned) {
        (4, 64, true) => Some("matvec_q4_g64_aligned_bf16"),
        (4, 32, _) => Some("matvec_q4_g32_bf16"),
        (4, 64, _) => Some("matvec_q4_g64_bf16"),
        (4, 128, _) => Some("matvec_q4_g128_bf16"),
        (8, 32, _) => Some("matvec_q8_g32_bf16"),
        (8, 64, _) => Some("matvec_q8_g64_bf16"),
        (8, 128, _) => Some("matvec_q8_g128_bf16"),
        _ => None,
    }
}

// Host validation is independent of GPU availability and prevents malformed ABI inputs
// from becoming unchecked shader memory access. All indexing stays within uint range.
fn dispatch_layout(name: &str, p: &[u32]) -> Result<(Vec<usize>, usize)> {
    let count = match name {
        "matvec_f16" | "embed_f16" | "rms_norm" | "conv_silu" | "split_q_gate" | "softmax" => 2,
        "matvec_affine" | "embed_affine" | "matvec_affine_bf16" | "embed_affine_bf16"
        | "delta_step" | "attn_scores" | "attn_values" => 4,
        "delta_norm" | "gated_rms" | "head_rms" | "kv_append" | "matmul_f16" => 3,
        "rope" | "matmul_affine" | "matmul_affine_bf16" | "delta_step_block" => 5,
        "add" | "swiglu" | "copy_f32" => 1,
        _ => bail!("Unknown GPU kernel {name}"),
    };
    ensure!(p.len() == count, "{name}: expected {count} parameters");
    let n = |i: usize| p[i] as usize;
    let product = |v: &[usize]| -> Result<usize> {
        let size = v
            .iter()
            .try_fold(1usize, |a, &b| a.checked_mul(b))
            .context("GPU shape overflow")?;
        ensure!(size <= u32::MAX as usize, "GPU element index overflow");
        Ok(size)
    };
    let f = |elements: usize| -> Result<usize> {
        elements.checked_mul(4).context("GPU byte size overflow")
    };
    let positive = |indices: &[usize]| -> Result<()> {
        ensure!(
            indices.iter().all(|&i| p[i] > 0),
            "{name}: dimensions must be nonzero"
        );
        Ok(())
    };
    let eps = |i: usize| -> Result<()> {
        let value = f32::from_bits(p[i]);
        ensure!(
            value.is_finite() && value > 0.,
            "{name}: epsilon must be finite and positive"
        );
        Ok(())
    };
    Ok(match name {
        "matvec_f16" | "embed_f16" | "matvec_affine" | "embed_affine" | "matvec_affine_bf16"
        | "embed_affine_bf16" => {
            positive(&[1])?;
            let embedding = name.starts_with("embed");
            let rows = if embedding {
                n(0).checked_add(1).context("Embedding row overflow")?
            } else {
                positive(&[0])?;
                n(0)
            };
            let total = product(&[rows, n(1)])?;
            let output = if embedding { n(1) } else { n(0) };
            let bf16_metadata = name.ends_with("_bf16");
            let mut sizes = if name.ends_with("affine") || bf16_metadata {
                ensure!(
                    matches!(p[2], 4 | 8) && p[3] > 0,
                    "Only affine 4/8-bit quantization supported"
                );
                ensure!(
                    n(1) % (32 / n(2)) == 0 && n(1) % n(3) == 0,
                    "Invalid packed matrix alignment"
                );
                ensure!(
                    !bf16_metadata || supported_bf16_quantization(p[2], n(3)),
                    "BF16 metadata supports only affine Q4/Q8 groups 32, 64 or 128"
                );
                let metadata_bytes = (total / n(3))
                    .checked_mul(if bf16_metadata { 2 } else { 4 })
                    .context("Affine metadata byte size overflow")?;
                vec![
                    total
                        .checked_mul(n(2))
                        .context("Packed byte size overflow")?
                        / 8,
                    metadata_bytes,
                    metadata_bytes,
                ]
            } else {
                vec![total.checked_mul(2).context("Dense byte size overflow")?]
            };
            if !embedding {
                sizes.push(f(n(1))?);
            }
            sizes.push(f(output)?);
            (
                sizes,
                if embedding {
                    output
                } else {
                    product(&[output, 32])?
                },
            )
        }
        "matmul_affine" | "matmul_affine_bf16" | "matmul_f16" => {
            positive(&[0, 1])?;
            let affine = name != "matmul_f16";
            let batch = n(if affine { 4 } else { 2 });
            ensure!(
                (1..=4).contains(&batch),
                "Matrix blocks support one through four RHS"
            );
            let total = product(&[n(0), n(1)])?;
            let mut sizes = if affine {
                ensure!(
                    supported_bf16_quantization(p[2], n(3)),
                    "Matrix blocks support affine Q4/Q8 groups 32, 64 or 128"
                );
                ensure!(
                    n(1) % (32 / n(2)) == 0 && n(1) % n(3) == 0,
                    "Invalid packed matrix alignment"
                );
                let metadata = (total / n(3))
                    .checked_mul(if name.ends_with("_bf16") { 2 } else { 4 })
                    .context("Affine block metadata byte size overflow")?;
                vec![
                    total
                        .checked_mul(n(2))
                        .context("Packed block byte size overflow")?
                        / 8,
                    metadata,
                    metadata,
                ]
            } else {
                vec![
                    total
                        .checked_mul(2)
                        .context("Dense block byte size overflow")?,
                ]
            };
            sizes.push(f(product(&[batch, n(1)])?)?);
            sizes.push(f(product(&[batch, n(0)])?)?);
            (sizes, product(&[n(0), 32])?)
        }
        "copy_f32" => {
            positive(&[0])?;
            (vec![f(n(0))?; 2], n(0))
        }
        "rms_norm" => {
            positive(&[0])?;
            eps(1)?;
            (vec![f(n(0))?; 3], 32)
        }
        "add" | "swiglu" => {
            positive(&[0])?;
            (vec![f(n(0))?; 3], n(0))
        }
        "conv_silu" => {
            positive(&[0, 1])?;
            (
                vec![
                    f(n(0))?,
                    f(product(&[n(0), n(1)])?)?,
                    f(product(&[n(0), n(1) - 1])?)?,
                    f(n(0))?,
                ],
                n(0),
            )
        }
        "delta_norm" => {
            positive(&[0, 1])?;
            eps(2)?;
            (vec![f(product(&[2, n(0), n(1)])?)?], product(&[n(0), 32])?)
        }
        "delta_step" | "delta_step_block" => {
            positive(&[0, 1, 2, 3])?;
            ensure!(
                n(1) % n(0) == 0,
                "Value heads must be divisible by key heads"
            );
            let values = product(&[n(1), n(3)])?;
            let qkv = product(&[2, n(0), n(2)])?
                .checked_add(values)
                .context("QKV shape overflow")?;
            ensure!(qkv <= u32::MAX as usize, "QKV index overflow");
            let batch = if name == "delta_step_block" {
                ensure!(n(2) <= 128, "DeltaNet block key dimension must be <=128");
                ensure!(
                    (1..=4).contains(&n(4)),
                    "DeltaNet block width must be 1..=4"
                );
                n(4)
            } else {
                1
            };
            let mut sizes = vec![
                f(product(&[qkv, batch])?)?,
                f(product(&[n(1), batch])?)?,
                f(product(&[n(1), batch])?)?,
                f(n(1))?,
                f(n(1))?,
                f(product(&[values, n(2)])?)?,
                f(product(&[values, batch])?)?,
            ];
            if name == "delta_step_block" {
                sizes.push(f(product(&[values, n(2), batch])?)?);
            }
            (sizes, product(&[values, 32])?)
        }
        "gated_rms" => {
            positive(&[0, 1])?;
            eps(2)?;
            let len = f(product(&[n(0), n(1)])?)?;
            (vec![len, len, f(n(1))?, len], product(&[n(0), 32])?)
        }
        "head_rms" => {
            positive(&[0, 1])?;
            eps(2)?;
            (
                vec![f(product(&[n(0), n(1)])?)?, f(n(1))?],
                product(&[n(0), 32])?,
            )
        }
        "split_q_gate" => {
            positive(&[0, 1])?;
            let len = product(&[n(0), n(1)])?;
            (vec![f(product(&[len, 2])?)?, f(len)?, f(len)?], len)
        }
        "rope" => {
            positive(&[0, 1, 2])?;
            ensure!(p[2] % 2 == 0 && p[2] <= p[1], "Invalid rotary dimension");
            let theta = f32::from_bits(p[4]);
            ensure!(theta.is_finite() && theta > 0., "Invalid RoPE theta");
            (
                vec![f(product(&[n(0), n(1)])?)?],
                product(&[n(0), n(2) / 2])?,
            )
        }
        "kv_append" => {
            positive(&[0, 1])?;
            let len = product(&[n(0), n(1)])?;
            let cap = product(&[n(2) + 1, len])?;
            (vec![f(len)?, f(len)?, f(cap)?, f(cap)?], len)
        }
        "attn_scores" | "attn_values" => {
            positive(&[0, 1, 2, 3])?;
            ensure!(
                n(0) % n(1) == 0,
                "Attention heads must be divisible by KV heads"
            );
            let q = f(product(&[n(0), n(2)])?)?;
            let kv = f(product(&[n(3), n(1), n(2)])?)?;
            let score = product(&[n(0), n(3)])?;
            if name == "attn_scores" {
                (vec![q, kv, f(score)?], product(&[score, 32])?)
            } else {
                (vec![f(score)?, kv, q, q], product(&[n(0), n(2)])?)
            }
        }
        "softmax" => {
            positive(&[0, 1])?;
            (vec![f(product(&[n(0), n(1)])?)?], product(&[n(0), 32])?)
        }
        _ => unreachable!(),
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn block_kernel_modes_are_independent_and_reject_unknown_values() -> Result<()> {
        assert_eq!(
            BlockKernelMode::parse("legacy", "batched")?,
            BlockKernelMode::default()
        );
        for (matmul, delta, shared_matmul, batched_delta) in [
            ("legacy", "sequential", false, false),
            ("shared", "sequential", true, false),
            ("legacy", "batched", false, true),
            ("shared", "batched", true, true),
        ] {
            assert_eq!(
                BlockKernelMode::parse(matmul, delta)?,
                BlockKernelMode {
                    shared_matmul,
                    batched_delta
                }
            );
        }
        for invalid in ["", "auto", "SHARED", "shared "] {
            assert!(BlockKernelMode::parse(invalid, "sequential").is_err());
            assert!(BlockKernelMode::parse("legacy", invalid).is_err());
        }
        Ok(())
    }

    #[test]
    fn shared_matmul_selection_changes_only_supported_aligned_batches() {
        for batch in 1..=4 {
            for name in ["matmul_affine", "matmul_affine_bf16"] {
                let params = [20, 5120, 4, 64, batch];
                let legacy = specialized_matmul(name, &params, true, false).unwrap();
                let shared = specialized_matmul(name, &params, true, true).unwrap();
                assert!(!legacy.contains("_shared"));
                assert_eq!(shared.contains("_shared"), matches!(batch, 2 | 3));
                if matches!(batch, 1 | 4) {
                    assert_eq!(legacy, shared);
                }
                for (fallback, aligned) in [
                    (params, false),
                    ([19, 5120, 4, 64, batch], true),
                    ([20, 5120, 8, 64, batch], true),
                    ([20, 5120, 4, 32, batch], true),
                ] {
                    assert_eq!(
                        specialized_matmul(name, &fallback, aligned, false),
                        specialized_matmul(name, &fallback, aligned, true),
                    );
                }
            }
        }
    }
    #[test]
    fn buffer_views_reject_misalignment_bounds_and_integer_wraparound() {
        assert!(checked_buffer_range(4, 12, 16).is_ok());
        assert!(checked_buffer_range(0, 16, 16).is_ok());
        assert!(checked_buffer_range(4, 16, 16).is_err());
        assert!(checked_buffer_range(2, 12, 16).is_err());
        assert!(checked_buffer_range(usize::MAX - 3, 8, usize::MAX).is_err());
        assert!(ranges_overlap(0, 8, 4, 8));
        assert!(!ranges_overlap(0, 8, 8, 8));
    }
    #[test]
    fn block_specialization_preserves_alignment_and_all_batch_sizes() {
        for batch in 1..=4 {
            let params = [20, 5120, 4, 64, batch];
            assert!(
                specialized_matmul("matmul_affine", &params, true, false)
                    .unwrap()
                    .ends_with("_aligned")
            );
            assert!(
                specialized_matmul("matmul_affine_bf16", &params, true, false)
                    .unwrap()
                    .ends_with("_aligned_bf16")
            );
            assert!(
                !specialized_matmul("matmul_affine", &params, false, false)
                    .unwrap()
                    .ends_with("_aligned")
            );
            assert!(
                !specialized_matmul("matmul_affine", &[19, 5120, 4, 64, batch], true, false)
                    .unwrap()
                    .ends_with("_aligned")
            );
            assert!(
                !specialized_matmul("matmul_affine", &[20, 5120, 8, 64, batch], true, false)
                    .unwrap()
                    .ends_with("_aligned")
            );
        }
    }
    #[test]
    fn batch_dispatch_layout_sizes_all_rhs_and_rejects_invalid_batches() -> Result<()> {
        for batch in 1..=4 {
            let (sizes, threads) = dispatch_layout("matmul_affine", &[3, 128, 4, 64, batch])?;
            assert_eq!(
                sizes,
                vec![192, 24, 24, 512 * batch as usize, 12 * batch as usize]
            );
            assert_eq!(threads, 96);
            let (sizes, _) = dispatch_layout("matmul_affine_bf16", &[3, 128, 8, 32, batch])?;
            assert_eq!(
                sizes,
                vec![384, 24, 24, 512 * batch as usize, 12 * batch as usize]
            );
            let (sizes, _) = dispatch_layout("matmul_f16", &[3, 127, batch])?;
            assert_eq!(sizes, vec![762, 508 * batch as usize, 12 * batch as usize]);
        }
        for batch in [0, 5, u32::MAX] {
            assert!(dispatch_layout("matmul_affine", &[3, 128, 4, 64, batch]).is_err());
            assert!(dispatch_layout("matmul_f16", &[3, 127, batch]).is_err());
        }
        for p in [
            [3, 127, 4, 64, 2],
            [3, 256, 4, 256, 2],
            [3, 128, 2, 64, 2],
            [u32::MAX, 128, 4, 64, 4],
        ] {
            assert!(dispatch_layout("matmul_affine", &p).is_err());
        }
        assert!(dispatch_layout("matmul_f16", &[1, u32::MAX, 4]).is_err());
        assert_eq!(dispatch_layout("copy_f32", &[7])?, (vec![28, 28], 7));
        assert!(dispatch_layout("copy_f32", &[0]).is_err());
        Ok(())
    }

    #[test]
    fn bf16_dispatch_keeps_metadata_typed_and_rejects_unsupported_groups() -> Result<()> {
        let (sizes, threads) = dispatch_layout("matvec_affine_bf16", &[20, 5120, 4, 64])?;
        assert_eq!(sizes, vec![51200, 3200, 3200, 20480, 80]);
        assert_eq!(threads, 640);
        let (sizes, threads) = dispatch_layout("embed_affine_bf16", &[249999, 5120, 4, 64])?;
        assert_eq!(sizes, vec![640_000_000, 40_000_000, 40_000_000, 20480]);
        assert_eq!(threads, 5120);
        for name in ["matvec_affine_bf16", "embed_affine_bf16"] {
            assert!(dispatch_layout(name, &[2, 256, 4, 256]).is_err());
            assert!(dispatch_layout(name, &[2, 128, 4, 8]).is_err());
            assert!(dispatch_layout(name, &[2, 128, 2, 64]).is_err());
            assert!(dispatch_layout(name, &[2, 127, 4, 64]).is_err());
        }
        Ok(())
    }

    #[test]
    fn bf16_aligned_kernel_requires_q4_group64_and_safe_dimensions() {
        assert_eq!(
            specialized_matvec_bf16(&[20, 5120, 4, 64], true),
            Some("matvec_q4_g64_aligned_bf16")
        );
        for p in [[19, 5120, 4, 64], [20, 192, 4, 64], [524288, 8192, 4, 64]] {
            assert_eq!(
                specialized_matvec_bf16(&p, true),
                Some("matvec_q4_g64_bf16")
            );
        }
        for aligned in [true, false] {
            assert_eq!(
                specialized_matvec_bf16(&[20, 5120, 4, 32], aligned),
                Some("matvec_q4_g32_bf16")
            );
            assert_eq!(
                specialized_matvec_bf16(&[20, 5120, 4, 128], aligned),
                Some("matvec_q4_g128_bf16")
            );
            assert_eq!(
                specialized_matvec_bf16(&[20, 5120, 8, 64], aligned),
                Some("matvec_q8_g64_bf16")
            );
            assert_eq!(specialized_matvec_bf16(&[20, 5120, 4, 256], aligned), None);
        }
        assert_eq!(
            specialized_matvec_bf16(&[20, 5120, 4, 64], false),
            Some("matvec_q4_g64_bf16")
        );
    }

    #[test]
    fn aligned_dispatch_requires_safe_dimensions() {
        assert_eq!(
            specialized_matvec(&[20, 5120, 4, 64], true),
            Some("matvec_q4_g64_aligned")
        );
        for p in [[19, 5120, 4, 64], [20, 192, 4, 64], [524288, 8192, 4, 64]] {
            assert_eq!(specialized_matvec(&p, true), Some("matvec_q4_g64"));
        }
        assert_eq!(specialized_matvec(&[20, 5120, 4, 256], true), None);
        assert_eq!(
            specialized_matvec(&[20, 5120, 8, 64], true),
            Some("matvec_q8_g64")
        );
        assert_eq!(
            specialized_matvec(&[20, 5120, 4, 64], false),
            Some("matvec_q4_g64")
        );
    }
    #[test]
    #[ignore = "requires a real Metal GPU; explicitly run with --ignored"]
    fn affine_q4_little_nibbles_and_bias() -> anyhow::Result<()> {
        let gpu = Gpu::new()?;
        let w = gpu.upload_bytes(bytemuck::cast_slice(&[0x76543210u32, 0xfedcba98]))?;
        let scale = gpu.upload_f32(&[2., 0.5])?;
        let bias = gpu.upload_f32(&[-1., -4.])?;
        let x = gpu.upload_f32(&[1.; 8])?;
        let y = gpu.alloc_f32(2)?;
        let cmd = gpu.begin();
        let e = cmd.new_compute_command_encoder();
        gpu.encode(
            e,
            "matvec_affine",
            &[&w, &scale, &bias, &x, &y],
            &[2, 8, 4, 8],
            64,
            128,
        )?;
        e.end_encoding();
        gpu.finish(cmd)?;
        assert_eq!(gpu.read_f32(&y, 2)?, vec![48., 14.]);
        Ok(())
    }
    fn near(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - e).abs() < 3e-5 * e.abs().max(1.),
                "index {i}: actual {a}, expected {e}"
            );
        }
    }
    fn dispatch(gpu: &Gpu, name: &str, buffers: &[&BufferRef], p: &[u32]) -> Result<()> {
        let cmd = gpu.begin();
        let e = cmd.new_compute_command_encoder();
        let (_, threads) = dispatch_layout(name, p)?;
        gpu.encode(e, name, buffers, p, threads, 128)?;
        e.end_encoding();
        gpu.finish(cmd)
    }
    #[test]
    fn rejects_invalid_shapes_and_index_overflow_before_dispatch() {
        assert!(dispatch_layout("matvec_affine", &[2, 7, 4, 8]).is_err());
        assert!(dispatch_layout("matvec_affine", &[2, 8, 3, 8]).is_err());
        assert!(dispatch_layout("matvec_f16", &[u32::MAX, 2]).is_err());
        assert!(dispatch_layout("delta_step", &[3, 4, 2, 2]).is_err());
        assert!(dispatch_layout("attn_scores", &[3, 2, 2, 2]).is_err());
        assert!(dispatch_layout("rope", &[1, 4, 6, 0, 10000f32.to_bits()]).is_err());
        assert!(dispatch_layout("head_rms", &[1, 2, f32::NAN.to_bits()]).is_err());
        assert!(dispatch_layout("conv_silu", &[2, 0]).is_err());
        assert!(dispatch_layout("typo", &[2]).is_err());
    }
    #[test]
    fn packed_large_vocabulary_and_cache_bounds_use_checked_sizes() -> Result<()> {
        // A 250k x 5120 packed embedding is valid; bit count exceeds u32.
        let (sizes, threads) = dispatch_layout("embed_affine", &[249999, 5120, 4, 64])?;
        assert_eq!(sizes, vec![640_000_000, 80_000_000, 80_000_000, 20480]);
        assert_eq!(threads, 5120);
        let (sizes, _) = dispatch_layout("kv_append", &[2, 4, 9])?;
        assert_eq!(sizes, vec![32, 32, 320, 320]);
        Ok(())
    }
    #[test]
    #[ignore = "requires a real Metal GPU; explicitly run with --ignored"]
    fn dense_and_q8_matvec_and_embedding() -> Result<()> {
        let gpu = Gpu::new()?;
        let dense: Vec<u16> = [1., 2., 3., 4., -1., 0., 1., 2.]
            .into_iter()
            .map(|v| half::f16::from_f32(v).to_bits())
            .collect();
        let w = gpu.upload_bytes(bytemuck::cast_slice(&dense))?;
        let x = gpu.upload_f32(&[1., 2., 3., 4.])?;
        let y = gpu.alloc_f32(2)?;
        dispatch(&gpu, "matvec_f16", &[&w, &x, &y], &[2, 4])?;
        near(&gpu.read_f32(&y, 2)?, &[30., 10.]);
        let emb = gpu.alloc_f32(4)?;
        dispatch(&gpu, "embed_f16", &[&w, &emb], &[1, 4])?;
        near(&gpu.read_f32(&emb, 4)?, &[-1., 0., 1., 2.]);
        let q = gpu.upload_bytes(bytemuck::cast_slice(&[0xff800100u32, 0x04030201]))?;
        let scale = gpu.upload_f32(&[0.5, 2.])?;
        let bias = gpu.upload_f32(&[-1., -3.])?;
        dispatch(
            &gpu,
            "matvec_affine",
            &[&q, &scale, &bias, &x, &y],
            &[2, 4, 8, 4],
        )?;
        near(&gpu.read_f32(&y, 2)?, &[693., 30.]);
        dispatch(
            &gpu,
            "embed_affine",
            &[&q, &scale, &bias, &emb],
            &[0, 4, 8, 4],
        )?;
        near(&gpu.read_f32(&emb, 4)?, &[-1., -0.5, 63., 126.5]);
        Ok(())
    }
    #[test]
    #[ignore = "requires a real Metal GPU; explicitly run with --ignored"]
    fn conv_state_shift_and_delta_update_before_readout_across_tokens() -> Result<()> {
        let gpu = Gpu::new()?;
        let w = gpu.upload_f32(&[1., 2., 3., 4., 5., 6.])?;
        let state = gpu.alloc_f32(4)?;
        let y = gpu.alloc_f32(2)?;
        let x = gpu.upload_f32(&[1., -1.])?;
        dispatch(&gpu, "conv_silu", &[&x, &w, &state, &y], &[2, 3])?;
        let silu = |v: f32| v / (1. + (-v).exp());
        near(&gpu.read_f32(&y, 2)?, &[silu(3.), silu(-6.)]);
        let x = gpu.upload_f32(&[2., 3.])?;
        dispatch(&gpu, "conv_silu", &[&x, &w, &state, &y], &[2, 3])?;
        near(&gpu.read_f32(&y, 2)?, &[silu(8.), silu(13.)]);
        near(&gpu.read_f32(&state, 4)?, &[1., 2., -1., 3.]);
        let state = gpu.alloc_f32(4)?;
        let zeros = gpu.upload_f32(&[0.])?;
        // g=exp(-softplus(0))=1/2, beta=sigmoid(0)=1/2.
        let qkv = gpu.upload_f32(&[1., 0., 1., 0., 2., 4.])?;
        dispatch(
            &gpu,
            "delta_step",
            &[&qkv, &zeros, &zeros, &zeros, &zeros, &state, &y],
            &[1, 1, 2, 2],
        )?;
        near(&gpu.read_f32(&y, 2)?, &[1., 2.]);
        near(&gpu.read_f32(&state, 4)?, &[1., 0., 2., 0.]);
        let qkv = gpu.upload_f32(&[0., 1., 0., 1., 6., 8.])?;
        dispatch(
            &gpu,
            "delta_step",
            &[&qkv, &zeros, &zeros, &zeros, &zeros, &state, &y],
            &[1, 1, 2, 2],
        )?;
        near(&gpu.read_f32(&y, 2)?, &[3., 4.]);
        near(&gpu.read_f32(&state, 4)?, &[0.5, 3., 1., 4.]);
        // Nonzero prediction verifies subtraction uses decayed state, then readout.
        let qkv = gpu.upload_f32(&[1., 1., 1., 0., 0., 0.])?;
        dispatch(
            &gpu,
            "delta_step",
            &[&qkv, &zeros, &zeros, &zeros, &zeros, &state, &y],
            &[1, 1, 2, 2],
        )?;
        near(&gpu.read_f32(&y, 2)?, &[1.625, 2.25]);
        near(&gpu.read_f32(&state, 4)?, &[0.125, 1.5, 0.25, 2.]);
        Ok(())
    }
    #[test]
    #[ignore = "requires a real Metal GPU; explicitly run with --ignored"]
    fn norms_rope_and_elementwise_agree_with_scalar_fixtures() -> Result<()> {
        let gpu = Gpu::new()?;
        let eps = 1e-6f32;
        let x = gpu.upload_f32(&[3., 4.])?;
        let w = gpu.upload_f32(&[2., 1.])?;
        let y = gpu.alloc_f32(2)?;
        dispatch(&gpu, "rms_norm", &[&x, &w, &y], &[2, eps.to_bits()])?;
        let denom = (12.5 + eps).sqrt();
        near(&gpu.read_f32(&y, 2)?, &[6. / denom, 4. / denom]);
        dispatch(&gpu, "head_rms", &[&x, &w], &[1, 2, eps.to_bits()])?;
        near(&gpu.read_f32(&x, 2)?, &[6. / denom, 4. / denom]);
        let qkv = gpu.upload_f32(&[3., 4., 0., 5., 99.])?;
        dispatch(&gpu, "delta_norm", &[&qkv], &[1, 2, eps.to_bits()])?;
        let d = (25. + eps).sqrt();
        near(
            &gpu.read_f32(&qkv, 5)?,
            &[3. / d / 2f32.sqrt(), 4. / d / 2f32.sqrt(), 0., 5. / d, 99.],
        );
        let x = gpu.upload_f32(&[1., 2., 3., 4., 5., 6.])?;
        dispatch(&gpu, "rope", &[&x], &[1, 6, 4, 1, 1f32.to_bits()])?;
        let (s, c) = 1f32.sin_cos();
        near(
            &gpu.read_f32(&x, 6)?,
            &[
                c - 3. * s,
                2. * c - 4. * s,
                s + 3. * c,
                2. * s + 4. * c,
                5.,
                6.,
            ],
        );
        let x = gpu.upload_f32(&[3., 4.])?;
        let z = gpu.upload_f32(&[0., 1.])?;
        dispatch(&gpu, "gated_rms", &[&x, &z, &w, &y], &[1, 2, eps.to_bits()])?;
        near(
            &gpu.read_f32(&y, 2)?,
            &[0., 4. / denom / (1. + (-1f32).exp())],
        );
        dispatch(&gpu, "swiglu", &[&z, &x, &y], &[2])?;
        near(&gpu.read_f32(&y, 2)?, &[0., 4. / (1. + (-1f32).exp())]);
        dispatch(&gpu, "add", &[&z, &x, &y], &[2])?;
        near(&gpu.read_f32(&y, 2)?, &[3., 5.]);
        Ok(())
    }
    #[test]
    #[ignore = "requires a real Metal GPU; explicitly run with --ignored"]
    fn gqa_cache_and_sigmoid_gate_in_one_command_buffer() -> Result<()> {
        let gpu = Gpu::new()?;
        let projection = gpu.upload_f32(&[
            0.,
            0.,
            0.,
            3f32.ln(),
            0.,
            0.,
            0.,
            3f32.ln(),
            0.,
            0.,
            0.,
            3f32.ln(),
            0.,
            0.,
            0.,
            3f32.ln(),
        ])?;
        let q = gpu.alloc_f32(8)?;
        let gate = gpu.alloc_f32(8)?;
        let k = gpu.upload_f32(&[1., 2., 3., 4.])?;
        let v0 = gpu.upload_f32(&[2., 4., 10., 20.])?;
        let v1 = gpu.upload_f32(&[6., 8., 30., 40.])?;
        let kc = gpu.alloc_f32(8)?;
        let vc = gpu.alloc_f32(8)?;
        let scores = gpu.alloc_f32(8)?;
        let y = gpu.alloc_f32(8)?;
        let cmd = gpu.begin();
        let e = cmd.new_compute_command_encoder();
        gpu.encode(
            e,
            "split_q_gate",
            &[&projection, &q, &gate],
            &[4, 2],
            8,
            128,
        )?;
        gpu.encode(e, "kv_append", &[&k, &v0, &kc, &vc], &[2, 2, 0], 4, 128)?;
        gpu.encode(e, "kv_append", &[&k, &v1, &kc, &vc], &[2, 2, 1], 4, 128)?;
        gpu.encode(
            e,
            "attn_scores",
            &[&q, &kc, &scores],
            &[4, 2, 2, 2],
            256,
            128,
        )?;
        gpu.encode(e, "softmax", &[&scores], &[4, 2], 128, 128)?;
        gpu.encode(
            e,
            "attn_values",
            &[&scores, &vc, &gate, &y],
            &[4, 2, 2, 2],
            8,
            128,
        )?;
        e.end_encoding();
        gpu.finish(cmd)?;
        near(
            &gpu.read_f32(&y, 8)?,
            &[2., 4.5, 2., 4.5, 10., 22.5, 10., 22.5],
        );
        near(&gpu.read_f32(&scores, 8)?, &[0.5; 8]);
        Ok(())
    }
    #[test]
    #[ignore = "requires a real Metal GPU; explicitly run with --ignored"]
    fn odd_width_reductions_and_nonuniform_long_attention() -> Result<()> {
        let gpu = Gpu::new()?;
        // More than a SIMD width, and five rows: exercise loop tail and padded groups.
        let raw: Vec<f32> = (0..5 * 37).map(|i| ((i % 13) as f32 - 6.) / 4.).collect();
        let half: Vec<u16> = raw
            .iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect();
        let input: Vec<f32> = (0..37).map(|i| ((i % 7) as f32 - 3.) / 8.).collect();
        let expected: Vec<f32> = raw
            .chunks_exact(37)
            .map(|row| row.iter().zip(&input).map(|(a, b)| a * b).sum())
            .collect();
        let w = gpu.upload_bytes(bytemuck::cast_slice(&half))?;
        let x = gpu.upload_f32(&input)?;
        let y = gpu.alloc_f32(5)?;
        dispatch(&gpu, "matvec_f16", &[&w, &x, &y], &[5, 37])?;
        near(&gpu.read_f32(&y, 5)?, &expected);
        // 35 causal keys span two SIMD chunks; query heads share one KV head.
        let query = [1., -1., 0.5, -0.5, 2., 1.];
        let keys: Vec<f32> = (0..35 * 3).map(|i| (i % 17) as f32 / 7. - 1.).collect();
        let values: Vec<f32> = (0..35 * 3).map(|i| (i % 11) as f32 - 5.).collect();
        let q = gpu.upload_f32(&query)?;
        let k = gpu.upload_f32(&keys)?;
        let v = gpu.upload_f32(&values)?;
        let scores = gpu.alloc_f32(70)?;
        let gate = gpu.upload_f32(&[0.; 6])?;
        let y = gpu.alloc_f32(6)?;
        let cmd = gpu.begin();
        let e = cmd.new_compute_command_encoder();
        gpu.encode(
            e,
            "attn_scores",
            &[&q, &k, &scores],
            &[2, 1, 3, 35],
            2 * 35 * 32,
            128,
        )?;
        gpu.encode(e, "softmax", &[&scores], &[2, 35], 64, 128)?;
        gpu.encode(
            e,
            "attn_values",
            &[&scores, &v, &gate, &y],
            &[2, 1, 3, 35],
            6,
            128,
        )?;
        e.end_encoding();
        gpu.finish(cmd)?;
        let mut expected = vec![];
        for head in query.chunks_exact(3) {
            let logits: Vec<f64> = keys
                .chunks_exact(3)
                .map(|key| {
                    head.iter()
                        .zip(key)
                        .map(|(&q, &k)| q as f64 * k as f64)
                        .sum::<f64>()
                        / 3f64.sqrt()
                })
                .collect();
            let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = logits.iter().map(|x| (x - max).exp()).collect();
            let sum: f64 = exps.iter().sum();
            for d in 0..3 {
                expected.push(
                    (exps
                        .iter()
                        .enumerate()
                        .map(|(t, p)| p / sum * values[t * 3 + d] as f64)
                        .sum::<f64>()
                        * 0.5) as f32,
                );
            }
        }
        near(&gpu.read_f32(&y, 6)?, &expected);
        Ok(())
    }
}
