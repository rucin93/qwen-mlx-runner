//! Independent FP64 expansion tests for real shared-weight matrix blocks.
use anyhow::Result;
use qwen_metal::gpu::Gpu;

fn assert_close(got: &[f32], want: &[f64], description: &str) {
    assert_eq!(got.len(), want.len());
    for (i, (&actual, &expected)) in got.iter().zip(want).enumerate() {
        assert!(
            (actual as f64 - expected).abs() <= 4e-4 + expected.abs() * 3e-5,
            "{description} element {i}: {actual} vs {expected}"
        );
    }
}

#[test]
#[ignore = "requires a real Apple Metal GPU"]
fn affine_blocks_share_weights_for_all_batches_formats_tails_and_offsets() -> Result<()> {
    let gpu = Gpu::new_with_variant(false, "aligned")?;
    for bits in [4u32, 8] {
        for group in [32usize, 64, 128] {
            for (rows, cols) in [(1usize, group), (3, 3 * group), (7, 512), (20, 5120)] {
                let pack = 32 / bits as usize;
                let weights: Vec<u32> = (0..rows * cols / pack)
                    .map(|i| (i as u32).wrapping_mul(0x9e3779b9).wrapping_add(0x1478abcf))
                    .collect();
                let scales: Vec<f32> = (0..rows * cols / group)
                    .map(|i| (1 + i % 11) as f32 * 0.001953125)
                    .collect();
                let biases: Vec<f32> = (0..scales.len())
                    .map(|i| (i % 9) as f32 * 0.00390625 - 0.0625)
                    .collect();
                let x: Vec<f32> = (0..4 * cols)
                    .map(|i| ((i * 13 % 101) as f32 - 50.) / 50.)
                    .collect();
                let expected: Vec<f64> = (0..4 * rows)
                    .map(|i| {
                        let (token, row) = (i / rows, i % rows);
                        (0..cols)
                            .map(|c| {
                                let q = (weights[row * (cols / pack) + c / pack]
                                    >> ((c % pack) * bits as usize))
                                    & ((1 << bits) - 1);
                                let gi = row * (cols / group) + c / group;
                                (q as f64 * scales[gi] as f64 + biases[gi] as f64)
                                    * x[token * cols + c] as f64
                            })
                            .sum()
                    })
                    .collect();
                for bf16 in [false, true] {
                    // All offsets are nonzero; X starts at 4-byte but not float4 alignment.
                    let mut wb = vec![0u8; 4];
                    wb.extend(bytemuck::cast_slice(&weights));
                    let metadata = |v: &[f32]| -> Vec<u8> {
                        let mut bytes = vec![0u8; 4];
                        if bf16 {
                            let packed: Vec<u16> =
                                v.iter().map(|x| (x.to_bits() >> 16) as u16).collect();
                            bytes.extend(bytemuck::cast_slice(&packed));
                        } else {
                            bytes.extend(bytemuck::cast_slice(v));
                        }
                        bytes
                    };
                    let w = gpu.upload_bytes(&wb)?;
                    let s = gpu.upload_bytes(&metadata(&scales))?;
                    let b = gpu.upload_bytes(&metadata(&biases))?;
                    let mut xv = vec![99.];
                    xv.extend(&x);
                    xv.push(99.);
                    let xb = gpu.upload_f32(&xv)?;
                    for batch in 1..=4 {
                        let yb = gpu.upload_f32(&vec![12345.; batch * rows + 2])?;
                        let command = gpu.begin();
                        let encoder = gpu.begin_encoding(command);
                        let name = if bf16 {
                            "matmul_affine_bf16"
                        } else {
                            "matmul_affine"
                        };
                        encoder.encode_offsets(
                            name,
                            &[&w, &s, &b, &xb, &yb],
                            &[4; 5],
                            &[rows as u32, cols as u32, bits, group as u32, batch as u32],
                            rows * 32,
                            128,
                        )?;
                        encoder.end_encoding()?;
                        gpu.finish(command)?;
                        let got = gpu.read_f32(&yb, batch * rows + 2)?;
                        assert_eq!(got[0], 12345.);
                        assert_eq!(got[batch * rows + 1], 12345.);
                        assert_close(
                            &got[1..=batch * rows],
                            &expected[..batch * rows],
                            &format!("{name} B{batch} Q{bits} g{group} {rows}x{cols}"),
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires a real Apple Metal GPU"]
fn dense_blocks_and_offset_copies_keep_guard_values() -> Result<()> {
    let gpu = Gpu::new_with_reference(false)?;
    for rows in [1usize, 3, 7, 20] {
        for cols in [1usize, 31, 127, 512] {
            let weights: Vec<u16> = (0..rows * cols)
                .map(|i| half::f16::from_f32(((i * 7 % 41) as f32 - 20.) / 32.).to_bits())
                .collect();
            let x: Vec<f32> = (0..4 * cols)
                .map(|i| ((i * 13 % 31) as f32 - 15.) / 16.)
                .collect();
            let expected: Vec<f64> = (0..4 * rows)
                .map(|i| {
                    (0..cols)
                        .map(|c| {
                            half::f16::from_bits(weights[(i % rows) * cols + c]).to_f64()
                                * x[(i / rows) * cols + c] as f64
                        })
                        .sum()
                })
                .collect();
            let mut wv = vec![0u8; 4];
            wv.extend(bytemuck::cast_slice(&weights));
            let w = gpu.upload_bytes(&wv)?;
            let mut xv = vec![99.];
            xv.extend(x);
            let xb = gpu.upload_f32(&xv)?;
            for batch in 1..=4 {
                let y = gpu.upload_f32(&vec![12345.; batch * rows + 2])?;
                let copy = gpu.upload_f32(&vec![54321.; batch * rows + 2])?;
                let command = gpu.begin();
                let e = command.new_compute_command_encoder();
                gpu.encode_offsets(
                    e,
                    "matmul_f16",
                    &[&w, &xb, &y],
                    &[4; 3],
                    &[rows as u32, cols as u32, batch as u32],
                    rows * 32,
                    128,
                )?;
                gpu.encode_offsets(
                    e,
                    "copy_f32",
                    &[&y, &copy],
                    &[4, 4],
                    &[(batch * rows) as u32],
                    batch * rows,
                    128,
                )?;
                e.end_encoding();
                gpu.finish(command)?;
                let got = gpu.read_f32(&copy, batch * rows + 2)?;
                assert_eq!(got[0], 54321.);
                assert_eq!(got[batch * rows + 1], 54321.);
                assert_close(
                    &got[1..=batch * rows],
                    &expected[..batch * rows],
                    "dense block copied",
                );
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires a real Apple Metal GPU"]
fn offset_dispatch_rejects_bounds_and_unsafe_aliases_and_preserves_scalar_fallback() -> Result<()> {
    let mut gpu = Gpu::new_with_reference(false)?;
    gpu.set_parallel_norm(true);
    let input = gpu.upload_f32(&vec![2.; 1028])?;
    let output = gpu.alloc_f32(1028)?;
    let cmd = gpu.begin();
    let e = gpu.begin_encoding(cmd);
    for offsets in [vec![], vec![0], vec![2, 0], vec![0, usize::MAX], vec![4, 4]] {
        assert!(
            e.encode_offsets("copy_f32", &[&input, &output], &offsets, &[1028], 1028, 128)
                .is_err()
        );
    }
    assert!(
        e.encode_offsets("copy_f32", &[&input, &input], &[0, 4], &[32], 32, 128)
            .is_err()
    );
    e.encode_offsets(
        "rms_norm",
        &[&input, &input, &output],
        &[4, 4, 4],
        &[1024, 1e-6f32.to_bits()],
        32,
        128,
    )?;
    let w = gpu.upload_bytes(&vec![0u8; 512 * 4 / 2])?;
    let s = gpu.upload_f32(&vec![1.; 32])?;
    let b = gpu.upload_f32(&vec![1.; 32])?;
    e.encode_offsets(
        "matvec_affine",
        &[&w, &s, &b, &input, &output],
        &[0, 0, 0, 4, 4],
        &[4, 512, 4, 64],
        128,
        128,
    )?;
    let s16 = gpu.upload_bytes(&vec![0u8; 64])?;
    assert!(
        e.encode_offsets(
            "matvec_affine_bf16",
            &[&w, &s16, &s16, &input, &output],
            &[0, 0, 0, 4, 4],
            &[4, 512, 4, 64],
            128,
            128
        )
        .is_err()
    );
    e.end_encoding()?;
    gpu.finish(cmd)?;
    let got = gpu.read_f32(&output, 1028)?;
    assert_eq!(got[0], 0.);
    assert_eq!(got[1027], 0.);
    assert_close(&got[1..5], &[1024.; 4], "scalar offset matvec");
    assert_close(&got[5..1025], &vec![2.; 1020], "scalar offset RMS");
    Ok(())
}

#[test]
#[ignore = "requires a real Apple Metal GPU"]
fn aligned_q4_blocks_match_independent_oracle_and_single_token_path() -> Result<()> {
    let mut gpu = Gpu::new_with_variant(false, "aligned")?;
    let mut legacy_outputs: Vec<Vec<f32>> = Vec::new();
    for matmul in [
        qwen_metal::gpu::BlockMatmulMode::Legacy,
        qwen_metal::gpu::BlockMatmulMode::Shared,
        qwen_metal::gpu::BlockMatmulMode::MlpR2,
    ] {
        gpu.set_block_kernel_mode(qwen_metal::gpu::BlockKernelMode {
            matmul,
            batched_delta: false,
        });
        let mut case_index = 0;
        for cols in [512usize, 5120, 17408] {
            let rows = 20;
            let weights: Vec<u32> = (0..rows * cols / 8)
                .map(|i| (i as u32).wrapping_mul(0x1478abcf).wrapping_add(0x9e3779b9))
                .collect();
            let scales: Vec<f32> = (0..rows * cols / 64)
                .map(|i| (1 + i % 11) as f32 * 0.001953125)
                .collect();
            let biases: Vec<f32> = (0..scales.len())
                .map(|i| (i % 9) as f32 * 0.00390625 - 0.0625)
                .collect();
            let x: Vec<f32> = (0..4 * cols)
                .map(|i| ((i * 13 % 101) as f32 - 50.) / 50.)
                .collect();
            let expected: Vec<f64> = (0..4 * rows)
                .map(|i| {
                    (0..cols)
                        .map(|c| {
                            let q = (weights[(i % rows) * cols / 8 + c / 8] >> ((c % 8) * 4)) & 15;
                            let gi = (i % rows) * cols / 64 + c / 64;
                            (q as f64 * scales[gi] as f64 + biases[gi] as f64)
                                * x[(i / rows) * cols + c] as f64
                        })
                        .sum()
                })
                .collect();
            let wb = gpu.upload_bytes(bytemuck::cast_slice(&weights))?;
            let xb = gpu.upload_f32(&x)?;
            for bf16 in [false, true] {
                let upload = |v: &[f32]| {
                    if bf16 {
                        let values: Vec<u16> =
                            v.iter().map(|x| (x.to_bits() >> 16) as u16).collect();
                        gpu.upload_bytes(bytemuck::cast_slice(&values))
                    } else {
                        gpu.upload_f32(v)
                    }
                };
                let sb = upload(&scales)?;
                let bb = upload(&biases)?;
                for batch in 1..=4 {
                    let y = gpu.alloc_f32(rows * batch)?;
                    let sequential = gpu.alloc_f32(rows * batch)?;
                    let cmd = gpu.begin();
                    let encoder = gpu.begin_encoding(cmd);
                    let matmul_name = if bf16 {
                        "matmul_affine_bf16"
                    } else {
                        "matmul_affine"
                    };
                    let matvec = if bf16 {
                        "matvec_affine_bf16"
                    } else {
                        "matvec_affine"
                    };
                    encoder.encode(
                        matmul_name,
                        &[&wb, &sb, &bb, &xb, &y],
                        &[rows as u32, cols as u32, 4, 64, batch as u32],
                        rows * 32,
                        128,
                    )?;
                    for b in 0..batch {
                        encoder.encode_offsets(
                            matvec,
                            &[&wb, &sb, &bb, &xb, &sequential],
                            &[0, 0, 0, b * cols * 4, b * rows * 4],
                            &[rows as u32, cols as u32, 4, 64],
                            rows * 32,
                            128,
                        )?;
                    }
                    encoder.end_encoding()?;
                    gpu.finish(cmd)?;
                    let actual = gpu.read_f32(&y, rows * batch)?;
                    assert_close(
                        &actual,
                        &expected[..rows * batch],
                        "aligned batch independent oracle",
                    );
                    if matmul != qwen_metal::gpu::BlockMatmulMode::Legacy {
                        assert_eq!(
                            actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                            legacy_outputs[case_index]
                                .iter()
                                .map(|x| x.to_bits())
                                .collect::<Vec<_>>(),
                            "{matmul:?} schedule must preserve every bit B{batch} cols{cols} BF16={bf16}"
                        );
                    } else {
                        legacy_outputs.push(actual.clone());
                    }
                    case_index += 1;
                    let scalar = gpu.read_f32(&sequential, rows * batch)?;
                    for (a, b) in actual.iter().zip(scalar) {
                        assert!(
                            (a - b).abs() <= 1e-5 + b.abs() * 3e-6,
                            "aligned Q4 B{batch} cols{cols} BF16={bf16}: block {a}, sequential {b}"
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires a real Apple Metal GPU"]
fn stream_offset_weights_fall_back_to_packed_alignment() -> Result<()> {
    let gpu = Gpu::new_with_variant(false, "stream")?;
    let mut values = vec![0u32];
    values.extend(vec![0x76543210u32; 4 * 512 / 8]);
    let w = gpu.upload_bytes(bytemuck::cast_slice(&values))?;
    let scales = gpu.upload_f32(&vec![1.; 32])?;
    let biases = gpu.upload_f32(&vec![0.; 32])?;
    let x = gpu.upload_f32(&vec![1.; 512])?;
    let y = gpu.alloc_f32(4)?;
    let command = gpu.begin();
    let encoder = gpu.begin_encoding(command);
    encoder.encode_offsets(
        "matvec_affine",
        &[&w, &scales, &biases, &x, &y],
        &[4, 0, 0, 0, 0],
        &[4, 512, 4, 64],
        128,
        128,
    )?;
    encoder.end_encoding()?;
    gpu.finish(command)?;
    assert_eq!(gpu.read_f32(&y, 4)?, vec![1792.; 4]);
    Ok(())
}

// Frozen ef38979 R4 schedule: independent of the production template and route.
// Original full matmul_block.metal SHA256:
// 67e593a242ec5aa3fc5a390e28cb4c30da305184d4bbc2b002f8357e07e8abee
const FROZEN_R4_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

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

#[test]
#[ignore = "requires a real Apple Metal GPU"]
fn selective_mlp_r2_real_shapes_match_frozen_r4_bits_and_fp64_with_guards() -> Result<()> {
    use anyhow::anyhow;
    use qwen_metal::gpu::{BlockKernelMode, BlockMatmulMode};
    let mut gpu = Gpu::new_with_variant(false, "aligned")?;
    let options = metal::CompileOptions::new();
    options.set_fast_math_enabled(true);
    let library = gpu
        .device
        .new_library_with_source(FROZEN_R4_SOURCE, &options)
        .map_err(|e| anyhow!("frozen oracle compilation: {e}"))?;
    let function = library
        .get_function("baseline_q4_b3_bf16", None)
        .map_err(|e| anyhow!("frozen oracle function: {e}"))?;
    let descriptor = metal::ComputePipelineDescriptor::new();
    descriptor.set_compute_function(Some(&function));
    descriptor.set_thread_group_size_is_multiple_of_thread_execution_width(true);
    descriptor.set_max_total_threads_per_threadgroup(64);
    let pipeline = gpu
        .device
        .new_compute_pipeline_state(&descriptor)
        .map_err(|e| anyhow!("frozen oracle pipeline: {e}"))?;
    assert_eq!(pipeline.thread_execution_width(), 32);
    // Only one full matrix is resident per iteration (~50 MiB packed+metadata).
    // Both actual routed MLP shapes are necessary: down has a much longer dot.
    for (rows, cols) in [(17408usize, 5120usize), (5120, 17408)] {
        objc::rc::autoreleasepool(|| -> Result<()> {
            let weights: Vec<u32> = (0..rows * cols / 8)
                .map(|i| (i as u32).wrapping_mul(0x1478abcf).wrapping_add(0x9e3779b9))
                .collect();
            let scales: Vec<f32> = (0..rows * cols / 64)
                .map(|i| (1 + i % 11) as f32 * 0.001953125)
                .collect();
            let biases: Vec<f32> = (0..scales.len())
                .map(|i| (i % 9) as f32 * 0.00390625 - 0.0625)
                .collect();
            let x: Vec<f32> = (0..3 * cols)
                .map(|i| ((i * 13 % 101) as f32 - 50.) / 50.)
                .collect();
            let guarded = |bytes: &[u8]| -> Result<metal::Buffer> {
                let mut data = vec![0xa5u8; 16];
                data.extend_from_slice(bytes);
                data.extend_from_slice(&[0xa5u8; 16]);
                gpu.upload_bytes(&data)
            };
            let upload_metadata = |values: &[f32]| -> Result<metal::Buffer> {
                let packed: Vec<u16> = values
                    .iter()
                    .map(|value| {
                        assert_eq!(
                            value.to_bits() & 0xffff,
                            0,
                            "fixture metadata is exact BF16"
                        );
                        (value.to_bits() >> 16) as u16
                    })
                    .collect();
                guarded(bytemuck::cast_slice(&packed))
            };
            let wb = guarded(bytemuck::cast_slice(&weights))?;
            let sb = upload_metadata(&scales)?;
            let bb = upload_metadata(&biases)?;
            let xb = guarded(bytemuck::cast_slice(&x))?;
            let outputs = (0..3)
                .map(|_| gpu.upload_f32(&vec![12345.; rows * 3 + 8]))
                .collect::<Result<Vec<_>>>()?;
            let params = [rows as u32, cols as u32, 4, 64, 3];
            let command = gpu.begin();
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            for (i, buffer) in [&wb, &sb, &bb, &xb, &outputs[0]].iter().enumerate() {
                encoder.set_buffer(i as u64, Some(buffer), 16);
            }
            encoder.set_bytes(
                15,
                std::mem::size_of_val(&params) as u64,
                params.as_ptr().cast(),
            );
            encoder.dispatch_thread_groups(
                metal::MTLSize::new((rows / 4 * 32 / 64) as u64, 1, 1),
                metal::MTLSize::new(64, 1, 1),
            );
            encoder.end_encoding();
            gpu.finish(command)?;
            for (index, matmul) in [BlockMatmulMode::Legacy, BlockMatmulMode::MlpR2]
                .into_iter()
                .enumerate()
            {
                gpu.set_block_kernel_mode(BlockKernelMode {
                    matmul,
                    batched_delta: true,
                });
                let command = gpu.begin();
                let encoder = gpu.begin_encoding(command);
                encoder.encode_offsets(
                    "matmul_affine_bf16",
                    &[&wb, &sb, &bb, &xb, &outputs[index + 1]],
                    &[16; 5],
                    &params,
                    rows * 32,
                    128,
                )?;
                encoder.end_encoding()?;
                gpu.finish(command)?;
            }
            let frozen = gpu.read_f32(&outputs[0], rows * 3 + 8)?;
            for (index, output) in outputs.iter().enumerate() {
                let actual = gpu.read_f32(output, rows * 3 + 8)?;
                assert!(
                    actual[..4]
                        .iter()
                        .chain(&actual[rows * 3 + 4..])
                        .all(|&v| v == 12345.),
                    "output guards changed {rows}x{cols} mode {index}"
                );
                for (i, (&got, &want)) in actual[4..rows * 3 + 4]
                    .iter()
                    .zip(&frozen[4..rows * 3 + 4])
                    .enumerate()
                {
                    assert!(got.is_finite());
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "frozen R4 mismatch {rows}x{cols} mode {index} output {i}"
                    );
                }
            }
            // Independent scalar dequantization/FP64 accumulation at both edges
            // and an evenly spaced sample of the complete matrix for every RHS.
            for batch in 0..3 {
                for sample in 0..32 {
                    let row = sample * (rows - 1) / 31;
                    let expected: f64 = (0..cols)
                        .map(|col| {
                            let q = (weights[row * cols / 8 + col / 8] >> (4 * (col % 8))) & 15;
                            let group = row * cols / 64 + col / 64;
                            (q as f64 * scales[group] as f64 + biases[group] as f64)
                                * x[batch * cols + col] as f64
                        })
                        .sum();
                    assert_close(
                        &[frozen[4 + batch * rows + row]],
                        &[expected],
                        "full MLP independent oracle",
                    );
                }
            }
            for buffer in [&wb, &sb, &bb, &xb] {
                // Safe after the completed command buffers; storage is shared.
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        buffer.contents().cast::<u8>(),
                        buffer.length() as usize,
                    )
                };
                assert!(
                    bytes[..16]
                        .iter()
                        .chain(&bytes[bytes.len() - 16..])
                        .all(|&b| b == 0xa5),
                    "input guard changed"
                );
            }
            let input = gpu.read_f32(&xb, x.len() + 8)?;
            assert_eq!(&input[4..x.len() + 4], x);
            Ok(())
        })?;
    }
    Ok(())
}
