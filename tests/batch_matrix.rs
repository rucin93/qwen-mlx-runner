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
    let gpu = Gpu::new_with_variant(false, "aligned")?;
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
                    let values: Vec<u16> = v.iter().map(|x| (x.to_bits() >> 16) as u16).collect();
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
                let matmul = if bf16 {
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
                    matmul,
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
