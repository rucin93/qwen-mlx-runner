//! The oracle expands exactly the BF16 metadata stored in each GPU buffer.
use anyhow::Result;
use qwen_metal::gpu::Gpu;

#[test]
#[ignore = "requires a real Apple Metal GPU"]
fn bf16_metadata_matches_scalar_for_packing_groups_shapes_and_embeddings() -> Result<()> {
    for mode in ["aligned", "packed4", "stream"] {
        let gpu = Gpu::new_with_variant(false, mode)?;
        for bits in [4u32, 8] {
            for group in [32usize, 64, 128] {
                for cols in [group, 3 * group, 5120] {
                    for rows in [1usize, 3, 7, 19, 20] {
                        let pack = 32 / bits as usize;
                        let mut seed = 0x1478_abcfu32;
                        let weights: Vec<u32> = (0..rows * cols / pack)
                            .map(|_| {
                                seed ^= seed << 13;
                                seed ^= seed >> 17;
                                seed ^= seed << 5;
                                seed
                            })
                            .collect();
                        let scales: Vec<f32> = (0..rows * cols / group)
                            .map(|i| {
                                f32::from_bits(
                                    (0.001 * (1 + i % 11) as f32).to_bits() & 0xffff_0000,
                                )
                            })
                            .collect();
                        let biases: Vec<f32> = (0..scales.len())
                            .map(|i| {
                                f32::from_bits(
                                    (-0.07 + (i % 9) as f32 * 0.003).to_bits() & 0xffff_0000,
                                )
                            })
                            .collect();
                        let x: Vec<f32> = (0..cols)
                            .map(|i| ((i * 13 % 101) as f32 - 50.) / 50.)
                            .collect();
                        let value = |r: usize, c: usize| {
                            let q = (weights[r * (cols / pack) + c / pack]
                                >> ((c % pack) * bits as usize))
                                & ((1 << bits) - 1);
                            let gi = r * (cols / group) + c / group;
                            q as f64 * scales[gi] as f64 + biases[gi] as f64
                        };
                        let expected: Vec<f32> = (0..rows)
                            .map(|r| {
                                (0..cols).map(|c| value(r, c) * x[c] as f64).sum::<f64>() as f32
                            })
                            .collect();
                        let s16: Vec<u16> =
                            scales.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
                        let b16: Vec<u16> =
                            biases.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
                        let wb = gpu.upload_bytes(bytemuck::cast_slice(&weights))?;
                        let sb = gpu.upload_bytes(bytemuck::cast_slice(&s16))?;
                        let bb = gpu.upload_bytes(bytemuck::cast_slice(&b16))?;
                        let xb = gpu.upload_f32(&x)?;
                        let yb = gpu.alloc_f32(rows)?;
                        let eb = gpu.alloc_f32(cols)?;
                        let cmd = gpu.begin();
                        let e = cmd.new_compute_command_encoder();
                        gpu.encode(
                            e,
                            "matvec_affine_bf16",
                            &[&wb, &sb, &bb, &xb, &yb],
                            &[rows as u32, cols as u32, bits, group as u32],
                            rows * 32,
                            128,
                        )?;
                        gpu.encode(
                            e,
                            "embed_affine_bf16",
                            &[&wb, &sb, &bb, &eb],
                            &[(rows - 1) as u32, cols as u32, bits, group as u32],
                            cols,
                            128,
                        )?;
                        e.end_encoding();
                        gpu.finish(cmd)?;
                        for (got, want) in gpu.read_f32(&yb, rows)?.into_iter().zip(expected) {
                            assert!(
                                (got - want).abs() < 2e-4 + want.abs() * 2e-5,
                                "{mode}: Q{bits}, group{group}, {rows}x{cols}: {got} vs {want}"
                            );
                        }
                        for (c, got) in gpu.read_f32(&eb, cols)?.into_iter().enumerate() {
                            let want = value(rows - 1, c) as f32;
                            assert!((got - want).abs() < 2e-6 + want.abs() * 2e-6);
                        }
                    }
                }
            }
        }
    }
    Ok(())
}
