//! Independent FP64 dot products catch unpack, group-boundary and SIMD row-tail bugs.
use anyhow::Result;
use qwen_metal::gpu::Gpu;

#[test]
#[ignore = "requires a real Metal GPU"]
fn specialized_q4_q8_match_scalar_across_groups_and_row_tails() -> Result<()> {
    let fast = Gpu::new_with_variant(false, "packed4")?;
    let aligned = Gpu::new_with_variant(false, "aligned")?;
    let reference = Gpu::new_with_reference(true)?;
    for bits in [4u32, 8] {
        for group in [32usize, 64, 128] {
            for cols in [group, 3 * group, 5120usize] {
                for rows in [1usize, 3, 7, 19, 20] {
                    let pack = 32 / bits as usize;
                    let mut seed = 0x8421_3957u32
                        ^ bits
                        ^ (group as u32).rotate_left(7)
                        ^ (cols as u32).rotate_left(13)
                        ^ (rows as u32).rotate_left(19);
                    let w: Vec<u32> = (0..rows * cols / pack)
                        .map(|_| {
                            seed ^= seed << 13;
                            seed ^= seed >> 17;
                            seed ^= seed << 5;
                            seed
                        })
                        .collect();
                    let scales: Vec<f32> = (0..rows * cols / group)
                        .map(|i| 0.001 * (1 + (i % 11)) as f32)
                        .collect();
                    let biases: Vec<f32> = (0..scales.len())
                        .map(|i| -0.07 + (i % 9) as f32 * 0.003)
                        .collect();
                    let x: Vec<f32> = (0..cols)
                        .map(|i| ((i * 13 % 101) as f32 - 50.) / 50.)
                        .collect();
                    let expect: Vec<f32> = (0..rows)
                        .map(|r| {
                            (0..cols)
                                .map(|c| {
                                    let q = (w[r * (cols / pack) + c / pack]
                                        >> ((c % pack) * bits as usize))
                                        & ((1 << bits) - 1);
                                    let gi = r * (cols / group) + c / group;
                                    (q as f64 * scales[gi] as f64 + biases[gi] as f64) * x[c] as f64
                                })
                                .sum::<f64>() as f32
                        })
                        .collect();
                    for gpu in [&fast, &aligned, &reference] {
                        let wb = gpu.upload_bytes(bytemuck::cast_slice(&w))?;
                        let sb = gpu.upload_f32(&scales)?;
                        let bb = gpu.upload_f32(&biases)?;
                        let xb = gpu.upload_f32(&x)?;
                        let yb = gpu.alloc_f32(rows)?;
                        let cmd = gpu.begin();
                        let enc = cmd.new_compute_command_encoder();
                        gpu.encode(
                            enc,
                            "matvec_affine",
                            &[&wb, &sb, &bb, &xb, &yb],
                            &[rows as u32, cols as u32, bits, group as u32],
                            rows * 32,
                            128,
                        )?;
                        enc.end_encoding();
                        gpu.finish(cmd)?;
                        for (r, (got, want)) in
                            gpu.read_f32(&yb, rows)?.iter().zip(&expect).enumerate()
                        {
                            assert!(
                                (got - want).abs() < 2e-4 + want.abs() * 2e-5,
                                "{} Q{bits}/g{group}, cols {cols}, rows {rows}, row {r}: {got} != {want}",
                                gpu.kernel_mode()
                            );
                        }
                    }
                }
            }
        }
    }
    Ok(())
}
