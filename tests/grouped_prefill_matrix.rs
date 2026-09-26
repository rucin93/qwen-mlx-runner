//! Synthetic-weight real-shape comparison of grouped existing matrix kernels.
use anyhow::Result;
use qwen_metal::gpu::Gpu;
#[test]
#[ignore = "requires a real Apple Metal GPU; bounded real-shape matrix microbenchmark"]
fn grouped_prefill_real_shape_microbenchmark() -> Result<()> {
    use std::time::Instant;
    let gpu = Gpu::new_with_variant(false, "aligned")?;
    for (rows, cols) in [
        (17408usize, 5120usize),
        (5120, 17408),
        (10240, 5120),
        (5120, 6144),
        (48, 5120),
    ] {
        let w: Vec<u32> = (0..rows * cols / 8)
            .map(|i| (i as u32).wrapping_mul(0x9e3779b9).wrapping_add(0x1478abcf))
            .collect();
        let scales: Vec<u16> = (0..rows * cols / 64)
            .map(|i| (((1 + i % 11) as f32 / 512.).to_bits() >> 16) as u16)
            .collect();
        let biases: Vec<u16> = (0..scales.len())
            .map(|i| (((i % 9) as f32 / 256. - 0.0625).to_bits() >> 16) as u16)
            .collect();
        let w = gpu.upload_bytes(bytemuck::cast_slice(&w))?;
        let s = gpu.upload_bytes(bytemuck::cast_slice(&scales))?;
        let b = gpu.upload_bytes(bytemuck::cast_slice(&biases))?;
        let x = gpu.upload_f32(
            &(0..48 * cols)
                .map(|i| ((i * 13 % 101) as f32 - 50.) / 50.)
                .collect::<Vec<_>>(),
        )?;
        let y = gpu.alloc_f32(48 * rows)?;
        let measure = |width: usize| -> Result<(f64, f64)> {
            objc::rc::autoreleasepool(|| {
                let started = Instant::now();
                let command = gpu.begin();
                let encoder = gpu.begin_encoding(command);
                for block in (0..48).step_by(width) {
                    let end = (block + width).min(48);
                    for token in (block..end).step_by(3) {
                        let narrow = (end - token).min(3);
                        encoder.encode_offsets(
                            "matmul_affine_bf16",
                            &[&w, &s, &b, &x, &y],
                            &[0, 0, 0, token * cols * 4, token * rows * 4],
                            &[rows as u32, cols as u32, 4, 64, narrow as u32],
                            rows * 32,
                            64,
                        )?;
                    }
                }
                encoder.end_encoding()?;
                gpu.finish(command)?;
                Ok((
                    gpu.last_frame_timing().unwrap().gpu_seconds,
                    started.elapsed().as_secs_f64(),
                ))
            })
        };
        measure(3)?;
        let expected = gpu.read_f32(&y, 48 * rows)?;
        for width in [8, 16] {
            measure(width)?;
            let actual = gpu.read_f32(&y, 48 * rows)?;
            assert!(
                actual
                    .iter()
                    .zip(&expected)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "real-shape grouped prompt B{width} diverged from Legacy B3"
            );
        }
        for _ in 0..4 {
            for width in [3, 8, 16] {
                measure(width)?;
            }
        }
        let mut timings: [Vec<(f64, f64)>; 3] = std::array::from_fn(|_| vec![]);
        for iteration in 0..20 {
            for index in 0..3 {
                let slot = (index + iteration) % 3;
                timings[slot].push(measure([3, 8, 16][slot])?);
            }
        }
        let median = |v: Vec<f64>| {
            let mut v = v;
            v.sort_by(f64::total_cmp);
            (v[9] + v[10]) / 2.
        };
        let modes:Vec<_>=timings.iter().enumerate().map(|(i,t)|serde_json::json!({"batch":([3,8,16][i]),"gpu_ms_per_48_inputs":median(t.iter().map(|t|t.0*1000.).collect()),"wall_ms_per_48_inputs":median(t.iter().map(|t|t.1*1000.).collect())})).collect();
        println!(
            "{}",
            serde_json::json!({"kind":"grouped_known_prompt_real_shape_matrix_microbenchmark","device":gpu.device.name(),"rows":rows,"cols":cols,"modes":modes,"iterations":20,"bitwise_equal":true,"note":"Synthetic weights and real matrix dimensions; one command per48inputs; existing B3 kernels grouped at prompt widths3/8/16; does not measure fullmodel throughput or M5 speedup."})
        );
    }
    Ok(())
}
