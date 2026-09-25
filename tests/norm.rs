use anyhow::Result;
use qwen_metal::gpu::Gpu;

#[test]
#[ignore = "requires a real Apple Metal GPU"]
fn parallel_rms_matches_fp64_for_large_odd_shapes_and_alias_fallback() -> Result<()> {
    let mut gpu = Gpu::new_with_reference(false)?;
    gpu.set_parallel_norm(true);
    for n in [1usize, 3, 32, 257, 1024, 1025, 1733, 5120, 17408] {
        for zero in [false, true] {
            let x: Vec<f32> = (0..n)
                .map(|i| {
                    if zero {
                        0.
                    } else {
                        ((i * 17 % 127) as f32 - 63.) / 31.
                    }
                })
                .collect();
            let w: Vec<f32> = (0..n).map(|i| 0.2 + (i % 13) as f32 * 0.1).collect();
            let eps = 1e-6f32;
            let inv = 1.
                / (x.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / n as f64 + eps as f64)
                    .sqrt();
            let expected: Vec<f32> = x
                .iter()
                .zip(&w)
                .map(|(&x, &w)| (x as f64 * inv * w as f64) as f32)
                .collect();
            for inplace in [false, true] {
                let xb = gpu.upload_f32(&x)?;
                let wb = gpu.upload_f32(&w)?;
                let output = if inplace {
                    xb.clone()
                } else {
                    gpu.alloc_f32(n)?
                };
                let cmd = gpu.begin();
                let e = cmd.new_compute_command_encoder();
                gpu.encode(
                    e,
                    "rms_norm",
                    &[&xb, &wb, &output],
                    &[n as u32, eps.to_bits()],
                    32,
                    32,
                )?;
                e.end_encoding();
                gpu.finish(cmd)?;
                for (got, want) in gpu.read_f32(&output, n)?.into_iter().zip(&expected) {
                    assert!(
                        (got - want).abs() < 2e-5 + want.abs() * 2e-5,
                        "n={n}, inplace={inplace}, zero={zero}: {got} != {want}"
                    );
                }
                let timing = gpu
                    .last_frame_timing()
                    .expect("real GPU command timestamps");
                assert!(timing.gpu_seconds > 0. && timing.cpu_encode_seconds >= 0.);
            }
        }
    }
    Ok(())
}
