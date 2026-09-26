//! Snapshot-free prompt recurrence must retain all outputs and final state.
use anyhow::Result;
use qwen_metal::gpu::Gpu;

fn values(count: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut seed = seed;
    (0..count)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed >> 8) as f32 / 16777216. - 0.5) * scale
        })
        .collect()
}

fn close(actual: &[f32], expected: &[f64], label: &str) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        assert!(
            got.is_finite() && (got as f64 - want).abs() < 2e-6 + 2e-5 * want.abs(),
            "{label} {i}: {got} != {want}"
        );
    }
}

#[test]
#[ignore = "requires a real Apple Metal GPU"]
fn known_prompt_delta_matches_fp64_and_scalar_state_and_outputs() -> Result<()> {
    let gpu = Gpu::new_with_reference(false)?;
    // Wrong token strides, grouping, final-state writes, prefix zero, or masked
    // register tails each alter an independently computed state/output below.
    for (kh, vh, k, v) in [
        (1, 2, 1, 3),
        (2, 6, 31, 7),
        (1, 2, 32, 5),
        (2, 4, 33, 3),
        (1, 3, 127, 9),
        (16, 48, 128, 128),
    ] {
        for batch in [1, 3, 8, 16] {
            let rows = vh * v;
            let qwidth = 2 * kh * k + rows;
            let elements = rows * k;
            let qkv = values(batch * qwidth, 91, 0.25);
            let a = values(batch * vh, 18, 1.5);
            let b = values(batch * vh, 44, 2.);
            let alog = values(vh, 52, 0.8);
            let dt = values(vh, 17, 0.5);
            let initial = values(elements, 71, 0.16);
            let mut expected: Vec<f64> = initial.iter().map(|&x| x as f64).collect();
            let mut output = vec![0.; batch * rows];
            for token in 0..batch {
                for row in 0..rows {
                    let h = row / v;
                    let key_head = h / (vh / kh);
                    let qo = token * qwidth + key_head * k;
                    let ko = qo + kh * k;
                    let decay = (-f64::from(alog[h]).exp()
                        * (f64::from(a[token * vh + h]) + f64::from(dt[h]))
                            .exp()
                            .ln_1p())
                    .exp();
                    let beta = 1. / (1. + (-f64::from(b[token * vh + h])).exp());
                    let predicted: f64 = (0..k)
                        .map(|i| expected[row * k + i] * decay * f64::from(qkv[ko + i]))
                        .sum();
                    let delta =
                        (f64::from(qkv[token * qwidth + 2 * kh * k + row]) - predicted) * beta;
                    for i in 0..k {
                        expected[row * k + i] =
                            expected[row * k + i] * decay + f64::from(qkv[ko + i]) * delta;
                        output[token * rows + row] +=
                            expected[row * k + i] * f64::from(qkv[qo + i]);
                    }
                }
            }
            let guarded = |data: &[f32]| {
                let mut data_with_guards = vec![9876.];
                data_with_guards.extend_from_slice(data);
                data_with_guards.push(9876.);
                gpu.upload_f32(&data_with_guards)
            };
            let q = guarded(&qkv)?;
            let ab = guarded(&a)?;
            let bb = guarded(&b)?;
            let al = guarded(&alog)?;
            let db = guarded(&dt)?;
            let state = guarded(&initial)?;
            let old_state = guarded(&initial)?;
            let y = guarded(&vec![0.; batch * rows])?;
            let old_y = guarded(&vec![0.; batch * rows])?;
            let command = gpu.begin();
            let encoder = gpu.begin_encoding(command);
            encoder.encode_offsets(
                "delta_step_prefill",
                &[&q, &ab, &bb, &al, &db, &state, &y],
                &[4; 7],
                &[kh as u32, vh as u32, k as u32, v as u32, batch as u32],
                rows * 32,
                128,
            )?;
            for token in 0..batch {
                encoder.encode_offsets(
                    "delta_step",
                    &[&q, &ab, &bb, &al, &db, &old_state, &old_y],
                    &[
                        4 + token * qwidth * 4,
                        4 + token * vh * 4,
                        4 + token * vh * 4,
                        4,
                        4,
                        4,
                        4 + token * rows * 4,
                    ],
                    &[kh as u32, vh as u32, k as u32, v as u32],
                    rows * 32,
                    128,
                )?;
            }
            encoder.end_encoding()?;
            gpu.finish(command)?;
            for (actual, ordinary, wanted, count, label) in [
                (&state, &old_state, &expected, elements, "state"),
                (&y, &old_y, &output, batch * rows, "output"),
            ] {
                let actual = gpu.read_f32(actual, count + 2)?;
                let ordinary = gpu.read_f32(ordinary, count + 2)?;
                assert_eq!(actual[0], 9876., "{label} leading guard");
                assert_eq!(actual[count + 1], 9876., "{label} trailing guard");
                close(&actual[1..=count], wanted, label);
                for (i, (&actual, &ordinary)) in actual.iter().zip(&ordinary).enumerate() {
                    assert_eq!(
                        actual.to_bits(),
                        ordinary.to_bits(),
                        "B{batch} K{k} {label} {i} ordinary FP32 trajectory"
                    );
                }
            }
        }
    }
    Ok(())
}
