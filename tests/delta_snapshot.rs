//! Recurrent blocks retain every rollback prefix and the ordinary FP32 trajectory.
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
fn delta_blocks_match_scalar_outputs_states_and_every_pre_update_snapshot() -> Result<()> {
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
        for batch in 1..=4 {
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
            let mut snapshots = Vec::new();
            for token in 0..batch {
                snapshots.extend_from_slice(&expected);
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
            let snapshot = guarded(&vec![0.; batch * elements])?;
            let old_snapshot = guarded(&vec![0.; batch * elements])?;
            let command = gpu.begin();
            let encoder = gpu.begin_encoding(command);
            encoder.encode_offsets(
                "delta_step_block",
                &[&q, &ab, &bb, &al, &db, &state, &y, &snapshot],
                &[4; 8],
                &[kh as u32, vh as u32, k as u32, v as u32, batch as u32],
                rows * 32,
                128,
            )?;
            for token in 0..batch {
                encoder.encode_offsets(
                    "copy_f32",
                    &[&old_state, &old_snapshot],
                    &[4, 4 + token * elements * 4],
                    &[elements as u32],
                    elements,
                    128,
                )?;
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
                (
                    &snapshot,
                    &old_snapshot,
                    &snapshots,
                    batch * elements,
                    "snapshot",
                ),
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

#[test]
#[ignore = "requires a real Apple Metal GPU"]
fn delta_block_rejects_unsupported_shapes_bounds_and_overlapping_writes() -> Result<()> {
    let gpu = Gpu::new_with_reference(false)?;
    let input = gpu.alloc_f32(4096)?;
    let state = gpu.alloc_f32(4096)?;
    let output = gpu.alloc_f32(4096)?;
    let snapshots = gpu.alloc_f32(4096)?;
    let command = gpu.begin();
    let encoder = gpu.begin_encoding(command);
    let buffers = [
        &*input,
        &*input,
        &*input,
        &*input,
        &*input,
        &*state,
        &*output,
        &*snapshots,
    ];
    for shape in [
        [1, 2, 129, 3, 2],
        [1, 2, 32, 3, 0],
        [1, 2, 32, 3, 5],
        [3, 4, 32, 3, 2],
        [1, 2, 0, 3, 2],
        [1, 2, 32, 0, 2],
    ] {
        assert!(
            encoder
                .encode("delta_step_block", &buffers, &shape, 192, 128)
                .is_err()
        );
    }
    let shape = [1, 2, 32, 3, 2];
    for index in 0..8 {
        for offset in [2, 4096 * 4 - 4, usize::MAX] {
            let mut offsets = [0; 8];
            offsets[index] = offset;
            assert!(
                encoder
                    .encode_offsets("delta_step_block", &buffers, &offsets, &shape, 192, 128)
                    .is_err(),
                "buffer {index}, offset {offset}"
            );
        }
    }
    for destination in [5, 6, 7] {
        for source in 0..8 {
            if source == destination {
                continue;
            }
            let mut alias = buffers;
            alias[destination] = alias[source];
            assert!(
                encoder
                    .encode("delta_step_block", &alias, &shape, 192, 128)
                    .is_err(),
                "write {destination} overlaps buffer {source}"
            );
        }
    }
    encoder.end_encoding()?;
    gpu.finish(command)?;
    Ok(())
}
