//! Optional aggregation of ordinary inference timings, without GPU profiling.

use crate::gpu::FrameTiming;
use anyhow::{Result, ensure};
use serde_json::{Value, json};

const BLOCK_SIZE: usize = 32;
const METRICS: [&str; 7] = [
    "forward",
    "sampling",
    "cpu_encode",
    "cpu_commit",
    "completion_wait",
    "gpu",
    "unclassified_host",
];

/// Durations measured around one ordinary forward call and its token selection.
pub struct StepTiming {
    pub history: usize,
    pub sampling_seconds: f64,
    pub forward_seconds: f64,
    pub frame: Option<FrameTiming>,
}

struct Sample {
    history: usize,
    seconds: [Option<f64>; METRICS.len()],
}

/// Stores only scalar measurements. Reporting and sorting happen after a phase.
pub struct PhaseTimings {
    samples: Vec<Sample>,
    totals: [f64; METRICS.len()],
}

impl PhaseTimings {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            samples: Vec::with_capacity(capacity),
            totals: [0.0; METRICS.len()],
        }
    }

    /// Reject invalid samples atomically; missing Metal timings remain missing.
    pub fn push(&mut self, step: StepTiming) -> Result<()> {
        valid_duration("forward", step.forward_seconds)?;
        valid_duration("sampling", step.sampling_seconds)?;
        let mut seconds = [None; METRICS.len()];
        seconds[0] = Some(step.forward_seconds);
        seconds[1] = Some(step.sampling_seconds);
        if let Some(frame) = step.frame {
            valid_duration("cpu_encode", frame.cpu_encode_seconds)?;
            valid_duration("cpu_commit", frame.cpu_commit_seconds)?;
            valid_duration("completion_wait", frame.completion_wait_seconds)?;
            valid_duration("gpu", frame.gpu_seconds)?;
            ensure!(frame.gpu_seconds > 0.0, "GPU duration must be positive");
            let classified =
                frame.cpu_encode_seconds + frame.cpu_commit_seconds + frame.completion_wait_seconds;
            ensure!(classified.is_finite(), "CPU timing sum overflowed");
            // GPU execution overlaps the completion wait. Only disjoint CPU
            // intervals are subtracted from the outer forward wall duration.
            let residual = step.forward_seconds - classified;
            let tolerance = 8.0 * f64::EPSILON * step.forward_seconds.max(classified);
            ensure!(
                residual >= -tolerance,
                "Forward duration {} is shorter than its CPU timing components {classified}",
                step.forward_seconds
            );
            seconds[2] = Some(frame.cpu_encode_seconds);
            seconds[3] = Some(frame.cpu_commit_seconds);
            seconds[4] = Some(frame.completion_wait_seconds);
            seconds[5] = Some(frame.gpu_seconds);
            seconds[6] = Some(residual.max(0.0));
        }
        let mut totals = self.totals;
        for (index, duration) in seconds.iter().enumerate() {
            if let Some(duration) = duration {
                totals[index] += duration;
                ensure!(
                    totals[index].is_finite(),
                    "{} total overflowed",
                    METRICS[index]
                );
            }
        }
        self.samples.push(Sample {
            history: step.history,
            seconds,
        });
        self.totals = totals;
        Ok(())
    }

    pub fn report(&self) -> Value {
        let mut report = sample_counts(&self.samples);
        report["percentile_method"] = json!("nearest_rank");
        report["gpu_timing_note"] =
            json!("GPU execution and completion wait overlap; do not add their durations.");
        report["unclassified_host_description"] = json!(
            "Forward wall time minus CPU encode, commit, and completion wait; includes shared readback, autorelease, timing, and validation work, not an isolated readback measurement."
        );
        for (index, name) in METRICS.iter().enumerate() {
            report[*name] = statistics(&self.samples, index);
        }
        report["blocks"] = Value::Array(
            self.samples
                .chunks(BLOCK_SIZE)
                .enumerate()
                .map(|(index, samples)| {
                    let mut block = sample_counts(samples);
                    block["step_range"] =
                        json!([index * BLOCK_SIZE + 1, index * BLOCK_SIZE + samples.len()]);
                    block["history_range"] =
                        json!([samples[0].history, samples[samples.len() - 1].history]);
                    for (metric, name) in METRICS.iter().enumerate() {
                        block[format!("{name}_mean_milliseconds")] =
                            complete_values(samples, metric).map_or(Value::Null, |values| {
                                json!(values.iter().sum::<f64>() / values.len() as f64 * 1000.0)
                            });
                    }
                    block
                })
                .collect(),
        );
        report
    }
}

fn valid_duration(name: &str, seconds: f64) -> Result<()> {
    ensure!(
        seconds.is_finite() && seconds >= 0.0 && (seconds * 1000.0).is_finite(),
        "Invalid {name} duration: {seconds}"
    );
    Ok(())
}

fn sample_counts(samples: &[Sample]) -> Value {
    let valid = samples
        .iter()
        .filter(|sample| sample.seconds[5].is_some())
        .count();
    json!({
        "sample_count": samples.len(),
        "valid_gpu_samples": valid,
        "missing_gpu_samples": samples.len() - valid,
    })
}

fn complete_values(samples: &[Sample], metric: usize) -> Option<Vec<f64>> {
    if samples.is_empty() {
        return None;
    }
    samples
        .iter()
        .map(|sample| sample.seconds[metric])
        .collect()
}

fn statistics(samples: &[Sample], metric: usize) -> Value {
    let Some(mut values) = complete_values(samples, metric) else {
        return Value::Null;
    };
    let total = values.iter().sum::<f64>();
    values.sort_unstable_by(f64::total_cmp);
    let percentile = |percent: usize| {
        let rank = (values.len() / 100 * percent) + (values.len() % 100 * percent).div_ceil(100);
        values[rank - 1] * 1000.0
    };
    json!({
        "total_seconds": total,
        "mean_milliseconds": total / values.len() as f64 * 1000.0,
        "p50_milliseconds": percentile(50),
        "p95_milliseconds": percentile(95),
        "min_milliseconds": values[0] * 1000.0,
        "max_milliseconds": values[values.len() - 1] * 1000.0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step(history: usize, forward: f64) -> StepTiming {
        StepTiming {
            history,
            sampling_seconds: 0.001,
            forward_seconds: forward,
            frame: Some(FrameTiming {
                cpu_encode_seconds: 0.001,
                cpu_commit_seconds: 0.002,
                completion_wait_seconds: 0.003,
                gpu_seconds: 0.02,
            }),
        }
    }

    fn near(value: &Value, expected: f64) {
        let actual = value.as_f64().expect("numeric duration");
        assert!(
            (actual - expected).abs() < 1e-10,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn empty_phase_has_counts_and_null_stats() {
        let report = PhaseTimings::with_capacity(0).report();
        assert_eq!(report["sample_count"], 0);
        assert_eq!(report["valid_gpu_samples"], 0);
        assert_eq!(report["missing_gpu_samples"], 0);
        for field in [
            "forward",
            "sampling",
            "cpu_encode",
            "cpu_commit",
            "completion_wait",
            "gpu",
            "unclassified_host",
        ] {
            assert!(report[field].is_null(), "{field}");
        }
        assert_eq!(report["blocks"], json!([]));
    }

    #[test]
    fn reports_known_totals_means_and_nearest_rank_percentiles() {
        let mut phase = PhaseTimings::with_capacity(4);
        for (i, duration) in [0.01, 0.02, 0.03, 0.04].into_iter().enumerate() {
            let mut sample = step(512 + i, duration);
            sample.sampling_seconds = 0.001 * (i + 1) as f64;
            phase.push(sample).unwrap();
        }
        let report = phase.report();
        assert_eq!(report["sample_count"], 4);
        assert_eq!(report["valid_gpu_samples"], 4);
        assert_eq!(report["missing_gpu_samples"], 0);
        near(&report["forward"]["total_seconds"], 0.1);
        near(&report["forward"]["mean_milliseconds"], 25.0);
        near(&report["forward"]["p50_milliseconds"], 20.0);
        near(&report["forward"]["p95_milliseconds"], 40.0);
        near(&report["forward"]["min_milliseconds"], 10.0);
        near(&report["forward"]["max_milliseconds"], 40.0);
        near(&report["sampling"]["total_seconds"], 0.01);
        near(&report["cpu_encode"]["total_seconds"], 0.004);
        near(&report["cpu_commit"]["total_seconds"], 0.008);
        near(&report["completion_wait"]["total_seconds"], 0.012);
        near(&report["gpu"]["total_seconds"], 0.08);
        near(&report["unclassified_host"]["total_seconds"], 0.076);
        assert_eq!(report["percentile_method"], "nearest_rank");
    }

    #[test]
    fn missing_frames_do_not_publish_partial_aggregates_as_complete() {
        let mut phase = PhaseTimings::with_capacity(33);
        for i in 0..33 {
            let mut sample = step(512 + i, 0.01);
            if i == 10 {
                sample.frame = None;
            }
            phase.push(sample).unwrap();
        }
        let report = phase.report();
        assert_eq!(report["valid_gpu_samples"], 32);
        assert_eq!(report["missing_gpu_samples"], 1);
        near(&report["forward"]["total_seconds"], 0.33);
        for field in [
            "cpu_encode",
            "cpu_commit",
            "completion_wait",
            "gpu",
            "unclassified_host",
        ] {
            assert!(report[field].is_null(), "{field}");
            assert!(
                report["blocks"][0][format!("{field}_mean_milliseconds")].is_null(),
                "{field}"
            );
            assert!(
                report["blocks"][1][format!("{field}_mean_milliseconds")].is_number(),
                "{field}"
            );
        }
        assert_eq!(report["blocks"][0]["missing_gpu_samples"], 1);
        assert_eq!(report["blocks"][1]["missing_gpu_samples"], 0);
    }

    #[test]
    fn partitions_one_thirty_one_thirty_two_and_thirty_three_samples() {
        for (count, expected_counts) in [
            (1, vec![1]),
            (31, vec![31]),
            (32, vec![32]),
            (33, vec![32, 1]),
        ] {
            let mut phase = PhaseTimings::with_capacity(count);
            for i in 0..count {
                phase.push(step(512 + i, 0.01)).unwrap();
            }
            let report = phase.report();
            let blocks = report["blocks"].as_array().unwrap();
            assert_eq!(blocks.len(), expected_counts.len());
            let mut first = 0;
            for (block, count) in blocks.iter().zip(expected_counts) {
                assert_eq!(block["sample_count"], count);
                assert_eq!(block["step_range"], json!([first + 1, first + count]));
                assert_eq!(
                    block["history_range"],
                    json!([512 + first, 511 + first + count])
                );
                near(&block["forward_mean_milliseconds"], 10.0);
                first += count;
            }
        }
    }

    #[test]
    fn rejects_invalid_durations_without_recording_the_sample() {
        for bad in [-0.01, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for field in 0..6 {
                let mut phase = PhaseTimings::with_capacity(1);
                let mut sample = step(512, 0.01);
                match field {
                    0 => sample.forward_seconds = bad,
                    1 => sample.sampling_seconds = bad,
                    2 => sample.frame.as_mut().unwrap().cpu_encode_seconds = bad,
                    3 => sample.frame.as_mut().unwrap().cpu_commit_seconds = bad,
                    4 => sample.frame.as_mut().unwrap().completion_wait_seconds = bad,
                    5 => sample.frame.as_mut().unwrap().gpu_seconds = bad,
                    _ => unreachable!(),
                }
                assert!(phase.push(sample).is_err(), "field {field}, value {bad}");
                assert_eq!(phase.report()["sample_count"], 0);
            }
        }
        let mut phase = PhaseTimings::with_capacity(1);
        let mut sample = step(512, 0.01);
        sample.frame.as_mut().unwrap().gpu_seconds = 0.0;
        assert!(phase.push(sample).is_err());
    }

    #[test]
    fn accepts_only_floating_roundoff_for_negative_host_remainder() {
        let mut sample = step(512, 0.6);
        sample.frame.as_mut().unwrap().cpu_encode_seconds = 0.1;
        sample.frame.as_mut().unwrap().cpu_commit_seconds = 0.2;
        sample.frame.as_mut().unwrap().completion_wait_seconds = 0.3;
        let mut phase = PhaseTimings::with_capacity(1);
        phase.push(sample).unwrap();
        near(&phase.report()["unclassified_host"]["total_seconds"], 0.0);
        assert!(phase.push(step(513, 0.0059)).is_err());
        assert_eq!(phase.report()["sample_count"], 1);
    }

    #[test]
    fn gpu_interval_is_not_added_to_wait_or_subtracted_from_host_remainder() {
        let mut sample = step(512, 0.01);
        sample.frame.as_mut().unwrap().gpu_seconds = 0.5;
        let mut phase = PhaseTimings::with_capacity(1);
        phase.push(sample).unwrap();
        let report = phase.report();
        near(&report["unclassified_host"]["total_seconds"], 0.004);
        near(&report["forward"]["total_seconds"], 0.01);
        near(&report["gpu"]["total_seconds"], 0.5);
        assert!(
            report["gpu_timing_note"]
                .as_str()
                .unwrap()
                .contains("overlap")
        );
    }
}
