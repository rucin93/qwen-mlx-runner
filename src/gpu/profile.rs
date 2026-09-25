//! Opt-in diagnostic timings. A separate compute pass per dispatch changes
//! scheduling, so these timings must not be presented as normal inference speed.

use anyhow::{Context, Result, anyhow, ensure};
use metal::{
    Buffer, CommandBuffer, CommandBufferRef, ComputeCommandEncoderRef, ComputePassDescriptor,
    CounterSampleBuffer, CounterSampleBufferDescriptor, Device, DeviceRef, MTLCommandBufferStatus,
    MTLCounterSamplingPoint, MTLDispatchType, MTLResourceOptions, MTLStorageMode, NSRange,
};
use serde::Serialize;
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;

// A Qwen 27B step has fewer than 1,200 dispatches, each using two timestamps.
const MAX_SAMPLES: usize = 4096;
const TIMESTAMP_BYTES: usize = std::mem::size_of::<u64>();

#[derive(Debug, Clone, Serialize)]
pub struct ProfileRow {
    pub kernel: String,
    pub parameters: Vec<u32>,
    pub dispatches: usize,
    pub raw_gpu_ticks: u64,
    /// Calibrated CPU nanoseconds, rounded to the nearest integer.
    pub gpu_nanoseconds: u64,
    pub gpu_seconds: f64,
    pub mean_gpu_microseconds: f64,
    pub max_gpu_microseconds: f64,
}

struct Entry {
    kernel: String,
    parameters: Vec<u32>,
}

#[derive(Clone, Copy)]
struct TimestampPair {
    cpu: u64,
    gpu: u64,
}

#[derive(Clone, Copy)]
struct ClockCalibration {
    cpu_span: u64,
    gpu_span: u64,
}

impl ClockCalibration {
    fn new(start: TimestampPair, end: TimestampPair) -> Result<Self> {
        ensure!(
            [start.cpu, start.gpu, end.cpu, end.gpu]
                .iter()
                .all(|&value| value != 0 && value != u64::MAX),
            "Metal returned an invalid clock-calibration timestamp"
        );
        let cpu_span = end.cpu.checked_sub(start.cpu);
        let gpu_span = end.gpu.checked_sub(start.gpu);
        ensure!(
            cpu_span.is_some_and(|span| span > 0) && gpu_span.is_some_and(|span| span > 0),
            "Metal clock-calibration spans must be positive and monotonic"
        );
        Ok(Self {
            cpu_span: cpu_span.unwrap(),
            gpu_span: gpu_span.unwrap(),
        })
    }

    fn nanoseconds(&self, ticks: u64) -> Result<u64> {
        // Calculate the delta before converting units. Integer arithmetic avoids
        // losing precision by subtracting large floating-point absolute times.
        let numerator = u128::from(ticks) * u128::from(self.cpu_span);
        let rounded = (numerator + u128::from(self.gpu_span / 2)) / u128::from(self.gpu_span);
        u64::try_from(rounded).context("Calibrated Metal duration exceeds u64 nanoseconds")
    }

    fn fractional_nanoseconds(&self, ticks: u64) -> f64 {
        ticks as f64 / self.gpu_span as f64 * self.cpu_span as f64
    }
}

pub struct StageProfiler {
    device: Device,
    samples: CounterSampleBuffer,
    results: Buffer,
    entries: RefCell<Vec<Entry>>,
    resolved_samples: Cell<Option<usize>>,
    // Retaining the last command allows report() to reject a premature CPU read.
    resolved_command: RefCell<Option<CommandBuffer>>,
    start_pair: Cell<Option<TimestampPair>>,
    calibration: Cell<Option<ClockCalibration>>,
}

impl StageProfiler {
    pub fn new(device: &DeviceRef) -> Result<Self> {
        ensure!(
            device.supports_counter_sampling(MTLCounterSamplingPoint::AtStageBoundary),
            "GPU does not support Metal stage-boundary counter sampling"
        );
        let counter_sets = device.counter_sets();
        let timestamp_set = counter_sets
            .iter()
            .find(|set| set.name().eq_ignore_ascii_case("timestamp"))
            .context("GPU does not expose the Metal timestamp counter set")?;
        let descriptor = CounterSampleBufferDescriptor::new();
        descriptor.set_counter_set(timestamp_set);
        descriptor.set_sample_count(MAX_SAMPLES as u64);
        descriptor.set_storage_mode(MTLStorageMode::Shared);
        descriptor.set_label("qwen diagnostic dispatch timestamps");
        let samples = device
            .new_counter_sample_buffer_with_descriptor(&descriptor)
            .map_err(|error| anyhow!("Cannot create Metal timestamp buffer: {error}"))?;
        let results = device.new_buffer(
            (MAX_SAMPLES * TIMESTAMP_BYTES) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        results.set_label("qwen resolved diagnostic timestamps");
        Ok(Self {
            device: device.to_owned(),
            samples,
            results,
            entries: RefCell::new(Vec::new()),
            resolved_samples: Cell::new(None),
            resolved_command: RefCell::new(None),
            start_pair: Cell::new(None),
            calibration: Cell::new(None),
        })
    }

    /// Begin a new diagnostic frame after the previous command has completed.
    /// No result-buffer writes occur here; encoder() also checks GPU completion
    /// before permitting the buffer to be reused.
    pub fn reset(&self) {
        self.entries.borrow_mut().clear();
        self.resolved_samples.set(None);
        self.start_pair.set(None);
        self.calibration.set(None);
    }

    /// The caller must end this encoder before requesting the next one. All
    /// encoders and resolve() belong to the same command buffer for this frame.
    pub fn encoder<'a>(
        &self,
        command: &'a CommandBufferRef,
        name: &str,
        parameters: &[u32],
    ) -> Result<&'a ComputeCommandEncoderRef> {
        ensure!(
            self.resolved_samples.get().is_none(),
            "Reset the stage profiler before encoding another frame"
        );
        if let Some(previous) = self.resolved_command.borrow().as_ref() {
            ensure!(
                matches!(
                    previous.status(),
                    MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error
                ),
                "Previous profiled GPU command is still running"
            );
        }
        let mut entries = self.entries.borrow_mut();
        let sample_index = entries
            .len()
            .checked_mul(2)
            .context("Stage profiler sample index overflow")?;
        ensure!(
            sample_index + 2 <= MAX_SAMPLES,
            "Stage profiler capacity exceeded: at most {} dispatches per frame",
            MAX_SAMPLES / 2
        );
        if self.start_pair.get().is_none() {
            self.start_pair.set(Some(self.sample_clocks()));
        }
        let descriptor = ComputePassDescriptor::new();
        descriptor.set_dispatch_type(MTLDispatchType::Serial);
        let attachment = descriptor
            .sample_buffer_attachments()
            .object_at(0)
            .context("Metal did not provide a compute-pass counter attachment")?;
        attachment.set_sample_buffer(&self.samples);
        attachment.set_start_of_encoder_sample_index(sample_index as u64);
        attachment.set_end_of_encoder_sample_index((sample_index + 1) as u64);
        let encoder = command.compute_command_encoder_with_descriptor(descriptor);
        encoder.set_label(name);
        entries.push(Entry {
            kernel: name.to_owned(),
            parameters: parameters.to_vec(),
        });
        Ok(encoder)
    }

    /// Encode resolution after all compute encoders have ended, before commit.
    /// It neither commits the command nor waits for the GPU.
    pub fn resolve(&self, command: &CommandBufferRef) -> Result<()> {
        ensure!(
            self.resolved_samples.get().is_none(),
            "Stage profiler timestamps have already been resolved for this frame"
        );
        let count = self.entries.borrow().len() * 2;
        if count != 0 {
            let blit = command.new_blit_command_encoder();
            blit.set_label("resolve qwen diagnostic timestamps");
            // metal 0.32's CPU resolve_counter_range wrapper copies zero bytes
            // into an uninitialized Vec. Use Metal's GPU resolution API instead.
            blit.resolve_counters(
                &self.samples,
                NSRange::new(0, count as u64),
                &self.results,
                0,
            );
            blit.end_encoding();
        }
        self.resolved_samples.set(Some(count));
        *self.resolved_command.borrow_mut() = Some(command.to_owned());
        Ok(())
    }

    fn sample_clocks(&self) -> TimestampPair {
        let (mut cpu, mut gpu) = (0, 0);
        self.device.sample_timestamps(&mut cpu, &mut gpu);
        TimestampPair { cpu, gpu }
    }

    /// Read only after the caller has committed and completed the GPU command.
    pub fn report(&self) -> Result<Vec<ProfileRow>> {
        let count = self
            .resolved_samples
            .get()
            .context("Stage profiler timestamps have not been resolved")?;
        let command = self.resolved_command.borrow();
        let command = command
            .as_ref()
            .context("Stage profiler has no resolved GPU command")?;
        ensure!(
            command.status() == MTLCommandBufferStatus::Completed,
            "Profiled GPU command must complete successfully before reading timestamps"
        );
        let entries = self.entries.borrow();
        ensure!(
            count == entries.len() * 2 && count <= MAX_SAMPLES,
            "Stage profiler entry/sample count mismatch"
        );
        if entries.is_empty() {
            return Ok(Vec::new());
        }
        let calibration = match self.calibration.get() {
            Some(calibration) => calibration,
            None => {
                let start = self
                    .start_pair
                    .get()
                    .context("Stage profiler has no initial clock sample")?;
                // Sample only once after completion and cache the resulting
                // calibration so repeated reports use the same time window.
                // Apple's conversion guide explicitly specifies that the CPU
                // timestamps returned here are nanoseconds, not raw Mach ticks:
                // https://developer.apple.com/documentation/metal/converting-gpu-timestamps-into-cpu-time
                // Therefore no additional mach_timebase_info conversion applies.
                let calibration = ClockCalibration::new(start, self.sample_clocks())?;
                self.calibration.set(Some(calibration));
                calibration
            }
        };
        // This Shared buffer is written only by the completed resolve command.
        // Metal aligns buffer contents adequately for u64 counter results.
        let timestamps =
            unsafe { std::slice::from_raw_parts(self.results.contents().cast::<u64>(), count) };
        aggregate(&entries, timestamps, calibration)
    }
}

fn aggregate(
    entries: &[Entry],
    timestamps: &[u64],
    calibration: ClockCalibration,
) -> Result<Vec<ProfileRow>> {
    ensure!(
        timestamps.len() == entries.len() * 2,
        "Stage profiler entry/sample count mismatch"
    );
    let mut grouped: BTreeMap<(String, Vec<u32>), (usize, u64, u64)> = BTreeMap::new();
    for (entry, pair) in entries.iter().zip(timestamps.chunks_exact(2)) {
        let (start, end) = (pair[0], pair[1]);
        ensure!(
            start != u64::MAX && end != u64::MAX && start != 0 && end != 0 && end >= start,
            "Invalid Metal timestamp pair for {}: start={start}, end={end}",
            entry.kernel
        );
        let elapsed = end - start;
        let row = grouped
            .entry((entry.kernel.clone(), entry.parameters.clone()))
            .or_default();
        row.0 += 1;
        row.1 = row
            .1
            .checked_add(elapsed)
            .context("Stage profiler total duration overflow")?;
        row.2 = row.2.max(elapsed);
    }
    let mut rows: Vec<_> = grouped
        .into_iter()
        .map(|((kernel, parameters), (dispatches, total, maximum))| {
            let nanoseconds = calibration.fractional_nanoseconds(total);
            Ok(ProfileRow {
                kernel,
                parameters,
                dispatches,
                raw_gpu_ticks: total,
                gpu_nanoseconds: calibration.nanoseconds(total)?,
                gpu_seconds: nanoseconds / 1e9,
                mean_gpu_microseconds: nanoseconds / dispatches as f64 / 1e3,
                max_gpu_microseconds: calibration.fractional_nanoseconds(maximum) / 1e3,
            })
        })
        .collect::<Result<_>>()?;
    rows.sort_by(|a, b| b.raw_gpu_ticks.cmp(&a.raw_gpu_ticks));
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    const UNIT_CLOCK: ClockCalibration = ClockCalibration {
        cpu_span: 1,
        gpu_span: 1,
    };

    fn entry(kernel: &str, parameters: &[u32]) -> Entry {
        Entry {
            kernel: kernel.into(),
            parameters: parameters.into(),
        }
    }

    #[test]
    fn totals_preserve_distinct_matrix_shapes_and_sort_by_duration() {
        let entries = [
            entry("matvec", &[16, 32]),
            entry("matvec", &[16, 32]),
            entry("matvec", &[64, 32]),
            entry("norm", &[32]),
        ];
        let rows = aggregate(
            &entries,
            &[1, 1001, 1002, 4002, 4003, 14003, 14004, 14504],
            UNIT_CLOCK,
        )
        .expect("valid timestamps");
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].parameters, [64, 32]);
        assert_eq!(rows[0].gpu_nanoseconds, 10_000);
        assert_eq!(rows[1].dispatches, 2);
        assert_eq!(rows[1].gpu_nanoseconds, 4000);
        assert_eq!(rows[1].mean_gpu_microseconds, 2.0);
        assert_eq!(rows[1].max_gpu_microseconds, 3.0);
        assert_eq!(rows[2].kernel, "norm");
    }

    #[test]
    fn invalid_or_missing_timestamps_are_rejected() {
        let entries = [entry("test", &[])];
        for timestamps in [[u64::MAX, 2], [1, u64::MAX], [2, 1], [0, 1], [1, 0]] {
            assert!(aggregate(&entries, &timestamps, UNIT_CLOCK).is_err());
        }
        assert!(aggregate(&entries, &[], UNIT_CLOCK).is_err());
        assert!(aggregate(&[], &[], UNIT_CLOCK).unwrap().is_empty());
    }

    #[test]
    fn non_unit_gpu_clock_is_calibrated_to_cpu_nanoseconds() {
        let calibration = ClockCalibration::new(
            TimestampPair { cpu: 100, gpu: 500 },
            TimestampPair {
                cpu: 3100,
                gpu: 2500,
            },
        )
        .unwrap();
        let rows = aggregate(
            &[entry("test", &[]), entry("test", &[])],
            &[600, 1600, 1700, 2200],
            calibration,
        )
        .unwrap();
        assert_eq!(rows[0].raw_gpu_ticks, 1500);
        assert_eq!(rows[0].gpu_nanoseconds, 2250);
        assert_eq!(rows[0].gpu_seconds, 2250.0 / 1e9);
        assert_eq!(rows[0].mean_gpu_microseconds, 1.125);
        assert_eq!(rows[0].max_gpu_microseconds, 1.5);
    }

    #[test]
    fn calibration_rejects_invalid_zero_or_reversed_spans() {
        let start = TimestampPair { cpu: 100, gpu: 200 };
        for end in [
            TimestampPair { cpu: 100, gpu: 201 },
            TimestampPair { cpu: 101, gpu: 200 },
            TimestampPair { cpu: 99, gpu: 201 },
            TimestampPair { cpu: 101, gpu: 199 },
            TimestampPair { cpu: 0, gpu: 201 },
            TimestampPair { cpu: 101, gpu: 0 },
            TimestampPair {
                cpu: u64::MAX,
                gpu: 201,
            },
            TimestampPair {
                cpu: 101,
                gpu: u64::MAX,
            },
        ] {
            assert!(ClockCalibration::new(start, end).is_err());
        }
        assert!(
            ClockCalibration::new(
                TimestampPair { cpu: 0, gpu: 100 },
                TimestampPair { cpu: 1, gpu: 101 },
            )
            .is_err()
        );
    }

    #[test]
    fn calibration_preserves_small_spans_at_large_absolute_times() {
        let calibration = ClockCalibration::new(
            TimestampPair {
                cpu: u64::MAX - 100,
                gpu: u64::MAX - 100,
            },
            TimestampPair {
                cpu: u64::MAX - 40,
                gpu: u64::MAX - 80,
            },
        )
        .unwrap();
        assert_eq!(calibration.nanoseconds(10).unwrap(), 30);
        assert!(calibration.nanoseconds(u64::MAX).is_err());
        let fractional = ClockCalibration {
            cpu_span: 1,
            gpu_span: 3,
        };
        assert_eq!(fractional.nanoseconds(1).unwrap(), 0);
        assert_eq!(fractional.nanoseconds(2).unwrap(), 1);
        assert_eq!(fractional.fractional_nanoseconds(1), 1.0 / 3.0);
    }
}
