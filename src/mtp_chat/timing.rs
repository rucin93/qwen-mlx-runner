//! Sums the command-buffer timings already collected by the GPU backend.
use crate::gpu::FrameTiming;
use serde::Serialize;

#[derive(Clone, Debug, Default, Serialize)]
pub struct TargetDecodeTiming {
    pub command_buffers: usize,
    pub timed_command_buffers: usize,
    pub missing_command_buffers: usize,
    /// Sums cover only timed command buffers; missing samples are never zeros.
    pub observed_cpu_encode_seconds: f64,
    pub observed_cpu_commit_seconds: f64,
    pub observed_gpu_seconds: f64,
    /// May overlap GPU execution. Never add this to observed_gpu_seconds.
    pub observed_completion_wait_seconds: f64,
}

impl TargetDecodeTiming {
    pub(super) fn record(&mut self, frame: Option<FrameTiming>) {
        self.command_buffers += 1;
        if let Some(frame) = frame {
            self.timed_command_buffers += 1;
            self.observed_cpu_encode_seconds += frame.cpu_encode_seconds;
            self.observed_cpu_commit_seconds += frame.cpu_commit_seconds;
            self.observed_gpu_seconds += frame.gpu_seconds;
            self.observed_completion_wait_seconds += frame.completion_wait_seconds;
        } else {
            self.missing_command_buffers += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_timestamps_are_counted_separately_from_measured_sums() {
        let mut total = TargetDecodeTiming::default();
        total.record(None);
        total.record(Some(FrameTiming {
            cpu_encode_seconds: 0.125,
            cpu_commit_seconds: 0.25,
            gpu_seconds: 0.5,
            completion_wait_seconds: 0.75,
        }));
        total.record(None);
        total.record(Some(FrameTiming {
            cpu_encode_seconds: 0.125,
            cpu_commit_seconds: 0.25,
            gpu_seconds: 0.5,
            completion_wait_seconds: 0.75,
        }));
        assert_eq!(total.command_buffers, 4);
        assert_eq!(total.timed_command_buffers, 2);
        assert_eq!(total.missing_command_buffers, 2);
        assert_eq!(total.observed_cpu_encode_seconds, 0.25);
        assert_eq!(total.observed_cpu_commit_seconds, 0.5);
        assert_eq!(total.observed_gpu_seconds, 1.0);
        assert_eq!(total.observed_completion_wait_seconds, 1.5);
    }
}
