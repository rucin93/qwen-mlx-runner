//! Command-buffer timing without changing compute-pass scheduling.

use anyhow::{Result, ensure};
use metal::{CommandBufferRef, MTLCommandBufferStatus};
use objc::{msg_send, sel, sel_impl};
use serde::Serialize;

#[derive(Debug, Clone, Copy, Serialize)]
pub struct FrameTiming {
    /// CPU wall time spent preparing and encoding this command buffer.
    pub cpu_encode_seconds: f64,
    /// CPU wall time inside Metal's commit call, separate from the final wait.
    pub cpu_commit_seconds: f64,
    /// Metal's GPU end timestamp minus its GPU start timestamp, in seconds.
    /// This is the entire command-buffer execution interval, not kernel-only
    /// active time, and may overlap CPU work or the completion wait.
    pub gpu_seconds: f64,
    /// CPU wall time spent waiting for command-buffer completion. This can
    /// include queueing as well as GPU execution; do not add it to gpu_seconds.
    pub completion_wait_seconds: f64,
}

/// Read Metal's GPU timestamps after successful command-buffer completion.
///
/// The caller measures the CPU intervals using a monotonic clock. This
/// function neither commits nor waits for the command buffer. Missing, zero,
/// non-finite, or reversed GPU timestamps return an error, never a CPU-derived
/// substitute. Ordinary inference can discard an unavailable timing result;
/// diagnostic commands should propagate the error instead of reporting it as
/// measured GPU time.
///
/// Metal exposes GPUStartTime and GPUEndTime as CFTimeInterval (a double in
/// seconds) on macOS 10.15 and later. metal 0.32 has no wrappers for them.
pub fn command_timing(
    command: &CommandBufferRef,
    cpu_encode_seconds: f64,
    cpu_commit_seconds: f64,
    completion_wait_seconds: f64,
) -> Result<FrameTiming> {
    ensure!(
        command.status() == MTLCommandBufferStatus::Completed,
        "GPU command must complete successfully before reading its timestamps"
    );
    ensure!(
        cpu_encode_seconds.is_finite() && cpu_encode_seconds >= 0.0,
        "Invalid CPU encode duration: {cpu_encode_seconds}"
    );
    ensure!(
        cpu_commit_seconds.is_finite() && cpu_commit_seconds >= 0.0,
        "Invalid CPU commit duration: {cpu_commit_seconds}"
    );
    ensure!(
        completion_wait_seconds.is_finite() && completion_wait_seconds >= 0.0,
        "Invalid CPU completion-wait duration: {completion_wait_seconds}"
    );

    // SAFETY: CommandBufferRef is a live object implementing MTLCommandBuffer.
    // These no-argument selectors return CFTimeInterval, whose ABI is f64.
    // The successfully completed command owns stable execution timestamps.
    let start: f64 = unsafe { msg_send![command, GPUStartTime] };
    let end: f64 = unsafe { msg_send![command, GPUEndTime] };
    ensure!(
        start.is_finite() && end.is_finite() && start > 0.0 && end > start,
        "Metal GPU timestamps unavailable or invalid: start={start}, end={end}"
    );
    let gpu_seconds = end - start;
    ensure!(
        gpu_seconds.is_finite() && gpu_seconds > 0.0,
        "Invalid Metal GPU duration: {gpu_seconds}"
    );

    Ok(FrameTiming {
        cpu_encode_seconds,
        cpu_commit_seconds,
        gpu_seconds,
        completion_wait_seconds,
    })
}
