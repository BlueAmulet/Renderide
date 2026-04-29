//! Lightweight per-frame timing for the **Frame timing** ImGui window (FPS, wall interval,
//! CPU/GPU submit splits, RAM/VRAM, and a rolling frametime graph — MangoHud-style overlay).
//!
//! Unlike [`super::FrameDiagnosticsSnapshot`], this avoids the heavy shader-routes / allocator-report
//! gathering and is safe to populate every tick.

use std::collections::VecDeque;

use crate::gpu::GpuContext;

use super::frame_diagnostics_snapshot::HostCpuMemoryHud;

/// Frametime history length used for the sparkline plot (power of two for predictable caps).
pub const FRAME_TIME_HISTORY_LEN: usize = 128;

/// Rolling frametime ring used to feed the HUD sparkline. Samples are milliseconds.
#[derive(Clone, Debug, Default)]
pub struct FrameTimeHistory {
    samples: VecDeque<f32>,
}

impl FrameTimeHistory {
    /// Empty history (next [`Self::push`] starts filling the ring).
    pub fn new() -> Self {
        Self {
            samples: VecDeque::with_capacity(FRAME_TIME_HISTORY_LEN),
        }
    }

    /// Appends a sample in milliseconds, evicting the oldest when capacity is hit.
    pub fn push(&mut self, ms: f32) {
        if self.samples.len() == FRAME_TIME_HISTORY_LEN {
            self.samples.pop_front();
        }
        self.samples.push_back(ms);
    }

    /// Clones the current samples oldest-first for consumers that want a contiguous slice.
    pub fn to_vec(&self) -> Vec<f32> {
        self.samples.iter().copied().collect()
    }

    /// Number of stored samples (`0..=FRAME_TIME_HISTORY_LEN`).
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// `true` when no samples have been pushed yet.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

/// Minimal HUD payload: wall-clock roundtrip, CPU/GPU per-frame ms, memory totals, and frametime graph.
#[derive(Clone, Debug, Default)]
pub struct FrameTimingHudSnapshot {
    /// Wall-clock roundtrip between consecutive winit ticks (ms): the time between when one frame
    /// started and the next one started. FPS = `1000.0 / wall_frame_time_ms`.
    pub wall_frame_time_ms: f64,
    /// CPU per-frame ms: from the start of the winit tick (CPU begins preparing the frame) to
    /// the moment `Queue::submit` returns on the driver thread for that tick's last submit.
    ///
    /// Comes from the most recent frame whose submit has reached the driver thread, so it may
    /// lag the current tick by one frame; see
    /// [`crate::gpu::frame_cpu_gpu_timing::FrameCpuGpuTiming`].
    pub cpu_frame_ms: Option<f64>,
    /// GPU per-frame ms: from `Queue::submit` returning on the driver thread to the
    /// `on_submitted_work_done` callback firing for that submit (i.e. wgpu reports the GPU has
    /// no more work for this frame).
    ///
    /// Comes from the most recent frame whose completion callback has fired, so it may lag the
    /// current tick by one or more frames.
    pub gpu_frame_ms: Option<f64>,
    /// Rolling frametime samples (ms, oldest-first) for the sparkline plot.
    pub frame_time_history: Vec<f32>,
    /// Global host CPU usage 0–100 (sysinfo, throttled).
    pub host_cpu_usage_percent: f32,
    /// Total system RAM in bytes (sysinfo).
    pub host_ram_total_bytes: u64,
    /// Used system RAM in bytes (sysinfo).
    pub host_ram_used_bytes: u64,
    /// Resident memory of the renderer process in bytes (sysinfo; `None` when unavailable).
    pub process_ram_bytes: Option<u64>,
}

impl FrameTimingHudSnapshot {
    /// Reads GPU timing and pairs them with the supplied host / history state.
    pub fn capture(
        gpu: &GpuContext,
        wall_frame_time_ms: f64,
        host: &HostCpuMemoryHud,
        history: &FrameTimeHistory,
    ) -> Self {
        profiling::scope!("hud::build_timing_snapshot");
        let (cpu_frame_ms, gpu_frame_ms) = gpu.frame_cpu_gpu_ms_for_hud();
        Self {
            wall_frame_time_ms,
            cpu_frame_ms,
            gpu_frame_ms,
            frame_time_history: history.to_vec(),
            host_cpu_usage_percent: host.cpu_usage_percent,
            host_ram_total_bytes: host.ram_total_bytes,
            host_ram_used_bytes: host.ram_used_bytes,
            process_ram_bytes: host.process_ram_bytes,
        }
    }

    /// FPS from wall-clock interval between redraws (matches [`super::FrameDiagnosticsSnapshot::fps_from_wall`]).
    pub fn fps_from_wall(&self) -> f64 {
        if self.wall_frame_time_ms <= f64::EPSILON {
            0.0
        } else {
            1000.0 / self.wall_frame_time_ms
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{FrameTimeHistory, FrameTimingHudSnapshot, FRAME_TIME_HISTORY_LEN};

    #[test]
    fn fps_from_wall_matches_inverse_ms() {
        let s = FrameTimingHudSnapshot {
            wall_frame_time_ms: 16.0,
            cpu_frame_ms: Some(2.0),
            gpu_frame_ms: Some(1.0),
            ..Default::default()
        };
        assert!((s.fps_from_wall() - 62.5).abs() < 0.01);
    }

    #[test]
    fn fps_from_wall_zero_interval() {
        let s = FrameTimingHudSnapshot::default();
        assert_eq!(s.fps_from_wall(), 0.0);
    }

    #[test]
    fn history_caps_at_configured_length() {
        let mut h = FrameTimeHistory::new();
        for i in 0..(FRAME_TIME_HISTORY_LEN + 10) {
            h.push(i as f32);
        }
        assert_eq!(h.len(), FRAME_TIME_HISTORY_LEN);
        let v = h.to_vec();
        assert_eq!(v.first().copied(), Some(10.0));
        assert_eq!(v.last().copied(), Some((FRAME_TIME_HISTORY_LEN + 9) as f32));
    }
}
