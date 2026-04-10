//! Lightweight per-frame timing for the **Frame timing** ImGui window (FPS, wall interval, CPU/GPU submit splits).
//!
//! Unlike [`super::FrameDiagnosticsSnapshot`], this avoids backend enumeration and host sampling.

use crate::gpu::GpuContext;

/// Minimal HUD payload: wall-clock pacing plus CPU/GPU wall intervals around queue submits.
#[derive(Clone, Debug, Default)]
pub struct FrameTimingHudSnapshot {
    /// Wall-clock time between consecutive redraw ticks (ms); FPS = `1000.0 / wall_frame_time_ms`.
    pub wall_frame_time_ms: f64,
    /// Wall-clock from the start of the winit frame tick to the last tracked `Queue::submit` (ms).
    pub cpu_frame_until_submit_ms: Option<f64>,
    /// Wall-clock from that last submit until the GPU finishes that submission (`on_submitted_work_done`) (ms).
    pub gpu_frame_after_submit_ms: Option<f64>,
}

impl FrameTimingHudSnapshot {
    /// Reads GPU timing splits and uses `wall_frame_time_ms` from the app (same as full diagnostics).
    pub fn capture(gpu: &GpuContext, wall_frame_time_ms: f64) -> Self {
        let (cpu_frame_until_submit_ms, gpu_frame_after_submit_ms) = gpu.frame_cpu_gpu_ms_for_hud();
        Self {
            wall_frame_time_ms,
            cpu_frame_until_submit_ms,
            gpu_frame_after_submit_ms,
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
    use super::FrameTimingHudSnapshot;

    #[test]
    fn fps_from_wall_matches_inverse_ms() {
        let s = FrameTimingHudSnapshot {
            wall_frame_time_ms: 16.0,
            cpu_frame_until_submit_ms: Some(2.0),
            gpu_frame_after_submit_ms: Some(1.0),
        };
        assert!((s.fps_from_wall() - 62.5).abs() < 0.01);
    }

    #[test]
    fn fps_from_wall_zero_interval() {
        let s = FrameTimingHudSnapshot {
            wall_frame_time_ms: 0.0,
            cpu_frame_until_submit_ms: None,
            gpu_frame_after_submit_ms: None,
        };
        assert_eq!(s.fps_from_wall(), 0.0);
    }
}
