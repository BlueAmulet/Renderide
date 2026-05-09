//! Runtime-owned diagnostics accumulation state.

use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::diagnostics::{GpuAllocatorHud, GpuAllocatorReportHud, HostHudGatherer};
use crate::gpu::GpuContext;

/// How often [`wgpu::Device::generate_allocator_report`] replaces the GPU memory tab payload.
const GPU_ALLOCATOR_FULL_REPORT_INTERVAL: Duration = Duration::from_secs(2);

/// Diagnostics state that belongs to runtime orchestration rather than the backend HUD widget.
pub(super) struct RuntimeDiagnosticsState {
    /// Throttled host CPU/RAM sampling for the debug HUD.
    pub(super) host_hud: HostHudGatherer,
    /// Rolling per-frame wall time history that feeds the frame timing sparkline.
    pub(super) frame_time_history: crate::diagnostics::FrameTimeHistory,
    /// Persistent EMA state for frame timing scalar readouts.
    pub(super) frame_timing_ema: crate::diagnostics::FrameTimingEma,
    /// `FrameSubmitData::render_tasks` length from the last applied frame submit.
    pub(super) last_submit_render_task_count: usize,
    /// Camera readback tasks currently waiting to be drained before the next begin-frame send.
    pub(super) pending_camera_readbacks: usize,
    /// Cumulative camera readback tasks successfully written to host shared memory.
    pub(super) completed_camera_readbacks: u64,
    /// Cumulative camera readback tasks failed and zero-filled when possible.
    pub(super) failed_camera_readbacks: u64,
    /// Cached full allocator report for the GPU memory HUD tab.
    pub(super) allocator_report_hud: Option<GpuAllocatorReportHud>,
    /// Cached allocator totals from the same throttled report.
    pub(super) allocator_report_totals: GpuAllocatorHud,
    /// Wall clock when a GPU memory tab refresh was last attempted.
    pub(super) allocator_report_last_refresh: Option<Instant>,
    /// Count of failed frame-submit apply or cache-flush operations after host submits.
    pub(super) frame_submit_apply_failures: u64,
}

impl RuntimeDiagnosticsState {
    /// Creates empty runtime diagnostics state.
    pub(super) fn new() -> Self {
        Self {
            host_hud: HostHudGatherer::default(),
            frame_time_history: crate::diagnostics::FrameTimeHistory::new(),
            frame_timing_ema: crate::diagnostics::FrameTimingEma::default(),
            last_submit_render_task_count: 0,
            pending_camera_readbacks: 0,
            completed_camera_readbacks: 0,
            failed_camera_readbacks: 0,
            allocator_report_hud: None,
            allocator_report_totals: GpuAllocatorHud::default(),
            allocator_report_last_refresh: None,
            frame_submit_apply_failures: 0,
        }
    }

    /// Updates the latest render-task count for the HUD.
    pub(super) fn set_last_submit_render_task_count(&mut self, n: usize) {
        self.last_submit_render_task_count = n;
    }

    /// Replaces the current pending camera readback count.
    pub(super) fn set_pending_camera_readbacks(&mut self, n: usize) {
        self.pending_camera_readbacks = n;
    }

    /// Adds completed and failed camera readback counts to the cumulative HUD counters.
    pub(super) fn note_camera_readback_results(&mut self, completed: u64, failed: u64) {
        self.completed_camera_readbacks = self.completed_camera_readbacks.saturating_add(completed);
        self.failed_camera_readbacks = self.failed_camera_readbacks.saturating_add(failed);
    }

    /// Increments the cumulative scene-apply failure counter.
    pub(super) fn note_frame_submit_apply_failure(&mut self) {
        self.frame_submit_apply_failures = self.frame_submit_apply_failures.saturating_add(1);
    }

    /// Refreshes the sorted GPU allocator report when the interval elapses.
    pub(super) fn refresh_gpu_allocator_report_hud(&mut self, gpu: &GpuContext, now: Instant) {
        let should_refresh = self
            .allocator_report_last_refresh
            .is_none_or(|t| now.duration_since(t) >= GPU_ALLOCATOR_FULL_REPORT_INTERVAL);
        if !should_refresh {
            return;
        }
        self.allocator_report_last_refresh = Some(now);
        if let Some(rep) = gpu.device().generate_allocator_report() {
            self.allocator_report_totals = GpuAllocatorHud {
                allocated_bytes: Some(rep.total_allocated_bytes),
                reserved_bytes: Some(rep.total_reserved_bytes),
            };
            let mut order: Vec<usize> = (0..rep.allocations.len()).collect();
            order.sort_by_key(|&i| std::cmp::Reverse(rep.allocations[i].size));
            self.allocator_report_hud = Some(GpuAllocatorReportHud {
                report: Arc::new(rep),
                allocation_indices_by_size: order.into(),
            });
        }
    }

    /// Seconds until the next full allocator refresh should be attempted.
    pub(super) fn allocator_report_next_refresh_in_secs(&self, now: Instant) -> f32 {
        self.allocator_report_last_refresh.map_or(
            GPU_ALLOCATOR_FULL_REPORT_INTERVAL.as_secs_f32(),
            |t| {
                let elapsed = now.saturating_duration_since(t);
                GPU_ALLOCATOR_FULL_REPORT_INTERVAL
                    .saturating_sub(elapsed)
                    .as_secs_f32()
            },
        )
    }

    /// Clears main-HUD allocator report state when the main HUD is disabled.
    pub(super) fn clear_allocator_report(&mut self) {
        self.allocator_report_hud = None;
        self.allocator_report_totals = GpuAllocatorHud::default();
        self.allocator_report_last_refresh = None;
    }
}
