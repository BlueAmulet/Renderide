//! GPU allocator totals and full-report refresh state fragment of
//! [`super::FrameDiagnosticsSnapshot`].

use std::sync::Arc;

use crate::gpu::GpuContext;

/// Optional wgpu allocator totals when the backend exposes a report.
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuAllocatorHud {
    /// Sum of live allocation sizes from the device allocator report.
    pub allocated_bytes: Option<u64>,
    /// Reserved capacity including internal fragmentation.
    pub reserved_bytes: Option<u64>,
}

/// Full GPU allocator report plus sort order for the **GPU memory** HUD tab.
///
/// The report is refreshed on a timer in [`crate::runtime::RendererRuntime`] (not every frame);
/// [`GpuAllocatorHud`] totals on the **Stats** tab are still sampled each capture via
/// [`GpuContext::gpu_allocator_bytes`].
#[derive(Clone, Debug)]
pub struct GpuAllocatorReportHud {
    /// Live [`wgpu::Device::generate_allocator_report`] payload when the backend supports it.
    pub report: Arc<wgpu::AllocatorReport>,
    /// Indices into [`wgpu::AllocatorReport::allocations`], sorted by descending allocation size.
    pub allocation_indices_by_size: Arc<[usize]>,
}

/// Throttled full GPU allocator report and refresh timer for the **GPU memory** HUD tab.
#[derive(Clone, Debug, Default)]
pub struct GpuAllocatorHudRefresh {
    /// Live allocator report when supported (`None` before first refresh).
    pub gpu_allocator_report: Option<GpuAllocatorReportHud>,
    /// Seconds until the runtime replaces [`Self::gpu_allocator_report`] on the next capture.
    pub gpu_allocator_report_next_refresh_in_secs: f32,
}

/// GPU allocator fragment: per-tick totals plus the throttled full report.
///
/// Stats-tab totals ([`Self::totals`]) refresh every capture; the full report
/// ([`Self::report`]) follows the runtime's throttled refresh interval driven by
/// [`Self::report_next_refresh_in_secs`].
#[derive(Clone, Debug, Default)]
pub struct GpuAllocatorFragment {
    /// Per-tick allocated/reserved totals (refreshed every capture).
    pub totals: GpuAllocatorHud,
    /// Throttled full allocator report for the **GPU memory** tab (`None` if unsupported or
    /// before first refresh).
    pub report: Option<GpuAllocatorReportHud>,
    /// Seconds until the runtime replaces [`Self::report`] on the next capture.
    pub report_next_refresh_in_secs: f32,
}

impl GpuAllocatorFragment {
    /// Builds the fragment from current device totals plus the runtime's throttled refresh state.
    pub fn capture(gpu: &GpuContext, refresh: GpuAllocatorHudRefresh) -> Self {
        let (allocated_bytes, reserved_bytes) = gpu.gpu_allocator_bytes();
        Self {
            totals: GpuAllocatorHud {
                allocated_bytes,
                reserved_bytes,
            },
            report: refresh.gpu_allocator_report,
            report_next_refresh_in_secs: refresh.gpu_allocator_report_next_refresh_in_secs,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{GpuAllocatorFragment, GpuAllocatorHud};

    #[test]
    fn fragment_default_has_zero_refresh_countdown_and_no_report() {
        let f = GpuAllocatorFragment::default();
        assert_eq!(f.totals.allocated_bytes, None);
        assert_eq!(f.totals.reserved_bytes, None);
        assert!(f.report.is_none());
        assert_eq!(f.report_next_refresh_in_secs, 0.0);
    }

    #[test]
    fn fragment_can_carry_totals_independently_of_report() {
        let f = GpuAllocatorFragment {
            totals: GpuAllocatorHud {
                allocated_bytes: Some(123),
                reserved_bytes: Some(456),
            },
            report: None,
            report_next_refresh_in_secs: 1.5,
        };
        assert_eq!(f.totals.allocated_bytes, Some(123));
        assert_eq!(f.totals.reserved_bytes, Some(456));
        assert_eq!(f.report_next_refresh_in_secs, 1.5);
    }
}
