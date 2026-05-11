//! Tracy plots for asset-integration backlog and budget pressure.
//!
//! Plot names emitted here are an external contract with the Tracy GUI and dashboards; do not
//! rename them.

use super::tracy_plot::tracy_plot;

/// Asset-integration backlog and budget-exhaustion counters for one drain.
#[derive(Clone, Copy, Debug, Default)]
pub struct AssetIntegrationProfileSample {
    /// High-priority tasks still queued after the drain.
    pub high_priority_queued: usize,
    /// Normal-priority tasks still queued after the drain.
    pub normal_priority_queued: usize,
    /// Whether the high-priority emergency ceiling stopped the drain.
    pub high_priority_budget_exhausted: bool,
    /// Whether the normal-priority frame budget stopped the drain.
    pub normal_priority_budget_exhausted: bool,
}

/// Records asset-integration backlog and budget pressure for the current frame.
#[inline]
pub fn plot_asset_integration(sample: AssetIntegrationProfileSample) {
    tracy_plot!(
        "asset_integration::high_priority_queued",
        sample.high_priority_queued as f64
    );
    tracy_plot!(
        "asset_integration::normal_priority_queued",
        sample.normal_priority_queued as f64
    );
    tracy_plot!(
        "asset_integration::high_priority_budget_exhausted",
        if sample.high_priority_budget_exhausted {
            1.0
        } else {
            0.0
        }
    );
    tracy_plot!(
        "asset_integration::normal_priority_budget_exhausted",
        if sample.normal_priority_budget_exhausted {
            1.0
        } else {
            0.0
        }
    );
}
