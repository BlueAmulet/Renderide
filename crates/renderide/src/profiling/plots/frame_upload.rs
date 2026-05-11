//! Tracy plots for per-frame upload traffic and world-mesh batch compression.
//!
//! Plot names emitted here are an external contract with the Tracy GUI and dashboards; do not
//! rename them.

use super::tracy_plot::tracy_plot;

/// Records, per call to `crate::passes::world_mesh_forward::encode::draw_subset`,
/// how many instance batches and how many input draws were submitted in that subpass.
///
/// One sample lands on the Tracy timeline per opaque or intersection subpass record, so the
/// plot trace shows fragmentation visually: when batches ~= draws, the merge isn't compressing;
/// when batches << draws, instancing is collapsing same-mesh runs as intended. Pair with
/// [`crate::world_mesh::WorldMeshDrawStats::gpu_instances_emitted`] in the HUD for a
/// per-frame integral. Expands to nothing when the `tracy` feature is off.
#[inline]
pub fn plot_world_mesh_subpass(batches: usize, draws: usize) {
    tracy_plot!("world_mesh::subpass_batches", batches as f64);
    tracy_plot!("world_mesh::subpass_draws", draws as f64);
}

/// Records deferred queue-write traffic for one frame.
#[inline]
pub fn plot_frame_upload_batch(writes: usize, bytes: usize) {
    tracy_plot!("frame_upload::writes", writes as f64);
    tracy_plot!("frame_upload::bytes", bytes as f64);
}
