//! Cooperative renderer shutdown hooks.

use crate::diagnostics::crash_context::{self, TickPhase};

use super::RendererRuntime;

impl RendererRuntime {
    /// Starts nonblocking teardown for runtime-owned resources that can outlive the frame loop.
    pub(crate) fn begin_graceful_shutdown(&mut self) {
        profiling::scope!("runtime::begin_graceful_shutdown");
        crash_context::set_tick_phase(TickPhase::Shutdown);
        self.log_compact_renderer_summary("graceful-shutdown-begin");
        self.backend.begin_video_shutdown();
    }

    /// Returns `true` once runtime-owned shutdown work has quiesced.
    pub(crate) fn graceful_shutdown_complete(&mut self) -> bool {
        profiling::scope!("runtime::graceful_shutdown_complete");
        self.backend.video_shutdown_complete()
    }

    /// Emits one compact state line for shutdown and fatal boundaries.
    pub(crate) fn log_compact_renderer_summary(&self, reason: &'static str) {
        let (primary_drop_streak, background_drop_streak) =
            self.frontend.ipc_consecutive_outbound_drop_streaks();
        logger::info!(
            "Renderer summary reason={reason}: last_host_frame={} init_state={:?} graph_passes={} graph_topo_levels={} pending_asset_work={} ipc_drop_streaks=primary:{} background:{} active_render_spaces={} mesh_renderables={} pending_camera_readbacks={} pending_reflection_probe_tasks={} completed_camera_readbacks={} failed_camera_readbacks={} frame_submit_apply_failures={} {}",
            self.host_camera.frame_index,
            self.frontend.init_state(),
            self.backend.frame_graph_pass_count(),
            self.backend.frame_graph_topo_levels(),
            self.backend.has_pending_asset_work(),
            primary_drop_streak,
            background_drop_streak,
            self.scene.render_space_count(),
            self.scene.total_mesh_renderable_count(),
            self.diagnostics.pending_camera_readbacks,
            self.tick_state.pending_reflection_probe_render_tasks.len(),
            self.diagnostics.completed_camera_readbacks,
            self.diagnostics.failed_camera_readbacks,
            self.diagnostics.frame_submit_apply_failures,
            crash_context::format_snapshot().trim_end(),
        );
    }
}
