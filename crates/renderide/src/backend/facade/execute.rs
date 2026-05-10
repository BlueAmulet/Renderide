//! Compiled render graph execution (multiview entry point).

use crate::gpu::GpuContext;
use crate::render_graph::{FrameView, GraphExecuteError};
use crate::scene::SceneCoordinator;

use super::RenderBackend;

impl RenderBackend {
    /// Clears mapped-buffer owners after wgpu reports that mapped staging/readback buffers are invalid.
    pub(crate) fn reset_mapped_buffer_recovery_state(&mut self, generation: u64, source: &str) {
        logger::warn!(
            "backend mapped-buffer recovery: generation={generation} source={source} resetting upload arena and Hi-Z readbacks"
        );
        self.graph_state.reset_upload_arena();
        self.occlusion.clear_pending_hi_z_readbacks();
    }

    /// Unified multi-view entry: one Hi-Z readback (unless skipped), one encoder, one submit.
    ///
    /// When `skip_hi_z_begin_readback` is `false`, drains Hi-Z `map_async` readbacks first
    /// ([`crate::occlusion::OcclusionSystem::hi_z_begin_frame_readback`]). Set to `true` when the
    /// caller already invoked readback this tick (e.g. the runtime drains Hi-Z once at the top
    /// of the app driver's redraw tick via
    /// [`crate::runtime::RendererRuntime::drain_hi_z_readback`]).
    ///
    /// `views` is not consumed; callers can clear and repopulate the same [`Vec`] each frame to
    /// retain capacity. Each [`FrameView`] routes to its own target -- desktop swapchain, external
    /// OpenXR multiview, or host render-texture offscreen -- without changing the backend entry
    /// point.
    pub fn execute_multi_view_frame(
        &mut self,
        gpu: &mut GpuContext,
        scene: &SceneCoordinator,
        views: &mut Vec<FrameView<'_>>,
        skip_hi_z_begin_readback: bool,
    ) -> Result<(), GraphExecuteError> {
        profiling::scope!("backend::execute_multi_view_frame");
        if !skip_hi_z_begin_readback {
            let mapped_buffer_recovery = gpu.begin_mapped_buffer_recovery_frame();
            if mapped_buffer_recovery.invalidated {
                self.reset_mapped_buffer_recovery_state(
                    mapped_buffer_recovery.generation,
                    "frame begin",
                );
            }
            if !mapped_buffer_recovery.avoid_mapped_buffers {
                self.hi_z_begin_frame_readback(gpu.device());
                if gpu.observe_mapped_buffer_invalidation_during_frame() {
                    self.reset_mapped_buffer_recovery_state(
                        gpu.mapped_buffer_invalidation_generation(),
                        "Hi-Z readback",
                    );
                }
            }
        }
        self.graph_state.history_registry_mut().advance_frame();
        // Live HUD edits to `[post_processing]` only take effect when the graph is rebuilt; check
        // each tick so signature flips (effect added or removed) take effect on the next frame.
        // Parameter-only edits do not flip the signature and avoid the rebuild cost.
        let multiview_stereo = views.iter().any(FrameView::is_multiview_stereo_active);
        self.ensure_frame_graph_in_sync(multiview_stereo);
        let Some(mut graph) = self.graph_state.frame_graph_cache.take_graph() else {
            return Err(GraphExecuteError::NoFrameGraph);
        };
        let res = {
            let mut backend_access = self.graph_access();
            graph.execute_multi_view(gpu, scene, &mut backend_access, views.as_mut_slice())
        };
        self.graph_state.frame_graph_cache.restore_graph(graph);
        res
    }
}
