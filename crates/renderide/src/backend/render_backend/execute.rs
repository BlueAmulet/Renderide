//! Compiled render graph execution (desktop, multiview, offscreen).

use crate::gpu::GpuContext;
use crate::render_graph::{
    CameraTransformDrawFilter, CompiledRenderGraph, ExternalFrameTargets, ExternalOffscreenTargets,
    GraphExecuteError, WorldMeshDrawCollection,
};
use crate::scene::SceneCoordinator;
use winit::window::Window;

use super::RenderBackend;

impl RenderBackend {
    /// Runs `run` with a taken [`CompiledRenderGraph`], restoring it afterward. Invokes Hi-Z readback begin.
    fn with_compiled_graph<R>(
        &mut self,
        gpu: &mut GpuContext,
        run: impl FnOnce(
            &mut CompiledRenderGraph,
            &mut GpuContext,
            &mut RenderBackend,
        ) -> Result<R, GraphExecuteError>,
    ) -> Result<R, GraphExecuteError> {
        self.occlusion.hi_z_begin_frame_readback(gpu.device());
        let Some(mut graph) = self.frame_graph.take() else {
            return Err(GraphExecuteError::NoFrameGraph);
        };
        let res = run(&mut graph, gpu, self);
        self.frame_graph = Some(graph);
        res
    }

    /// Records and presents one frame using the compiled render graph (deform compute + forward mesh pass).
    ///
    /// Returns [`GraphExecuteError::NoFrameGraph`] if graph build failed during [`crate::backend::RenderBackend::attach`].
    pub fn execute_frame_graph(
        &mut self,
        gpu: &mut GpuContext,
        window: &Window,
        scene: &SceneCoordinator,
        host_camera: crate::render_graph::HostCameraFrame,
    ) -> Result<(), GraphExecuteError> {
        self.with_compiled_graph(gpu, |graph, gpu_ctx, backend| {
            graph.execute(gpu_ctx, window, scene, backend, host_camera)
        })
    }

    /// Renders the frame graph to pre-acquired OpenXR multiview array targets (no surface present).
    pub fn execute_frame_graph_external_multiview(
        &mut self,
        gpu: &mut GpuContext,
        window: &Window,
        scene: &SceneCoordinator,
        host_camera: crate::render_graph::HostCameraFrame,
        external: ExternalFrameTargets<'_>,
    ) -> Result<(), GraphExecuteError> {
        self.with_compiled_graph(gpu, |graph, gpu_ctx, backend| {
            graph.execute_external_multiview(gpu_ctx, window, scene, backend, host_camera, external)
        })
    }

    /// Renders the default graph to a single-view render texture (secondary camera).
    ///
    /// When `prefetched_world_mesh_draws` is [`Some`], the world mesh forward pass skips CPU draw
    /// collection and uses the provided list (see [`crate::render_graph::FrameRenderParams::prefetched_world_mesh_draws`]).
    #[allow(clippy::too_many_arguments)]
    pub fn execute_frame_graph_offscreen_single_view(
        &mut self,
        gpu: &mut GpuContext,
        window: &Window,
        scene: &SceneCoordinator,
        host_camera: crate::render_graph::HostCameraFrame,
        external: ExternalOffscreenTargets<'_>,
        transform_filter: Option<CameraTransformDrawFilter>,
        prefetched_world_mesh_draws: Option<WorldMeshDrawCollection>,
    ) -> Result<(), GraphExecuteError> {
        self.with_compiled_graph(gpu, |graph, gpu_ctx, backend| {
            graph.execute_offscreen_single_view(
                gpu_ctx,
                window,
                scene,
                backend,
                host_camera,
                external,
                transform_filter,
                prefetched_world_mesh_draws,
            )
        })
    }
}
