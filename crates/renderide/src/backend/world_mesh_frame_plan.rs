//! Backend-owned world-mesh forward frame planning.

use std::sync::Arc;

use crate::diagnostics::PerViewHudOutputs;
use crate::gpu::GpuLimits;
use crate::passes::{
    PreparedWorldMeshForwardFrame, WorldMeshForwardPrepareContext, WorldMeshForwardSkyboxRenderer,
    prepare_world_mesh_forward_frame,
};
use crate::render_graph::WorldMeshDrawPlan;
use crate::render_graph::frame_params::{GraphPassFrame, PerViewFramePlan};
use crate::render_graph::frame_upload_batch::FrameUploadBatch;
use crate::world_mesh::PrefetchedWorldMeshViewDraws;

/// Backend-owned world-mesh forward preparation caches.
pub(crate) struct BackendWorldMeshFramePlanner {
    /// Skybox/background preparation cache shared across frame plans.
    skybox: WorldMeshForwardSkyboxRenderer,
}

/// Per-view world-mesh packet prepared before graph pass recording.
pub(crate) struct WorldMeshPreparedView {
    /// Forward draw state consumed by graph raster and helper passes.
    pub(crate) prepared: Option<PreparedWorldMeshForwardFrame>,
    /// Optional HUD output produced while building this view's draw packet.
    pub(crate) hud_outputs: Option<PerViewHudOutputs>,
}

/// Ordered backend-owned world-mesh plan for one graph frame.
pub(crate) struct WorldMeshFramePlan {
    /// Prepared world-mesh packets in submitted view order.
    views: Vec<WorldMeshPreparedView>,
}

impl BackendWorldMeshFramePlanner {
    /// Creates an empty world-mesh frame planner.
    pub(crate) fn new() -> Self {
        Self {
            skybox: WorldMeshForwardSkyboxRenderer::default(),
        }
    }

    /// Releases view-scoped cached planning resources.
    pub(crate) fn release_view_resources(&self, retired_views: &[crate::camera::ViewId]) {
        self.skybox.release_view_resources(retired_views);
    }

    /// Builds one per-view world-mesh packet from an explicit draw plan.
    pub(crate) fn prepare_view(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        upload_batch: &FrameUploadBatch,
        gpu_limits: &GpuLimits,
        frame: &GraphPassFrame<'_>,
        inputs: WorldMeshPrepareViewInputs<'_>,
    ) -> WorldMeshPreparedView {
        let frame_plan = PerViewFramePlan {
            frame_bind_group: Arc::clone(inputs.frame_plan.frame_bind_group),
            frame_uniform_buffer: inputs.frame_plan.frame_uniform_buffer.clone(),
            view_idx: inputs.frame_plan.view_idx,
        };
        let prefetched = match inputs.draw_plan {
            WorldMeshDrawPlan::Prefetched(draws) => *draws,
            WorldMeshDrawPlan::Empty => PrefetchedWorldMeshViewDraws::empty(),
        };
        let prepared = prepare_world_mesh_forward_frame(
            WorldMeshForwardPrepareContext {
                device,
                queue,
                upload_batch,
                gpu_limits,
                frame,
                frame_plan: &frame_plan,
                skybox_renderer: &self.skybox,
            },
            prefetched,
        );
        WorldMeshPreparedView {
            prepared: prepared.prepared,
            hud_outputs: prepared.hud_outputs,
        }
    }
}

impl Default for BackendWorldMeshFramePlanner {
    fn default() -> Self {
        Self::new()
    }
}

impl WorldMeshFramePlan {
    /// Creates an empty frame plan with capacity for `capacity` views.
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            views: Vec::with_capacity(capacity),
        }
    }

    /// Appends a prepared per-view packet.
    pub(crate) fn push(&mut self, view: WorldMeshPreparedView) {
        self.views.push(view);
    }

    /// Consumes the frame plan and returns its ordered per-view packets.
    pub(crate) fn into_views(self) -> Vec<WorldMeshPreparedView> {
        self.views
    }
}

/// Per-view world-mesh planning inputs beyond shared GPU/frame context.
pub(crate) struct WorldMeshPrepareViewInputs<'a> {
    /// Per-view frame bind resources.
    pub(crate) frame_plan: PerViewFramePlanInputs<'a>,
    /// Explicit draw plan for this view.
    pub(crate) draw_plan: WorldMeshDrawPlan,
}

/// Per-view frame bind inputs used while preparing world-mesh frame data.
pub(crate) struct PerViewFramePlanInputs<'a> {
    /// Frame bind group that will be bound at `@group(0)` for this view.
    pub(crate) frame_bind_group: &'a Arc<wgpu::BindGroup>,
    /// Frame uniform buffer backing `frame_bind_group`.
    pub(crate) frame_uniform_buffer: &'a wgpu::Buffer,
    /// Index of this view in the submitted multi-view batch.
    pub(crate) view_idx: usize,
}
