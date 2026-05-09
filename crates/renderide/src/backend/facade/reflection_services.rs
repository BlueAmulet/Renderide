//! Reflection-probe services owned behind the backend facade.

use crate::backend::AssetTransferQueue;
use crate::backend::frame_gpu::ReflectionProbeSpecularResources;
use crate::gpu::GpuContext;
use crate::ipc::SharedMemoryAccessor;
use crate::materials::MaterialSystem;
use crate::reflection_probes::ReflectionProbeSh2System;
use crate::reflection_probes::specular::{
    ReflectionProbeFrameSelection, ReflectionProbeSpecularSystem, RuntimeReflectionProbeCapture,
};
use crate::scene::SceneCoordinator;
use crate::shared::{FrameSubmitData, RenderingContext};

/// Nonblocking reflection-probe projection, bake, cache, and selection services.
pub(super) struct ReflectionProbeServices {
    /// Nonblocking reflection-probe SH2 GPU projection service.
    sh2: ReflectionProbeSh2System,
    /// Reflection-probe specular IBL bake/cache/selection system.
    specular: ReflectionProbeSpecularSystem,
}

impl ReflectionProbeServices {
    /// Creates empty reflection-probe services.
    pub(super) fn new() -> Self {
        Self {
            sh2: ReflectionProbeSh2System::new(),
            specular: ReflectionProbeSpecularSystem::new(),
        }
    }

    /// Starts SH2 projection pipeline builds early so first probe use does not discover them lazily.
    pub(super) fn pre_warm_sh2_projection_pipelines(
        &mut self,
        device: &std::sync::Arc<wgpu::Device>,
    ) {
        self.sh2.pre_warm_projection_pipelines(device);
    }

    /// Answers host SH2 task rows for the latest frame submit without blocking GPU readback.
    pub(super) fn answer_sh2_frame_submit_tasks(
        &mut self,
        shm: &mut SharedMemoryAccessor,
        scene: &SceneCoordinator,
        materials: &MaterialSystem,
        asset_transfers: &AssetTransferQueue,
        data: &FrameSubmitData,
    ) {
        let captures = self.specular.capture_store();
        self.sh2
            .answer_frame_submit_tasks(shm, scene, materials, asset_transfers, captures, data);
    }

    /// Advances nonblocking SH2 GPU jobs and schedules queued projection work.
    pub(super) fn maintain_sh2_jobs(
        &mut self,
        gpu: &mut GpuContext,
        asset_transfers: &AssetTransferQueue,
    ) {
        self.sh2.maintain_gpu_jobs(gpu, asset_transfers);
    }

    /// Advances reflection-probe specular IBL jobs and returns frame-global probe bindings.
    pub(super) fn maintain_specular_jobs(
        &mut self,
        gpu: &mut GpuContext,
        scene: &SceneCoordinator,
        materials: &MaterialSystem,
        asset_transfers: &AssetTransferQueue,
        render_context: RenderingContext,
    ) -> Option<ReflectionProbeSpecularResources> {
        self.specular.maintain(
            gpu,
            scene,
            materials,
            asset_transfers,
            render_context,
            &mut self.sh2,
        );
        self.specular.resources()
    }

    /// CPU selection snapshot used by draw collection.
    pub(super) fn selection(&self) -> &ReflectionProbeFrameSelection {
        self.specular.selection()
    }

    /// Registers a completed OnChanges runtime cubemap capture.
    pub(super) fn register_runtime_capture(&mut self, capture: RuntimeReflectionProbeCapture) {
        self.specular.register_runtime_capture(capture);
    }
}
