//! GPU and host-facing resource layer: pools, material tables, uploads, preprocess pipelines.
//!
//! This module owns **wgpu** [`wgpu::Device`] / [`wgpu::Queue`], mesh and texture pools, the
//! [`MaterialPropertyStore`](crate::materials::host_data::MaterialPropertyStore), the compiled
//! [`CompiledRenderGraph`](crate::render_graph::CompiledRenderGraph) after attach, and code paths
//! that turn shared-memory asset payloads into resident GPU resources. [`light_gpu`](crate::backend::light_gpu)
//! packs scene [`ResolvedLight`](crate::scene::ResolvedLight) values for future storage-buffer upload. It does **not**
//! own IPC queues, [`SharedMemoryAccessor`](crate::ipc::SharedMemoryAccessor), or scene graph state;
//! callers pass those in where a command requires both transport and GPU work.

mod cluster_gpu;
mod debug_hud_bundle;
mod facade;
pub(crate) mod frame_gpu;
mod frame_gpu_bindings;
mod frame_gpu_error;
mod frame_resource_manager;
pub(crate) mod gpu_jobs;
mod history_registry;
mod light_gpu;
pub(crate) mod material_property_reader;
mod per_draw_resources;
mod per_view_resource_map;
mod view_resource_registry;
mod world_mesh_frame_plan;

pub use cluster_gpu::{CLUSTER_PARAMS_UNIFORM_SIZE, MAX_LIGHTS_PER_TILE};
pub(crate) use facade::{BackendGraphAccess, ExtractedFrameShared, WorldMeshForwardEncodeRefs};
pub use facade::{RenderBackend, RenderBackendAttachDesc};
pub use frame_gpu::{FrameGpuResources, empty_material_bind_group_layout};
pub use frame_gpu_bindings::FrameGpuBindingsError;
pub use frame_resource_manager::{FrameResourceManager, PreRecordViewResourceLayout};
pub(crate) use gpu_jobs::{
    GpuJobResources, GpuReadbackJobs, GpuReadbackOutcomes, SubmittedReadbackJob,
};
pub use history_registry::{
    HistoryRegistry, HistoryRegistryError, HistoryResourceScope, HistoryTextureMipViews,
    TextureHistorySpec,
};
pub use light_gpu::GpuLight;
pub(crate) use view_resource_registry::ViewResourceRegistry;
pub(crate) use world_mesh_frame_plan::{
    BackendWorldMeshFramePlanner, PerViewFramePlanInputs, WorldMeshFramePlan,
    WorldMeshPrepareViewInputs, WorldMeshPreparedView,
};
