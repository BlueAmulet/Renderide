//! Renderide: host–renderer IPC, window loop, and GPU presentation (skeleton).
//!
//! The library exposes [`run`] for the `renderide` binary. Shared IPC types live in [`shared`] and
//! are generated; do not edit `shared/shared.rs` by hand.

pub mod app;
pub mod assets;
pub mod connection;
pub mod gpu;
pub mod ipc;
pub mod materials;
pub mod pipelines;
pub mod present;
pub mod resources;
pub mod runtime;
pub mod scene;

pub mod shared;

pub use assets::material::{
    parse_materials_update_batch_into_store, MaterialBatchBlobLoader, MaterialDictionary,
    MaterialPropertyLookupIds, MaterialPropertySemanticHook, MaterialPropertyStore,
    MaterialPropertyValue, ParseMaterialBatchOptions, PropertyIdRegistry,
};
pub use connection::{
    get_connection_parameters, try_claim_renderer_singleton, ConnectionParams, InitError,
    DEFAULT_QUEUE_CAPACITY,
};
pub use gpu::MeshPreprocessPipelines;
pub use ipc::DualQueueIpc;
pub use materials::{
    compose_wgsl, MaterialFamilyId, MaterialPipelineCache, MaterialPipelineCacheKey,
    MaterialPipelineDesc, MaterialPipelineFamily, MaterialRegistry, MaterialRouter,
    SolidColorFamily, WgslPatch, SOLID_COLOR_FAMILY_ID,
};
pub use resources::{
    GpuResource, GpuTexture2d, MeshPool, MeshResidencyMeta, NoopStreamingPolicy, ResidencyTier,
    StreamingPolicy, TexturePool, TextureResidencyMeta, VramAccounting, VramResourceKind,
};
pub use runtime::{InitState, RendererRuntime};
pub use scene::{
    MeshMaterialSlot, RenderSpaceId, SceneCoordinator, SkinnedMeshRenderer, StaticMeshRenderer,
    TransformRemovalEvent,
};

/// Runs the renderer process: logging, optional IPC, winit loop, and wgpu presentation.
///
/// Returns [`None`] when the event loop exits without a host-requested exit code; otherwise
/// returns an exit code for [`std::process::exit`].
pub fn run() -> Option<i32> {
    app::run()
}
