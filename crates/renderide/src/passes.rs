//! Concrete render passes registered on a [`crate::render_graph::CompiledRenderGraph`].
//!
//! Each pass implements one of the typed pass traits:
//! - [`crate::render_graph::pass::RasterPass`] -- raster render passes
//! - [`crate::render_graph::pass::ComputePass`] -- encoder-driven compute

mod clustered_light;
mod helpers;
mod hi_z_build;
mod mesh_deform;
pub mod post_processing;
mod scene_color_compose;
mod world_mesh_forward;

pub use clustered_light::{ClusteredLightGraphResources, ClusteredLightPass};
pub use hi_z_build::{HiZBuildGraphResources, HiZBuildPass};
pub use mesh_deform::MeshDeformPass;
pub use post_processing::{AcesTonemapEffect, AutoExposureEffect, BloomEffect, GtaoEffect};
pub use scene_color_compose::{SceneColorComposeGraphResources, SceneColorComposePass};
pub(crate) use world_mesh_forward::{
    GTAO_VIEW_NORMAL_FORMAT, PreparedWorldMeshForwardFrame, WorldMeshForwardPlanSlot,
    WorldMeshForwardPrepareContext, WorldMeshForwardSkyboxRenderer,
    prepare_world_mesh_forward_frame,
};
pub use world_mesh_forward::{
    WorldMeshColorSnapshotPass, WorldMeshDepthSnapshotPass,
    WorldMeshForwardColorResolveGraphResources, WorldMeshForwardColorResolvePass,
    WorldMeshForwardDepthResolvePass, WorldMeshForwardGraphResources,
    WorldMeshForwardIntersectPass, WorldMeshForwardNormalGraphResources,
    WorldMeshForwardNormalPass, WorldMeshForwardOpaquePass, WorldMeshForwardTransparentPass,
};
