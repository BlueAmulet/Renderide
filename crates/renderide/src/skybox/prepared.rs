//! Prepared skybox draws produced during world-mesh forward preparation.

use std::sync::Arc;

/// Prepared draw that fills the forward color target after opaque world meshes.
pub enum PreparedSkybox {
    /// Host material-driven skybox draw.
    Material(PreparedMaterialSkybox),
    /// Solid color background for host cameras using `CameraClearMode::Color`.
    ClearColor(PreparedClearColorSkybox),
}

/// Prepared material-driven skybox resources.
pub struct PreparedMaterialSkybox {
    /// Cached render pipeline for the skybox family and view target layout.
    pub pipeline: Arc<wgpu::RenderPipeline>,
    /// `@group(1)` material bind group resolved from the host material store.
    pub material_bind_group: Arc<wgpu::BindGroup>,
    /// Dynamic offset for the material uniform arena, when the skybox material has one.
    pub material_uniform_dynamic_offset: Option<u32>,
    /// `@group(2)` draw-local skybox view uniform bind group.
    pub view_bind_group: Arc<wgpu::BindGroup>,
    /// Vertex count used by the fullscreen skybox draw.
    pub vertex_count: u32,
}

/// Prepared solid-color background resources.
pub struct PreparedClearColorSkybox {
    /// Cached render pipeline for the color background draw.
    pub pipeline: Arc<wgpu::RenderPipeline>,
    /// `@group(0)` bind group carrying the background color uniform.
    pub view_bind_group: Arc<wgpu::BindGroup>,
}
