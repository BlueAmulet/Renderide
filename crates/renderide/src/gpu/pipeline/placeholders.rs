//! Placeholder pipelines for material and PBR (reserved for future use).

use super::core::RenderPipeline;

/// Material-based pipeline stub. Reserved for future use.
pub struct MaterialPipeline;

impl MaterialPipeline {
    /// Creates a material pipeline stub.
    pub fn new(_device: &wgpu::Device, _config: &wgpu::SurfaceConfiguration) -> Self {
        Self
    }
}

impl RenderPipeline for MaterialPipeline {
    fn bind(
        &self,
        _pass: &mut wgpu::RenderPass,
        _batch_index: Option<u32>,
        _frame_index: u64,
        _draw_bind_group: Option<&wgpu::BindGroup>,
    ) {
        // Stub: no-op
    }
}

/// PBR pipeline stub. Reserved for future use.
pub struct PbrPipeline;

impl PbrPipeline {
    /// Creates a PBR pipeline stub.
    pub fn new(_device: &wgpu::Device, _config: &wgpu::SurfaceConfiguration) -> Self {
        Self
    }
}

impl RenderPipeline for PbrPipeline {
    fn bind(
        &self,
        _pass: &mut wgpu::RenderPass,
        _batch_index: Option<u32>,
        _frame_index: u64,
        _draw_bind_group: Option<&wgpu::BindGroup>,
    ) {
        // Stub: no-op
    }
}
