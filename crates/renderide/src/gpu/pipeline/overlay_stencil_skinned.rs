//! Skinned overlay stencil pipelines for GraphicsChunk masking.

use nalgebra::Matrix4;

use super::core::{RenderPipeline, UniformData};
use super::overlay_stencil::OverlayStencilPhase;
use super::skinned::SkinnedPipeline;

/// Skinned overlay with stencil for GraphicsChunk masking.
///
/// Same stencil semantics as [`super::overlay_stencil::OverlayStencilPipeline`] (Content phase). Call
/// `set_stencil_reference` before each draw.
pub struct OverlayStencilSkinnedPipeline {
    inner: SkinnedPipeline,
}

impl OverlayStencilSkinnedPipeline {
    /// Creates a skinned overlay stencil pipeline for the Content phase.
    pub fn new(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> Self {
        let inner =
            SkinnedPipeline::new(device, config, Some(OverlayStencilPhase::Content), false);
        Self { inner }
    }
}

/// Skinned overlay stencil MaskWrite pipeline.
pub struct OverlayStencilMaskWriteSkinnedPipeline {
    inner: SkinnedPipeline,
}

impl OverlayStencilMaskWriteSkinnedPipeline {
    /// Creates a skinned overlay stencil MaskWrite pipeline.
    pub fn new(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> Self {
        let inner =
            SkinnedPipeline::new(device, config, Some(OverlayStencilPhase::MaskWrite), false);
        Self { inner }
    }
}

/// Skinned overlay stencil MaskClear pipeline.
pub struct OverlayStencilMaskClearSkinnedPipeline {
    inner: SkinnedPipeline,
}

impl OverlayStencilMaskClearSkinnedPipeline {
    /// Creates a skinned overlay stencil MaskClear pipeline.
    pub fn new(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> Self {
        let inner =
            SkinnedPipeline::new(device, config, Some(OverlayStencilPhase::MaskClear), false);
        Self { inner }
    }
}

impl RenderPipeline for OverlayStencilMaskWriteSkinnedPipeline {
    fn bind(
        &self,
        pass: &mut wgpu::RenderPass,
        batch_index: Option<u32>,
        frame_index: u64,
        draw_bind_group: Option<&wgpu::BindGroup>,
    ) {
        self.inner.bind(pass, batch_index, frame_index, draw_bind_group);
    }

    fn create_skinned_draw_bind_group(
        &self,
        device: &wgpu::Device,
        buffers: &super::super::mesh::GpuMeshBuffers,
    ) -> Option<wgpu::BindGroup> {
        self.inner.create_skinned_draw_bind_group(device, buffers)
    }

    fn draw_skinned(
        &self,
        pass: &mut wgpu::RenderPass,
        buffers: &super::super::mesh::GpuMeshBuffers,
        uniforms: &UniformData<'_>,
    ) {
        self.inner.draw_skinned(pass, buffers, uniforms);
    }

    fn upload_skinned_batch(
        &self,
        queue: &wgpu::Queue,
        items: &[(Matrix4<f32>, &[[[f32; 4]; 4]], Option<&[f32]>, u32)],
        frame_index: u64,
    ) {
        self.inner.upload_skinned_batch(queue, items, frame_index);
    }
}

impl RenderPipeline for OverlayStencilMaskClearSkinnedPipeline {
    fn bind(
        &self,
        pass: &mut wgpu::RenderPass,
        batch_index: Option<u32>,
        frame_index: u64,
        draw_bind_group: Option<&wgpu::BindGroup>,
    ) {
        self.inner.bind(pass, batch_index, frame_index, draw_bind_group);
    }

    fn create_skinned_draw_bind_group(
        &self,
        device: &wgpu::Device,
        buffers: &super::super::mesh::GpuMeshBuffers,
    ) -> Option<wgpu::BindGroup> {
        self.inner.create_skinned_draw_bind_group(device, buffers)
    }

    fn draw_skinned(
        &self,
        pass: &mut wgpu::RenderPass,
        buffers: &super::super::mesh::GpuMeshBuffers,
        uniforms: &UniformData<'_>,
    ) {
        self.inner.draw_skinned(pass, buffers, uniforms);
    }

    fn upload_skinned_batch(
        &self,
        queue: &wgpu::Queue,
        items: &[(Matrix4<f32>, &[[[f32; 4]; 4]], Option<&[f32]>, u32)],
        frame_index: u64,
    ) {
        self.inner.upload_skinned_batch(queue, items, frame_index);
    }
}

impl RenderPipeline for OverlayStencilSkinnedPipeline {
    fn bind(
        &self,
        pass: &mut wgpu::RenderPass,
        batch_index: Option<u32>,
        frame_index: u64,
        draw_bind_group: Option<&wgpu::BindGroup>,
    ) {
        self.inner.bind(pass, batch_index, frame_index, draw_bind_group);
    }

    fn create_skinned_draw_bind_group(
        &self,
        device: &wgpu::Device,
        buffers: &super::super::mesh::GpuMeshBuffers,
    ) -> Option<wgpu::BindGroup> {
        self.inner.create_skinned_draw_bind_group(device, buffers)
    }

    fn draw_skinned(
        &self,
        pass: &mut wgpu::RenderPass,
        buffers: &super::super::mesh::GpuMeshBuffers,
        uniforms: &UniformData<'_>,
    ) {
        self.inner.draw_skinned(pass, buffers, uniforms);
    }

    fn upload_skinned_batch(
        &self,
        queue: &wgpu::Queue,
        items: &[(Matrix4<f32>, &[[[f32; 4]; 4]], Option<&[f32]>, u32)],
        frame_index: u64,
    ) {
        self.inner.upload_skinned_batch(queue, items, frame_index);
    }
}
