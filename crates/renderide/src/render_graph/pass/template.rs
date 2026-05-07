//! Render-pass attachment templates produced during graph setup and consumed while recording.

use crate::render_graph::resources::{TextureAttachmentResolve, TextureAttachmentTarget};

/// Compiled render-pass attachment template.
#[derive(Clone, Debug)]
pub struct RenderPassTemplate {
    /// Color attachments in declaration order.
    pub color_attachments: Vec<ColorAttachmentTemplate>,
    /// Optional depth/stencil attachment.
    pub depth_stencil_attachment: Option<DepthAttachmentTemplate>,
    /// Optional multiview mask.
    pub multiview_mask: Option<std::num::NonZeroU32>,
}

/// Color attachment template.
#[derive(Clone, Debug)]
pub struct ColorAttachmentTemplate {
    /// Color target handle.
    pub target: TextureAttachmentTarget,
    /// Load operation.
    pub load: wgpu::LoadOp<wgpu::Color>,
    /// Store operation.
    pub store: wgpu::StoreOp,
    /// Optional resolve target.
    pub resolve_to: Option<TextureAttachmentResolve>,
}

/// Depth/stencil attachment template.
#[derive(Clone, Debug)]
pub struct DepthAttachmentTemplate {
    /// Depth/stencil target handle.
    pub target: TextureAttachmentTarget,
    /// Depth operations.
    pub depth: wgpu::Operations<f32>,
    /// Optional stencil operations.
    pub stencil: Option<wgpu::Operations<u32>>,
}
