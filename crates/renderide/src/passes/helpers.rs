//! Shared setup and error helpers for concrete render-graph passes.

use crate::render_graph::error::RenderPassError;
use crate::render_graph::pass::PassBuilder;
use crate::render_graph::resources::{
    ImportedTextureHandle, TextureAccess, TextureHandle, TextureResourceHandle,
};

/// Declares a transient texture read by a fragment shader.
pub(super) fn read_fragment_sampled_texture(b: &mut PassBuilder<'_>, handle: TextureHandle) {
    b.read_texture_resource(
        handle,
        TextureAccess::Sampled {
            stages: wgpu::ShaderStages::FRAGMENT,
        },
    );
}

/// Declares a color attachment write with no resolve target.
pub(super) fn color_attachment(
    b: &mut PassBuilder<'_>,
    handle: impl Into<TextureResourceHandle>,
    load: wgpu::LoadOp<wgpu::Color>,
) {
    let mut r = b.raster();
    r.color(
        handle,
        wgpu::Operations {
            load,
            store: wgpu::StoreOp::Store,
        },
        Option::<TextureHandle>::None,
    );
}

/// Declares an imported color attachment write with no resolve target.
pub(super) fn imported_color_attachment(
    b: &mut PassBuilder<'_>,
    handle: ImportedTextureHandle,
    load: wgpu::LoadOp<wgpu::Color>,
) {
    let mut r = b.raster();
    r.color(
        handle,
        wgpu::Operations {
            load,
            store: wgpu::StoreOp::Store,
        },
        Option::<ImportedTextureHandle>::None,
    );
}

/// Builds a missing-frame-params error with pass-specific context.
pub(super) fn missing_pass_resource(pass: &str, detail: impl std::fmt::Display) -> RenderPassError {
    RenderPassError::MissingFrameParams {
        pass: format!("{pass} ({detail})"),
    }
}
