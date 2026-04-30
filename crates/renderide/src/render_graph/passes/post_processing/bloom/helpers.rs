//! Small shared helpers for the bloom pass implementations.

use std::num::NonZeroU32;

use crate::render_graph::compiled::RenderPassTemplate;
use crate::render_graph::context::{GraphResolvedResources, RasterPassCtx};
use crate::render_graph::gpu_cache::raster_stereo_mask_override;

/// Returns `NonZeroU32::new(3)` (both stereo layers) when the current frame is multiview stereo,
/// otherwise forwards the template's preset. Matches the policy used by
/// [`super::super::aces_tonemap::AcesTonemapPass::multiview_mask_override`].
pub(super) fn stereo_mask_override(
    ctx: &RasterPassCtx<'_, '_>,
    template: &RenderPassTemplate,
) -> Option<NonZeroU32> {
    raster_stereo_mask_override(ctx, template)
}

/// Resolves the color attachment format for a transient handle; falls back to the bloom texture
/// format (`Rg11b10Ufloat`) when the handle has no current mapping (graph build error).
pub(super) fn attachment_format(
    graph_resources: &GraphResolvedResources,
    handle: crate::render_graph::resources::TextureHandle,
) -> wgpu::TextureFormat {
    graph_resources
        .transient_texture(handle)
        .map_or(wgpu::TextureFormat::Rg11b10Ufloat, |t| t.texture.format())
}
