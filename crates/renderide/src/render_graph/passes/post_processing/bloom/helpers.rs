//! Small shared helpers for the bloom pass implementations.

use std::num::NonZeroU32;

use crate::render_graph::compiled::RenderPassTemplate;
use crate::render_graph::context::{GraphResolvedResources, RasterPassCtx};

/// Returns `NonZeroU32::new(3)` (both stereo layers) when the current frame is multiview stereo,
/// otherwise forwards the template's preset. Matches the policy used by
/// [`super::super::aces_tonemap::AcesTonemapPass::multiview_mask_override`].
pub(super) fn stereo_mask_override(
    ctx: &RasterPassCtx<'_, '_>,
    template: &RenderPassTemplate,
) -> Option<NonZeroU32> {
    let stereo = ctx
        .frame
        .as_ref()
        .is_some_and(|frame| frame.view.multiview_stereo);
    if stereo {
        NonZeroU32::new(3)
    } else {
        template.multiview_mask
    }
}

/// Resolves the color attachment format for a transient handle; falls back to the bloom texture
/// format (`Rg11b10Ufloat`) when the handle has no current mapping (graph build error).
pub(super) fn attachment_format(
    graph_resources: &GraphResolvedResources,
    handle: crate::render_graph::resources::TextureHandle,
) -> wgpu::TextureFormat {
    graph_resources
        .transient_texture(handle)
        .map(|t| t.texture.format())
        .unwrap_or(wgpu::TextureFormat::Rg11b10Ufloat)
}
