//! Cached pipelines and bind layout for [`super::AcesTonemapPass`].
//!
//! Delegates to the shared [`FullscreenD2ArraySampledPipelineCache`]: one filterable D2-array
//! texture + linear-clamp sampler, mono/multiview pipelines keyed by output format. WGSL is
//! sourced from the build-time embedded shader registry.

use std::sync::Arc;

use crate::embedded_shaders::embedded_wgsl;
use crate::passes::helpers::{
    FullscreenD2ArrayPipelineLabels, FullscreenD2ArraySampledPipelineCache,
    FullscreenD2ArrayShaders,
};

/// Upper bound for cached ACES bind groups before the cache is flushed.
///
/// The scene-color transient texture is stable across most frames -- the cache normally holds
/// one or two entries (mono + multiview). This cap protects against unbounded growth when the
/// swapchain / MSAA setting flips repeatedly and the transient pool cycles allocations.
const MAX_CACHED_BIND_GROUPS: usize = 8;

const LABELS: FullscreenD2ArrayPipelineLabels = FullscreenD2ArrayPipelineLabels {
    base: "aces_tonemap",
    sampled_view: "aces_tonemap_sampled",
};

/// GPU state shared by all ACES tonemap passes (bind layout + sampler + per-format pipelines).
pub(super) struct AcesTonemapPipelineCache(FullscreenD2ArraySampledPipelineCache);

impl Default for AcesTonemapPipelineCache {
    fn default() -> Self {
        Self(FullscreenD2ArraySampledPipelineCache::new(
            LABELS,
            FullscreenD2ArrayShaders {
                mono_label: "aces_tonemap_default",
                mono_source: embedded_wgsl!("aces_tonemap_default"),
                multiview_label: "aces_tonemap_multiview",
                multiview_source: embedded_wgsl!("aces_tonemap_multiview"),
            },
            MAX_CACHED_BIND_GROUPS,
        ))
    }
}

impl AcesTonemapPipelineCache {
    /// Returns or builds a render pipeline for `output_format` and multiview stereo.
    pub(super) fn pipeline(
        &self,
        device: &wgpu::Device,
        output_format: wgpu::TextureFormat,
        multiview_stereo: bool,
    ) -> Arc<wgpu::RenderPipeline> {
        self.0.pipeline(device, output_format, multiview_stereo)
    }

    /// Bind group for one frame's scene-color texture, cached by `(Texture, multiview_stereo)`.
    pub(super) fn bind_group(
        &self,
        device: &wgpu::Device,
        scene_color_texture: &wgpu::Texture,
        multiview_stereo: bool,
    ) -> wgpu::BindGroup {
        self.0
            .bind_group(device, scene_color_texture, multiview_stereo, || {
                crate::profiling::note_resource_churn!(
                    BindGroup,
                    "passes::aces_tonemap_bind_group"
                );
            })
    }
}
