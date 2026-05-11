//! Cached pipelines and bind layout for [`super::SceneColorComposePass`].
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

/// Upper bound for cached scene-color-compose bind groups before the cache is flushed.
///
/// Normally one or two entries (mono + multiview). The cap protects against unbounded growth
/// when the transient pool cycles allocations (resize / MSAA toggle).
const MAX_CACHED_BIND_GROUPS: usize = 8;

const LABELS: FullscreenD2ArrayPipelineLabels = FullscreenD2ArrayPipelineLabels {
    base: "scene_color_compose",
    sampled_view: "scene_color_compose_sampled",
};

/// GPU state shared by all compose passes (bind layout + sampler + per-format pipelines).
pub(super) struct SceneColorComposePipelineCache(FullscreenD2ArraySampledPipelineCache);

impl Default for SceneColorComposePipelineCache {
    fn default() -> Self {
        Self(FullscreenD2ArraySampledPipelineCache::new(
            LABELS,
            FullscreenD2ArrayShaders {
                mono_label: "scene_color_compose_default",
                mono_source: embedded_wgsl!("scene_color_compose_default"),
                multiview_label: "scene_color_compose_multiview",
                multiview_source: embedded_wgsl!("scene_color_compose_multiview"),
            },
            MAX_CACHED_BIND_GROUPS,
        ))
    }
}

impl SceneColorComposePipelineCache {
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
                    "passes::scene_color_compose_bind_group"
                );
            })
    }
}
