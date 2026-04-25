//! Stage 3 of the GTAO sub-graph: applies the (optionally denoised) AO term to the chain's HDR
//! scene color and writes the chain output.
//!
//! Reads the chain input (HDR scene color), the AO-term texture left over by the main / denoise
//! pipeline, and the shared GTAO params UBO (`intensity` + `albedo_multibounce`). Carries the
//! Jiménez Eq. 10 multi-bounce fit that used to live inside the legacy single-pass GTAO shader;
//! placing it after the linear-space bilateral denoise keeps the filter's weighted average
//! meaningful and only re-introduces the nonlinear shaping at the very end.

use std::num::NonZeroU32;

use super::pipeline::{GtaoPipelineCache, GtaoStage};
use crate::render_graph::compiled::RenderPassTemplate;
use crate::render_graph::context::RasterPassCtx;
use crate::render_graph::error::{RenderPassError, SetupError};
use crate::render_graph::pass::{PassBuilder, RasterPass};
use crate::render_graph::resources::{TextureAccess, TextureHandle};

/// Graph handles for [`GtaoApplyPass`].
#[derive(Clone, Copy, Debug)]
pub(super) struct GtaoApplyResources {
    /// Sampled HDR scene color (chain input). Mono samples layer 0.
    pub scene_color: TextureHandle,
    /// Sampled AO term written by the main pass or the last denoise pass.
    pub ao_term: TextureHandle,
    /// Color attachment: the post-process chain's modulated HDR output.
    pub output: TextureHandle,
}

/// Fragment-shader pass that multiplies HDR scene color by the AO visibility factor.
pub(super) struct GtaoApplyPass {
    /// Graph handles for this pass instance.
    resources: GtaoApplyResources,
    /// Process-wide cached pipelines / bind-group layouts / sampler / params UBO.
    pipelines: &'static GtaoPipelineCache,
}

impl GtaoApplyPass {
    /// Constructs a new instance.
    pub(super) fn new(
        resources: GtaoApplyResources,
        pipelines: &'static GtaoPipelineCache,
    ) -> Self {
        Self {
            resources,
            pipelines,
        }
    }
}

impl RasterPass for GtaoApplyPass {
    fn name(&self) -> &str {
        "GtaoApply"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        b.read_texture_resource(
            self.resources.scene_color,
            TextureAccess::Sampled {
                stages: wgpu::ShaderStages::FRAGMENT,
            },
        );
        b.read_texture_resource(
            self.resources.ao_term,
            TextureAccess::Sampled {
                stages: wgpu::ShaderStages::FRAGMENT,
            },
        );
        let mut r = b.raster();
        r.color(
            self.resources.output,
            wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
            Option::<TextureHandle>::None,
        );
        Ok(())
    }

    fn multiview_mask_override(
        &self,
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

    fn record(
        &self,
        ctx: &mut RasterPassCtx<'_, '_>,
        rpass: &mut wgpu::RenderPass<'_>,
    ) -> Result<(), RenderPassError> {
        profiling::scope!("post_processing::gtao::apply");
        let Some(frame) = ctx.frame.as_ref() else {
            return Err(RenderPassError::MissingFrameParams {
                pass: self.name().to_string(),
            });
        };
        let Some(graph_resources) = ctx.graph_resources else {
            return Err(RenderPassError::MissingFrameParams {
                pass: self.name().to_string(),
            });
        };
        let Some(scene_tex) = graph_resources.transient_texture(self.resources.scene_color) else {
            return Err(RenderPassError::MissingFrameParams {
                pass: format!("{} (missing scene input)", self.name()),
            });
        };
        let Some(ao_tex) = graph_resources.transient_texture(self.resources.ao_term) else {
            return Err(RenderPassError::MissingFrameParams {
                pass: format!("{} (missing ao term)", self.name()),
            });
        };

        let multiview_stereo = frame.view.multiview_stereo;
        let output_format = graph_resources
            .transient_texture(self.resources.output)
            .map(|t| t.texture.format())
            .unwrap_or(wgpu::TextureFormat::Rgba16Float);

        let pipeline = self.pipelines.pipeline(
            ctx.device,
            GtaoStage::Apply,
            output_format,
            multiview_stereo,
        );
        let bind_group = self.pipelines.bind_group_apply(
            ctx.device,
            multiview_stereo,
            &scene_tex.texture,
            &ao_tex.texture,
        );
        rpass.set_pipeline(pipeline.as_ref());
        rpass.set_bind_group(0, &bind_group, &[]);
        rpass.draw(0..3, 0..1);
        Ok(())
    }
}
