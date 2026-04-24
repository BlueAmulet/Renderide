//! Bloom upsample pass: 3×3 tent filter blended into the target mip with a per-pass blend factor.

use std::num::NonZeroU32;

use super::helpers::{attachment_format, stereo_mask_override};
use super::pipeline::{BloomPipelineCache, BloomPipelineKind};
use crate::config::BloomCompositeMode;
use crate::render_graph::compiled::RenderPassTemplate;
use crate::render_graph::context::RasterPassCtx;
use crate::render_graph::error::{RenderPassError, SetupError};
use crate::render_graph::pass::{PassBuilder, RasterPass};
use crate::render_graph::resources::{TextureAccess, TextureHandle};

/// Reads bloom mip `i` (input) and blends into bloom mip `i-1` (output) using a constant-factor
/// blend whose strength is configured via [`wgpu::RenderPass::set_blend_constant`]. The factor
/// is precomputed on CPU (see [`super::BloomEffect::register`]) so the shader itself only
/// produces the scattered sample — the blend unit handles the low-frequency boost curve and
/// composite-mode math.
pub(super) struct BloomUpsamplePass {
    input: TextureHandle,
    output: TextureHandle,
    blend_constant: f32,
    composite_mode: BloomCompositeMode,
    pipelines: &'static BloomPipelineCache,
}

impl BloomUpsamplePass {
    pub(super) fn new(
        input: TextureHandle,
        output: TextureHandle,
        blend_constant: f32,
        composite_mode: BloomCompositeMode,
        pipelines: &'static BloomPipelineCache,
    ) -> Self {
        Self {
            input,
            output,
            blend_constant: blend_constant.clamp(0.0, 1.0),
            composite_mode,
            pipelines,
        }
    }
}

impl RasterPass for BloomUpsamplePass {
    fn name(&self) -> &str {
        "BloomUpsample"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        b.read_texture_resource(
            self.input,
            TextureAccess::Sampled {
                stages: wgpu::ShaderStages::FRAGMENT,
            },
        );
        let mut r = b.raster();
        // Upsample blends into the target mip; load the existing contents so the blend unit can
        // combine `src * C` with `dst * (1-C)` or `dst * 1` depending on composite mode.
        r.color(
            self.output,
            wgpu::Operations {
                load: wgpu::LoadOp::Load,
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
        stereo_mask_override(ctx, template)
    }

    fn record(
        &self,
        ctx: &mut RasterPassCtx<'_, '_>,
        rpass: &mut wgpu::RenderPass<'_>,
    ) -> Result<(), RenderPassError> {
        profiling::scope!("post_processing::bloom::upsample");
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
        let Some(input_tex) = graph_resources.transient_texture(self.input) else {
            return Err(RenderPassError::MissingFrameParams {
                pass: format!("{} (missing transient input)", self.name()),
            });
        };
        let multiview_stereo = frame.view.multiview_stereo;
        let output_format = attachment_format(graph_resources, self.output);

        let kind = match self.composite_mode {
            BloomCompositeMode::EnergyConserving => BloomPipelineKind::UpsampleEnergyConserving,
            BloomCompositeMode::Additive => BloomPipelineKind::UpsampleAdditive,
        };
        let pipeline = self
            .pipelines
            .pipeline(ctx.device, kind, output_format, multiview_stereo);
        let bind_group =
            self.pipelines
                .group0_bind_group(ctx.device, &input_tex.texture, multiview_stereo);
        rpass.set_pipeline(pipeline.as_ref());
        rpass.set_bind_group(0, &bind_group, &[]);
        let c = f64::from(self.blend_constant);
        rpass.set_blend_constant(wgpu::Color {
            r: c,
            g: c,
            b: c,
            a: c,
        });
        rpass.draw(0..3, 0..1);
        Ok(())
    }
}
