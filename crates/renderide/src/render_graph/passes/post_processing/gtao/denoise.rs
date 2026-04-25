//! Stage 2 of the GTAO sub-graph: bilateral denoise of the AO term.
//!
//! Each instance reads the previous AO term + the packed-edges side-channel and writes the next
//! AO term. Pipeline ping-pongs between two `R16Float` transients owned by the GTAO effect; the
//! parity-correct handle is fed into the apply pass at register time.
//!
//! Two pass kinds share this single struct via [`GtaoDenoiseStage`]: `Intermediate` (centre-pixel
//! weight `beta / 5`) and `Final` (centre-pixel weight `beta`, scaled by
//! `XE_GTAO_OCCLUSION_TERM_SCALE`). The Rust struct picks the right entry-point pipeline; the
//! kernel body is implemented once in `post/gtao_denoise.wgsl`.

use std::num::NonZeroU32;

use super::pipeline::{GtaoPipelineCache, GtaoStage};
use crate::render_graph::compiled::RenderPassTemplate;
use crate::render_graph::context::RasterPassCtx;
use crate::render_graph::error::{RenderPassError, SetupError};
use crate::render_graph::pass::{PassBuilder, RasterPass};
use crate::render_graph::resources::{TextureAccess, TextureHandle};

/// Distinguishes intermediate denoise passes from the final pass. Selects the pipeline entry
/// point and the centre-pixel weight in the shader.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum GtaoDenoiseStage {
    /// Used for every denoise pass before the last when `denoise_passes >= 2`.
    Intermediate,
    /// Used for the last (or only) denoise pass; applies the XeGTAO occlusion-term scaling.
    Final,
}

impl GtaoDenoiseStage {
    /// Maps the denoise stage to the cache pipeline-stage key.
    fn pipeline_stage(self) -> GtaoStage {
        match self {
            Self::Intermediate => GtaoStage::DenoiseIntermediate,
            Self::Final => GtaoStage::DenoiseFinal,
        }
    }

    /// Stable pass name used by the graph executor for logging and diagnostics.
    fn pass_name(self) -> &'static str {
        match self {
            Self::Intermediate => "GtaoDenoiseIntermediate",
            Self::Final => "GtaoDenoiseFinal",
        }
    }
}

/// Graph handles for [`GtaoDenoisePass`]. All three handles are transient textures owned by the
/// GTAO effect; the chain doesn't see them.
#[derive(Clone, Copy, Debug)]
pub(super) struct GtaoDenoiseResources {
    /// Sampled input AO term.
    pub ao_in: TextureHandle,
    /// Sampled packed edges (constant across the denoise loop).
    pub edges: TextureHandle,
    /// Color attachment: next AO term.
    pub ao_out: TextureHandle,
}

/// Bilateral denoise raster pass.
pub(super) struct GtaoDenoisePass {
    /// Graph handles for this pass instance.
    resources: GtaoDenoiseResources,
    /// Intermediate vs final.
    stage: GtaoDenoiseStage,
    /// Process-wide cached pipelines / bind-group layouts / params UBO.
    pipelines: &'static GtaoPipelineCache,
}

impl GtaoDenoisePass {
    /// Constructs a new instance.
    pub(super) fn new(
        resources: GtaoDenoiseResources,
        stage: GtaoDenoiseStage,
        pipelines: &'static GtaoPipelineCache,
    ) -> Self {
        Self {
            resources,
            stage,
            pipelines,
        }
    }
}

impl RasterPass for GtaoDenoisePass {
    fn name(&self) -> &str {
        self.stage.pass_name()
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        b.read_texture_resource(
            self.resources.ao_in,
            TextureAccess::Sampled {
                stages: wgpu::ShaderStages::FRAGMENT,
            },
        );
        b.read_texture_resource(
            self.resources.edges,
            TextureAccess::Sampled {
                stages: wgpu::ShaderStages::FRAGMENT,
            },
        );
        let mut r = b.raster();
        r.color(
            self.resources.ao_out,
            wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
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
        // Single profiling zone for both intermediate and final denoise dispatches; the graph
        // executor already differentiates them by pass name (`GtaoDenoiseIntermediate` vs
        // `GtaoDenoiseFinal`) when emitting per-pass GPU timestamps.
        profiling::scope!("post_processing::gtao::denoise");
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
        let Some(ao_in) = graph_resources.transient_texture(self.resources.ao_in) else {
            return Err(RenderPassError::MissingFrameParams {
                pass: format!("{} (missing ao input)", self.name()),
            });
        };
        let Some(edges) = graph_resources.transient_texture(self.resources.edges) else {
            return Err(RenderPassError::MissingFrameParams {
                pass: format!("{} (missing edges input)", self.name()),
            });
        };

        let multiview_stereo = frame.view.multiview_stereo;
        let output_format = graph_resources
            .transient_texture(self.resources.ao_out)
            .map(|t| t.texture.format())
            .unwrap_or(wgpu::TextureFormat::R16Float);

        let pipeline = self.pipelines.pipeline(
            ctx.device,
            self.stage.pipeline_stage(),
            output_format,
            multiview_stereo,
        );
        let bind_group = self.pipelines.bind_group_denoise(
            ctx.device,
            multiview_stereo,
            &ao_in.texture,
            &edges.texture,
        );
        rpass.set_pipeline(pipeline.as_ref());
        rpass.set_bind_group(0, &bind_group, &[]);
        rpass.draw(0..3, 0..1);
        Ok(())
    }
}
