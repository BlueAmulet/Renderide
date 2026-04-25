//! Stage 1 of the GTAO sub-graph: reads scene depth, runs the analytic horizon-cosine integral,
//! and writes a raw AO term + the packed-edges side-channel that downstream stages consume.
//!
//! Doesn't read the chain's HDR scene color — the apply pass owns the final modulation. Splitting
//! the pipeline this way is the prerequisite for porting XeGTAO's bilateral denoise: the denoise
//! kernel has to operate on the AO term in isolation, since cross-bilateral weights are derived
//! from depth (via the `edges` texture), not scene radiance.

use std::num::NonZeroU32;

use super::pipeline::{GtaoParamsGpu, GtaoPipelineCache, GtaoStage};
use crate::config::GtaoSettings;
use crate::render_graph::compiled::RenderPassTemplate;
use crate::render_graph::context::RasterPassCtx;
use crate::render_graph::error::{RenderPassError, SetupError};
use crate::render_graph::frame_params::{GtaoSettingsSlot, PerViewFramePlanSlot};
use crate::render_graph::pass::{PassBuilder, RasterPass};
use crate::render_graph::resources::{
    BufferAccess, ImportedBufferHandle, ImportedTextureHandle, TextureAccess, TextureHandle,
};

/// Graph handles for [`GtaoMainPass`].
#[derive(Clone, Copy, Debug)]
pub(super) struct GtaoMainResources {
    /// MRT 0: single-channel AO term (`R16Float`). Consumed by the denoise stages or, when
    /// denoise is disabled, by the apply pass.
    pub ao_term: TextureHandle,
    /// MRT 1: packed LRTB edge stoppers (`R8Unorm`). Read by every denoise pass.
    pub edges: TextureHandle,
    /// Imported scene depth handle. Sampled via a depth-only `TextureView` built at record time.
    pub depth: ImportedTextureHandle,
    /// Imported per-view frame uniforms (fallback path; the per-view-plan path is preferred).
    pub frame_uniforms: ImportedBufferHandle,
}

/// Fragment-shader pass that produces the AO term and packed edges from depth.
pub(super) struct GtaoMainPass {
    /// Graph handles for this pass instance.
    resources: GtaoMainResources,
    /// Live GTAO tunables captured at chain-build time. Used to seed the GPU UBO when the
    /// blackboard slot is not populated (tests / pre-lifecycle paths).
    fallback_settings: GtaoSettings,
    /// Process-wide cached pipelines, bind-group layouts, sampler, and params UBO.
    pipelines: &'static GtaoPipelineCache,
}

impl GtaoMainPass {
    /// Constructs a new instance.
    pub(super) fn new(
        resources: GtaoMainResources,
        fallback_settings: GtaoSettings,
        pipelines: &'static GtaoPipelineCache,
    ) -> Self {
        Self {
            resources,
            fallback_settings,
            pipelines,
        }
    }
}

impl RasterPass for GtaoMainPass {
    fn name(&self) -> &str {
        "GtaoMain"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        b.import_texture(
            self.resources.depth,
            TextureAccess::Sampled {
                stages: wgpu::ShaderStages::FRAGMENT,
            },
        );
        b.import_buffer(
            self.resources.frame_uniforms,
            BufferAccess::Uniform {
                stages: wgpu::ShaderStages::FRAGMENT,
                dynamic_offset: false,
            },
        );
        let mut r = b.raster();
        r.color(
            self.resources.ao_term,
            wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                store: wgpu::StoreOp::Store,
            },
            Option::<TextureHandle>::None,
        );
        r.color(
            self.resources.edges,
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
        profiling::scope!("post_processing::gtao::main");
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

        // Bind the per-view frame-uniforms buffer when the per-view plan is populated; fall back
        // to the imported handle otherwise. Mirrors the policy used by the legacy GTAO record
        // path so per-eye projection coefficients remain correct under multiview.
        let per_view_buffer = ctx
            .blackboard
            .get::<PerViewFramePlanSlot>()
            .map(|plan| plan.frame_uniform_buffer.clone());
        let frame_uniform_buffer = match per_view_buffer {
            Some(buf) => buf,
            None => match graph_resources.imported_buffer(self.resources.frame_uniforms) {
                Some(resolved) => resolved.buffer.clone(),
                None => {
                    return Err(RenderPassError::MissingFrameParams {
                        pass: format!("{} (frame_uniforms not resolved)", self.name()),
                    });
                }
            },
        };

        let multiview_stereo = frame.view.multiview_stereo;
        let ao_format = graph_resources
            .transient_texture(self.resources.ao_term)
            .map(|t| t.texture.format())
            .unwrap_or(wgpu::TextureFormat::R16Float);

        // Update the shared GTAO params UBO once per frame from the live blackboard slot. Routed
        // through the deferred upload sink so every queue write goes through the driver thread
        // (project's single-producer invariant for `wgpu::Queue::write_buffer`).
        let settings = ctx
            .blackboard
            .get::<GtaoSettingsSlot>()
            .map(|slot| slot.0)
            .unwrap_or(self.fallback_settings);
        let params = GtaoParamsGpu::from_settings(&settings);
        let params_buffer = self.pipelines.params_buffer(ctx.device);
        ctx.upload_batch
            .write_buffer(params_buffer, 0, bytemuck::bytes_of(&params));

        let pipeline =
            self.pipelines
                .pipeline(ctx.device, GtaoStage::Main, ao_format, multiview_stereo);
        let bind_group = self.pipelines.bind_group_main(
            ctx.device,
            multiview_stereo,
            frame.view.depth_texture,
            &frame_uniform_buffer,
        );
        rpass.set_pipeline(pipeline.as_ref());
        rpass.set_bind_group(0, &bind_group, &[]);
        rpass.draw(0..3, 0..1);
        Ok(())
    }
}
