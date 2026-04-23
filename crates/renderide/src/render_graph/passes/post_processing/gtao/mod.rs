//! Ground-Truth Ambient Occlusion (Jimenez et al. 2016) render pass.
//!
//! Reads the chain's HDR scene color and the scene depth buffer, reconstructs view-space normals
//! from depth derivatives, evaluates the analytic cosine-weighted GTAO integral with a single
//! horizon direction per pixel (spatially jittered 4×4 + per-frame phase rotation), applies the
//! multi-bounce fit for near-field indirect light, and writes the HDR scene color modulated by
//! the resulting visibility factor. Must run **before** tonemapping so AO acts on linear light.
//!
//! Multiview is handled the same way as [`crate::render_graph::passes::AcesTonemapPass`]: two
//! pipeline variants (mono / multiview) picked via a `multiview_mask_override` of
//! `NonZeroU32::new(3)` in stereo, with `#ifdef MULTIVIEW` in the shader selecting
//! `@builtin(view_index)` and the depth-array sample path.

mod pipeline;

use std::num::NonZeroU32;
use std::sync::OnceLock;

use pipeline::{GtaoParamsGpu, GtaoPipelineCache};

use crate::config::{GtaoSettings, PostProcessingSettings};
use crate::render_graph::compiled::RenderPassTemplate;
use crate::render_graph::context::RasterPassCtx;
use crate::render_graph::error::{RenderPassError, SetupError};
use crate::render_graph::pass::{PassBuilder, RasterPass};
use crate::render_graph::post_processing::{PostProcessEffect, PostProcessEffectId};
use crate::render_graph::resources::{
    BufferAccess, ImportedBufferHandle, TextureAccess, TextureHandle,
};

/// Graph handles for [`GtaoPass`].
#[derive(Clone, Copy, Debug)]
pub struct GtaoGraphResources {
    /// HDR scene-color input (chain stage's previous output, or the forward HDR target).
    pub input: TextureHandle,
    /// HDR chain output written by this pass (input × AO factor).
    pub output: TextureHandle,
    /// Shared frame-uniform buffer (`FrameGpuUniforms`) for per-eye proj coefficients, near/far,
    /// viewport, and frame index. Imported from the backend's frame resources.
    pub frame_uniforms: ImportedBufferHandle,
}

/// Fullscreen render pass applying GTAO to `input`, writing modulated HDR to `output`.
pub struct GtaoPass {
    /// Graph handles for this pass instance.
    resources: GtaoGraphResources,
    /// Live GTAO tunables captured at chain-build time and rewritten into the GPU UBO each record.
    settings: GtaoSettings,
    /// Process-wide cached pipelines, bind-group layouts, sampler, and params UBO.
    pipelines: &'static GtaoPipelineCache,
}

impl GtaoPass {
    /// Creates a new GTAO pass instance capturing the current settings snapshot.
    pub fn new(resources: GtaoGraphResources, settings: GtaoSettings) -> Self {
        Self {
            resources,
            settings,
            pipelines: gtao_pipelines(),
        }
    }
}

/// Process-wide pipeline cache shared by every GTAO pass instance.
fn gtao_pipelines() -> &'static GtaoPipelineCache {
    static CACHE: OnceLock<GtaoPipelineCache> = OnceLock::new();
    CACHE.get_or_init(GtaoPipelineCache::default)
}

impl RasterPass for GtaoPass {
    fn name(&self) -> &str {
        "Gtao"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        b.read_texture_resource(
            self.resources.input,
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
        profiling::scope!("post_processing::gtao");
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
        let Some(input_tex) = graph_resources.transient_texture(self.resources.input) else {
            return Err(RenderPassError::MissingFrameParams {
                pass: format!(
                    "{} (missing transient input {:?})",
                    self.name(),
                    self.resources.input
                ),
            });
        };
        let Some(frame_uniforms) = graph_resources.imported_buffer(self.resources.frame_uniforms)
        else {
            return Err(RenderPassError::MissingFrameParams {
                pass: format!("{} (frame_uniforms not resolved)", self.name()),
            });
        };

        let multiview_stereo = frame.view.multiview_stereo;
        let target_format = output_attachment_format(self.resources.output, graph_resources);

        let params = GtaoParamsGpu {
            radius_world: self.settings.radius_meters.max(0.0),
            max_pixel_radius: self.settings.max_pixel_radius.max(1.0),
            intensity: self.settings.intensity.max(0.0),
            step_count: self.settings.step_count.max(1),
            thickness_heuristic: self.settings.thickness_heuristic.clamp(0.0, 1.0),
            albedo_multibounce: self.settings.albedo_multibounce.clamp(0.0, 0.99),
            align_pad_tail: [0.0; 2],
        };
        let params_buffer = self.pipelines.params_buffer(ctx.device);
        ctx.queue
            .write_buffer(params_buffer, 0, bytemuck::bytes_of(&params));

        let scene_color_view = input_tex.view_for_sampled_2d_array(multiview_stereo);
        let scene_depth_view = depth_sample_view(frame.view.depth_texture, multiview_stereo);

        let pipeline = self
            .pipelines
            .pipeline(ctx.device, target_format, multiview_stereo);
        let bind_group = self.pipelines.bind_group(
            ctx.device,
            multiview_stereo,
            &scene_color_view,
            &scene_depth_view,
            &frame_uniforms.buffer,
        );
        rpass.set_pipeline(pipeline.as_ref());
        rpass.set_bind_group(0, &bind_group, &[]);
        rpass.draw(0..3, 0..1);
        Ok(())
    }
}

/// Creates a depth-only sample view matching the active multiview state.
fn depth_sample_view(depth_texture: &wgpu::Texture, multiview_stereo: bool) -> wgpu::TextureView {
    let (dimension, array_layer_count) = if multiview_stereo {
        (wgpu::TextureViewDimension::D2Array, Some(2))
    } else {
        (wgpu::TextureViewDimension::D2, Some(1))
    };
    depth_texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("gtao-depth-sample"),
        dimension: Some(dimension),
        base_array_layer: 0,
        array_layer_count,
        aspect: wgpu::TextureAspect::DepthOnly,
        ..Default::default()
    })
}

/// Resolves the wgpu format the GTAO color attachment is bound to this frame.
fn output_attachment_format(
    output: TextureHandle,
    graph_resources: &crate::render_graph::context::GraphResolvedResources,
) -> wgpu::TextureFormat {
    graph_resources
        .transient_texture(output)
        .map(|t| t.texture.format())
        .unwrap_or(wgpu::TextureFormat::Rgba16Float)
}

/// Effect descriptor that contributes a [`GtaoPass`] to the post-processing chain.
pub struct GtaoEffect {
    /// Snapshot of the GTAO settings used when building the pass for this frame.
    pub settings: GtaoSettings,
    /// Imported frame-uniforms buffer handle (from the main graph's buffer imports).
    pub frame_uniforms: ImportedBufferHandle,
}

impl PostProcessEffect for GtaoEffect {
    fn id(&self) -> PostProcessEffectId {
        PostProcessEffectId::Gtao
    }

    fn is_enabled(&self, settings: &PostProcessingSettings) -> bool {
        settings.enabled && settings.gtao.enabled
    }

    fn build_pass(&self, input: TextureHandle, output: TextureHandle) -> Box<dyn RasterPass> {
        Box::new(GtaoPass::new(
            GtaoGraphResources {
                input,
                output,
                frame_uniforms: self.frame_uniforms,
            },
            self.settings,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render_graph::pass::node::PassKind;
    use crate::render_graph::pass::PassBuilder;
    use crate::render_graph::resources::{
        AccessKind, BufferImportSource, ImportedBufferDecl, TextureAccess, TransientArrayLayers,
        TransientExtent, TransientSampleCount, TransientTextureDesc, TransientTextureFormat,
    };
    use crate::render_graph::GraphBuilder;

    fn fake_graph() -> (
        GraphBuilder,
        TextureHandle,
        TextureHandle,
        ImportedBufferHandle,
    ) {
        let mut builder = GraphBuilder::new();
        let desc = || TransientTextureDesc {
            label: "pp_hdr",
            format: TransientTextureFormat::SceneColorHdr,
            extent: TransientExtent::Backbuffer,
            mip_levels: 1,
            sample_count: TransientSampleCount::Fixed(1),
            dimension: wgpu::TextureDimension::D2,
            array_layers: TransientArrayLayers::Frame,
            base_usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING,
            alias: true,
        };
        let input = builder.create_texture(desc());
        let output = builder.create_texture(desc());
        let frame_uniforms = builder.import_buffer(ImportedBufferDecl {
            label: "frame_uniforms",
            source: BufferImportSource::BackendFrameResource(
                crate::render_graph::resources::BackendFrameBufferKind::FrameUniforms,
            ),
            initial_access: BufferAccess::Uniform {
                stages: wgpu::ShaderStages::FRAGMENT,
                dynamic_offset: false,
            },
            final_access: BufferAccess::Uniform {
                stages: wgpu::ShaderStages::FRAGMENT,
                dynamic_offset: false,
            },
        });
        (builder, input, output, frame_uniforms)
    }

    #[test]
    fn setup_declares_sampled_input_and_raster_output_and_uniform_buffer() {
        let (_builder, input, output, frame_uniforms) = fake_graph();
        let mut pass = GtaoPass::new(
            GtaoGraphResources {
                input,
                output,
                frame_uniforms,
            },
            GtaoSettings::default(),
        );
        let mut b = PassBuilder::new("Gtao");
        pass.setup(&mut b).expect("setup");
        let setup = b.finish().expect("finish");
        assert_eq!(setup.kind, PassKind::Raster);
        assert!(
            setup.accesses.iter().any(|a| matches!(
                &a.access,
                AccessKind::Texture(TextureAccess::Sampled {
                    stages: wgpu::ShaderStages::FRAGMENT,
                    ..
                })
            )),
            "expected sampled HDR input read"
        );
        assert!(
            setup.accesses.iter().any(|a| matches!(
                &a.access,
                AccessKind::Buffer(BufferAccess::Uniform {
                    stages: wgpu::ShaderStages::FRAGMENT,
                    ..
                })
            )),
            "expected uniform-buffer read of frame_uniforms"
        );
        assert_eq!(setup.color_attachments.len(), 1);
    }

    #[test]
    fn gtao_effect_id_label() {
        let e = GtaoEffect {
            settings: GtaoSettings::default(),
            frame_uniforms: ImportedBufferHandle(0),
        };
        assert_eq!(e.id(), PostProcessEffectId::Gtao);
        assert_eq!(e.id().label(), "GTAO");
    }

    #[test]
    fn gtao_effect_is_gated_by_master_and_per_effect_enable() {
        let e = GtaoEffect {
            settings: GtaoSettings::default(),
            frame_uniforms: ImportedBufferHandle(0),
        };
        let mut s = PostProcessingSettings::default();
        assert!(!e.is_enabled(&s), "defaults: master off, gtao off");
        s.enabled = true;
        assert!(!e.is_enabled(&s), "master on but gtao off");
        s.gtao.enabled = true;
        assert!(e.is_enabled(&s), "master on + gtao on");
        s.enabled = false;
        assert!(!e.is_enabled(&s), "master off disables even if gtao on");
    }
}
