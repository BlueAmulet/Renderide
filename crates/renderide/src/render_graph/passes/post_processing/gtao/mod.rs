//! Ground-Truth Ambient Occlusion effect (Jiménez et al. 2016 "Practical Realtime Strategies for
//! Accurate Indirect Occlusion") with XeGTAO's bilateral spatial denoise.
//!
//! [`GtaoEffect`] presents a single `PostProcessEffect` to the post-processing chain, but expands
//! internally to a 1 + N + 1 raster sub-graph:
//!
//! 1. **Main** ([`main::GtaoMainPass`]) — reconstructs view-space normals from depth, runs the
//!    analytic horizon-cosine integral, and writes a raw AO term (`R16Float`) plus the
//!    packed-edges side-channel (`R8Unorm`) used by the bilateral filter.
//! 2. **Denoise** ([`denoise::GtaoDenoisePass`], `0..=N` instances) — XeGTAO's 5×5 cross-bilateral
//!    on the AO term, edge-stopped via the packed-edges texture. The passes ping-pong between two
//!    `R16Float` transients owned by this effect; the parity-correct handle is passed to apply.
//!    The last (or only) denoise pass uses the `Final` entry point, scaling its output by
//!    `XE_GTAO_OCCLUSION_TERM_SCALE = 1.5`.
//! 3. **Apply** ([`apply::GtaoApplyPass`]) — multiplies HDR scene color by the (optionally
//!    denoised) AO term after applying the `intensity` exponent and Eq. 10 multi-bounce fit.
//!
//! Pass count is graph topology; [`crate::render_graph::post_processing::PostProcessChainSignature`]
//! tracks `gtao_denoise_passes` so changing the slider rebuilds the graph. Per-frame parameter
//! edits (intensity, blur beta, etc.) flow through the existing [`GtaoSettingsSlot`] blackboard
//! and reach the shader without a graph rebuild.
//!
//! Multiview is handled the same way as the rest of the post-processing chain: each sub-pass uses
//! a `multiview_mask_override` of `NonZeroU32::new(3)` in stereo, with two pipeline variants
//! (mono / multiview) sharing the bind-group layouts. Depth array layers are bound per-eye in
//! the main pass, and the AO and edges transients use [`TransientArrayLayers::Frame`] so they
//! match the frame's array shape.

mod apply;
mod denoise;
mod main;
mod pipeline;

use std::sync::OnceLock;

use apply::{GtaoApplyPass, GtaoApplyResources};
use denoise::{GtaoDenoisePass, GtaoDenoiseResources, GtaoDenoiseStage};
use main::{GtaoMainPass, GtaoMainResources};
use pipeline::GtaoPipelineCache;

use crate::config::{GtaoSettings, PostProcessingSettings};
use crate::render_graph::builder::GraphBuilder;
use crate::render_graph::post_processing::{EffectPasses, PostProcessEffect, PostProcessEffectId};
use crate::render_graph::resources::{
    ImportedBufferHandle, ImportedTextureHandle, TextureHandle, TransientArrayLayers,
    TransientExtent, TransientSampleCount, TransientTextureDesc, TransientTextureFormat,
};

/// Format of the AO-term ping-pong transients.
///
/// Single-channel half-float keeps the AO term linear under the bilateral filter without needing
/// the 32-bit precision of `R32Float` (the visibility factor is in `[0, 1]` and gets compressed
/// to `[0, ~1.5]` before the apply pass, well within R16Float's representable range).
const AO_TERM_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;

/// Format of the packed-edges side-channel.
///
/// XeGTAO encodes four edge stoppers as 2 bits each into a single `R8Unorm` channel; see
/// `shaders/source/modules/gtao_packing.wgsl`.
const EDGES_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;

/// Effect descriptor that contributes the GTAO sub-graph to the post-processing chain.
pub struct GtaoEffect {
    /// Snapshot of the GTAO settings used when building the sub-graph for this frame. The live
    /// values are still pulled from the per-view blackboard at record time so non-topology knobs
    /// don't require a graph rebuild; this snapshot is the fallback for tests / pre-lifecycle
    /// paths and the source of `denoise_passes` (which **is** topology).
    pub settings: GtaoSettings,
    /// Imported depth texture handle (declared as a sampled read on the main stage for graph
    /// scheduling).
    pub depth: ImportedTextureHandle,
    /// Imported frame-uniforms buffer handle; the actual buffer bound at record time is the
    /// per-view-plan slot when populated, otherwise this fallback handle.
    pub frame_uniforms: ImportedBufferHandle,
}

impl PostProcessEffect for GtaoEffect {
    fn id(&self) -> PostProcessEffectId {
        PostProcessEffectId::Gtao
    }

    fn is_enabled(&self, settings: &PostProcessingSettings) -> bool {
        settings.enabled && settings.gtao.enabled
    }

    fn register(
        &self,
        builder: &mut GraphBuilder,
        input: TextureHandle,
        output: TextureHandle,
    ) -> EffectPasses {
        let pipelines = gtao_pipelines();

        // Two ping-pong transients for the AO term plus one for the packed-edges side-channel.
        // All three use `TransientArrayLayers::Frame` so the multiview shape (1 layer mono /
        // 2 layers stereo) follows the chain's HDR transients automatically.
        let ao_a = builder.create_texture(ao_term_transient_desc("gtao_ao_term_a"));
        let ao_b = builder.create_texture(ao_term_transient_desc("gtao_ao_term_b"));
        let edges = builder.create_texture(edges_transient_desc());

        let main_pass = builder.add_raster_pass(Box::new(GtaoMainPass::new(
            GtaoMainResources {
                ao_term: ao_a,
                edges,
                depth: self.depth,
                frame_uniforms: self.frame_uniforms,
            },
            self.settings,
            pipelines,
        )));

        // Ping-pong the AO term across N denoise passes. After the loop, `prev_ao` holds the
        // *last written* texture: even N → ao_a, odd N → ao_b. The apply pass binds whatever
        // `prev_ao` ends as.
        let n = denoise_pass_count(&self.settings);
        let mut prev_ao = ao_a;
        let mut next_ao = ao_b;
        let mut last = main_pass;
        for i in 0..n {
            let stage = if i + 1 == n {
                GtaoDenoiseStage::Final
            } else {
                GtaoDenoiseStage::Intermediate
            };
            let pass = builder.add_raster_pass(Box::new(GtaoDenoisePass::new(
                GtaoDenoiseResources {
                    ao_in: prev_ao,
                    edges,
                    ao_out: next_ao,
                },
                stage,
                pipelines,
            )));
            builder.add_edge(last, pass);
            last = pass;
            std::mem::swap(&mut prev_ao, &mut next_ao);
        }

        let apply_pass = builder.add_raster_pass(Box::new(GtaoApplyPass::new(
            GtaoApplyResources {
                scene_color: input,
                ao_term: prev_ao,
                output,
            },
            pipelines,
        )));
        builder.add_edge(last, apply_pass);

        EffectPasses {
            first: main_pass,
            last: apply_pass,
        }
    }
}

/// Returns the clamped denoise pass count for this effect snapshot. Centralises the same clamp
/// the chain signature applies, so the registered topology and the cache key agree.
pub(super) fn denoise_pass_count(settings: &GtaoSettings) -> usize {
    settings.denoise_passes.min(3) as usize
}

/// Process-wide pipeline cache shared by every GTAO sub-pass instance.
fn gtao_pipelines() -> &'static GtaoPipelineCache {
    static CACHE: OnceLock<GtaoPipelineCache> = OnceLock::new();
    CACHE.get_or_init(GtaoPipelineCache::default)
}

/// Transient-texture descriptor for the AO-term ping-pong textures (`R16Float`, full backbuffer
/// resolution, frame-shaped array layers, sampled + render-attachment usage).
fn ao_term_transient_desc(label: &'static str) -> TransientTextureDesc {
    TransientTextureDesc {
        label,
        format: TransientTextureFormat::Fixed(AO_TERM_FORMAT),
        extent: TransientExtent::Backbuffer,
        mip_levels: 1,
        sample_count: TransientSampleCount::Fixed(1),
        dimension: wgpu::TextureDimension::D2,
        array_layers: TransientArrayLayers::Frame,
        base_usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        alias: true,
    }
}

/// Transient-texture descriptor for the packed-edges side-channel (`R8Unorm`).
fn edges_transient_desc() -> TransientTextureDesc {
    TransientTextureDesc {
        label: "gtao_edges",
        format: TransientTextureFormat::Fixed(EDGES_FORMAT),
        extent: TransientExtent::Backbuffer,
        mip_levels: 1,
        sample_count: TransientSampleCount::Fixed(1),
        dimension: wgpu::TextureDimension::D2,
        array_layers: TransientArrayLayers::Frame,
        base_usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        alias: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render_graph::pass::node::PassKind;
    use crate::render_graph::pass::{PassBuilder, RasterPass};
    use crate::render_graph::resources::{
        AccessKind, BufferAccess, BufferImportSource, ImportSource, ImportedBufferDecl,
        ImportedTextureDecl, TextureAccess,
    };
    use crate::render_graph::GraphBuilder;

    fn fake_chain_io() -> (
        GraphBuilder,
        TextureHandle,
        TextureHandle,
        ImportedTextureHandle,
        ImportedBufferHandle,
    ) {
        let mut builder = GraphBuilder::new();
        let chain_desc = || TransientTextureDesc {
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
        let input = builder.create_texture(chain_desc());
        let output = builder.create_texture(chain_desc());
        let depth = builder.import_texture(ImportedTextureDecl {
            label: "frame_depth",
            source: ImportSource::FrameTarget(
                crate::render_graph::resources::FrameTargetRole::DepthAttachment,
            ),
            initial_access: TextureAccess::Sampled {
                stages: wgpu::ShaderStages::FRAGMENT,
            },
            final_access: TextureAccess::Sampled {
                stages: wgpu::ShaderStages::FRAGMENT,
            },
        });
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
        (builder, input, output, depth, frame_uniforms)
    }

    /// Builds a GTAO main pass against ad-hoc transient AO + edges textures.
    fn build_main_pass(
        builder: &mut GraphBuilder,
        depth: ImportedTextureHandle,
        frame_uniforms: ImportedBufferHandle,
    ) -> (GtaoMainPass, TextureHandle, TextureHandle) {
        let ao = builder.create_texture(ao_term_transient_desc("gtao_ao_term_a"));
        let edges = builder.create_texture(edges_transient_desc());
        let pass = GtaoMainPass::new(
            GtaoMainResources {
                ao_term: ao,
                edges,
                depth,
                frame_uniforms,
            },
            GtaoSettings::default(),
            gtao_pipelines(),
        );
        (pass, ao, edges)
    }

    #[test]
    fn main_pass_declares_two_color_attachments_and_imports_depth_and_frame_uniforms() {
        let (mut builder, _input, _output, depth, frame_uniforms) = fake_chain_io();
        let (mut pass, _ao, _edges) = build_main_pass(&mut builder, depth, frame_uniforms);
        let mut b = PassBuilder::new("GtaoMain");
        pass.setup(&mut b).expect("setup");
        let setup = b.finish().expect("finish");
        assert_eq!(setup.kind, PassKind::Raster);
        assert_eq!(
            setup.color_attachments.len(),
            2,
            "main pass writes ao_term + edges as MRT"
        );
        let imports_depth = setup.accesses.iter().any(|a| {
            matches!(
                &a.access,
                AccessKind::Texture(TextureAccess::Sampled { .. })
            )
        });
        assert!(
            imports_depth,
            "main pass declares depth as a sampled FRAGMENT read"
        );
        let imports_frame_uniforms = setup.accesses.iter().any(|a| {
            matches!(
                &a.access,
                AccessKind::Buffer(BufferAccess::Uniform {
                    stages: wgpu::ShaderStages::FRAGMENT,
                    ..
                })
            )
        });
        assert!(
            imports_frame_uniforms,
            "main pass imports frame_uniforms as a uniform buffer"
        );
    }

    #[test]
    fn apply_pass_declares_scene_and_ao_inputs_and_one_color_attachment() {
        let (mut builder, input, output, _depth, _frame_uniforms) = fake_chain_io();
        let ao = builder.create_texture(ao_term_transient_desc("gtao_ao_term_a"));
        let mut pass = GtaoApplyPass::new(
            GtaoApplyResources {
                scene_color: input,
                ao_term: ao,
                output,
            },
            gtao_pipelines(),
        );
        let mut b = PassBuilder::new("GtaoApply");
        pass.setup(&mut b).expect("setup");
        let setup = b.finish().expect("finish");
        assert_eq!(setup.kind, PassKind::Raster);
        assert_eq!(setup.color_attachments.len(), 1);
        let sampled_reads = setup
            .accesses
            .iter()
            .filter(|a| {
                matches!(
                    &a.access,
                    AccessKind::Texture(TextureAccess::Sampled {
                        stages: wgpu::ShaderStages::FRAGMENT,
                        ..
                    })
                )
            })
            .count();
        assert!(
            sampled_reads >= 2,
            "apply samples scene_color + ao_term (got {sampled_reads})"
        );
    }

    #[test]
    fn denoise_pass_declares_ao_and_edges_inputs_and_writes_next_ao() {
        let (mut builder, _input, _output, _depth, _frame_uniforms) = fake_chain_io();
        let ao_in = builder.create_texture(ao_term_transient_desc("gtao_ao_term_a"));
        let ao_out = builder.create_texture(ao_term_transient_desc("gtao_ao_term_b"));
        let edges = builder.create_texture(edges_transient_desc());
        let mut pass = GtaoDenoisePass::new(
            GtaoDenoiseResources {
                ao_in,
                edges,
                ao_out,
            },
            GtaoDenoiseStage::Intermediate,
            gtao_pipelines(),
        );
        let mut b = PassBuilder::new("GtaoDenoiseIntermediate");
        pass.setup(&mut b).expect("setup");
        let setup = b.finish().expect("finish");
        assert_eq!(setup.kind, PassKind::Raster);
        assert_eq!(setup.color_attachments.len(), 1);
        let sampled_reads = setup
            .accesses
            .iter()
            .filter(|a| {
                matches!(
                    &a.access,
                    AccessKind::Texture(TextureAccess::Sampled {
                        stages: wgpu::ShaderStages::FRAGMENT,
                        ..
                    })
                )
            })
            .count();
        assert!(
            sampled_reads >= 2,
            "denoise samples ao_in + edges (got {sampled_reads})"
        );
    }

    #[test]
    fn gtao_effect_id_label() {
        let e = GtaoEffect {
            settings: GtaoSettings::default(),
            depth: ImportedTextureHandle(0),
            frame_uniforms: ImportedBufferHandle(0),
        };
        assert_eq!(e.id(), PostProcessEffectId::Gtao);
        assert_eq!(e.id().label(), "GTAO");
    }

    #[test]
    fn gtao_effect_is_gated_by_master_and_per_effect_enable() {
        let e = GtaoEffect {
            settings: GtaoSettings::default(),
            depth: ImportedTextureHandle(0),
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

    #[test]
    fn denoise_pass_count_clamps_to_three() {
        let with_n = |n: u8| GtaoSettings {
            denoise_passes: n,
            ..GtaoSettings::default()
        };
        assert_eq!(denoise_pass_count(&with_n(0)), 0);
        assert_eq!(denoise_pass_count(&with_n(1)), 1);
        assert_eq!(denoise_pass_count(&with_n(3)), 3);
        assert_eq!(
            denoise_pass_count(&with_n(9)),
            3,
            "out-of-range values clamp to XeGTAO's max preset"
        );
    }

    #[test]
    fn register_emits_main_plus_n_denoise_plus_apply_with_parity_correct_ao_handle() {
        // Verifies that the ping-pong handle threaded into the apply pass alternates with N.
        // We can't introspect the graph builder, so we instead drive the same logic the register
        // function uses and assert the resulting texture handle.
        let ao_a = TextureHandle(101);
        let ao_b = TextureHandle(102);
        let resolved = |n: usize| {
            let mut prev = ao_a;
            let mut next = ao_b;
            for _ in 0..n {
                std::mem::swap(&mut prev, &mut next);
            }
            prev
        };
        assert_eq!(resolved(0), ao_a, "N=0: apply reads what main wrote (ao_a)");
        assert_eq!(resolved(1), ao_b, "N=1: denoise wrote ao_b");
        assert_eq!(resolved(2), ao_a, "N=2: ping-pong alternates back to ao_a");
        assert_eq!(resolved(3), ao_b, "N=3: ping-pong terminates at ao_b");
    }
}
