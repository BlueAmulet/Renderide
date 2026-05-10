//! Canonical main render graph: imports, transient declarations, pass topology, and the default
//! post-processing chain.
//!
//! This module is the *application* of the render-graph framework: it wires the renderer's
//! built-in passes together to produce the frame the host expects (mesh deform -> clustered lights
//! -> forward -> Hi-Z -> post-processing -> HDR scene-color compose). The framework primitives it
//! consumes (builder, compiled graph, resources, post-processing chain) live in their respective
//! sibling modules.

use crate::render_graph::GraphCacheKey;
use crate::render_graph::builder::GraphBuilder;
use crate::render_graph::compiled::CompiledRenderGraph;
use crate::render_graph::error::GraphBuildError;
use crate::render_graph::ids::PassId;
use crate::render_graph::post_process_chain;
use crate::render_graph::resources::{
    BackendFrameBufferKind, BufferAccess, BufferHandle, BufferImportSource, BufferSizePolicy,
    FrameTargetRole, HistorySlotId, ImportSource, ImportedBufferDecl, ImportedBufferHandle,
    ImportedTextureDecl, ImportedTextureHandle, StorageAccess, TextureAccess, TextureHandle,
    TransientArrayLayers, TransientBufferDesc, TransientExtent, TransientSampleCount,
    TransientTextureDesc, TransientTextureFormat,
};

/// Long-lived resources shared by post-processing passes across main-graph rebuilds.
#[derive(Clone, Default)]
pub(crate) struct MainGraphPostProcessingResources {
    auto_exposure_state_cache:
        std::sync::Arc<crate::passes::post_processing::AutoExposureStateCache>,
}

impl MainGraphPostProcessingResources {
    /// Shared auto-exposure state cache used by graph instances built for the same backend.
    pub(crate) fn auto_exposure_state_cache(
        &self,
    ) -> std::sync::Arc<crate::passes::post_processing::AutoExposureStateCache> {
        std::sync::Arc::clone(&self.auto_exposure_state_cache)
    }

    /// Releases view-scoped post-processing resources for views that are no longer active.
    pub(crate) fn retire_views(&self, retired_views: &[crate::camera::ViewId]) {
        self.auto_exposure_state_cache.retire_views(retired_views);
    }
}

/// Imported buffers/transients wired into [`build_main_graph`].
struct MainGraphHandles {
    color: ImportedTextureHandle,
    depth: ImportedTextureHandle,
    hi_z_current: ImportedTextureHandle,
    lights: ImportedBufferHandle,
    cluster_light_counts: ImportedBufferHandle,
    cluster_light_indices: ImportedBufferHandle,
    per_draw_slab: ImportedBufferHandle,
    frame_uniforms: ImportedBufferHandle,
    cluster_params: BufferHandle,
    /// Single-sample HDR scene color (forward resolve target + compose input).
    scene_color_hdr: TextureHandle,
    /// Multisampled HDR scene color for forward when MSAA is active.
    scene_color_hdr_msaa: TextureHandle,
    forward_msaa_depth: TextureHandle,
    forward_msaa_depth_r32: TextureHandle,
}

/// Handles for imported backend buffers (lights, cluster tables, per-draw slab, frame uniforms).
struct MainGraphBufferImports {
    lights: ImportedBufferHandle,
    cluster_light_counts: ImportedBufferHandle,
    cluster_light_indices: ImportedBufferHandle,
    per_draw_slab: ImportedBufferHandle,
    frame_uniforms: ImportedBufferHandle,
}

fn import_main_graph_textures(
    builder: &mut GraphBuilder,
) -> (
    ImportedTextureHandle,
    ImportedTextureHandle,
    ImportedTextureHandle,
) {
    let color = builder.import_texture(ImportedTextureDecl {
        label: "frame_color",
        source: ImportSource::Frame(FrameTargetRole::ColorAttachment),
        initial_access: TextureAccess::ColorAttachment {
            load: wgpu::LoadOp::Load,
            store: wgpu::StoreOp::Store,
            resolve_to: None,
        },
        final_access: TextureAccess::Present,
    });
    let depth = builder.import_texture(ImportedTextureDecl {
        label: "frame_depth",
        source: ImportSource::Frame(FrameTargetRole::DepthAttachment),
        initial_access: TextureAccess::DepthAttachment {
            depth: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
            stencil: None,
        },
        final_access: TextureAccess::Sampled {
            stages: wgpu::ShaderStages::COMPUTE,
        },
    });
    let hi_z_current = builder.import_texture(ImportedTextureDecl {
        label: "hi_z_current",
        source: ImportSource::PingPong(HistorySlotId::HI_Z),
        initial_access: TextureAccess::Storage {
            stages: wgpu::ShaderStages::COMPUTE,
            access: StorageAccess::WriteOnly,
        },
        final_access: TextureAccess::Storage {
            stages: wgpu::ShaderStages::COMPUTE,
            access: StorageAccess::WriteOnly,
        },
    });
    (color, depth, hi_z_current)
}

fn import_main_graph_buffers(builder: &mut GraphBuilder) -> MainGraphBufferImports {
    let lights = builder.import_buffer(ImportedBufferDecl {
        label: "lights",
        source: BufferImportSource::Frame(BackendFrameBufferKind::Lights),
        initial_access: BufferAccess::Storage {
            stages: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
            access: StorageAccess::ReadOnly,
        },
        final_access: BufferAccess::Storage {
            stages: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
            access: StorageAccess::ReadOnly,
        },
    });
    let cluster_light_counts = builder.import_buffer(ImportedBufferDecl {
        label: "cluster_light_counts",
        source: BufferImportSource::Frame(BackendFrameBufferKind::ClusterLightCounts),
        initial_access: BufferAccess::Storage {
            stages: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
            access: StorageAccess::WriteOnly,
        },
        final_access: BufferAccess::Storage {
            stages: wgpu::ShaderStages::FRAGMENT,
            access: StorageAccess::ReadOnly,
        },
    });
    let cluster_light_indices = builder.import_buffer(ImportedBufferDecl {
        label: "cluster_light_indices",
        source: BufferImportSource::Frame(BackendFrameBufferKind::ClusterLightIndices),
        initial_access: BufferAccess::Storage {
            stages: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
            access: StorageAccess::WriteOnly,
        },
        final_access: BufferAccess::Storage {
            stages: wgpu::ShaderStages::FRAGMENT,
            access: StorageAccess::ReadOnly,
        },
    });
    let per_draw_slab = builder.import_buffer(ImportedBufferDecl {
        label: "per_draw_slab",
        source: BufferImportSource::Frame(BackendFrameBufferKind::PerDrawSlab),
        initial_access: BufferAccess::Storage {
            stages: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            access: StorageAccess::ReadOnly,
        },
        final_access: BufferAccess::Storage {
            stages: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            access: StorageAccess::ReadOnly,
        },
    });
    let frame_uniforms = builder.import_buffer(ImportedBufferDecl {
        label: "frame_uniforms",
        source: BufferImportSource::Frame(BackendFrameBufferKind::FrameUniforms),
        initial_access: BufferAccess::Uniform {
            stages: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            dynamic_offset: false,
        },
        final_access: BufferAccess::Uniform {
            stages: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            dynamic_offset: false,
        },
    });
    MainGraphBufferImports {
        lights,
        cluster_light_counts,
        cluster_light_indices,
        per_draw_slab,
        frame_uniforms,
    }
}

/// Declares cluster buffers and HDR forward transients for [`build_main_graph`].
///
/// Forward MSAA depth targets use [`TransientArrayLayers::Frame`] (not a fixed layer count from
/// [`GraphCacheKey::multiview_stereo`]) so the same compiled graph can run mono desktop and stereo
/// OpenXR without mismatched multiview attachment layers.
fn create_main_graph_transient_resources(
    builder: &mut GraphBuilder,
) -> (
    BufferHandle,
    TextureHandle,
    TextureHandle,
    TextureHandle,
    TextureHandle,
) {
    let cluster_params = builder.create_buffer(TransientBufferDesc {
        label: "cluster_params",
        size_policy: BufferSizePolicy::Fixed(crate::gpu::CLUSTER_PARAMS_UNIFORM_SIZE * 2),
        base_usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        alias: true,
    });
    // Use [`TransientExtent::Backbuffer`] for forward MSAA targets: [`build_default_main_graph`]
    // uses a placeholder [`GraphCacheKey::surface_extent`]; baking that into `Custom` extent would
    // allocate 1x1 textures while resolve / imported frame color stay at the real swapchain size.
    // Execute-time resolution uses each view's viewport (see [`crate::render_graph::compiled::helpers::resolve_transient_extent`]).
    //
    // Multisampled forward attachments use [`TransientSampleCount::Frame`] so pool allocations match
    // the live MSAA tier; [`GraphCacheKey::msaa_sample_count`] still invalidates [`crate::render_graph::GraphCache`].
    let extent_backbuffer = TransientExtent::Backbuffer;
    let scene_color_hdr = builder.create_texture(TransientTextureDesc {
        label: "scene_color_hdr",
        format: TransientTextureFormat::SceneColorHdr,
        extent: extent_backbuffer,
        mip_levels: 1,
        sample_count: TransientSampleCount::Fixed(1),
        dimension: wgpu::TextureDimension::D2,
        array_layers: TransientArrayLayers::Frame,
        base_usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        alias: true,
    });
    let scene_color_hdr_msaa = builder.create_texture(TransientTextureDesc {
        label: "scene_color_hdr_msaa",
        format: TransientTextureFormat::SceneColorHdr,
        extent: extent_backbuffer,
        mip_levels: 1,
        sample_count: TransientSampleCount::Frame,
        dimension: wgpu::TextureDimension::D2,
        array_layers: TransientArrayLayers::Frame,
        base_usage: wgpu::TextureUsages::empty(),
        alias: true,
    });
    let mut forward_msaa_depth = TransientTextureDesc::frame_depth_stencil_sampled_texture_2d(
        "forward_msaa_depth",
        extent_backbuffer,
        wgpu::TextureUsages::empty(),
    );
    forward_msaa_depth.sample_count = TransientSampleCount::Frame;
    // Same layer policy as scene color MSAA: execute-time stereo (e.g. OpenXR) must not disagree
    // with a graph built under a mono [`GraphCacheKey`].
    forward_msaa_depth.array_layers = TransientArrayLayers::Frame;
    let forward_msaa_depth = builder.create_texture(forward_msaa_depth);
    let forward_msaa_depth_r32 = builder.create_texture(
        TransientTextureDesc::texture_2d(
            "forward_msaa_depth_r32",
            wgpu::TextureFormat::R32Float,
            extent_backbuffer,
            1,
            wgpu::TextureUsages::empty(),
        )
        .with_frame_array_layers(),
    );
    (
        cluster_params,
        scene_color_hdr,
        scene_color_hdr_msaa,
        forward_msaa_depth,
        forward_msaa_depth_r32,
    )
}

fn gtao_post_processing_active(settings: &crate::config::PostProcessingSettings) -> bool {
    settings.enabled && settings.gtao.enabled
}

fn create_gtao_view_normal_transients(
    builder: &mut GraphBuilder,
) -> (TextureHandle, TextureHandle) {
    let extent = TransientExtent::Backbuffer;
    let normals = builder.create_texture(TransientTextureDesc {
        label: "gtao_view_normals",
        format: TransientTextureFormat::Fixed(crate::passes::GTAO_VIEW_NORMAL_FORMAT),
        extent,
        mip_levels: 1,
        sample_count: TransientSampleCount::Fixed(1),
        dimension: wgpu::TextureDimension::D2,
        array_layers: TransientArrayLayers::Frame,
        base_usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        alias: true,
    });
    let normals_msaa = builder.create_texture(TransientTextureDesc {
        label: "gtao_view_normals_msaa",
        format: TransientTextureFormat::Fixed(crate::passes::GTAO_VIEW_NORMAL_FORMAT),
        extent,
        mip_levels: 1,
        sample_count: TransientSampleCount::Frame,
        dimension: wgpu::TextureDimension::D2,
        array_layers: TransientArrayLayers::Frame,
        base_usage: wgpu::TextureUsages::empty(),
        alias: true,
    });
    (normals, normals_msaa)
}

struct GtaoNormalPrepassNode {
    view_normals: TextureHandle,
    pass: PassId,
}

fn main_forward_resources(h: &MainGraphHandles) -> crate::passes::WorldMeshForwardGraphResources {
    crate::passes::WorldMeshForwardGraphResources {
        scene_color_hdr: h.scene_color_hdr,
        scene_color_hdr_msaa: h.scene_color_hdr_msaa,
        depth: h.depth,
        msaa_depth: h.forward_msaa_depth,
        msaa_depth_r32: h.forward_msaa_depth_r32,
        cluster_light_counts: h.cluster_light_counts,
        cluster_light_indices: h.cluster_light_indices,
        lights: h.lights,
        per_draw_slab: h.per_draw_slab,
        frame_uniforms: h.frame_uniforms,
    }
}

fn main_depth_prepass_resources(
    h: &MainGraphHandles,
) -> crate::passes::WorldMeshForwardDepthPrepassGraphResources {
    crate::passes::WorldMeshForwardDepthPrepassGraphResources {
        depth: h.depth,
        msaa_depth: h.forward_msaa_depth,
        per_draw_slab: h.per_draw_slab,
    }
}

fn add_gtao_normal_prepass_if_active(
    builder: &mut GraphBuilder,
    h: &MainGraphHandles,
    post_processing_settings: &crate::config::PostProcessingSettings,
) -> Option<GtaoNormalPrepassNode> {
    if !gtao_post_processing_active(post_processing_settings) {
        return None;
    }
    let (view_normals, normals_msaa) = create_gtao_view_normal_transients(builder);
    let pass = builder.add_raster_pass(Box::new(crate::passes::WorldMeshForwardNormalPass::new(
        crate::passes::WorldMeshForwardNormalGraphResources {
            normals: view_normals,
            normals_msaa,
            depth: h.depth,
            msaa_depth: h.forward_msaa_depth,
            per_draw_slab: h.per_draw_slab,
        },
    )));
    Some(GtaoNormalPrepassNode { view_normals, pass })
}

fn add_world_mesh_depth_prepass(builder: &mut GraphBuilder, h: &MainGraphHandles) -> PassId {
    builder.add_raster_pass(Box::new(crate::passes::WorldMeshForwardDepthPrepass::new(
        main_depth_prepass_resources(h),
    )))
}

/// Wires imported frame targets and main-graph transients into `builder` for [`build_main_graph`].
fn import_main_graph_resources(builder: &mut GraphBuilder) -> MainGraphHandles {
    let (color, depth, hi_z_current) = import_main_graph_textures(builder);
    let buf = import_main_graph_buffers(builder);
    let (
        cluster_params,
        scene_color_hdr,
        scene_color_hdr_msaa,
        forward_msaa_depth,
        forward_msaa_depth_r32,
    ) = create_main_graph_transient_resources(builder);
    MainGraphHandles {
        color,
        depth,
        hi_z_current,
        lights: buf.lights,
        cluster_light_counts: buf.cluster_light_counts,
        cluster_light_indices: buf.cluster_light_indices,
        per_draw_slab: buf.per_draw_slab,
        frame_uniforms: buf.frame_uniforms,
        cluster_params,
        scene_color_hdr,
        scene_color_hdr_msaa,
        forward_msaa_depth,
        forward_msaa_depth_r32,
    }
}

fn connect_post_processing_edges(
    builder: &mut GraphBuilder,
    forward_tail: PassId,
    chain_output: post_process_chain::ChainOutput,
    compose: PassId,
) {
    if let Some((first_post, last_post)) = chain_output.pass_range() {
        builder.add_edge(forward_tail, first_post);
        builder.add_edge(last_post, compose);
    } else {
        builder.add_edge(forward_tail, compose);
    }
}

fn add_main_graph_passes_and_edges(
    mut builder: GraphBuilder,
    h: MainGraphHandles,
    post_processing_settings: &crate::config::PostProcessingSettings,
    post_processing_resources: &MainGraphPostProcessingResources,
    msaa_sample_count: u8,
    multiview_stereo: bool,
) -> Result<CompiledRenderGraph, GraphBuildError> {
    let deform = builder.add_compute_pass(Box::new(crate::passes::MeshDeformPass::new()));
    let clustered = builder.add_compute_pass(Box::new(crate::passes::ClusteredLightPass::new(
        crate::passes::ClusteredLightGraphResources {
            lights: h.lights,
            cluster_light_counts: h.cluster_light_counts,
            cluster_light_indices: h.cluster_light_indices,
            params: h.cluster_params,
        },
    )));
    let forward_resources = main_forward_resources(&h);
    let depth_prepass = add_world_mesh_depth_prepass(&mut builder, &h);
    let forward_opaque = builder.add_raster_pass(Box::new(
        crate::passes::WorldMeshForwardOpaquePass::new(forward_resources),
    ));
    let gtao_normals =
        add_gtao_normal_prepass_if_active(&mut builder, &h, post_processing_settings);
    let depth_snapshot = builder.add_compute_pass(Box::new(
        crate::passes::WorldMeshDepthSnapshotPass::new(forward_resources),
    ));
    let forward_intersect = builder.add_raster_pass(Box::new(
        crate::passes::WorldMeshForwardIntersectPass::new(forward_resources),
    ));
    // Color resolve replaces the wgpu automatic linear `resolve_target`. The pre-grab resolve
    // makes a single-sample HDR snapshot available to grab-pass shaders; the final resolve moves
    // any grab-pass transparent MSAA color back into the single-sample HDR target consumed by
    // post-processing. In 1x mode each forward pass writes `scene_color_hdr` directly.
    let color_resolve_resources = crate::passes::WorldMeshForwardColorResolveGraphResources {
        scene_color_hdr_msaa: h.scene_color_hdr_msaa,
        scene_color_hdr: h.scene_color_hdr,
    };
    let pre_grab_color_resolve = (msaa_sample_count > 1).then(|| {
        builder.add_raster_pass(Box::new(
            crate::passes::WorldMeshForwardColorResolvePass::new_pre_grab(color_resolve_resources),
        ))
    });
    let color_snapshot = builder.add_compute_pass(Box::new(
        crate::passes::WorldMeshColorSnapshotPass::new(forward_resources),
    ));
    let forward_transparent = builder.add_raster_pass(Box::new(
        crate::passes::WorldMeshForwardTransparentPass::new(forward_resources),
    ));
    let final_color_resolve = (msaa_sample_count > 1).then(|| {
        builder.add_raster_pass(Box::new(
            crate::passes::WorldMeshForwardColorResolvePass::new_final(color_resolve_resources),
        ))
    });
    let depth_resolve = builder.add_compute_pass(Box::new(
        crate::passes::WorldMeshForwardDepthResolvePass::new(forward_resources),
    ));
    let hiz = builder.add_compute_pass(Box::new(crate::passes::HiZBuildPass::new(
        crate::passes::HiZBuildGraphResources {
            depth: h.depth,
            hi_z_current: h.hi_z_current,
        },
    )));

    let chain = build_default_post_processing_chain(
        &h,
        post_processing_settings,
        multiview_stereo,
        post_processing_resources,
        gtao_normals.as_ref().map(|node| node.view_normals),
    );
    let chain_output =
        chain.build_into_graph(&mut builder, h.scene_color_hdr, post_processing_settings);
    let compose_input = chain_output.final_handle();

    let compose = builder.add_raster_pass(Box::new(crate::passes::SceneColorComposePass::new(
        crate::passes::SceneColorComposeGraphResources {
            scene_color_hdr: h.scene_color_hdr,
            post_processed_scene_color_hdr: compose_input,
            frame_color: h.color,
        },
    )));
    builder.add_edge(deform, clustered);
    builder.add_edge(clustered, depth_prepass);
    builder.add_edge(depth_prepass, forward_opaque);
    if let Some(gtao_normals) = gtao_normals {
        builder.add_edge(forward_opaque, gtao_normals.pass);
        builder.add_edge(gtao_normals.pass, depth_snapshot);
    } else {
        builder.add_edge(forward_opaque, depth_snapshot);
    }
    builder.add_edge(depth_snapshot, forward_intersect);
    if let Some(pre_grab_color_resolve) = pre_grab_color_resolve {
        builder.add_edge(forward_intersect, pre_grab_color_resolve);
        builder.add_edge(pre_grab_color_resolve, color_snapshot);
    } else {
        builder.add_edge(forward_intersect, color_snapshot);
    }
    builder.add_edge(color_snapshot, forward_transparent);
    if let Some(final_color_resolve) = final_color_resolve {
        builder.add_edge(forward_transparent, final_color_resolve);
        builder.add_edge(final_color_resolve, depth_resolve);
    } else {
        builder.add_edge(forward_transparent, depth_resolve);
    }
    builder.add_edge(depth_resolve, hiz);
    connect_post_processing_edges(&mut builder, hiz, chain_output, compose);
    builder.build()
}

/// Builds the canonical post-processing chain shipped with the renderer.
///
/// Execution order is GTAO -> auto-exposure -> bloom -> ACES tonemap. GTAO runs first so ambient
/// occlusion modulates linear HDR light before metering; auto-exposure meters and scales the HDR
/// scene before bloom; bloom scatters exposed HDR light; then ACES compresses the final exposed HDR
/// signal to display-referred `[0, 1]`. Each effect gates itself via
/// [`crate::render_graph::post_process_chain::PostProcessEffect::is_enabled`] against the live
/// [`crate::config::PostProcessingSettings`].
///
/// `GtaoEffect` is parameterised with the current [`crate::config::GtaoSettings`] snapshot and
/// the imported `frame_uniforms` handle (used to access per-eye projection coefficients and the
/// frame index at record time). It is registered only when the graph also created the matching
/// view-normal texture. `BloomEffect` captures a [`crate::config::BloomSettings`] snapshot for its
/// shared params UBO and per-mip blend constants.
fn build_default_post_processing_chain(
    h: &MainGraphHandles,
    post_processing_settings: &crate::config::PostProcessingSettings,
    multiview_stereo: bool,
    post_processing_resources: &MainGraphPostProcessingResources,
    gtao_view_normals: Option<TextureHandle>,
) -> post_process_chain::PostProcessChain {
    let mut chain = post_process_chain::PostProcessChain::new();
    if let Some(view_normals) = gtao_view_normals {
        chain.push(Box::new(crate::passes::GtaoEffect {
            settings: post_processing_settings.gtao,
            depth: h.depth,
            view_normals,
            frame_uniforms: h.frame_uniforms,
            multiview_stereo,
        }));
    }
    chain.push(Box::new(crate::passes::AutoExposureEffect::new(
        post_processing_resources.auto_exposure_state_cache(),
    )));
    chain.push(Box::new(crate::passes::BloomEffect {
        settings: post_processing_settings.bloom,
    }));
    chain.push(Box::new(crate::passes::AcesTonemapEffect));
    chain.push(Box::new(crate::passes::AgxTonemapEffect));
    chain
}

/// Builds the main frame graph: mesh deform compute, clustered lights, world forward, Hi-Z readback,
/// then HDR scene-color compose into the display target.
///
/// Forward MSAA transients use [`TransientExtent::Backbuffer`] and [`TransientSampleCount::Frame`] so
/// sizes match the current view at execute time (the graph is often built with
/// [`build_default_main_graph`]'s placeholder [`GraphCacheKey::surface_extent`]). HDR scene color
/// uses [`TransientTextureFormat::SceneColorHdr`]; the resolved format follows
/// [`crate::config::RenderingSettings::scene_color_format`] at execute time (see
/// [`GraphCacheKey::scene_color_format`] for graph-cache identity). `key` still drives graph-cache
/// identity ([`GraphCacheKey::surface_format`], [`GraphCacheKey::multiview_stereo`],
/// [`GraphCacheKey::msaa_sample_count`]). Imported sources resolve at execute time via
/// [`crate::backend::FrameResourceManager`].
#[cfg(test)]
pub fn build_main_graph(
    key: GraphCacheKey,
    post_processing_settings: &crate::config::PostProcessingSettings,
) -> Result<CompiledRenderGraph, GraphBuildError> {
    build_main_graph_with_resources(
        key,
        post_processing_settings,
        &MainGraphPostProcessingResources::default(),
    )
}

/// Builds the main frame graph using caller-owned post-processing resources.
pub(crate) fn build_main_graph_with_resources(
    key: GraphCacheKey,
    post_processing_settings: &crate::config::PostProcessingSettings,
    post_processing_resources: &MainGraphPostProcessingResources,
) -> Result<CompiledRenderGraph, GraphBuildError> {
    logger::info!(
        "main render graph: scene color HDR format = {:?}, post-processing = {} effect(s)",
        key.scene_color_format,
        key.post_processing.active_count()
    );
    let mut builder = GraphBuilder::new();
    let handles = import_main_graph_resources(&mut builder);
    let msaa_handles = [handles.forward_msaa_depth, handles.forward_msaa_depth_r32];
    let mut graph = add_main_graph_passes_and_edges(
        builder,
        handles,
        post_processing_settings,
        post_processing_resources,
        key.msaa_sample_count,
        key.multiview_stereo,
    )?;
    graph.set_main_graph_msaa_transient_handles(msaa_handles);
    Ok(graph)
}

#[cfg(test)]
mod tests;
