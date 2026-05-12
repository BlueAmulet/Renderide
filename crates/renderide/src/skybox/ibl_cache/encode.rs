//! Command encoding for IBL mip-0 producers and GGX convolve passes.

use bytemuck::{Pod, Zeroable};

use crate::profiling::GpuProfilerHandle;
use crate::skybox::params::SkyboxEvaluatorParams;
use crate::skybox::specular::{CubemapIblSource, RuntimeCubemapIblSource};

use super::bind_groups::{
    build_input_output_bind_group, build_sampled_bind_group, build_storage_bind_group,
    make_uniform_buffer,
};
use super::key::convolve_sample_count;
use super::mip_loop::{MipChainConfig, PerMipBindings, dispatch_mip0_pass, run_mip_chain};
use super::pipeline::ComputePipeline;
use super::resources::{
    PendingBakeResources, create_mip_array_sample_view, create_mip_storage_view,
};

/// Uniform payload shared by the cubemap and convolve mip-0 producers.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Mip0CubeParams {
    dst_size: u32,
    src_face_size: u32,
    storage_v_inverted: u32,
    _pad0: u32,
}

/// Uniform payload for one GGX convolve mip dispatch.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ConvolveParams {
    dst_size: u32,
    mip_index: u32,
    mip_count: u32,
    sample_count: u32,
    src_face_size: u32,
    src_max_lod: f32,
    _pad0: u32,
    _pad1: u32,
}

/// Uniform payload for one source-pyramid downsample dispatch.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct DownsampleParams {
    dst_size: u32,
    src_size: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Inputs for [`encode_analytic_mip0`].
pub(super) struct AnalyticEncodeContext<'a> {
    pub(super) device: &'a wgpu::Device,
    pub(super) encoder: &'a mut wgpu::CommandEncoder,
    pub(super) pipeline: &'a ComputePipeline,
    pub(super) texture: &'a wgpu::Texture,
    pub(super) face_size: u32,
    pub(super) params: &'a SkyboxEvaluatorParams,
    pub(super) profiler: Option<&'a GpuProfilerHandle>,
}

/// Encodes mip 0 from evaluator parameters for constant-color probe sources.
pub(super) fn encode_analytic_mip0(
    ctx: AnalyticEncodeContext<'_>,
    resources: &mut PendingBakeResources,
) {
    profiling::scope!("skybox_ibl::encode_mip0_analytic");
    let params = ctx.params.with_sample_size(ctx.face_size);
    let params_buffer = make_uniform_buffer(ctx.device, "skybox_ibl analytic params", &params);
    crate::profiling::note_resource_churn!(Buffer, "skybox::ibl_analytic_params_buffer");
    let mip0_storage = create_mip_storage_view(ctx.texture, 0);
    let bind_group = build_storage_bind_group(
        ctx.device,
        &ctx.pipeline.layout,
        "skybox_ibl analytic bind group",
        &params_buffer,
        &mip0_storage,
    );
    crate::profiling::note_resource_churn!(BindGroup, "skybox::ibl_analytic_bind_group");
    dispatch_mip0_pass(
        ctx.encoder,
        ctx.pipeline,
        &bind_group,
        ctx.face_size,
        "skybox_ibl analytic mip0",
        ctx.profiler,
        "skybox_ibl::mip0_analytic",
    );
    resources.buffers.push(params_buffer);
    resources.bind_groups.push(bind_group);
    resources.texture_views.push(mip0_storage);
}

/// Inputs for [`encode_cube_mip0`].
pub(super) struct CubeEncodeContext<'a> {
    pub(super) device: &'a wgpu::Device,
    pub(super) encoder: &'a mut wgpu::CommandEncoder,
    pub(super) pipeline: &'a ComputePipeline,
    pub(super) texture: &'a wgpu::Texture,
    pub(super) face_size: u32,
    pub(super) src: CubemapIblSource,
    pub(super) sampler: &'a wgpu::Sampler,
    pub(super) profiler: Option<&'a GpuProfilerHandle>,
}

/// Encodes mip 0 by resampling a host cubemap source.
pub(super) fn encode_cube_mip0(ctx: CubeEncodeContext<'_>, resources: &mut PendingBakeResources) {
    profiling::scope!("skybox_ibl::encode_mip0_cube");
    let params = Mip0CubeParams {
        dst_size: ctx.face_size,
        src_face_size: ctx.src.face_size,
        storage_v_inverted: u32::from(ctx.src.storage_v_inverted),
        _pad0: 0,
    };
    let params_buffer = make_uniform_buffer(ctx.device, "skybox_ibl cube mip0 params", &params);
    crate::profiling::note_resource_churn!(Buffer, "skybox::ibl_cube_mip0_params_buffer");
    let mip0_storage = create_mip_storage_view(ctx.texture, 0);
    let bind_group = build_sampled_bind_group(
        ctx.device,
        &ctx.pipeline.layout,
        "skybox_ibl cube mip0 bind group",
        &params_buffer,
        ctx.src.view.as_ref(),
        ctx.sampler,
        &mip0_storage,
    );
    crate::profiling::note_resource_churn!(BindGroup, "skybox::ibl_cube_mip0_bind_group");
    dispatch_mip0_pass(
        ctx.encoder,
        ctx.pipeline,
        &bind_group,
        ctx.face_size,
        "skybox_ibl cube mip0",
        ctx.profiler,
        "skybox_ibl::mip0_cube",
    );
    resources.buffers.push(params_buffer);
    resources.bind_groups.push(bind_group);
    resources.texture_views.push(mip0_storage);
    resources.source_views.push(ctx.src.view);
}

/// Inputs for [`encode_runtime_cube_mip0`].
pub(super) struct RuntimeCubeEncodeContext<'a> {
    pub(super) device: &'a wgpu::Device,
    pub(super) encoder: &'a mut wgpu::CommandEncoder,
    pub(super) pipeline: &'a ComputePipeline,
    pub(super) texture: &'a wgpu::Texture,
    pub(super) face_size: u32,
    pub(super) src: RuntimeCubemapIblSource,
    pub(super) sampler: &'a wgpu::Sampler,
    pub(super) profiler: Option<&'a GpuProfilerHandle>,
}

/// Encodes mip 0 by resampling a renderer-captured cubemap source.
pub(super) fn encode_runtime_cube_mip0(
    ctx: RuntimeCubeEncodeContext<'_>,
    resources: &mut PendingBakeResources,
) {
    profiling::scope!("skybox_ibl::encode_mip0_runtime_cube");
    let params = Mip0CubeParams {
        dst_size: ctx.face_size,
        src_face_size: ctx.src.face_size,
        storage_v_inverted: 0,
        _pad0: 0,
    };
    let params_buffer =
        make_uniform_buffer(ctx.device, "skybox_ibl runtime cube mip0 params", &params);
    crate::profiling::note_resource_churn!(Buffer, "skybox::ibl_runtime_cube_mip0_params_buffer");
    let mip0_storage = create_mip_storage_view(ctx.texture, 0);
    let bind_group = build_sampled_bind_group(
        ctx.device,
        &ctx.pipeline.layout,
        "skybox_ibl runtime cube mip0 bind group",
        &params_buffer,
        ctx.src.view.as_ref(),
        ctx.sampler,
        &mip0_storage,
    );
    crate::profiling::note_resource_churn!(BindGroup, "skybox::ibl_runtime_cube_mip0_bind_group");
    dispatch_mip0_pass(
        ctx.encoder,
        ctx.pipeline,
        &bind_group,
        ctx.face_size,
        "skybox_ibl runtime cube mip0",
        ctx.profiler,
        "skybox_ibl::mip0_runtime_cube",
    );
    resources.buffers.push(params_buffer);
    resources.bind_groups.push(bind_group);
    resources.texture_views.push(mip0_storage);
    resources.textures.push(ctx.src.texture);
    resources.source_views.push(ctx.src.view);
}

/// Inputs for [`encode_downsample_mips`].
pub(super) struct DownsampleEncodeContext<'a> {
    pub(super) device: &'a wgpu::Device,
    pub(super) encoder: &'a mut wgpu::CommandEncoder,
    pub(super) pipeline: &'a ComputePipeline,
    pub(super) texture: &'a wgpu::Texture,
    pub(super) face_size: u32,
    pub(super) mip_levels: u32,
    pub(super) profiler: Option<&'a GpuProfilerHandle>,
}

/// Encodes sequential per-face downsample passes for mips `1..mip_levels` of the source cube.
pub(super) fn encode_downsample_mips(
    ctx: DownsampleEncodeContext<'_>,
    resources: &mut PendingBakeResources,
) {
    profiling::scope!("skybox_ibl::encode_downsample_mips");
    let DownsampleEncodeContext {
        device,
        encoder,
        pipeline,
        texture,
        face_size,
        mip_levels,
        profiler,
    } = ctx;
    run_mip_chain(
        MipChainConfig {
            encoder,
            pipeline,
            face_size,
            mip_levels,
            profiler,
            resources,
            pass_label: "skybox_ibl downsample mip",
            profiler_label_prefix: "skybox_ibl::downsample_mip",
        },
        |mip, dst_size, src_size| {
            let params = DownsampleParams {
                dst_size,
                src_size,
                _pad0: 0,
                _pad1: 0,
            };
            let params_buffer =
                make_uniform_buffer(device, "skybox_ibl downsample params", &params);
            crate::profiling::note_resource_churn!(Buffer, "skybox::ibl_downsample_params_buffer");
            let src_view = create_mip_array_sample_view(texture, mip - 1);
            let dst_view = create_mip_storage_view(texture, mip);
            let bind_group = build_input_output_bind_group(
                device,
                &pipeline.layout,
                "skybox_ibl downsample bind group",
                &params_buffer,
                &src_view,
                &dst_view,
            );
            crate::profiling::note_resource_churn!(BindGroup, "skybox::ibl_downsample_bind_group");
            PerMipBindings {
                params: params_buffer,
                bind_group,
                src_view: Some(src_view),
                dst_view,
            }
        },
    );
}

/// Inputs for [`encode_convolve_mips`].
pub(super) struct ConvolveEncodeContext<'a> {
    pub(super) device: &'a wgpu::Device,
    pub(super) encoder: &'a mut wgpu::CommandEncoder,
    pub(super) pipeline: &'a ComputePipeline,
    pub(super) texture: &'a wgpu::Texture,
    pub(super) src_view: &'a wgpu::TextureView,
    pub(super) sampler: &'a wgpu::Sampler,
    pub(super) face_size: u32,
    pub(super) mip_levels: u32,
    pub(super) src_max_lod: f32,
    pub(super) profiler: Option<&'a GpuProfilerHandle>,
}

/// Encodes the GGX convolve passes for mips `1..mip_levels` of the destination cube.
pub(super) fn encode_convolve_mips(
    ctx: ConvolveEncodeContext<'_>,
    resources: &mut PendingBakeResources,
) {
    profiling::scope!("skybox_ibl::encode_convolve_mips");
    let ConvolveEncodeContext {
        device,
        encoder,
        pipeline,
        texture,
        src_view,
        sampler,
        face_size,
        mip_levels,
        src_max_lod,
        profiler,
    } = ctx;
    run_mip_chain(
        MipChainConfig {
            encoder,
            pipeline,
            face_size,
            mip_levels,
            profiler,
            resources,
            pass_label: "skybox_ibl convolve mip",
            profiler_label_prefix: "skybox_ibl::convolve_mip",
        },
        |mip, dst_size, _src_size| {
            let params = ConvolveParams {
                dst_size,
                mip_index: mip,
                mip_count: mip_levels,
                sample_count: convolve_sample_count(mip),
                src_face_size: face_size,
                src_max_lod,
                _pad0: 0,
                _pad1: 0,
            };
            let params_buffer = make_uniform_buffer(device, "skybox_ibl convolve params", &params);
            crate::profiling::note_resource_churn!(Buffer, "skybox::ibl_convolve_params_buffer");
            let dst_view = create_mip_storage_view(texture, mip);
            let bind_group = build_sampled_bind_group(
                device,
                &pipeline.layout,
                "skybox_ibl convolve bind group",
                &params_buffer,
                src_view,
                sampler,
                &dst_view,
            );
            crate::profiling::note_resource_churn!(BindGroup, "skybox::ibl_convolve_bind_group");
            PerMipBindings {
                params: params_buffer,
                bind_group,
                src_view: None,
                dst_view,
            }
        },
    );
}
