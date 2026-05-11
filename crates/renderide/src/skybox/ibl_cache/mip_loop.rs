//! Shared per-mip dispatch loop used by both source-pyramid downsampling and GGX convolve passes.

use crate::profiling::{GpuProfilerHandle, compute_pass_timestamp_writes};

use super::key::{dispatch_groups, mip_extent};
use super::pipeline::ComputePipeline;
use super::resources::PendingBakeResources;

/// One mip iteration's encoded bind group plus the per-mip views to retain.
pub(super) struct PerMipBindings {
    /// Uniform buffer retained until submit completion.
    pub(super) params: wgpu::Buffer,
    /// Bind group retained until submit completion.
    pub(super) bind_group: wgpu::BindGroup,
    /// Source view created for this mip's input (downsample paths only).
    pub(super) src_view: Option<wgpu::TextureView>,
    /// Storage view for the mip being written.
    pub(super) dst_view: wgpu::TextureView,
}

/// Static configuration for a [`run_mip_chain`] invocation.
pub(super) struct MipChainConfig<'a> {
    /// Command encoder receiving the per-mip compute passes.
    pub(super) encoder: &'a mut wgpu::CommandEncoder,
    /// Compute pipeline driving each mip's dispatch.
    pub(super) pipeline: &'a ComputePipeline,
    /// Destination cube face edge at mip 0.
    pub(super) face_size: u32,
    /// Total mip count; iterates `1..mip_levels`.
    pub(super) mip_levels: u32,
    /// Optional GPU profiler for per-mip timestamp queries.
    pub(super) profiler: Option<&'a GpuProfilerHandle>,
    /// Transient resource bag for retention until submit completion.
    pub(super) resources: &'a mut PendingBakeResources,
    /// Static label applied to every compute pass.
    pub(super) pass_label: &'static str,
    /// Profiler label prefix; the current mip index is appended per iteration.
    pub(super) profiler_label_prefix: &'a str,
}

/// Runs one compute pass per mip in `1..mip_levels`, dispatching `dispatch_groups(dst_size)^2 x 6`.
///
/// `build_per_mip` is invoked for each mip with `(mip, dst_size, src_size)` and returns the
/// uniform buffer, bind group, and texture views to retain across submit completion.
pub(super) fn run_mip_chain<F>(config: MipChainConfig<'_>, mut build_per_mip: F)
where
    F: FnMut(u32, u32, u32) -> PerMipBindings,
{
    let MipChainConfig {
        encoder,
        pipeline,
        face_size,
        mip_levels,
        profiler,
        resources,
        pass_label,
        profiler_label_prefix,
    } = config;
    if mip_levels <= 1 {
        return;
    }
    for mip in 1..mip_levels {
        let dst_size = mip_extent(face_size, mip);
        let src_size = mip_extent(face_size, mip - 1);
        let bindings = build_per_mip(mip, dst_size, src_size);
        let pass_query =
            profiler.map(|p| p.begin_pass_query(format!("{profiler_label_prefix}{mip}"), encoder));
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(pass_label),
                timestamp_writes: compute_pass_timestamp_writes(pass_query.as_ref()),
            });
            pass.set_pipeline(&pipeline.pipeline);
            pass.set_bind_group(0, &bindings.bind_group, &[]);
            pass.dispatch_workgroups(dispatch_groups(dst_size), dispatch_groups(dst_size), 6);
        }
        if let (Some(p), Some(q)) = (profiler, pass_query) {
            p.end_query(encoder, q);
        }
        resources.buffers.push(bindings.params);
        resources.bind_groups.push(bindings.bind_group);
        if let Some(src_view) = bindings.src_view {
            resources.texture_views.push(src_view);
        }
        resources.texture_views.push(bindings.dst_view);
    }
}

/// Runs the boilerplate of one mip-0 compute pass: profiler query, dispatch, query close.
pub(super) fn dispatch_mip0_pass(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &ComputePipeline,
    bind_group: &wgpu::BindGroup,
    face_size: u32,
    pass_label: &'static str,
    profiler: Option<&GpuProfilerHandle>,
    profiler_label: &'static str,
) {
    let pass_query = profiler.map(|p| p.begin_pass_query(profiler_label, encoder));
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(pass_label),
            timestamp_writes: compute_pass_timestamp_writes(pass_query.as_ref()),
        });
        pass.set_pipeline(&pipeline.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(dispatch_groups(face_size), dispatch_groups(face_size), 6);
    }
    if let (Some(p), Some(q)) = (profiler, pass_query) {
        p.end_query(encoder, q);
    }
}
