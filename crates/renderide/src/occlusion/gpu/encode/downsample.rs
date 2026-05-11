//! Hi-Z hierarchical downsample chain.
//!
//! Issues one min-reduction compute dispatch per mip transition (mip0 -> mip1, mip1 -> mip2, ...)
//! within the active pyramid layer, sharing the cached pipeline and uniform buffer in
//! [`super::EncodeSession`].

use bytemuck::{Pod, Zeroable};

use crate::occlusion::cpu::pyramid::mip_dimensions;

use super::{EncodeSession, PyramidSide};

/// Per-mip extent uniform consumed by the downsample compute shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DownsampleUniform {
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
}

/// Farthest-depth min-reduction chain from mip0 through the rest of the R32F pyramid.
pub(super) fn dispatch(
    session: &mut EncodeSession<'_>,
    pyramid_views: &[wgpu::TextureView],
    side: PyramidSide,
) {
    let (bw, bh) = session.scratch.extent;
    for mip in 0..session.scratch.mip_levels.saturating_sub(1) {
        let (sw, sh) = mip_dimensions(bw, bh, mip).unwrap_or((1, 1));
        let (dw, dh) = mip_dimensions(bw, bh, mip + 1).unwrap_or((1, 1));
        let du = DownsampleUniform {
            src_w: sw,
            src_h: sh,
            dst_w: dw,
            dst_h: dh,
        };
        session.uploads.write_buffer(
            &session.scratch.downsample_uniform,
            0,
            bytemuck::bytes_of(&du),
        );
        let device = session.device;
        let layout = &session.pipes.bgl_downsample;
        // Clone the uniform buffer handle so the bind-group build closure does not borrow
        // `session.scratch` for the duration of `downsample_*_or_build`'s `&mut bind_groups`
        // borrow.
        let downsample_uniform = session.scratch.downsample_uniform.clone();
        let build = || {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("hi_z_ds_bg"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&pyramid_views[mip as usize]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(
                            &pyramid_views[mip as usize + 1],
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: downsample_uniform.as_entire_binding(),
                    },
                ],
            });
            crate::profiling::note_resource_churn!(BindGroup, "occlusion::hi_z_downsample_bg");
            bind_group
        };
        let bg = match side {
            PyramidSide::DesktopOrLeft => session
                .scratch
                .bind_groups
                .downsample_desktop_or_build(mip, build),
            PyramidSide::Right => session
                .scratch
                .bind_groups
                .downsample_right_or_build(mip, build),
        };
        let pass_query = session
            .profiler
            .map(|p| p.begin_pass_query("hi_z_downsample", session.encoder));
        let timestamp_writes = crate::profiling::compute_pass_timestamp_writes(pass_query.as_ref());
        {
            let mut pass = session
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("hi_z_downsample"),
                    timestamp_writes,
                });
            pass.set_pipeline(&session.pipes.downsample);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(dw.div_ceil(8), dh.div_ceil(8), 1);
        };
        if let (Some(p), Some(q)) = (session.profiler, pass_query) {
            p.end_query(session.encoder, q);
        }
    }
}
