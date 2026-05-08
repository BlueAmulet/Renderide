//! Cached shader, bind layout, sampler, and per-format pipeline for the display-blit pass.

use std::sync::OnceLock;

use crate::embedded_shaders::embedded_wgsl;

/// Returns the cached bind group layout used by [`super::resources::DisplayBlitResources`].
///
/// `(0)` 2D texture, `(1)` filtering sampler, `(2)` `vec4<f32>` UV scale/offset uniform.
pub(super) fn surface_bind_group_layout(device: &wgpu::Device) -> &'static wgpu::BindGroupLayout {
    static LAYOUT: OnceLock<wgpu::BindGroupLayout> = OnceLock::new();
    LAYOUT.get_or_init(|| {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("display_blit_surface"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(16),
                    },
                    count: None,
                },
            ],
        })
    })
}

/// Builds (or rebuilds) the fragment-output-format-specific pipeline for the display blit pass.
pub(super) fn surface_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("display_blit"),
        source: wgpu::ShaderSource::Wgsl(embedded_wgsl!("display_blit").into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("display_blit"),
        bind_group_layouts: &[Some(surface_bind_group_layout(device))],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("display_blit"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    });
    crate::profiling::note_resource_churn!(RenderPipeline, "gpu::display_blit_pipeline");
    pipeline
}

/// Returns the cached linear+clamp sampler shared by every display-blit invocation.
///
/// Mirrors Unity's `Graphics.DrawTexture` default behavior (linear filtering, clamped UV).
pub(super) fn linear_sampler(device: &wgpu::Device) -> &'static wgpu::Sampler {
    static SAMPLER: OnceLock<wgpu::Sampler> = OnceLock::new();
    SAMPLER.get_or_init(|| {
        device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("display_blit_linear"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        })
    })
}
