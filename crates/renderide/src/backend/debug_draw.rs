//! Uniform buffers and bind group for [`crate::pipelines::raster::DebugWorldNormalsFamily`] draws.

use crate::pipelines::raster::DebugWorldNormalsFamily;

/// Per-frame mutable uniforms: view-projection (written once or per space) and model (per draw).
pub struct DebugDrawResources {
    /// `GlobalUniforms`: column-major `mat4` (64 bytes).
    pub globals_buffer: wgpu::Buffer,
    /// `DrawUniforms`: column-major `mat4` (64 bytes).
    pub model_buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
}

impl DebugDrawResources {
    pub fn new(device: &wgpu::Device) -> Self {
        let layout = DebugWorldNormalsFamily::bind_group_layout(device);
        let globals_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("debug_world_normals_globals"),
            size: 128,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let model_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("debug_world_normals_model"),
            size: 128,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("debug_world_normals_bind_group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: globals_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: model_buffer.as_entire_binding(),
                },
            ],
        });
        Self {
            globals_buffer,
            model_buffer,
            bind_group,
        }
    }
}
