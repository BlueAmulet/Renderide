//! Per-frame `@group(0)` resources: scene uniform, lights storage, clustered light buffers.

use std::num::NonZeroU64;
use std::sync::Arc;

use crate::backend::cluster_gpu::{ClusterBufferCache, ClusterBufferRefs, CLUSTER_COUNT_Z};
use crate::backend::light_gpu::{GpuLight, MAX_LIGHTS};
use crate::gpu::frame_globals::FrameGpuUniforms;

/// GPU buffers and bind group for [`FrameGpuUniforms`], [`GpuLight`] storage, and cluster lists.
pub struct FrameGpuResources {
    /// Uniform buffer for [`FrameGpuUniforms`] (64 bytes).
    pub frame_uniform: wgpu::Buffer,
    /// Storage buffer holding up to [`MAX_LIGHTS`] [`GpuLight`] records.
    pub lights_buffer: wgpu::Buffer,
    /// Cluster buffers and compute params; resized with viewport ([`Self::sync_cluster_viewport`]).
    pub cluster_cache: ClusterBufferCache,
    /// Bind group for `@group(0)` in composed mesh shaders.
    pub bind_group: Arc<wgpu::BindGroup>,
    cluster_bind_version: u64,
}

impl FrameGpuResources {
    /// Layout for `@group(0)`: uniform frame + lights + cluster counts + cluster indices.
    pub fn bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("frame_globals"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(
                            std::mem::size_of::<FrameGpuUniforms>() as u64
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(std::mem::size_of::<GpuLight>() as u64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(4),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(4),
                    },
                    count: None,
                },
            ],
        })
    }

    fn create_bind_group(
        device: &wgpu::Device,
        frame_uniform: &wgpu::Buffer,
        lights_buffer: &wgpu::Buffer,
        refs: ClusterBufferRefs<'_>,
    ) -> Arc<wgpu::BindGroup> {
        let layout = Self::bind_group_layout(device);
        Arc::new(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frame_globals_bind_group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: frame_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: lights_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: refs.cluster_light_counts.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: refs.cluster_light_indices.as_entire_binding(),
                },
            ],
        }))
    }

    /// Allocates frame uniform, lights storage, minimal cluster grid `(1×1×Z)`; builds [`Self::bind_group`].
    pub fn new(device: &wgpu::Device) -> Self {
        let frame_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame_globals_uniform"),
            size: std::mem::size_of::<FrameGpuUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let lights_size = (MAX_LIGHTS * std::mem::size_of::<GpuLight>()) as u64;
        let lights_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame_lights_storage"),
            size: lights_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut cluster_cache = ClusterBufferCache::new();
        cluster_cache
            .ensure_buffers(device, (1, 1), CLUSTER_COUNT_Z)
            .expect("cluster buffers for 1x1 viewport");
        let cluster_bind_version = cluster_cache.version;
        let refs = cluster_cache
            .get_buffers((1, 1), CLUSTER_COUNT_Z)
            .expect("cluster buffers for 1x1 viewport");
        let bind_group = Self::create_bind_group(device, &frame_uniform, &lights_buffer, refs);
        Self {
            frame_uniform,
            lights_buffer,
            cluster_cache,
            bind_group,
            cluster_bind_version,
        }
    }

    /// Resizes cluster buffers when `viewport` changes; rebuilds [`Self::bind_group`] when needed.
    ///
    /// Returns `true` if the bind group was recreated.
    pub fn sync_cluster_viewport(&mut self, device: &wgpu::Device, viewport: (u32, u32)) -> bool {
        if self
            .cluster_cache
            .ensure_buffers(device, viewport, CLUSTER_COUNT_Z)
            .is_none()
        {
            return false;
        }
        let ver = self.cluster_cache.version;
        if ver == self.cluster_bind_version {
            return false;
        }
        let refs = self
            .cluster_cache
            .get_buffers(viewport, CLUSTER_COUNT_Z)
            .expect("cluster buffers after ensure_buffers");
        self.bind_group =
            Self::create_bind_group(device, &self.frame_uniform, &self.lights_buffer, refs);
        self.cluster_bind_version = ver;
        true
    }

    /// Uploads [`FrameGpuUniforms`] and packed lights for this frame.
    pub fn write_frame_uniform_and_lights(
        &self,
        queue: &wgpu::Queue,
        uniforms: &FrameGpuUniforms,
        lights: &[GpuLight],
    ) {
        queue.write_buffer(&self.frame_uniform, 0, bytemuck::bytes_of(uniforms));
        Self::write_lights_buffer_inner(queue, &self.lights_buffer, lights);
    }

    /// Uploads only the lights storage buffer (used by [`crate::render_graph::passes::ClusteredLightPass`]).
    pub fn write_lights_buffer(&self, queue: &wgpu::Queue, lights: &[GpuLight]) {
        Self::write_lights_buffer_inner(queue, &self.lights_buffer, lights);
    }

    fn write_lights_buffer_inner(
        queue: &wgpu::Queue,
        lights_buffer: &wgpu::Buffer,
        lights: &[GpuLight],
    ) {
        let n = lights.len().min(MAX_LIGHTS);
        if n > 0 {
            let bytes = bytemuck::cast_slice(&lights[..n]);
            queue.write_buffer(lights_buffer, 0, bytes);
        } else {
            queue.write_buffer(lights_buffer, 0, &[0u8; 4]);
        }
    }
}

/// Empty `@group(1)` layout for materials that declare no per-material bindings yet.
pub fn empty_material_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("empty_material_slot"),
        entries: &[],
    })
}

/// Single reusable empty bind group for [`empty_material_bind_group_layout`].
pub fn empty_material_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("empty_material_bind_group"),
        layout,
        entries: &[],
    })
}

/// Cached empty material bind group layout + instance (one per device attach).
pub struct EmptyMaterialBindGroup {
    pub layout: wgpu::BindGroupLayout,
    pub bind_group: Arc<wgpu::BindGroup>,
}

impl EmptyMaterialBindGroup {
    /// Builds layout and bind group for `@group(1)` placeholder.
    pub fn new(device: &wgpu::Device) -> Self {
        let layout = empty_material_bind_group_layout(device);
        let bind_group = Arc::new(empty_material_bind_group(device, &layout));
        Self { layout, bind_group }
    }
}
