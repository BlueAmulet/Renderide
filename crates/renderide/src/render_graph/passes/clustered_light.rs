//! Clustered forward lighting: compute pass assigns light indices per view-space cluster.
//!
//! Dispatches over a 3D grid (`16×16` pixel tiles × exponential Z slices). Uses the same
//! [`GpuLight`] buffer and cluster storage as raster `@group(0)` ([`crate::backend::FrameGpuResources`]).
//!
//! WGSL source: `shaders/source/compute/clustered_light.wgsl` (included at compile time).

use std::num::NonZeroU64;
use std::sync::OnceLock;

use bytemuck::{Pod, Zeroable};
use glam::Mat4;

use crate::backend::{GpuLight, MAX_LIGHTS};
use crate::backend::{CLUSTER_COUNT_Z, CLUSTER_PARAMS_UNIFORM_SIZE, TILE_SIZE};
use crate::render_graph::camera::{
    clamp_desktop_fov_degrees, effective_head_output_clip_planes, reverse_z_perspective,
    view_matrix_from_render_transform,
};
use crate::render_graph::context::RenderPassContext;
use crate::render_graph::error::RenderPassError;
use crate::render_graph::pass::RenderPass;
use crate::render_graph::resources::{PassResources, ResourceSlot};

/// CPU layout for the compute shader `ClusterParams` uniform (WGSL `struct` + 16-byte tail pad).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ClusterParams {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    inv_proj: [[f32; 4]; 4],
    viewport_width: f32,
    viewport_height: f32,
    tile_size: u32,
    light_count: u32,
    cluster_count_x: u32,
    cluster_count_y: u32,
    cluster_count_z: u32,
    near_clip: f32,
    far_clip: f32,
    _pad: [u8; 16],
}

const CLUSTERED_LIGHT_SHADER_SRC: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/shaders/source/compute/clustered_light.wgsl"
));

fn compute_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("clustered_light_compute"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(CLUSTER_PARAMS_UNIFORM_SIZE),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(std::mem::size_of::<GpuLight>() as u64),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(4),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(4),
                },
                count: None,
            },
        ],
    })
}

fn ensure_compute_pipeline(
    device: &wgpu::Device,
) -> &'static (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    static CACHE: OnceLock<(wgpu::ComputePipeline, wgpu::BindGroupLayout)> = OnceLock::new();
    CACHE.get_or_init(|| {
        let bgl = compute_bind_group_layout(device);
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("clustered_light_pipeline_layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("clustered_light"),
            source: wgpu::ShaderSource::Wgsl(CLUSTERED_LIGHT_SHADER_SRC.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("clustered_light"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        (pipeline, bgl)
    })
}

fn write_cluster_params_padded(queue: &wgpu::Queue, buf: &wgpu::Buffer, params: &ClusterParams) {
    let mut padded = [0u8; CLUSTER_PARAMS_UNIFORM_SIZE as usize];
    let src = bytemuck::bytes_of(params);
    padded[..src.len()].copy_from_slice(src);
    queue.write_buffer(buf, 0, &padded);
}

/// Builds per-cluster light lists before the world forward pass.
#[derive(Debug, Default)]
pub struct ClusteredLightPass {
    logged_active_once: bool,
}

impl ClusteredLightPass {
    /// Creates a clustered light pass (pipeline is created lazily on first execute).
    pub fn new() -> Self {
        Self {
            logged_active_once: false,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn build_params(
        scene_view: Mat4,
        proj: Mat4,
        viewport: (u32, u32),
        cluster_count_x: u32,
        cluster_count_y: u32,
        light_count: u32,
        near: f32,
        far: f32,
    ) -> ClusterParams {
        let inv_proj = proj.inverse();
        ClusterParams {
            view: scene_view.to_cols_array_2d(),
            proj: proj.to_cols_array_2d(),
            inv_proj: inv_proj.to_cols_array_2d(),
            viewport_width: viewport.0 as f32,
            viewport_height: viewport.1 as f32,
            tile_size: TILE_SIZE,
            light_count,
            cluster_count_x,
            cluster_count_y,
            cluster_count_z: CLUSTER_COUNT_Z,
            near_clip: near.max(0.01),
            far_clip: far,
            _pad: [0; 16],
        }
    }
}

impl RenderPass for ClusteredLightPass {
    fn name(&self) -> &str {
        "ClusteredLight"
    }

    fn resources(&self) -> PassResources {
        PassResources {
            reads: Vec::new(),
            writes: vec![ResourceSlot::ClusterBuffers, ResourceSlot::LightBuffer],
        }
    }

    fn execute(&mut self, ctx: &mut RenderPassContext<'_>) -> Result<(), RenderPassError> {
        let Some(frame) = ctx.frame.as_mut() else {
            return Ok(());
        };

        let (vw, vh) = frame.viewport_px;
        if vw == 0 || vh == 0 {
            return Ok(());
        }

        let lights_upload: Vec<GpuLight> = frame.backend.frame_lights().to_vec();
        let Some(fgpu) = frame.backend.frame_gpu_mut() else {
            return Ok(());
        };

        fgpu.sync_cluster_viewport(ctx.device, (vw, vh));

        let lights = lights_upload.as_slice();
        {
            let queue = ctx.queue.lock().unwrap_or_else(|e| e.into_inner());
            fgpu.write_lights_buffer(&queue, lights);
        }

        let Some(refs) = fgpu.cluster_cache.get_buffers((vw, vh), CLUSTER_COUNT_Z) else {
            logger::trace!("ClusteredLight: cluster buffers missing after sync");
            return Ok(());
        };

        let hc = frame.host_camera;
        let scene = frame.scene;
        let (near, far) = effective_head_output_clip_planes(
            hc.near_clip,
            hc.far_clip,
            hc.output_device,
            scene
                .active_main_space()
                .map(|space| space.root_transform.scale),
        );
        let aspect = vw as f32 / vh.max(1) as f32;
        let fov_rad = clamp_desktop_fov_degrees(hc.desktop_fov_degrees).to_radians();
        let proj = reverse_z_perspective(aspect, fov_rad, near, far);
        let scene_view = scene
            .active_main_space()
            .map(|s| view_matrix_from_render_transform(&s.view_transform))
            .unwrap_or(Mat4::IDENTITY);

        let cluster_count_x = vw.div_ceil(TILE_SIZE);
        let cluster_count_y = vh.div_ceil(TILE_SIZE);
        let light_count = lights_upload.len().min(MAX_LIGHTS) as u32;

        let params = Self::build_params(
            scene_view,
            proj,
            (vw, vh),
            cluster_count_x,
            cluster_count_y,
            light_count,
            near,
            far,
        );

        let queue = ctx.queue.lock().unwrap_or_else(|e| e.into_inner());
        write_cluster_params_padded(&queue, refs.params_buffer, &params);

        let (pipeline, bgl) = ensure_compute_pipeline(ctx.device);
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("clustered_light_bind_group"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: refs.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: fgpu.lights_buffer.as_entire_binding(),
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
        });

        let mut pass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("clustered_light"),
                timestamp_writes: None,
            });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(
            cluster_count_x.div_ceil(8),
            cluster_count_y.div_ceil(8),
            CLUSTER_COUNT_Z,
        );
        drop(pass);

        if !self.logged_active_once {
            self.logged_active_once = true;
            logger::info!(
                "ClusteredLight active (grid {}x{}x{} lights={})",
                cluster_count_x,
                cluster_count_y,
                CLUSTER_COUNT_Z,
                light_count
            );
        }

        Ok(())
    }
}
