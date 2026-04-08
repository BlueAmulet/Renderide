//! Clustered forward lighting: compute pass assigns light indices per view-space cluster.
//!
//! Dispatches over a 3D grid (`16×16` pixel tiles × exponential Z slices). Uses the same
//! [`GpuLight`] buffer and cluster storage as raster `@group(0)` ([`crate::backend::FrameGpuResources`]).

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

const CLUSTERED_LIGHT_SHADER_SRC: &str = r#"
struct GpuLight {
    position: vec3f,
    _pad0: f32,
    direction: vec3f,
    _pad1: f32,
    color: vec3f,
    intensity: f32,
    range: f32,
    spot_cos_half_angle: f32,
    light_type: u32,
    _pad_before_shadow_params: u32,
    shadow_strength: f32,
    shadow_near_plane: f32,
    shadow_bias: f32,
    shadow_normal_bias: f32,
    shadow_type: u32,
    _pad_trailing: array<u32, 3>,
}

struct ClusterParams {
    view: mat4x4f,
    proj: mat4x4f,
    inv_proj: mat4x4f,
    viewport_width: f32,
    viewport_height: f32,
    tile_size: u32,
    light_count: u32,
    cluster_count_x: u32,
    cluster_count_y: u32,
    cluster_count_z: u32,
    near_clip: f32,
    far_clip: f32,
}

@group(0) @binding(0) var<uniform> params: ClusterParams;
@group(0) @binding(1) var<storage, read> lights: array<GpuLight>;
@group(0) @binding(2) var<storage, read_write> cluster_light_counts: array<u32>;
@group(0) @binding(3) var<storage, read_write> cluster_light_indices: array<u32>;

const MAX_LIGHTS_PER_TILE: u32 = 32u;

struct TileAabb {
    min_v: vec3f,
    max_v: vec3f,
}

fn ndc_to_view(ndc: vec3f) -> vec3f {
    let clip = params.inv_proj * vec4f(ndc.x, ndc.y, ndc.z, 1.0);
    return clip.xyz / clip.w;
}

fn line_intersect_z_plane(ray_point: vec3f, z_dist: f32) -> vec3f {
    let t = z_dist / ray_point.z;
    return ray_point * t;
}

fn get_cluster_aabb(cluster_x: u32, cluster_y: u32, cluster_z: u32) -> TileAabb {
    let w = params.viewport_width;
    let h = params.viewport_height;
    let near = params.near_clip;
    let far = params.far_clip;
    let num_z = f32(params.cluster_count_z);
    let z = f32(cluster_z);

    let tile_near = -near * pow(far / near, z / num_z);
    let tile_far = -near * pow(far / near, (z + 1.0) / num_z);

    let px_min = f32(cluster_x * params.tile_size) + 0.5;
    let px_max = f32((cluster_x + 1u) * params.tile_size) - 0.5;
    let py_min = f32(cluster_y * params.tile_size) + 0.5;
    let py_max = f32((cluster_y + 1u) * params.tile_size) - 0.5;
    let ndc_left = 2.0 * px_min / w - 1.0;
    let ndc_right = 2.0 * px_max / w - 1.0;
    let ndc_top = 1.0 - 2.0 * py_min / h;
    let ndc_bottom = 1.0 - 2.0 * py_max / h;

    let v_bl = ndc_to_view(vec3f(ndc_left, ndc_bottom, 1.0));
    let v_br = ndc_to_view(vec3f(ndc_right, ndc_bottom, 1.0));
    let v_tl = ndc_to_view(vec3f(ndc_left, ndc_top, 1.0));
    let v_tr = ndc_to_view(vec3f(ndc_right, ndc_top, 1.0));

    let p_near_bl = line_intersect_z_plane(v_bl, tile_near);
    let p_near_br = line_intersect_z_plane(v_br, tile_near);
    let p_near_tl = line_intersect_z_plane(v_tl, tile_near);
    let p_near_tr = line_intersect_z_plane(v_tr, tile_near);
    let p_far_bl = line_intersect_z_plane(v_bl, tile_far);
    let p_far_br = line_intersect_z_plane(v_br, tile_far);
    let p_far_tl = line_intersect_z_plane(v_tl, tile_far);
    let p_far_tr = line_intersect_z_plane(v_tr, tile_far);

    var min_v = min(min(min(p_near_bl, p_near_br), min(p_near_tl, p_near_tr)), min(min(p_far_bl, p_far_br), min(p_far_tl, p_far_tr)));
    var max_v = max(max(max(p_near_bl, p_near_br), max(p_near_tl, p_near_tr)), max(max(p_far_bl, p_far_br), max(p_far_tl, p_far_tr)));
    return TileAabb(min_v, max_v);
}

fn sphere_aabb_intersect(center: vec3f, radius: f32, aabb_min: vec3f, aabb_max: vec3f) -> bool {
    let closest = clamp(center, aabb_min, aabb_max);
    let d = center - closest;
    return dot(d, d) <= radius * radius;
}

fn spotlight_bounds_intersect_aabb(apex: vec3f, axis: vec3f, cos_half: f32, range: f32, aabb_min: vec3f, aabb_max: vec3f) -> bool {
    if cos_half >= 0.9999 {
        return sphere_aabb_intersect(apex, range, aabb_min, aabb_max);
    }
    let axis_n = normalize(axis);
    let sin_sq = max(0.0, 1.0 - cos_half * cos_half);
    let tan_sq = sin_sq / max(cos_half * cos_half, 1e-8);
    let radius = range * sqrt(0.25 + tan_sq);
    let center = apex + axis_n * (range * 0.5);
    return sphere_aabb_intersect(center, radius, aabb_min, aabb_max);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    let cluster_count_x = params.cluster_count_x;
    let cluster_count_y = params.cluster_count_y;
    let cluster_count_z = params.cluster_count_z;
    if global_id.x >= cluster_count_x || global_id.y >= cluster_count_y || global_id.z >= cluster_count_z {
        return;
    }
    let cluster_id = global_id.x + cluster_count_x * (global_id.y + cluster_count_y * global_id.z);
    let cluster_x = global_id.x;
    let cluster_y = global_id.y;
    let cluster_z = global_id.z;

    let aabb = get_cluster_aabb(cluster_x, cluster_y, cluster_z);
    let aabb_min = aabb.min_v;
    let aabb_max = aabb.max_v;

    var count: u32 = 0u;
    let base_idx = cluster_id * MAX_LIGHTS_PER_TILE;

    for (var i = 0u; i < params.light_count; i++) {
        if count >= MAX_LIGHTS_PER_TILE {
            break;
        }
        let light = lights[i];
        let pos_view = (params.view * vec4f(light.position.x, light.position.y, light.position.z, 1.0)).xyz;
        let dir_view = (params.view * vec4f(light.direction.x, light.direction.y, light.direction.z, 0.0)).xyz;

        var intersects = false;
        if light.light_type == 0u {
            intersects = sphere_aabb_intersect(pos_view, light.range, aabb_min, aabb_max);
        } else if light.light_type == 1u {
            intersects = true;
        } else {
            let dir_len_sq = dot(dir_view, dir_view);
            let axis = select(
                vec3f(0.0, 0.0, 1.0),
                dir_view * inverseSqrt(dir_len_sq),
                dir_len_sq > 1e-16
            );
            intersects = spotlight_bounds_intersect_aabb(pos_view, axis, light.spot_cos_half_angle, light.range, aabb_min, aabb_max);
        }

        if intersects {
            cluster_light_indices[base_idx + count] = i;
            count += 1u;
        }
    }

    cluster_light_counts[cluster_id] = count;
}
"#;

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
