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

use hashbrown::HashMap;

use crate::backend::ClusterBufferRefs;
use crate::backend::GpuLight;
use crate::backend::{CLUSTER_COUNT_Z, CLUSTER_PARAMS_UNIFORM_SIZE, TILE_SIZE};
use crate::gpu::GpuLimits;
use crate::render_graph::cluster_frame::{
    cluster_frame_params, cluster_frame_params_stereo, ClusterFrameParams,
};
use crate::render_graph::context::ComputePassCtx;
use crate::render_graph::error::{RenderPassError, SetupError};
use crate::render_graph::frame_params::HostCameraFrame;
use crate::render_graph::pass::{ComputePass, PassBuilder};
use crate::render_graph::resources::{
    BufferAccess, BufferHandle, ImportedBufferHandle, StorageAccess,
};
use crate::render_graph::OcclusionViewId;
use crate::scene::SceneCoordinator;

/// CPU layout for the compute shader `ClusterParams` uniform (WGSL `struct` + tail pad).
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
    cluster_offset: u32,
    _pad: [u8; 8],
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
                    has_dynamic_offset: true,
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

fn write_cluster_params_padded(
    queue: &wgpu::Queue,
    buf: &wgpu::Buffer,
    params: &ClusterParams,
    buf_offset: u64,
) {
    let mut padded = [0u8; CLUSTER_PARAMS_UNIFORM_SIZE as usize];
    let src = bytemuck::bytes_of(params);
    padded[..src.len()].copy_from_slice(src);
    queue.write_buffer(buf, buf_offset, &padded);
}

/// Descriptor for building the `ClusterParams` uniform from scene matrices and cluster grid metadata.
struct ClusterParamsDesc {
    scene_view: Mat4,
    proj: Mat4,
    viewport: (u32, u32),
    cluster_count_x: u32,
    cluster_count_y: u32,
    light_count: u32,
    near: f32,
    far: f32,
    cluster_offset: u32,
}

/// GPU and uniform state for per-eye clustered light compute dispatches.
struct ClusteredLightEyePassEnv<'a> {
    encoder: &'a mut wgpu::CommandEncoder,
    queue: &'a wgpu::Queue,
    pipeline: &'a wgpu::ComputePipeline,
    bind_group: &'a wgpu::BindGroup,
    params_buffer: &'a wgpu::Buffer,
    eye_params: &'a [ClusterFrameParams],
    clusters_per_eye: u32,
    light_count: u32,
    viewport: (u32, u32),
    gpu_limits: &'a GpuLimits,
}

/// Per-eye cluster compute dispatches (params upload + 3D grid).
fn run_clustered_light_eye_passes(env: ClusteredLightEyePassEnv<'_>) {
    for (eye_idx, cfp) in env.eye_params.iter().enumerate() {
        let cluster_offset = (eye_idx as u32) * env.clusters_per_eye;
        let buf_offset = (eye_idx as u64) * CLUSTER_PARAMS_UNIFORM_SIZE;
        let params = ClusteredLightPass::build_params(ClusterParamsDesc {
            scene_view: cfp.world_to_view,
            proj: cfp.proj,
            viewport: env.viewport,
            cluster_count_x: cfp.cluster_count_x,
            cluster_count_y: cfp.cluster_count_y,
            light_count: env.light_count,
            near: cfp.near_clip,
            far: cfp.far_clip,
            cluster_offset,
        });
        write_cluster_params_padded(env.queue, env.params_buffer, &params, buf_offset);

        let dx = cfp.cluster_count_x.div_ceil(8);
        let dy = cfp.cluster_count_y.div_ceil(8);
        let dz = CLUSTER_COUNT_Z;
        if !env.gpu_limits.compute_dispatch_fits(dx, dy, dz) {
            logger::warn!(
                "ClusteredLight: dispatch {}×{}×{} exceeds max_compute_workgroups_per_dimension ({})",
                dx,
                dy,
                dz,
                env.gpu_limits.max_compute_workgroups_per_dimension()
            );
            continue;
        }

        let mut pass = env
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("clustered_light"),
                timestamp_writes: None,
            });
        pass.set_pipeline(env.pipeline);
        pass.set_bind_group(0, env.bind_group, &[buf_offset as u32]);
        pass.dispatch_workgroups(dx, dy, dz);
    }
}

/// Resolves mono or stereo [`ClusterFrameParams`] rows for the current host camera and viewport.
fn clustered_light_eye_params_for_viewport(
    stereo: bool,
    hc: &HostCameraFrame,
    scene: &SceneCoordinator,
    viewport: (u32, u32),
) -> Option<Vec<ClusterFrameParams>> {
    if stereo {
        if let Some((left, right)) = cluster_frame_params_stereo(hc, scene, viewport) {
            Some(vec![left, right])
        } else {
            cluster_frame_params(hc, scene, viewport).map(|mono| vec![mono])
        }
    } else {
        cluster_frame_params(hc, scene, viewport).map(|mono| vec![mono])
    }
}

/// Builds per-cluster light lists before the world forward pass.
#[derive(Debug)]
pub struct ClusteredLightPass {
    resources: ClusteredLightGraphResources,
    logged_active_once: bool,
    /// Per-view compute bind group cache: maps [`OcclusionViewId`] to
    /// `(cluster_version, bind_group)`. Rebuilt when the per-view cluster buffer version changes.
    cached_compute_bind_groups: HashMap<OcclusionViewId, (u64, wgpu::BindGroup)>,
}

/// Graph resources used by [`ClusteredLightPass`].
#[derive(Clone, Copy, Debug)]
pub struct ClusteredLightGraphResources {
    /// Imported light storage buffer.
    pub lights: ImportedBufferHandle,
    /// Imported per-cluster light-count storage buffer.
    pub cluster_light_counts: ImportedBufferHandle,
    /// Imported per-cluster light-index storage buffer.
    pub cluster_light_indices: ImportedBufferHandle,
    /// Transient uniform buffer for per-eye cluster parameters.
    pub params: BufferHandle,
}

impl ClusteredLightPass {
    /// Creates a clustered light pass (pipeline is created lazily on first execute).
    pub fn new(resources: ClusteredLightGraphResources) -> Self {
        Self {
            resources,
            logged_active_once: false,
            cached_compute_bind_groups: HashMap::new(),
        }
    }

    fn build_params(desc: ClusterParamsDesc) -> ClusterParams {
        let inv_proj = desc.proj.inverse();
        let near_clip = desc.near.max(0.01);
        ClusterParams {
            view: desc.scene_view.to_cols_array_2d(),
            proj: desc.proj.to_cols_array_2d(),
            inv_proj: inv_proj.to_cols_array_2d(),
            viewport_width: desc.viewport.0 as f32,
            viewport_height: desc.viewport.1 as f32,
            tile_size: TILE_SIZE,
            light_count: desc.light_count,
            cluster_count_x: desc.cluster_count_x,
            cluster_count_y: desc.cluster_count_y,
            cluster_count_z: CLUSTER_COUNT_Z,
            near_clip,
            far_clip: desc.far,
            cluster_offset: desc.cluster_offset,
            _pad: [0u8; 8],
        }
    }

    /// Returns the compute bind group for `view_id`, rebuilding it when the cluster version changes.
    fn ensure_cluster_compute_bind_group(
        &mut self,
        device: &wgpu::Device,
        view_id: OcclusionViewId,
        cluster_ver: u64,
        refs: ClusterBufferRefs<'_>,
        lights_buffer: &wgpu::Buffer,
        bgl: &wgpu::BindGroupLayout,
    ) -> Result<&wgpu::BindGroup, RenderPassError> {
        let needs_rebuild = self
            .cached_compute_bind_groups
            .get(&view_id)
            .is_none_or(|(ver, _)| *ver != cluster_ver);
        if needs_rebuild {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("clustered_light_compute"),
                layout: bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: refs.params_buffer,
                            offset: 0,
                            size: NonZeroU64::new(CLUSTER_PARAMS_UNIFORM_SIZE),
                        }),
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
            });
            self.cached_compute_bind_groups
                .insert(view_id, (cluster_ver, bind_group));
        }
        self.cached_compute_bind_groups
            .get(&view_id)
            .map(|(_, bg)| bg)
            .ok_or(RenderPassError::ClusteredLightBindGroupMissing)
    }
}

impl ComputePass for ClusteredLightPass {
    fn name(&self) -> &str {
        "ClusteredLight"
    }

    fn setup(&mut self, b: &mut PassBuilder<'_>) -> Result<(), SetupError> {
        b.compute();
        b.import_buffer(
            self.resources.lights,
            BufferAccess::Storage {
                stages: wgpu::ShaderStages::COMPUTE,
                access: StorageAccess::ReadOnly,
            },
        );
        b.import_buffer(
            self.resources.cluster_light_counts,
            BufferAccess::Storage {
                stages: wgpu::ShaderStages::COMPUTE,
                access: StorageAccess::WriteOnly,
            },
        );
        b.import_buffer(
            self.resources.cluster_light_indices,
            BufferAccess::Storage {
                stages: wgpu::ShaderStages::COMPUTE,
                access: StorageAccess::WriteOnly,
            },
        );
        b.write_buffer(
            self.resources.params,
            BufferAccess::Uniform {
                stages: wgpu::ShaderStages::COMPUTE,
                dynamic_offset: true,
            },
        );
        Ok(())
    }

    fn record(&mut self, ctx: &mut ComputePassCtx<'_, '_, '_>) -> Result<(), RenderPassError> {
        let Some(frame) = ctx.frame.as_mut() else {
            return Ok(());
        };

        let (vw, vh) = frame.viewport_px;
        if vw == 0 || vh == 0 {
            return Ok(());
        }

        let hc = frame.host_camera;
        let scene = frame.scene;
        let stereo = hc.vr_active && hc.stereo_views.is_some() && frame.multiview_stereo;
        let view_id = frame.occlusion_view;

        let light_count = frame.frame_resources.frame_light_count_u32();

        let queue: &wgpu::Queue = ctx.queue.as_ref();

        // Sync global cluster viewport + coalesce lights upload.
        if frame
            .frame_resources
            .sync_cluster_viewport_ensure_lights_upload(ctx.device, queue, (vw, vh), stereo)
            .is_none()
        {
            return Ok(());
        };

        // Resolve per-view cluster refs (independent from the global cluster_cache).
        let Some(per_view_state) = frame.frame_resources.per_view_frame(view_id) else {
            logger::trace!("ClusteredLight: per-view frame state missing for {view_id:?}");
            return Ok(());
        };
        let Some(refs) = per_view_state.cluster_buffer_refs() else {
            logger::trace!("ClusteredLight: per-view cluster buffers missing for {view_id:?}");
            return Ok(());
        };
        let cluster_ver = per_view_state.cluster_cache.version;

        let viewport = (vw, vh);

        let Some(eye_params) =
            clustered_light_eye_params_for_viewport(stereo, &hc, scene, viewport)
        else {
            return Ok(());
        };

        let clusters_per_eye =
            eye_params[0].cluster_count_x * eye_params[0].cluster_count_y * CLUSTER_COUNT_Z;

        if light_count == 0 {
            let total_clusters = clusters_per_eye as u64 * eye_params.len() as u64;
            let counts_bytes = total_clusters * std::mem::size_of::<u32>() as u64;
            ctx.encoder
                .clear_buffer(refs.cluster_light_counts, 0, Some(counts_bytes));
            return Ok(());
        }

        // Resolve lights buffer from shared frame GPU resources.
        let Some(lights_buffer) = frame
            .frame_resources
            .frame_gpu()
            .map(|fgpu| fgpu.lights_buffer.clone())
        else {
            return Ok(());
        };

        let (pipeline, bgl) = ensure_compute_pipeline(ctx.device);
        let bind_group = self.ensure_cluster_compute_bind_group(
            ctx.device,
            view_id,
            cluster_ver,
            refs,
            &lights_buffer,
            bgl,
        )?;

        run_clustered_light_eye_passes(ClusteredLightEyePassEnv {
            encoder: ctx.encoder,
            queue,
            pipeline,
            bind_group,
            params_buffer: refs.params_buffer,
            eye_params: &eye_params,
            clusters_per_eye,
            light_count,
            viewport,
            gpu_limits: ctx.gpu_limits,
        });

        if !self.logged_active_once {
            self.logged_active_once = true;
            let eye_count = eye_params.len();
            logger::info!(
                "ClusteredLight active (grid {}x{}x{} lights={} eyes={})",
                eye_params[0].cluster_count_x,
                eye_params[0].cluster_count_y,
                CLUSTER_COUNT_Z,
                light_count,
                eye_count,
            );
        }

        Ok(())
    }
}
