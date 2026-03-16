//! Render graph: RenderPass trait, contexts, and pass implementations.
//!
//! Extension point for shadows, post-processing, UI, probes.

use nalgebra::{Matrix4, Vector3};

use super::SpaceDrawBatch;
use crate::gpu::{
    GpuMeshBuffers, GpuState, PipelineKey, PipelineManager, PipelineVariant, UniformData,
};
use crate::scene::render_transform_to_matrix;
use crate::session::Session;

/// Errors that can occur during render pass execution.
#[derive(Debug)]
pub enum RenderPassError {
    /// Wrapper for wgpu surface errors when acquiring the current texture.
    Surface(wgpu::SurfaceError),
}

impl From<wgpu::SurfaceError> for RenderPassError {
    fn from(e: wgpu::SurfaceError) -> Self {
        RenderPassError::Surface(e)
    }
}

/// Color and optional depth texture views for the current render target.
pub struct RenderTarget<'a> {
    /// Color attachment view.
    pub color_view: &'a wgpu::TextureView,
    /// Optional depth attachment view.
    pub depth_view: Option<&'a wgpu::TextureView>,
}

/// Per-pass context passed to `RenderPass::execute`.
pub struct RenderPassContext<'a> {
    /// GPU state including device, queue, mesh cache, and depth texture.
    pub gpu: &'a mut GpuState,
    /// Session for scene, assets, and view state.
    pub session: &'a Session,
    /// Draw batches for this frame.
    pub draw_batches: &'a [SpaceDrawBatch],
    /// Pipeline manager for mesh pipelines.
    pub pipeline_manager: &'a mut PipelineManager,
    /// Viewport dimensions (width, height).
    pub viewport: (u32, u32),
    /// Primary projection matrix; passes build view-proj per batch as needed.
    pub proj: Matrix4<f32>,
    /// Current color and depth attachments.
    pub render_target: RenderTarget<'a>,
    /// Command encoder for this frame; pass records into this.
    pub encoder: &'a mut wgpu::CommandEncoder,
    /// Optional timestamp query set for GPU pass timing.
    pub timestamp_query_set: Option<&'a wgpu::QuerySet>,
}

/// Frame-level context created at the start of `render_frame`.
pub struct RenderGraphContext<'a> {
    /// GPU state.
    pub gpu: &'a mut GpuState,
    /// Session.
    pub session: &'a Session,
    /// Draw batches.
    pub draw_batches: &'a [SpaceDrawBatch],
    /// Pipeline manager.
    pub pipeline_manager: &'a mut PipelineManager,
    /// Viewport (width, height).
    pub viewport: (u32, u32),
    /// Color attachment view.
    pub color_view: &'a wgpu::TextureView,
    /// Optional depth attachment view.
    pub depth_view: Option<&'a wgpu::TextureView>,
    /// Primary projection matrix.
    pub proj: Matrix4<f32>,
    /// Optional timestamp query set for GPU pass timing.
    pub timestamp_query_set: Option<&'a wgpu::QuerySet>,
    /// Optional resolve buffer for timestamp readback.
    pub timestamp_resolve_buffer: Option<&'a wgpu::Buffer>,
    /// Optional staging buffer for timestamp readback.
    pub timestamp_staging_buffer: Option<&'a wgpu::Buffer>,
}

/// Trait for render passes that can be executed by the render graph.
pub trait RenderPass {
    /// Human-readable name for debugging.
    fn name(&self) -> &str;

    /// Executes the pass, recording commands into the context's encoder.
    fn execute(&mut self, ctx: &mut RenderPassContext) -> Result<(), RenderPassError>;
}

/// Graph of render passes executed in sequence each frame.
pub struct RenderGraph {
    passes: Vec<Box<dyn RenderPass>>,
}

impl RenderGraph {
    /// Creates an empty render graph.
    pub fn new() -> Self {
        Self {
            passes: Vec::new(),
        }
    }

    /// Adds a pass to the graph. Passes execute in insertion order.
    pub fn add_pass(&mut self, pass: Box<dyn RenderPass>) {
        self.passes.push(pass);
    }

    /// Executes all passes in order, recording into a new command encoder.
    pub fn execute(&mut self, ctx: &mut RenderGraphContext) -> Result<(), RenderPassError> {
        let mut encoder = ctx
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        let render_target = RenderTarget {
            color_view: ctx.color_view,
            depth_view: ctx.depth_view,
        };

        let mut pass_ctx = RenderPassContext {
            gpu: ctx.gpu,
            session: ctx.session,
            draw_batches: ctx.draw_batches,
            pipeline_manager: ctx.pipeline_manager,
            viewport: ctx.viewport,
            proj: ctx.proj,
            render_target,
            encoder: &mut encoder,
            timestamp_query_set: ctx.timestamp_query_set,
        };

        for pass in &mut self.passes {
            pass.execute(&mut pass_ctx)?;
        }

        if let (Some(query_set), Some(resolve_buffer), Some(staging_buffer)) = (
            ctx.timestamp_query_set,
            ctx.timestamp_resolve_buffer,
            ctx.timestamp_staging_buffer,
        ) {
            encoder.resolve_query_set(query_set, 0..2, resolve_buffer, 0);
            encoder.copy_buffer_to_buffer(resolve_buffer, 0, staging_buffer, 0, resolve_buffer.size());
        }

        ctx.gpu.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }
}

impl Default for RenderGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Mesh render pass: draws meshes from draw batches using normal, UV, and skinned pipelines.
pub struct MeshRenderPass;

impl MeshRenderPass {
    /// Creates a new mesh render pass.
    pub fn new() -> Self {
        Self
    }
}

impl Default for MeshRenderPass {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderPass for MeshRenderPass {
    fn name(&self) -> &str {
        "mesh"
    }

    fn execute(&mut self, ctx: &mut RenderPassContext) -> Result<(), RenderPassError> {
        let mesh_assets = ctx.session.asset_registry();

        for batch in ctx.draw_batches {
            for d in &batch.draws {
                if d.mesh_asset_id < 0 {
                    continue;
                }
                let Some(mesh) = mesh_assets.get_mesh(d.mesh_asset_id) else {
                    continue;
                };
                if mesh.vertex_count <= 0 || mesh.index_count <= 0 {
                    continue;
                }
                if !ctx.gpu.mesh_buffer_cache.contains_key(&d.mesh_asset_id) {
                    let stride =
                        crate::assets::compute_vertex_stride(&mesh.vertex_attributes) as usize;
                    let stride = if stride > 0 {
                        stride
                    } else {
                        crate::gpu::compute_vertex_stride_from_mesh(mesh)
                    };
                    if let Some(b) = crate::gpu::create_mesh_buffers(&ctx.gpu.device, mesh, stride)
                    {
                        ctx.gpu.mesh_buffer_cache.insert(d.mesh_asset_id, b);
                    }
                }
            }
        }

        struct BatchedDraw<'a> {
            buffers: &'a GpuMeshBuffers,
            mvp: Matrix4<f32>,
            model: Matrix4<f32>,
            pipeline_variant: PipelineVariant,
        }
        /// Collected skinned draw for batch upload.
        struct SkinnedBatchedDraw<'a> {
            buffers: &'a GpuMeshBuffers,
            mvp: Matrix4<f32>,
            bone_matrices: Vec<[[f32; 4]; 4]>,
        }
        let mut non_skinned_draws: Vec<BatchedDraw<'_>> = Vec::new();
        let mut skinned_draws: Vec<SkinnedBatchedDraw<'_>> = Vec::new();
        let scene_graph = ctx.session.scene_graph();
        let debug_skinned = ctx.session.render_config().debug_skinned;
        let mut first_skinned_logged = false;

        let frame_index = ctx.pipeline_manager.advance_frame();

        let timestamp_writes = ctx.timestamp_query_set.map(|query_set| {
            wgpu::RenderPassTimestampWrites {
                query_set,
                beginning_of_pass_write_index: Some(0),
                end_of_pass_write_index: Some(1),
            }
        });

        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("mesh pass"),
            timestamp_writes: timestamp_writes,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: ctx.render_target.color_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.8,
                        b: 0.0,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: ctx.render_target.depth_view.map(|dv| {
                wgpu::RenderPassDepthStencilAttachment {
                    view: dv,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }
            }),
            occlusion_query_set: None,
            multiview_mask: None,
        });

        for batch in ctx.draw_batches {
            let mut batch_vt = batch.view_transform;
            batch_vt.scale = filter_scale(batch_vt.scale);
            let view_mat = render_transform_to_matrix(&batch_vt)
                .try_inverse()
                .unwrap_or_else(Matrix4::identity);
            let view_mat = apply_view_handedness_fix(view_mat);
            let view_proj = ctx.proj * view_mat;

            for d in &batch.draws {
                let (buffers_ref, mesh) = if d.mesh_asset_id >= 0 {
                    let Some(mesh) = mesh_assets.get_mesh(d.mesh_asset_id) else {
                        continue;
                    };
                    if mesh.vertex_count <= 0 || mesh.index_count <= 0 {
                        continue;
                    }
                    let Some(b) = ctx.gpu.mesh_buffer_cache.get(&d.mesh_asset_id) else {
                        continue;
                    };
                    (b, mesh)
                } else {
                    continue;
                };

                let model_mvp = view_proj * d.model_matrix;

                if d.is_skinned {
                    let Some(bind_poses) = mesh.bind_poses.as_ref() else {
                        logger::trace!(
                            "Skinned draw skipped: mesh missing bind_poses (mesh={})",
                            d.mesh_asset_id
                        );
                        continue;
                    };
                    let Some(ids) = d.bone_transform_ids.as_deref() else {
                        logger::trace!(
                            "Skinned draw skipped: bone_transform_ids missing or empty (mesh={})",
                            d.mesh_asset_id
                        );
                        continue;
                    };
                    if ids.is_empty() {
                        logger::trace!(
                            "Skinned draw skipped: bone_transform_ids missing or empty (mesh={})",
                            d.mesh_asset_id
                        );
                        continue;
                    }
                    if ids.len() > bind_poses.len() {
                        logger::trace!(
                            "Skinned draw skipped: bone_transform_ids.len()={} > bind_poses.len()={} (mesh={})",
                            ids.len(),
                            bind_poses.len(),
                            d.mesh_asset_id
                        );
                        continue;
                    }
                    let Some(_) = buffers_ref.vertex_buffer_skinned.as_ref() else {
                        logger::trace!(
                            "Skinned draw skipped: vertex_buffer_skinned missing (mesh={})",
                            d.mesh_asset_id
                        );
                        continue;
                    };
                    if debug_skinned && !first_skinned_logged {
                        first_skinned_logged = true;
                        let first_3_ids: Vec<i32> = ids.iter().take(3).copied().collect();
                        let first_bind = bind_poses.first().map(|b| format!("{:?}", b)).unwrap_or_else(|| "none".to_string());
                        let (first_vert_indices, first_vert_weights) = if let (Some(bc), Some(bw)) = (mesh.bone_counts.as_ref(), mesh.bone_weights.as_ref()) {
                            let n = bc.get(0).copied().unwrap_or(0) as usize;
                            let n = n.min(4);
                            let mut indices = [0i32; 4];
                            let mut weights = [0.0f32; 4];
                            for j in 0..n {
                                if j * 8 + 8 <= bw.len() {
                                    let idx = i32::from_le_bytes(bw[j * 8 + 4..j * 8 + 8].try_into().unwrap_or([0; 4]));
                                    let w = f32::from_le_bytes(bw[j * 8..j * 8 + 4].try_into().unwrap_or([0; 4]));
                                    indices[j] = idx;
                                    weights[j] = w;
                                }
                            }
                            (format!("{:?}", indices), format!("{:?}", weights))
                        } else {
                            ("n/a".to_string(), "n/a".to_string())
                        };
                        logger::debug!(
                            "skinned draw: mesh={} node_id={} bone_ids_len={} first_3_ids={:?} first_bind={} first_vert_indices={} first_vert_weights={} has_skinned_vb={}",
                            d.mesh_asset_id,
                            d.node_id,
                            ids.len(),
                            first_3_ids,
                            first_bind,
                            first_vert_indices,
                            first_vert_weights,
                            buffers_ref.vertex_buffer_skinned.is_some()
                        );
                    }
                    // Correct skinned MVP logic (matches Unity SkinnedMeshRenderer for modular avatars)
                    let mut skinned_mvp = if ctx.session.render_config().skinned_use_root_bone {
                        let root_id = d.root_bone_transform_id.filter(|&id| id >= 0);
                        match root_id.and_then(|id| {
                            scene_graph.get_world_matrix(batch.space_id, id as usize)
                        }) {
                            Some(root_world) => view_proj * root_world,
                            None => view_proj,
                        }
                    } else {
                        // Default path: vertices are already in world space (standard case)
                        view_proj
                    };
                    if ctx.session.render_config().skinned_flip_handedness {
                        let z_flip = Matrix4::new_nonuniform_scaling(&Vector3::new(1.0, 1.0, -1.0));
                        skinned_mvp = skinned_mvp * z_flip;
                    }
                    let root_bone = if ctx.session.render_config().skinned_use_root_bone {
                        d.root_bone_transform_id
                    } else {
                        None
                    };
                    let bone_matrices = scene_graph.compute_bone_matrices(
                        batch.space_id,
                        ids,
                        bind_poses,
                        root_bone,
                    );
                    skinned_draws.push(SkinnedBatchedDraw {
                        buffers: buffers_ref,
                        mvp: skinned_mvp,
                        bone_matrices,
                    });
                    continue;
                }

                non_skinned_draws.push(BatchedDraw {
                    buffers: buffers_ref,
                    mvp: model_mvp,
                    model: d.model_matrix,
                    pipeline_variant: d.pipeline_variant.clone(),
                });
            }
        }

        if !skinned_draws.is_empty() {
            if let Some(skinned) = ctx.pipeline_manager.get_pipeline(
                PipelineKey(None, PipelineVariant::Skinned),
                &ctx.gpu.device,
                &ctx.gpu.config,
            ) {
                let mvp_bones: Vec<_> = skinned_draws
                    .iter()
                    .map(|d| (d.mvp, d.bone_matrices.as_slice()))
                    .collect();
                skinned.upload_skinned_batch(&ctx.gpu.queue, &mvp_bones, frame_index);
                for (j, d) in skinned_draws.iter().enumerate() {
                    skinned.bind(&mut pass, Some(j as u32), frame_index);
                    skinned.draw_skinned(
                        &mut pass,
                        d.buffers,
                        &UniformData::Skinned {
                            mvp: d.mvp,
                            bone_matrices: &d.bone_matrices,
                        },
                    );
                }
            }
        }

        let mut i = 0;
        while i < non_skinned_draws.len() {
            let variant = non_skinned_draws[i].pipeline_variant.clone();
            let group_end = non_skinned_draws[i..]
                .iter()
                .take_while(|d| d.pipeline_variant == variant)
                .count();
            let group = &non_skinned_draws[i..i + group_end];

            let pipeline_key = PipelineKey(None, variant);
            let Some(pipeline) = ctx.pipeline_manager.get_pipeline(
                pipeline_key,
                &ctx.gpu.device,
                &ctx.gpu.config,
            ) else {
                i += group_end;
                continue;
            };

            let mvp_models: Vec<_> = group.iter().map(|d| (d.mvp, d.model)).collect();
            pipeline.upload_batch(&ctx.gpu.queue, &mvp_models, frame_index);

            for (j, d) in group.iter().enumerate() {
                pipeline.bind(&mut pass, Some(j as u32), frame_index);
                pipeline.draw_mesh(
                    &mut pass,
                    d.buffers,
                    &UniformData::Simple {
                        mvp: d.mvp,
                        model: d.model,
                    },
                );
            }

            i += group_end;
        }

        Ok(())
    }
}

fn filter_scale(scale: Vector3<f32>) -> Vector3<f32> {
    const MIN_SCALE: f32 = 1e-8;
    if scale.x.abs() < MIN_SCALE || scale.y.abs() < MIN_SCALE || scale.z.abs() < MIN_SCALE {
        Vector3::new(1.0, 1.0, 1.0)
    } else {
        scale
    }
}

fn apply_view_handedness_fix(view: Matrix4<f32>) -> Matrix4<f32> {
    let z_flip = Matrix4::new_nonuniform_scaling(&Vector3::new(1.0, 1.0, -1.0));
    z_flip * view
}

/// Reverse-Z projection matrix for the given aspect and frustum.
pub fn reverse_z_projection(
    aspect: f32,
    vertical_fov: f32,
    near: f32,
    far: f32,
) -> Matrix4<f32> {
    let vertical_half = vertical_fov / 2.0;
    let tan_vertical_half = vertical_half.tan();
    let horizontal_fov = (tan_vertical_half * aspect)
        .atan()
        .clamp(0.1, std::f32::consts::FRAC_PI_2 - 0.1)
        * 2.0;
    let tan_horizontal_half = (horizontal_fov / 2.0).tan();
    let f_x = 1.0 / tan_horizontal_half;
    let f_y = 1.0 / tan_vertical_half;

    Matrix4::new(
        f_x,
        0.0,
        0.0,
        0.0,
        0.0,
        f_y,
        0.0,
        0.0,
        0.0,
        0.0,
        near / (far - near),
        (far * near) / (far - near),
        0.0,
        0.0,
        -1.0,
        0.0,
    )
}
