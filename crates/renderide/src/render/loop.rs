//! Render loop: executes one frame of mesh rendering.

use nalgebra::{Matrix4, Vector3};

use crate::gpu::GpuState;
use crate::scene::{render_transform_to_matrix, SceneGraph};
use crate::session::Session;

/// Encapsulates the render frame logic.
pub struct RenderLoop;

impl RenderLoop {
    /// Renders one frame: clear, draw batches, present.
    pub fn render(
        gpu: &mut GpuState,
        session: &mut Session,
    ) -> Result<(), wgpu::SurfaceError> {
        let draw_batches = session.collect_draw_batches();

        let output = gpu.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        let mesh_assets = session.asset_registry();

        let depth_view = gpu
            .depth_texture
            .as_ref()
            .map(|t| t.create_view(&wgpu::TextureViewDescriptor::default()));

        let mut view_transform = session
            .primary_view_transform()
            .cloned()
            .unwrap_or_default();
        view_transform.scale = filter_scale(view_transform.scale);

        let aspect = gpu.config.width as f32 / gpu.config.height.max(1) as f32;
        let proj = reverse_z_projection(
            aspect,
            session.desktop_fov().to_radians(),
            session.near_clip().max(0.01),
            session.far_clip(),
        );

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("mesh pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
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
                depth_stencil_attachment: depth_view.as_ref().map(|dv| wgpu::RenderPassDepthStencilAttachment {
                    view: dv,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            let use_debug_uv = std::env::var("RENDERIDE_DEBUG_UV").is_ok();

            for batch in &draw_batches {
                for (_, mesh_asset_id, _is_skinned, _material_id, _) in &batch.draws {
                    if *mesh_asset_id < 0 {
                        continue;
                    }
                    let Some(mesh) = mesh_assets.get_mesh(*mesh_asset_id) else {
                        continue;
                    };
                    if mesh.vertex_count <= 0 || mesh.index_count <= 0 {
                        continue;
                    }
                    if !gpu.mesh_buffer_cache.contains_key(mesh_asset_id) {
                        let stride = crate::assets::compute_vertex_stride(&mesh.vertex_attributes) as usize;
                        let stride = if stride > 0 {
                            stride
                        } else {
                            crate::gpu::compute_vertex_stride_from_mesh(mesh)
                        };
                        if let Some(b) = crate::gpu::create_mesh_buffers(&gpu.device, mesh, stride) {
                            gpu.mesh_buffer_cache.insert(*mesh_asset_id, b);
                        }
                    }
                }
            }

            struct BatchedDraw<'a> {
                vertex_buffer: &'a wgpu::Buffer,
                index_buffer: &'a wgpu::Buffer,
                submeshes: &'a [(u32, u32)],
                index_format: wgpu::IndexFormat,
                use_debug_uv: bool,
                has_uvs: bool,
                is_skinned: bool,
                material_id: i32,
            }
            let mut batched_draws: Vec<BatchedDraw<'_>> = Vec::new();
            let mut mvp_models: Vec<(Matrix4<f32>, Matrix4<f32>)> = Vec::new();
            let scene_graph = session.scene_graph();

            for batch in &draw_batches {
                let mut batch_vt = batch.view_transform;
                batch_vt.scale = filter_scale(batch_vt.scale);
                let view_mat = render_transform_to_matrix(&batch_vt)
                    .try_inverse()
                    .unwrap_or_else(Matrix4::identity);
                let view_mat = apply_view_handedness_fix(view_mat);
                let view_proj = proj * view_mat;

                for (model, mesh_asset_id, is_skinned, material_id, bone_transform_ids) in &batch.draws {
                    let buffers_ref = if *mesh_asset_id >= 0 {
                        let Some(mesh) = mesh_assets.get_mesh(*mesh_asset_id) else {
                            continue;
                        };
                        if mesh.vertex_count <= 0 {
                            continue;
                        }
                        if mesh.index_count <= 0 {
                            continue;
                        }
                        let Some(b) = gpu.mesh_buffer_cache.get(mesh_asset_id) else {
                            continue;
                        };
                        (b, mesh)
                    } else {
                        continue;
                    };

                    let (buffers_ref, mesh) = buffers_ref;

                    let model_mvp = view_proj * model;
                    let skinned_mvp = view_proj;

                    if *is_skinned {
                        let has_vb = buffers_ref.vertex_buffer_skinned.is_some();
                        let has_bind_poses = mesh.bind_poses.is_some();
                        let has_bone_ids = bone_transform_ids.is_some();
                        if has_vb && has_bind_poses && has_bone_ids {
                            let vb_skinned = buffers_ref.vertex_buffer_skinned.as_ref().unwrap();
                            let bind_poses = mesh.bind_poses.as_ref().unwrap();
                            let ids = bone_transform_ids.as_ref().unwrap();
                            let bone_matrices = compute_bone_matrices(
                                scene_graph,
                                batch.space_id,
                                ids,
                                bind_poses,
                            );
                            gpu.mesh_pipeline.upload_skinned_uniforms(&gpu.queue, skinned_mvp, &bone_matrices);
                            gpu.mesh_pipeline.draw_mesh_skinned(
                                &mut pass,
                                vb_skinned.as_ref(),
                                buffers_ref.index_buffer.as_ref(),
                                &buffers_ref.submeshes,
                                buffers_ref.index_format,
                            );
                            continue;
                        }
                    }

                    let vb = if use_debug_uv && buffers_ref.has_uvs && !*is_skinned {
                        buffers_ref.vertex_buffer_uv.as_ref().map(|b| b.as_ref()).unwrap_or(buffers_ref.vertex_buffer.as_ref())
                    } else {
                        buffers_ref.vertex_buffer.as_ref()
                    };

                    mvp_models.push((model_mvp, *model));
                    batched_draws.push(BatchedDraw {
                        vertex_buffer: vb,
                        index_buffer: buffers_ref.index_buffer.as_ref(),
                        submeshes: &buffers_ref.submeshes,
                        index_format: buffers_ref.index_format,
                        use_debug_uv,
                        has_uvs: buffers_ref.has_uvs,
                        is_skinned: false,
                        material_id: *material_id,
                    });
                }
            }

            gpu.mesh_pipeline.upload_uniforms_batch(&gpu.queue, &mvp_models);
            for (i, d) in batched_draws.iter().enumerate() {
                gpu.mesh_pipeline.draw_mesh_with_offset(
                    &mut pass,
                    d.vertex_buffer,
                    d.index_buffer,
                    d.submeshes,
                    d.index_format,
                    i as u32,
                    d.use_debug_uv,
                    d.has_uvs,
                    d.is_skinned,
                    d.material_id,
                );
            }
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        Ok(())
    }
}

fn clamp_near_far(near: f32, far: f32) -> (f32, f32) {
    let near = near.max(0.001);
    let far = if far > near { far } else { near + 1.0 };
    (near, far)
}

fn compute_bone_matrices(
    scene_graph: &SceneGraph,
    space_id: i32,
    bone_transform_ids: &[i32],
    bind_poses: &[[[f32; 4]; 4]],
) -> Vec<[[f32; 4]; 4]> {
    let mut out = Vec::with_capacity(bone_transform_ids.len().min(bind_poses.len()));
    for (i, &tid) in bone_transform_ids.iter().enumerate() {
        let bind = bind_poses.get(i).copied().unwrap_or(nalgebra::Matrix4::identity().into());
        let bind_mat = Matrix4::from_fn(|r, c| bind[r][c]);
        let world = if tid >= 0 {
            scene_graph
                .get_world_matrix(space_id, tid as usize)
                .unwrap_or_else(Matrix4::identity)
        } else {
            Matrix4::identity()
        };
        let combined: [[f32; 4]; 4] = (world * bind_mat).into();
        out.push(combined);
    }
    out
}

fn reverse_z_projection(aspect: f32, vertical_fov: f32, near: f32, far: f32) -> Matrix4<f32> {
    let vertical_half = vertical_fov / 2.0;
    let tan_vertical_half = vertical_half.tan();
    let horizontal_fov = (tan_vertical_half * aspect)
        .atan()
        .clamp(0.1, std::f32::consts::FRAC_PI_2 - 0.1)
        * 2.0;
    let tan_horizontal_half = (horizontal_fov / 2.0).tan();
    let f_x = 1.0 / tan_horizontal_half;
    let f_y = 1.0 / tan_vertical_half;
    let proj = Matrix4::new(
        f_x, 0.0, 0.0, 0.0,
        0.0, f_y, 0.0, 0.0,
        0.0, 0.0, near / (far - near), (far * near) / (far - near),
        0.0, 0.0, -1.0, 0.0,
    );
    proj
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
