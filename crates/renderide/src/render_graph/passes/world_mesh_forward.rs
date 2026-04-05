//! Main forward pass: clear color + depth, debug normal shading for scene meshes.

use glam::Mat4;

use crate::materials::{MaterialPipelineDesc, DEBUG_WORLD_NORMALS_FAMILY_ID};
use crate::pipelines::ShaderPermutation;
use crate::present::SWAPCHAIN_CLEAR_COLOR;
use crate::render_graph::context::RenderPassContext;
use crate::render_graph::error::RenderPassError;
use crate::render_graph::pass::RenderPass;
use crate::render_graph::resources::{PassResources, ResourceSlot};
use crate::scene::SceneCoordinator;
use crate::scene::{render_transform_to_matrix, RenderSpaceId};

/// Clears the backbuffer and depth, then draws meshes with [`crate::pipelines::raster::DebugWorldNormalsFamily`].
#[derive(Debug, Default)]
pub struct WorldMeshForwardPass;

impl WorldMeshForwardPass {
    pub fn new() -> Self {
        Self
    }
}

impl RenderPass for WorldMeshForwardPass {
    fn name(&self) -> &str {
        "WorldMeshForward"
    }

    fn resources(&self) -> PassResources {
        PassResources {
            reads: Vec::new(),
            writes: vec![ResourceSlot::Backbuffer, ResourceSlot::Depth],
        }
    }

    fn execute(&mut self, ctx: &mut RenderPassContext<'_>) -> Result<(), RenderPassError> {
        let Some(bb) = ctx.backbuffer else {
            return Err(RenderPassError::MissingBackbuffer {
                pass: self.name().to_string(),
            });
        };
        let Some(depth) = ctx.depth_view else {
            return Err(RenderPassError::MissingDepth {
                pass: self.name().to_string(),
            });
        };
        let Some(frame) = ctx.frame.as_mut() else {
            return Err(RenderPassError::MissingFrameParams {
                pass: self.name().to_string(),
            });
        };

        let desc = MaterialPipelineDesc {
            surface_format: frame.surface_format,
            depth_stencil_format: Some(wgpu::TextureFormat::Depth32Float),
            sample_count: 1,
        };
        let backend = &mut frame.backend;
        let Some(reg) = backend.material_registry.as_mut() else {
            return Ok(());
        };
        let Some(dbg) = backend.debug_draw.as_ref() else {
            return Ok(());
        };
        let Some(pipeline) =
            reg.pipeline_for_family(DEBUG_WORLD_NORMALS_FAMILY_ID, &desc, ShaderPermutation(0))
        else {
            return Ok(());
        };
        let debug_bind_group = &dbg.bind_group;
        let globals_buf = &dbg.globals_buffer;
        let model_buf = &dbg.model_buffer;
        let mesh_pool = &backend.mesh_pool;

        let (vw, vh) = frame.viewport_px;
        let aspect = vw as f32 / vh.max(1) as f32;
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, aspect, 0.05, 10_000.0);

        let mut rpass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("world-mesh-forward"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: bb,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(SWAPCHAIN_CLEAR_COLOR),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
            multiview_mask: None,
        });
        rpass.set_pipeline(pipeline);
        rpass.set_bind_group(0, debug_bind_group, &[]);

        let queue = match ctx.queue.lock() {
            Ok(q) => q,
            Err(poisoned) => poisoned.into_inner(),
        };

        for space_id in frame.scene.render_space_ids() {
            let Some(space) = frame.scene.space(space_id) else {
                continue;
            };
            if !space.is_active {
                continue;
            }

            let cam = render_transform_to_matrix(&space.view_transform);
            let view = cam.inverse();
            let vp = proj * view;
            let vp_bytes: [f32; 16] = vp.to_cols_array();
            queue.write_buffer(globals_buf, 0, bytemuck::cast_slice(&vp_bytes));

            for r in &space.static_mesh_renderers {
                draw_mesh(
                    &mut rpass,
                    &queue,
                    model_buf,
                    frame.scene,
                    space_id,
                    r.mesh_asset_id,
                    r.node_id,
                    mesh_pool,
                    false,
                );
            }
            for skinned in &space.skinned_mesh_renderers {
                let r = &skinned.base;
                draw_mesh(
                    &mut rpass,
                    &queue,
                    model_buf,
                    frame.scene,
                    space_id,
                    r.mesh_asset_id,
                    r.node_id,
                    mesh_pool,
                    true,
                );
            }
        }

        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_mesh(
    rpass: &mut wgpu::RenderPass<'_>,
    queue: &wgpu::Queue,
    model_buf: &wgpu::Buffer,
    scene: &SceneCoordinator,
    space_id: RenderSpaceId,
    mesh_asset_id: i32,
    node_id: i32,
    mesh_pool: &crate::resources::MeshPool,
    from_skinned_table: bool,
) {
    if mesh_asset_id < 0 || node_id < 0 {
        return;
    }
    let Some(mesh) = mesh_pool.get_mesh(mesh_asset_id) else {
        return;
    };
    if !mesh.debug_streams_ready() {
        return;
    }
    let Some(normals) = mesh.normals_buffer.as_deref() else {
        return;
    };

    let use_deformed = from_skinned_table && mesh.has_skeleton;
    let use_blend_only = mesh.num_blendshapes > 0;

    let pos_buf = if use_deformed {
        mesh.deformed_positions_buffer.as_deref()
    } else if use_blend_only {
        mesh.deform_temp_buffer.as_deref()
    } else {
        mesh.positions_buffer.as_deref()
    };
    let Some(pos) = pos_buf else {
        return;
    };

    let node_u = node_id as usize;
    let model = scene
        .world_matrix_with_root(space_id, node_u)
        .unwrap_or(Mat4::IDENTITY);
    let m_bytes: [f32; 16] = model.to_cols_array();
    queue.write_buffer(model_buf, 0, bytemuck::cast_slice(&m_bytes));

    rpass.set_vertex_buffer(0, pos.slice(..));
    rpass.set_vertex_buffer(1, normals.slice(..));
    rpass.set_index_buffer(mesh.index_buffer.slice(..), mesh.index_format);

    for (first, count) in &mesh.submeshes {
        if *count == 0 {
            continue;
        }
        rpass.draw_indexed(*first..(*first + *count), 0, 0..1);
    }
}
