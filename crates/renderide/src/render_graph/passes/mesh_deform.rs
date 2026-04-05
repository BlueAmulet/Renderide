//! Blendshape and skinning compute dispatches before the main forward pass.

use std::sync::Arc;

use glam::Mat4;

use crate::render_graph::context::RenderPassContext;
use crate::render_graph::error::RenderPassError;
use crate::render_graph::pass::RenderPass;
use crate::render_graph::resources::PassResources;
use crate::scene::RenderSpaceId;

/// Encodes mesh deformation compute for all active render spaces.
#[derive(Debug, Default)]
pub struct MeshDeformPass;

impl MeshDeformPass {
    pub fn new() -> Self {
        Self
    }
}

/// GPU buffer handles + metadata copied from [`crate::assets::mesh::GpuMesh`] so we can hold
/// deform state without borrowing the mesh pool across preprocess/scratch access.
struct MeshDeformSnapshot {
    vertex_count: u32,
    num_blendshapes: u32,
    has_skeleton: bool,
    positions_buffer: Option<Arc<wgpu::Buffer>>,
    blendshape_buffer: Option<Arc<wgpu::Buffer>>,
    deform_temp_buffer: Option<Arc<wgpu::Buffer>>,
    deformed_positions_buffer: Option<Arc<wgpu::Buffer>>,
    bone_indices_buffer: Option<Arc<wgpu::Buffer>>,
    bone_weights_vec4_buffer: Option<Arc<wgpu::Buffer>>,
    inverse_bind_poses: Vec<Mat4>,
}

impl MeshDeformSnapshot {
    fn from_mesh(m: &crate::assets::mesh::GpuMesh) -> Self {
        Self {
            vertex_count: m.vertex_count,
            num_blendshapes: m.num_blendshapes,
            has_skeleton: m.has_skeleton,
            positions_buffer: m.positions_buffer.clone(),
            blendshape_buffer: m.blendshape_buffer.clone(),
            deform_temp_buffer: m.deform_temp_buffer.clone(),
            deformed_positions_buffer: m.deformed_positions_buffer.clone(),
            bone_indices_buffer: m.bone_indices_buffer.clone(),
            bone_weights_vec4_buffer: m.bone_weights_vec4_buffer.clone(),
            inverse_bind_poses: m.inverse_bind_poses.clone(),
        }
    }
}

struct DeformWorkItem {
    space_id: RenderSpaceId,
    mesh: MeshDeformSnapshot,
    skinned: Option<Vec<i32>>,
    blend_weights: Vec<f32>,
}

impl RenderPass for MeshDeformPass {
    fn name(&self) -> &str {
        "MeshDeform"
    }

    fn resources(&self) -> PassResources {
        PassResources {
            reads: Vec::new(),
            writes: Vec::new(),
        }
    }

    fn execute(&mut self, ctx: &mut RenderPassContext<'_>) -> Result<(), RenderPassError> {
        let Some(frame) = ctx.frame.as_mut() else {
            return Ok(());
        };

        let mut work: Vec<DeformWorkItem> = Vec::new();
        for space_id in frame.scene.render_space_ids() {
            let Some(space) = frame.scene.space(space_id) else {
                continue;
            };
            if !space.is_active {
                continue;
            }
            for r in &space.static_mesh_renderers {
                if r.mesh_asset_id < 0 {
                    continue;
                }
                let Some(m) = frame.backend.mesh_pool().get_mesh(r.mesh_asset_id) else {
                    continue;
                };
                work.push(DeformWorkItem {
                    space_id,
                    mesh: MeshDeformSnapshot::from_mesh(m),
                    skinned: None,
                    blend_weights: r.blend_shape_weights.clone(),
                });
            }
            for skinned in &space.skinned_mesh_renderers {
                let r = &skinned.base;
                if r.mesh_asset_id < 0 {
                    continue;
                }
                let Some(m) = frame.backend.mesh_pool().get_mesh(r.mesh_asset_id) else {
                    continue;
                };
                work.push(DeformWorkItem {
                    space_id,
                    mesh: MeshDeformSnapshot::from_mesh(m),
                    skinned: Some(skinned.bone_transform_indices.clone()),
                    blend_weights: r.blend_shape_weights.clone(),
                });
            }
        }

        let Some((pre, scratch)) = frame.backend.mesh_deform_pre_and_scratch() else {
            return Ok(());
        };

        let queue = match ctx.queue.lock() {
            Ok(q) => q,
            Err(poisoned) => poisoned.into_inner(),
        };

        for item in work {
            record_mesh_deform(
                &queue,
                ctx.device,
                ctx.encoder,
                pre,
                scratch,
                frame.scene,
                item.space_id,
                &item.mesh,
                item.skinned.as_deref(),
                &item.blend_weights,
            );
        }

        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn record_mesh_deform(
    queue: &wgpu::Queue,
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    pre: &crate::gpu::MeshPreprocessPipelines,
    scratch: &mut crate::backend::MeshDeformScratch,
    scene: &crate::scene::SceneCoordinator,
    space_id: RenderSpaceId,
    mesh: &MeshDeformSnapshot,
    bone_transform_indices: Option<&[i32]>,
    blend_weights: &[f32],
) {
    let Some(ref positions) = mesh.positions_buffer else {
        return;
    };
    let vc = mesh.vertex_count;
    if vc == 0 {
        return;
    }
    let wg = workgroup_count(vc);

    let needs_blend = mesh.num_blendshapes > 0
        && mesh.blendshape_buffer.is_some()
        && mesh.deform_temp_buffer.is_some();
    let needs_skin = bone_transform_indices.is_some()
        && mesh.has_skeleton
        && mesh.deformed_positions_buffer.is_some()
        && mesh.bone_indices_buffer.is_some()
        && mesh.bone_weights_vec4_buffer.is_some()
        && !mesh.inverse_bind_poses.is_empty();

    if !needs_blend && !needs_skin {
        return;
    }

    if needs_blend {
        let Some(ref temp) = mesh.deform_temp_buffer else {
            return;
        };
        let Some(ref deltas) = mesh.blendshape_buffer else {
            return;
        };
        let shape_count = mesh.num_blendshapes;
        if shape_count == 0 {
            return;
        }
        scratch.ensure_shape_weight_capacity(device, shape_count);
        let mut wbytes = vec![0u8; (shape_count as usize) * 4];
        for s in 0..shape_count as usize {
            let w = blend_weights.get(s).copied().unwrap_or(0.0);
            wbytes[s * 4..s * 4 + 4].copy_from_slice(&w.to_le_bytes());
        }
        queue.write_buffer(&scratch.blendshape_weights, 0, &wbytes);
        let params: [u8; 16] = build_blend_params(vc, shape_count);
        queue.write_buffer(&scratch.blendshape_params, 0, &params);

        let blend_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blendshape_bg"),
            layout: &pre.blendshape_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scratch.blendshape_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: positions.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: deltas.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: scratch.blendshape_weights.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: temp.as_entire_binding(),
                },
            ],
        });

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("blendshape"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pre.blendshape_pipeline);
        cpass.set_bind_group(0, &blend_bg, &[]);
        cpass.dispatch_workgroups(wg, 1, 1);
        drop(cpass);
    }

    if needs_skin {
        let Some(ref dst) = mesh.deformed_positions_buffer else {
            return;
        };
        let Some(ref bone_idx) = mesh.bone_indices_buffer else {
            return;
        };
        let Some(ref bone_wt) = mesh.bone_weights_vec4_buffer else {
            return;
        };
        let Some(indices) = bone_transform_indices else {
            return;
        };

        let bone_count_u = mesh.inverse_bind_poses.len() as u32;
        scratch.ensure_bone_capacity(device, bone_count_u);
        let mut palette: Vec<u8> = vec![0u8; (bone_count_u as usize) * 64];
        for bi in 0..bone_count_u as usize {
            let widx = indices.get(bi).copied().unwrap_or(0).max(0) as usize;
            let world = scene
                .world_matrix_with_root(space_id, widx)
                .unwrap_or(Mat4::IDENTITY);
            let pal = world * mesh.inverse_bind_poses[bi];
            let cols = pal.to_cols_array();
            palette[bi * 64..bi * 64 + 64].copy_from_slice(bytemuck::cast_slice(&cols));
        }
        queue.write_buffer(&scratch.bone_matrices, 0, &palette);

        let src_for_skin: &wgpu::Buffer = if needs_blend {
            mesh.deform_temp_buffer.as_deref().expect("blend temp")
        } else {
            positions.as_ref()
        };

        let skin_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("skinning_bg"),
            layout: &pre.skinning_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scratch.bone_matrices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: src_for_skin.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bone_idx.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: bone_wt.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: dst.as_entire_binding(),
                },
            ],
        });

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("skinning"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pre.skinning_pipeline);
        cpass.set_bind_group(0, &skin_bg, &[]);
        cpass.dispatch_workgroups(wg, 1, 1);
        drop(cpass);
    }
}

fn workgroup_count(vertex_count: u32) -> u32 {
    (vertex_count.saturating_add(63)) / 64
}

fn build_blend_params(vertex_count: u32, shape_count: u32) -> [u8; 16] {
    let v = vertex_count.to_le_bytes();
    let s = shape_count.to_le_bytes();
    let per_v = vertex_count.to_le_bytes();
    let pad = 0u32.to_le_bytes();
    let mut o = [0u8; 16];
    o[0..4].copy_from_slice(&v);
    o[4..8].copy_from_slice(&s);
    o[8..12].copy_from_slice(&per_v);
    o[12..16].copy_from_slice(&pad);
    o
}
