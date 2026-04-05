//! GPU-resident mesh: wgpu buffers only; host layout preserved in one interleaved vertex buffer.

use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::shared::{
    IndexBufferFormat, MeshUploadData, RenderBoundingBox, SubmeshBufferDescriptor,
};

use super::layout::{
    compute_index_count, compute_mesh_buffer_layout, compute_vertex_stride, extract_bind_poses,
    extract_blendshape_offsets, index_bytes_per_element, synthetic_bone_data_for_blendshape_only,
    MeshBufferLayout,
};

/// Resident mesh on GPU: no CPU geometry retained.
///
/// **Vertex groups** in Renderite are expressed through per-vertex bone influence streams
/// (`bone_counts` + `bone_weights`) when the host provides skeleton data.
#[derive(Debug)]
pub struct GpuMesh {
    pub asset_id: i32,
    /// Full interleaved vertices as sent by the host (`vertex_attributes` order).
    pub vertex_buffer: Arc<wgpu::Buffer>,
    pub index_buffer: Arc<wgpu::Buffer>,
    pub index_format: wgpu::IndexFormat,
    pub index_count: u32,
    /// Per-submesh `(first_index, index_count)` in elements of `index_format`.
    pub submeshes: Vec<(u32, u32)>,
    pub vertex_count: u32,
    pub vertex_stride: u32,
    pub bounds: RenderBoundingBox,
    /// Optional 1 byte per vertex (skinned / synthetic for blendshape-only).
    pub bone_counts_buffer: Option<Arc<wgpu::Buffer>>,
    /// Packed bone weights host layout (8 bytes per weight slot).
    pub bone_weights_buffer: Option<Arc<wgpu::Buffer>>,
    /// Column-major `float4x4` bind poses (64 bytes per bone).
    pub bind_poses_buffer: Option<Arc<wgpu::Buffer>>,
    /// Packed blendshape deltas (`BLENDSHAPE_OFFSET_GPU_STRIDE` × vertices × shapes).
    pub blendshape_buffer: Option<Arc<wgpu::Buffer>>,
    pub num_blendshapes: u32,
    /// Approximate VRAM (bytes), used by [`crate::resources::VramAccounting`].
    pub resident_bytes: u64,
}

impl GpuMesh {
    /// Uploads mesh data from a raw byte slice covering at least `layout.total_buffer_length`.
    ///
    /// `raw` must be the mapping for `data.buffer` only for the duration of this call.
    pub fn upload(
        device: &wgpu::Device,
        raw: &[u8],
        data: &MeshUploadData,
        layout: &MeshBufferLayout,
    ) -> Option<Self> {
        if raw.len() < layout.total_buffer_length {
            logger::error!(
                "mesh {}: raw too short (need {}, got {})",
                data.asset_id,
                layout.total_buffer_length,
                raw.len()
            );
            return None;
        }

        let vertex_stride = compute_vertex_stride(&data.vertex_attributes).max(1) as u32;
        let index_count = compute_index_count(&data.submeshes);
        let index_count_u32 = index_count.max(0) as u32;
        let use_blendshapes =
            data.upload_hint.flags.blendshapes() && !data.blendshape_buffers.is_empty();

        let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("mesh {} vertices", data.asset_id)),
            contents: &raw[..layout.vertex_size],
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let ib_slice =
            &raw[layout.index_buffer_start..layout.index_buffer_start + layout.index_buffer_length];
        let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("mesh {} indices", data.asset_id)),
            contents: ib_slice,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        });

        let index_format = wgpu_index_format(data.index_buffer_format);

        let (bone_counts_buffer, bone_weights_buffer, bind_poses_buffer) = if data.bone_count > 0 {
            let bp_raw =
                &raw[layout.bind_poses_start..layout.bind_poses_start + layout.bind_poses_length];
            let bind_poses = extract_bind_poses(bp_raw, data.bone_count as usize)?;
            let bp_bytes: Vec<u8> = bind_poses
                .iter()
                .flat_map(|m| bytemuck::bytes_of(m).iter().copied())
                .collect();

            let bc = &raw
                [layout.bone_counts_start..layout.bone_counts_start + layout.bone_counts_length];
            let bw = &raw
                [layout.bone_weights_start..layout.bone_weights_start + layout.bone_weights_length];

            let bc_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("mesh {} bone_counts", data.asset_id)),
                contents: bc,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            let bw_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("mesh {} bone_weights", data.asset_id)),
                contents: bw,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            let bp_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("mesh {} bind_poses", data.asset_id)),
                contents: &bp_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            (
                Some(Arc::new(bc_buf)),
                Some(Arc::new(bw_buf)),
                Some(Arc::new(bp_buf)),
            )
        } else if use_blendshapes && data.vertex_count > 0 {
            let (bind_poses, bone_counts, bone_weights) =
                synthetic_bone_data_for_blendshape_only(data.vertex_count);
            let bp_bytes: Vec<u8> = bind_poses
                .iter()
                .flat_map(|m| bytemuck::bytes_of(m).iter().copied())
                .collect();
            let bc_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("mesh {} bone_counts synth", data.asset_id)),
                contents: &bone_counts,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            let bw_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("mesh {} bone_weights synth", data.asset_id)),
                contents: &bone_weights,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            let bp_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("mesh {} bind_poses synth", data.asset_id)),
                contents: &bp_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            (
                Some(Arc::new(bc_buf)),
                Some(Arc::new(bw_buf)),
                Some(Arc::new(bp_buf)),
            )
        } else {
            (None, None, None)
        };

        let (blendshape_buffer, num_blendshapes) = if use_blendshapes {
            match extract_blendshape_offsets(
                raw,
                layout,
                &data.blendshape_buffers,
                data.vertex_count,
            ) {
                Some((pack, n)) if !pack.is_empty() => {
                    let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(&format!("mesh {} blendshapes", data.asset_id)),
                        contents: &pack,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    });
                    (Some(Arc::new(buf)), n.max(0) as u32)
                }
                _ => (None, 0),
            }
        } else {
            (None, 0)
        };

        let submeshes = validated_submesh_ranges(&data.submeshes, index_count_u32);

        let mut resident_bytes = vb.size() + ib.size();
        if let Some(ref b) = bone_counts_buffer {
            resident_bytes += b.size();
        }
        if let Some(ref b) = bone_weights_buffer {
            resident_bytes += b.size();
        }
        if let Some(ref b) = bind_poses_buffer {
            resident_bytes += b.size();
        }
        if let Some(ref b) = blendshape_buffer {
            resident_bytes += b.size();
        }

        Some(Self {
            asset_id: data.asset_id,
            vertex_buffer: Arc::new(vb),
            index_buffer: Arc::new(ib),
            index_format,
            index_count: index_count_u32,
            submeshes,
            vertex_count: data.vertex_count.max(0) as u32,
            vertex_stride,
            bounds: data.bounds,
            bone_counts_buffer,
            bone_weights_buffer,
            bind_poses_buffer,
            blendshape_buffer,
            num_blendshapes,
            resident_bytes,
        })
    }
}

/// Builds layout and uploads; returns [`GpuMesh`] if validation and GPU creation succeed.
pub fn try_upload_mesh_from_raw(
    device: &wgpu::Device,
    raw: &[u8],
    data: &MeshUploadData,
) -> Option<GpuMesh> {
    if data.buffer.length <= 0 {
        return None;
    }
    let vertex_stride = compute_vertex_stride(&data.vertex_attributes);
    if vertex_stride <= 0 {
        logger::error!("mesh {}: invalid vertex stride", data.asset_id);
        return None;
    }
    let index_count = compute_index_count(&data.submeshes);
    let index_bytes = index_bytes_per_element(data.index_buffer_format);
    let layout = match compute_mesh_buffer_layout(
        vertex_stride,
        data.vertex_count,
        index_count,
        index_bytes,
        data.bone_count,
        data.bone_weight_count,
        Some(&data.blendshape_buffers),
    ) {
        Ok(l) => l,
        Err(e) => {
            logger::error!("mesh {}: layout error: {}", data.asset_id, e);
            return None;
        }
    };

    let expected_bone_weights_len = (data.bone_weight_count.max(0) * 8) as usize;
    let expected_bind_poses_len = (data.bone_count.max(0) * 64) as usize;
    if layout.bone_weights_length != expected_bone_weights_len {
        logger::error!("mesh {}: bone_weights layout mismatch", data.asset_id);
        return None;
    }
    if layout.bind_poses_length != expected_bind_poses_len {
        logger::error!("mesh {}: bind_poses layout mismatch", data.asset_id);
        return None;
    }

    GpuMesh::upload(device, raw, data, &layout)
}

fn wgpu_index_format(f: IndexBufferFormat) -> wgpu::IndexFormat {
    match f {
        IndexBufferFormat::u_int16 => wgpu::IndexFormat::Uint16,
        IndexBufferFormat::u_int32 => wgpu::IndexFormat::Uint32,
    }
}

fn validated_submesh_ranges(
    submeshes: &[SubmeshBufferDescriptor],
    index_count_u32: u32,
) -> Vec<(u32, u32)> {
    if submeshes.is_empty() {
        if index_count_u32 > 0 {
            return vec![(0, index_count_u32)];
        }
        return Vec::new();
    }
    let valid: Vec<(u32, u32)> = submeshes
        .iter()
        .filter(|s| {
            s.index_count > 0
                && (s.index_start as i64 + s.index_count as i64) <= index_count_u32 as i64
        })
        .map(|s| (s.index_start as u32, s.index_count as u32))
        .collect();
    if valid.is_empty() && index_count_u32 > 0 {
        vec![(0, index_count_u32)]
    } else {
        valid
    }
}
