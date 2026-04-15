//! GPU attach: flush pending texture allocations and replay queued IPC payloads.

use std::sync::Arc;

use crate::ipc::{DualQueueIpc, SharedMemoryAccessor};
use crate::shared::{MeshUploadData, SetCubemapData, SetTexture2DData, SetTexture3DData};

use super::super::{drain_asset_tasks_unbounded, AssetTransferQueue};
use super::allocations::{
    flush_pending_cubemap_allocations, flush_pending_render_texture_allocations,
    flush_pending_texture3d_allocations, flush_pending_texture_allocations,
};
use super::cubemap::try_cubemap_upload_with_device;
use super::mesh::try_process_mesh_upload;
use super::texture2d::try_texture_upload_with_device;
use super::texture3d::try_texture3d_upload_with_device;

/// After GPU [`crate::backend::RenderBackend::attach`], allocate textures for pending
/// formats and replay queued mesh/texture payloads when shared memory is available, then
/// drain the asset integrator synchronously (no per-frame budget).
pub fn attach_flush_pending_asset_uploads(
    queue: &mut AssetTransferQueue,
    device: &Arc<wgpu::Device>,
    shm: Option<&mut SharedMemoryAccessor>,
) {
    flush_pending_texture_allocations(queue, device);
    flush_pending_texture3d_allocations(queue, device);
    flush_pending_cubemap_allocations(queue, device);
    flush_pending_render_texture_allocations(queue, device);
    let pending_tex: Vec<SetTexture2DData> = queue.pending_texture_uploads.drain(..).collect();
    let pending_tex3d: Vec<SetTexture3DData> = queue.pending_texture3d_uploads.drain(..).collect();
    let pending_cube: Vec<SetCubemapData> = queue.pending_cubemap_uploads.drain(..).collect();
    let pending_mesh: Vec<MeshUploadData> = queue.pending_mesh_uploads.drain(..).collect();
    if let Some(shm) = shm {
        for data in pending_tex {
            try_texture_upload_with_device(queue, data, shm, None, false);
        }
        for data in pending_tex3d {
            try_texture3d_upload_with_device(queue, data, shm, None, false);
        }
        for data in pending_cube {
            try_cubemap_upload_with_device(queue, data, shm, None, false);
        }
        for data in pending_mesh {
            try_process_mesh_upload(queue, data, shm, None);
        }
        let mut ipc_opt = None::<&mut DualQueueIpc>;
        drain_asset_tasks_unbounded(queue, shm, &mut ipc_opt);
    } else {
        for data in pending_tex {
            queue.pending_texture_uploads.push_back(data);
        }
        for data in pending_tex3d {
            queue.pending_texture3d_uploads.push_back(data);
        }
        for data in pending_cube {
            queue.pending_cubemap_uploads.push_back(data);
        }
        for data in pending_mesh {
            queue.pending_mesh_uploads.push_back(data);
        }
    }
}
