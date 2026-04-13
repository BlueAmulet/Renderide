//! Shared-memory ingestion and queue draining for [`super::AssetTransferQueue`].

use std::sync::Arc;
use std::time::Instant;

use crate::assets::mesh::{
    compute_and_validate_mesh_layout, mesh_upload_input_fingerprint, try_upload_mesh_from_raw,
};
use crate::assets::texture::write_texture2d_mips;
use crate::ipc::{DualQueueIpc, SharedMemoryAccessor};
use crate::resources::{GpuRenderTexture, GpuTexture2d};
use crate::shared::{
    MeshUnload, MeshUploadData, MeshUploadResult, RenderTextureResult, RendererCommand,
    SetRenderTextureFormat, SetTexture2DData, SetTexture2DFormat, SetTexture2DProperties,
    SetTexture2DResult, TextureUpdateResultType, UnloadRenderTexture, UnloadTexture2D,
};

use super::AssetTransferQueue;

/// Max queued [`MeshUploadData`] when GPU is not ready yet (host data stays in shared memory).
pub const MAX_PENDING_MESH_UPLOADS: usize = 256;

/// Max deferred low-priority mesh uploads when [`MESH_UPLOAD_NON_HIGH_PRIORITY_BUDGET_PER_POLL`] is hit.
pub const MAX_DEFERRED_MESH_UPLOADS: usize = 512;

/// Max deferred mesh uploads drained at the end of one [`crate::runtime::RendererRuntime::poll_ipc`]
/// (cross-tick backlog may span multiple polls).
pub const MAX_DEFERRED_MESH_UPLOADS_DRAIN_PER_POLL: usize = 64;

/// Max non-[`MeshUploadData::high_priority`] mesh uploads processed inline per
/// [`crate::runtime::RendererRuntime::poll_ipc`] before additional commands are deferred.
pub const MESH_UPLOAD_NON_HIGH_PRIORITY_BUDGET_PER_POLL: u32 = 32;

/// Max non-[`SetTexture2DData::high_priority`] texture data uploads processed inline per
/// [`crate::runtime::RendererRuntime::poll_ipc`] before additional commands are deferred.
pub const TEXTURE_UPLOAD_NON_HIGH_PRIORITY_BUDGET_PER_POLL: u32 = 32;

/// Max deferred low-priority texture uploads when [`TEXTURE_UPLOAD_NON_HIGH_PRIORITY_BUDGET_PER_POLL`] is hit.
pub const MAX_DEFERRED_TEXTURE_UPLOADS: usize = 512;

/// Max deferred texture uploads drained at the end of one [`crate::runtime::RendererRuntime::poll_ipc`].
pub const MAX_DEFERRED_TEXTURE_UPLOADS_DRAIN_PER_POLL: usize = 64;

/// Max queued texture data commands when GPU or format is not ready.
pub const MAX_PENDING_TEXTURE_UPLOADS: usize = 256;

/// After GPU [`crate::backend::RenderBackend::attach`], allocate textures for pending
/// formats and replay queued mesh/texture payloads when shared memory is available.
pub fn attach_flush_pending_asset_uploads(
    queue: &mut AssetTransferQueue,
    device: &Arc<wgpu::Device>,
    shm: Option<&mut SharedMemoryAccessor>,
) {
    flush_pending_texture_allocations(queue, device);
    flush_pending_render_texture_allocations(queue, device);
    let pending_tex: Vec<SetTexture2DData> = queue.pending_texture_uploads.drain(..).collect();
    let pending_mesh: Vec<MeshUploadData> = queue.pending_mesh_uploads.drain(..).collect();
    if let Some(shm) = shm {
        for data in pending_tex {
            try_texture_upload_with_device(queue, data, shm, None, false);
        }
        for data in pending_mesh {
            try_mesh_upload_with_device(queue, device, data, shm, None, false);
        }
    } else {
        for data in pending_tex {
            queue.pending_texture_uploads.push_back(data);
        }
        for data in pending_mesh {
            queue.pending_mesh_uploads.push_back(data);
        }
    }
}

fn flush_pending_texture_allocations(queue: &mut AssetTransferQueue, device: &Arc<wgpu::Device>) {
    let ids: Vec<i32> = queue.texture_formats.keys().copied().collect();
    for id in ids {
        if queue.texture_pool.get_texture(id).is_some() {
            continue;
        }
        let Some(fmt) = queue.texture_formats.get(&id).cloned() else {
            continue;
        };
        let props = queue.texture_properties.get(&id);
        let Some(tex) = GpuTexture2d::new_from_format(device.as_ref(), &fmt, props) else {
            logger::warn!("texture {id}: failed to allocate GPU texture on attach");
            continue;
        };
        let _ = queue.texture_pool.insert_texture(tex);
    }
}

fn flush_pending_render_texture_allocations(
    queue: &mut AssetTransferQueue,
    device: &Arc<wgpu::Device>,
) {
    let ids: Vec<i32> = queue.render_texture_formats.keys().copied().collect();
    for id in ids {
        if queue.render_texture_pool.get(id).is_some() {
            continue;
        }
        let Some(fmt) = queue.render_texture_formats.get(&id).cloned() else {
            continue;
        };
        let Some(tex) = GpuRenderTexture::new_from_format(device.as_ref(), &fmt) else {
            logger::warn!("render texture {id}: failed to allocate GPU targets on attach");
            continue;
        };
        let _ = queue.render_texture_pool.insert_texture(tex);
    }
}

fn send_render_texture_result(
    ipc: Option<&mut DualQueueIpc>,
    asset_id: i32,
    instance_changed: bool,
) {
    let Some(ipc) = ipc else {
        return;
    };
    ipc.send_background(RendererCommand::render_texture_result(
        RenderTextureResult {
            asset_id,
            instance_changed,
        },
    ));
}

/// Handle [`SetRenderTextureFormat`](crate::shared::SetRenderTextureFormat).
pub fn on_set_render_texture_format(
    queue: &mut AssetTransferQueue,
    f: SetRenderTextureFormat,
    ipc: Option<&mut DualQueueIpc>,
) {
    let id = f.asset_id;
    queue.render_texture_formats.insert(id, f.clone());
    let Some(device) = queue.gpu_device.clone() else {
        send_render_texture_result(ipc, id, queue.render_texture_pool.get(id).is_none());
        return;
    };
    let Some(tex) = GpuRenderTexture::new_from_format(device.as_ref(), &f) else {
        logger::warn!("render texture {id}: SetRenderTextureFormat rejected (bad size or device)");
        return;
    };
    let existed_before = queue.render_texture_pool.insert_texture(tex);
    send_render_texture_result(ipc, id, !existed_before);
    logger::trace!(
        "render texture {} {}×{} depth_bits={} (resident_bytes≈{})",
        id,
        f.size.x,
        f.size.y,
        f.depth,
        queue
            .render_texture_pool
            .accounting()
            .texture_resident_bytes()
    );
}

/// Remove a render texture asset from the CPU table and GPU pool.
pub fn on_unload_render_texture(queue: &mut AssetTransferQueue, u: UnloadRenderTexture) {
    let id = u.asset_id;
    queue.render_texture_formats.remove(&id);
    if queue.render_texture_pool.remove(id) {
        logger::info!(
            "render texture {id} unloaded (tex≈{} total≈{})",
            queue.texture_pool.accounting().texture_resident_bytes(),
            queue.mesh_pool.accounting().total_resident_bytes()
        );
    }
}

fn send_texture_2d_result(
    ipc: Option<&mut DualQueueIpc>,
    asset_id: i32,
    update: i32,
    instance_changed: bool,
) {
    let Some(ipc) = ipc else {
        return;
    };
    ipc.send_background(RendererCommand::set_texture_2d_result(SetTexture2DResult {
        asset_id,
        r#type: TextureUpdateResultType(update),
        instance_changed,
    }));
}

/// Handle [`SetTexture2DFormat`](crate::shared::SetTexture2DFormat).
pub fn on_set_texture_2d_format(
    queue: &mut AssetTransferQueue,
    f: SetTexture2DFormat,
    ipc: Option<&mut DualQueueIpc>,
) {
    let id = f.asset_id;
    queue.texture_formats.insert(id, f.clone());
    let props = queue.texture_properties.get(&id);
    let Some(device) = queue.gpu_device.clone() else {
        send_texture_2d_result(
            ipc,
            id,
            TextureUpdateResultType::FORMAT_SET,
            queue.texture_pool.get_texture(id).is_none(),
        );
        return;
    };
    let Some(tex) = GpuTexture2d::new_from_format(device.as_ref(), &f, props) else {
        logger::warn!("texture {id}: SetTexture2DFormat rejected (bad size or device)");
        return;
    };
    let existed_before = queue.texture_pool.insert_texture(tex);
    send_texture_2d_result(
        ipc,
        id,
        TextureUpdateResultType::FORMAT_SET,
        !existed_before,
    );
    logger::trace!(
        "texture {} format {:?} {}×{} mips={} (resident_bytes≈{})",
        id,
        f.format,
        f.width,
        f.height,
        f.mipmap_count,
        queue.texture_pool.accounting().texture_resident_bytes()
    );
}

/// Handle [`SetTexture2DProperties`](crate::shared::SetTexture2DProperties).
pub fn on_set_texture_2d_properties(
    queue: &mut AssetTransferQueue,
    p: SetTexture2DProperties,
    ipc: Option<&mut DualQueueIpc>,
) {
    let id = p.asset_id;
    queue.texture_properties.insert(id, p.clone());
    if let Some(t) = queue.texture_pool.get_texture_mut(id) {
        t.apply_properties(&p);
    }
    send_texture_2d_result(ipc, id, TextureUpdateResultType::PROPERTIES_SET, false);
}

/// Handle [`SetTexture2DData`](crate::shared::SetTexture2DData). Pass shared memory when available
/// so mips can be read from the host buffer; if GPU or texture is not ready, data is queued.
pub fn on_set_texture_2d_data(
    queue: &mut AssetTransferQueue,
    d: SetTexture2DData,
    shm: Option<&mut SharedMemoryAccessor>,
    ipc: Option<&mut DualQueueIpc>,
) {
    if d.data.length <= 0 {
        return;
    }
    if !queue.texture_formats.contains_key(&d.asset_id) {
        logger::warn!(
            "texture {}: SetTexture2DData before format; ignored",
            d.asset_id
        );
        return;
    }
    if queue.gpu_device.is_none() || queue.gpu_queue.is_none() {
        if queue.pending_texture_uploads.len() >= MAX_PENDING_TEXTURE_UPLOADS {
            logger::warn!(
                "texture {}: pending texture upload queue full; dropping",
                d.asset_id
            );
            return;
        }
        queue.pending_texture_uploads.push_back(d);
        return;
    }
    let Some(ref device) = queue.gpu_device.clone() else {
        return;
    };
    if queue.texture_pool.get_texture(d.asset_id).is_none() {
        flush_pending_texture_allocations(queue, device);
    }
    if queue.texture_pool.get_texture(d.asset_id).is_none() {
        if queue.pending_texture_uploads.len() >= MAX_PENDING_TEXTURE_UPLOADS {
            logger::warn!(
                "texture {}: no GPU texture and pending full; dropping data",
                d.asset_id
            );
            return;
        }
        queue.pending_texture_uploads.push_back(d);
        return;
    }
    if !d.high_priority && queue.texture_upload_budget_this_poll == 0 {
        if queue.deferred_texture_uploads.len() >= MAX_DEFERRED_TEXTURE_UPLOADS {
            logger::warn!(
                "texture {}: deferred texture upload queue full; dropping",
                d.asset_id
            );
            return;
        }
        logger::trace!(
            "texture {}: deferring low-priority texture upload (budget exhausted)",
            d.asset_id
        );
        queue.deferred_texture_uploads.push_back(d);
        return;
    }
    let Some(shm) = shm else {
        logger::warn!(
            "texture {}: SetTexture2DData needs shared memory for upload",
            d.asset_id
        );
        return;
    };
    try_texture_upload_with_device(queue, d, shm, ipc, true);
}

/// Upload texture mips from shared memory and optionally notify the host on the background queue.
///
/// When `consume_texture_upload_budget` is `true`, a successful upload decrements
/// [`AssetTransferQueue::texture_upload_budget_this_poll`] for non-[`SetTexture2DData::high_priority`]
/// payloads (mirroring [`try_mesh_upload_with_device`] `allow_defer`). Use `false` when draining
/// [`AssetTransferQueue::deferred_texture_uploads`] or replaying [`AssetTransferQueue::pending_texture_uploads`]
/// on attach so deferred work does not double-charge the budget.
pub fn try_texture_upload_with_device(
    queue: &mut AssetTransferQueue,
    data: SetTexture2DData,
    shm: &mut SharedMemoryAccessor,
    ipc: Option<&mut DualQueueIpc>,
    consume_texture_upload_budget: bool,
) {
    let id = data.asset_id;
    let Some(fmt) = queue.texture_formats.get(&id).cloned() else {
        logger::warn!("texture {id}: missing format");
        return;
    };
    let (tex_arc, wgpu_fmt) = match queue.texture_pool.get_texture(id) {
        Some(t) => (t.texture.clone(), t.wgpu_format),
        None => {
            logger::warn!("texture {id}: missing GPU texture");
            return;
        }
    };
    let Some(queue_arc) = queue.gpu_queue.as_ref() else {
        return;
    };
    logger::trace!(
        "texture_upload telemetry asset_id={} payload_bytes={} high_priority={} has_region={} hint_readable={} mip_count={} start_mip={}",
        id,
        data.data.length.max(0),
        data.high_priority,
        data.hint.has_region != 0,
        data.hint.readable != 0,
        data.mip_map_sizes.len(),
        data.start_mip_level,
    );
    let upload_out = shm.with_read_bytes(&data.data, |raw| {
        let q = queue_arc.lock().unwrap_or_else(|e| e.into_inner());
        Some(write_texture2d_mips(
            &q,
            tex_arc.as_ref(),
            &fmt,
            wgpu_fmt,
            &data,
            raw,
        ))
    });
    match upload_out {
        Some(Ok(0)) => {
            logger::trace!("texture {id}: upload skipped (empty region hint)");
        }
        Some(Ok(uploaded_mips)) => {
            if consume_texture_upload_budget && !data.high_priority && uploaded_mips > 0 {
                queue.texture_upload_budget_this_poll =
                    queue.texture_upload_budget_this_poll.saturating_sub(1);
            }
            if let Some(t) = queue.texture_pool.get_texture_mut(id) {
                let start = data.start_mip_level.max(0) as u32;
                let end_exclusive = start.saturating_add(uploaded_mips).min(t.mip_levels_total);
                t.mip_levels_resident = t.mip_levels_resident.max(end_exclusive);
            }
            send_texture_2d_result(ipc, id, TextureUpdateResultType::DATA_UPLOAD, false);
            logger::trace!("texture {id}: data upload ok ({uploaded_mips} mips)");
        }
        Some(Err(e)) => {
            logger::warn!("texture {id}: upload failed: {e}");
        }
        None => {
            logger::warn!("texture {id}: shared memory slice missing");
        }
    }
}

/// Resets the per-poll budget for non-high-priority mesh and texture uploads. Call at the start of
/// each [`crate::runtime::RendererRuntime::poll_ipc`].
pub fn begin_ipc_poll_mesh_upload_budget(queue: &mut AssetTransferQueue) {
    queue.mesh_upload_budget_this_poll = MESH_UPLOAD_NON_HIGH_PRIORITY_BUDGET_PER_POLL;
    queue.texture_upload_budget_this_poll = TEXTURE_UPLOAD_NON_HIGH_PRIORITY_BUDGET_PER_POLL;
}

/// Processes mesh uploads deferred during the current IPC batch (low-priority overflow).
pub fn drain_deferred_mesh_uploads_after_poll(
    queue: &mut AssetTransferQueue,
    shm: &mut SharedMemoryAccessor,
    ipc: Option<&mut DualQueueIpc>,
) {
    let Some(device) = queue.gpu_device.clone() else {
        return;
    };
    let mut drained = 0usize;
    if let Some(ipc_ref) = ipc {
        while drained < MAX_DEFERRED_MESH_UPLOADS_DRAIN_PER_POLL {
            let Some(data) = queue.deferred_mesh_uploads.pop_front() else {
                break;
            };
            try_mesh_upload_with_device(&mut *queue, &device, data, shm, Some(ipc_ref), false);
            drained += 1;
        }
    } else {
        while drained < MAX_DEFERRED_MESH_UPLOADS_DRAIN_PER_POLL {
            let Some(data) = queue.deferred_mesh_uploads.pop_front() else {
                break;
            };
            try_mesh_upload_with_device(&mut *queue, &device, data, shm, None, false);
            drained += 1;
        }
    }
}

/// Processes texture uploads deferred when the non-high-priority texture budget was exhausted
/// mid-batch.
pub fn drain_deferred_texture_uploads_after_poll(
    queue: &mut AssetTransferQueue,
    shm: &mut SharedMemoryAccessor,
    ipc: Option<&mut DualQueueIpc>,
) {
    let mut drained = 0usize;
    if let Some(ipc_ref) = ipc {
        while drained < MAX_DEFERRED_TEXTURE_UPLOADS_DRAIN_PER_POLL {
            let Some(data) = queue.deferred_texture_uploads.pop_front() else {
                break;
            };
            try_texture_upload_with_device(&mut *queue, data, shm, Some(ipc_ref), false);
            drained += 1;
        }
    } else {
        while drained < MAX_DEFERRED_TEXTURE_UPLOADS_DRAIN_PER_POLL {
            let Some(data) = queue.deferred_texture_uploads.pop_front() else {
                break;
            };
            try_texture_upload_with_device(&mut *queue, data, shm, None, false);
            drained += 1;
        }
    }
}

/// Remove a texture asset from CPU tables and the pool.
pub fn on_unload_texture_2d(queue: &mut AssetTransferQueue, u: UnloadTexture2D) {
    let id = u.asset_id;
    queue.texture_formats.remove(&id);
    queue.texture_properties.remove(&id);
    if queue.texture_pool.remove_texture(id) {
        logger::info!(
            "texture {id} unloaded (mesh≈{} tex≈{} total≈{})",
            queue.mesh_pool.accounting().mesh_resident_bytes(),
            queue.texture_pool.accounting().texture_resident_bytes(),
            queue.mesh_pool.accounting().total_resident_bytes()
        );
    }
}

/// Ingest mesh bytes from shared memory; notifies host when `ipc` is set.
pub fn try_process_mesh_upload(
    queue: &mut AssetTransferQueue,
    data: MeshUploadData,
    shm: &mut SharedMemoryAccessor,
    ipc: Option<&mut DualQueueIpc>,
) {
    if data.buffer.length <= 0 {
        return;
    }
    let Some(device) = queue.gpu_device.clone() else {
        if queue.pending_mesh_uploads.len() >= MAX_PENDING_MESH_UPLOADS {
            logger::warn!(
                "mesh upload pending queue full; dropping asset {}",
                data.asset_id
            );
            return;
        }
        queue.pending_mesh_uploads.push_back(data);
        return;
    };
    if !data.high_priority && queue.mesh_upload_budget_this_poll == 0 {
        if queue.deferred_mesh_uploads.len() >= MAX_DEFERRED_MESH_UPLOADS {
            logger::warn!(
                "mesh {}: deferred mesh upload queue full; dropping",
                data.asset_id
            );
            return;
        }
        logger::trace!(
            "mesh {}: deferring low-priority mesh upload (budget exhausted)",
            data.asset_id
        );
        queue.deferred_mesh_uploads.push_back(data);
        return;
    }
    try_mesh_upload_with_device(queue, &device, data, shm, ipc, true);
}

/// `allow_defer` must be `false` when draining [`AssetTransferQueue::deferred_mesh_uploads`] so work is not
/// re-queued.
pub fn try_mesh_upload_with_device(
    queue: &mut AssetTransferQueue,
    device: &Arc<wgpu::Device>,
    data: MeshUploadData,
    shm: &mut SharedMemoryAccessor,
    ipc: Option<&mut DualQueueIpc>,
    allow_defer: bool,
) {
    let hint = data.upload_hint.flags;
    let high_priority = data.high_priority;
    let asset_id = data.asset_id;
    let started = Instant::now();

    let input_fp = mesh_upload_input_fingerprint(&data);
    let layout = if let Some(l) = queue.mesh_pool.get_cached_mesh_layout(asset_id, input_fp) {
        l
    } else {
        let Some(l) = compute_and_validate_mesh_layout(&data) else {
            logger::error!("mesh {asset_id}: invalid mesh layout or buffer descriptor");
            return;
        };
        queue
            .mesh_pool
            .set_cached_mesh_layout(asset_id, input_fp, l);
        l
    };

    let existing = queue.mesh_pool.get_mesh(asset_id);
    let queue_guard = queue
        .gpu_queue
        .as_ref()
        .map(|q| q.lock().unwrap_or_else(|e| e.into_inner()));
    let queue_ref = queue_guard.as_deref();

    let upload_result = shm.with_read_bytes(&data.buffer, |raw| {
        try_upload_mesh_from_raw(device.as_ref(), queue_ref, raw, &data, existing, &layout)
    });

    let Some(mesh) = upload_result else {
        // A `success` (or failure) field on [`MeshUploadResult`] requires updating the IPC contract
        // via SharedTypeGenerator and the host `Renderite.Shared` assembly; until then the host
        // cannot distinguish failure from a missing callback (timeouts may apply upstream).
        logger::error!(
            "mesh {asset_id}: upload failed or rejected — host callback not completed (no MeshUploadResult sent)"
        );
        return;
    };

    if allow_defer && !high_priority {
        queue.mesh_upload_budget_this_poll = queue.mesh_upload_budget_this_poll.saturating_sub(1);
    }

    let existed_before = queue.mesh_pool.insert_mesh(mesh);
    if let Some(ipc) = ipc {
        ipc.send_background(RendererCommand::mesh_upload_result(MeshUploadResult {
            asset_id,
            instance_changed: !existed_before,
        }));
    }

    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    logger::trace!(
        "mesh_upload telemetry asset_id={} high_priority={} hint_flags=0x{:08x} (vlayout={} geo={} submesh={} dyn={}) replaced_existing={} time_ms={:.3}",
        asset_id,
        high_priority,
        hint.0,
        hint.vertex_layout(),
        hint.geometry(),
        hint.submesh_layout(),
        hint.dynamic(),
        existed_before,
        elapsed_ms
    );

    logger::trace!(
        "mesh {} uploaded (replaced={} resident_bytes≈{})",
        asset_id,
        existed_before,
        queue.mesh_pool.accounting().total_resident_bytes()
    );
}

/// Remove a mesh from the pool.
pub fn on_mesh_unload(queue: &mut AssetTransferQueue, u: MeshUnload) {
    if queue.mesh_pool.remove_mesh(u.asset_id) {
        logger::info!(
            "mesh {} unloaded (resident_bytes≈{})",
            u.asset_id,
            queue.mesh_pool.accounting().total_resident_bytes()
        );
    }
}
