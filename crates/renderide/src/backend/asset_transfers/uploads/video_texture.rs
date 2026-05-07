use super::super::AssetTransferQueue;
use crate::assets::video::player::VideoPlayer;
use renderide_shared::{
    UnloadVideoTexture, VideoTextureLoad, VideoTextureProperties, VideoTextureStartAudioTrack,
    VideoTextureUpdate,
};

/// Replays video texture state that arrived before GPU attach.
pub fn attach_flush_pending_video_textures(queue: &mut AssetTransferQueue) {
    let pending_properties: Vec<VideoTextureProperties> = queue
        .catalogs
        .video_texture_properties
        .values()
        .cloned()
        .collect();
    for props in pending_properties {
        if queue.ensure_video_texture_with_props(&props).is_none() {
            logger::warn!(
                "video texture {}: GPU device unavailable while flushing properties",
                props.asset_id
            );
        }
    }

    let pending_loads: Vec<VideoTextureLoad> = queue
        .pending
        .pending_video_texture_loads
        .drain()
        .map(|(_, load)| load)
        .collect();
    for load in pending_loads {
        on_video_texture_load(queue, load);
    }
}

/// Handle [`VideoTextureLoad`].
pub fn on_video_texture_load(queue: &mut AssetTransferQueue, v: VideoTextureLoad) {
    let id = v.asset_id;
    let Some(device) = queue.gpu.gpu_device.clone() else {
        queue.pending.pending_video_texture_loads.insert(id, v);
        return;
    };
    let Some(gpu_queue) = queue.gpu.gpu_queue.clone() else {
        queue.pending.pending_video_texture_loads.insert(id, v);
        return;
    };

    let props = queue.catalogs.video_texture_properties_or_default(id);
    if queue.ensure_video_texture_with_props(&props).is_none() {
        logger::warn!("video texture {id}: failed to create GPU placeholder before load");
        return;
    }

    if let Some(player) = VideoPlayer::new(v, device, gpu_queue) {
        queue.pending.pending_video_texture_loads.remove(&id);
        queue.video.video_players.insert(id, player);
    }
}

/// Handle [`VideoTextureUpdate`].
pub fn on_video_texture_update(queue: &mut AssetTransferQueue, v: VideoTextureUpdate) {
    let id = v.asset_id;
    if let Some(t) = queue.video.video_players.get_mut(&id) {
        t.handle_update(v);
    }
}

/// Handle [`VideoTextureProperties`].
pub fn on_video_texture_properties(queue: &mut AssetTransferQueue, p: VideoTextureProperties) {
    let id = p.asset_id;
    queue
        .catalogs
        .video_texture_properties
        .insert(id, p.clone());

    if let Some(tex) = queue.ensure_video_texture_with_props(&p) {
        tex.set_props(&p);
        logger::trace!(
            "video texture {} (resident_bytes~{})",
            id,
            queue
                .pools
                .video_texture_pool
                .accounting()
                .texture_resident_bytes()
        );
    }
}

/// Handle [`VideoTextureStartAudioTrack`].
pub fn on_video_texture_start_audio_track(
    queue: &mut AssetTransferQueue,
    s: VideoTextureStartAudioTrack,
) {
    let id = s.asset_id;
    if let Some(tex) = queue.video.video_players.get_mut(&id) {
        tex.handle_start_audio_track(s);
    }
}

/// Handle [`UnloadVideoTexture`].
pub fn on_unload_video_texture(queue: &mut AssetTransferQueue, u: UnloadVideoTexture) {
    let id = u.asset_id;
    queue.pending.pending_video_texture_loads.remove(&id);
    queue.catalogs.video_texture_properties.remove(&id);
    queue.video.video_players.remove(&id);
    if queue.pools.video_texture_pool.remove(id) {
        logger::info!(
            "video texture {id} unloaded (total~{})",
            queue
                .pools
                .video_texture_pool
                .accounting()
                .texture_resident_bytes(),
        );
    }
}
