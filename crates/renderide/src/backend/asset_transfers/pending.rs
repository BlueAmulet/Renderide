//! Upload commands deferred until GPU state, formats, or shared memory are available.

use std::collections::VecDeque;

use hashbrown::HashMap;

use crate::shared::{
    MeshUploadData, SetCubemapData, SetTexture2DData, SetTexture3DData, VideoTextureLoad,
};

/// Pre-GPU or not-yet-resident upload commands awaiting replay.
#[derive(Default)]
pub(crate) struct PendingAssetUploads {
    /// Mesh payloads waiting for GPU or shared memory.
    pub(crate) pending_mesh_uploads: VecDeque<MeshUploadData>,
    /// Texture2D payloads waiting for GPU allocation or shared memory.
    pub(crate) pending_texture_uploads: VecDeque<SetTexture2DData>,
    /// Texture3D payloads waiting for GPU allocation or shared memory.
    pub(crate) pending_texture3d_uploads: VecDeque<SetTexture3DData>,
    /// Cubemap payloads waiting for GPU allocation or shared memory.
    pub(crate) pending_cubemap_uploads: VecDeque<SetCubemapData>,
    /// Latest video load commands received before GPU attach.
    pub(crate) pending_video_texture_loads: HashMap<i32, VideoTextureLoad>,
}
