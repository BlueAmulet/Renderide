//! Host-authored asset descriptors and sampler/property catalogs.

use hashbrown::HashMap;

use crate::shared::{
    SetCubemapFormat, SetCubemapProperties, SetRenderTextureFormat, SetTexture2DFormat,
    SetTexture2DProperties, SetTexture3DFormat, SetTexture3DProperties, VideoTextureProperties,
};

/// Latest host format/property rows keyed by asset id.
#[derive(Default)]
pub(crate) struct AssetCatalogs {
    /// Latest render-texture format rows.
    pub(crate) render_texture_formats: HashMap<i32, SetRenderTextureFormat>,
    /// Latest Texture2D format rows.
    pub(crate) texture_formats: HashMap<i32, SetTexture2DFormat>,
    /// Latest Texture2D sampler/property rows.
    pub(crate) texture_properties: HashMap<i32, SetTexture2DProperties>,
    /// Latest Texture3D format rows.
    pub(crate) texture3d_formats: HashMap<i32, SetTexture3DFormat>,
    /// Latest Texture3D sampler/property rows.
    pub(crate) texture3d_properties: HashMap<i32, SetTexture3DProperties>,
    /// Latest cubemap format rows.
    pub(crate) cubemap_formats: HashMap<i32, SetCubemapFormat>,
    /// Latest cubemap sampler/property rows.
    pub(crate) cubemap_properties: HashMap<i32, SetCubemapProperties>,
    /// Latest video texture sampler/property rows.
    pub(crate) video_texture_properties: HashMap<i32, VideoTextureProperties>,
}

impl AssetCatalogs {
    /// Returns cached video texture properties, or stable defaults tagged with `asset_id`.
    pub(crate) fn video_texture_properties_or_default(
        &self,
        asset_id: i32,
    ) -> VideoTextureProperties {
        self.video_texture_properties
            .get(&asset_id)
            .cloned()
            .unwrap_or(VideoTextureProperties {
                asset_id,
                ..VideoTextureProperties::default()
            })
    }
}
