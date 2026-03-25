//! Shader asset type for host-uploaded WGSL (or future intermediate representations).
//!
//! Filled by [`super::AssetRegistry::handle_shader_upload`].

use super::Asset;
use super::AssetId;

/// Stored shader data for pipeline creation.
pub struct ShaderAsset {
    /// Unique identifier for this shader.
    pub id: AssetId,
    /// Optional WGSL source. Populated when shader_upload provides it.
    pub wgsl_source: Option<String>,
}

impl Asset for ShaderAsset {
    fn id(&self) -> AssetId {
        self.id
    }
}
