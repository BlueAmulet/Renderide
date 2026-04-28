//! Resolves [`ShaderUpload`](crate::shared::ShaderUpload) to a [`RasterPipelineKind`] for [`MaterialRegistry`](crate::materials::MaterialRegistry).
//!
//! The stock host sends an on-disk shader AssetBundle path in [`ShaderUpload::file`]. Routing reads
//! that bundle, extracts the Shader object's `m_Container` asset filename, and maps that filename to
//! an embedded WGSL stem.
//!
//! Names with an embedded `{asset_name}_default` WGSL target (see [`crate::materials::embedded_shader_stem`]) resolve to
//! [`RasterPipelineKind::EmbeddedStem`]; unresolved or non-embedded shaders use
//! [`RasterPipelineKind::Null`] (the black/grey checkerboard) as the **only** mesh fallback
//! (there is no separate solid-color pipeline).

use std::path::Path;
use std::sync::Arc;

use crate::materials::{embedded_default_stem_for_shader_asset_name, RasterPipelineKind};

use crate::shared::ShaderUpload;

use super::unity_asset;

/// Resolved upload: optional AssetBundle shader asset name plus the raster pipeline kind.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedShaderUpload {
    /// Shader asset filename or stem from the AssetBundle `m_Container` entry.
    pub shader_asset_name: Option<String>,
    /// Pipeline kind passed to [`crate::materials::MaterialRegistry::map_shader_route`].
    pub pipeline: RasterPipelineKind,
}

/// Full resolution pipeline for a host [`ShaderUpload`].
pub fn resolve_shader_upload(data: &ShaderUpload) -> ResolvedShaderUpload {
    let shader_asset_name = data
        .file
        .as_deref()
        .and_then(|file| unity_asset::try_resolve_shader_asset_name_from_path(Path::new(file)));
    let pipeline = match shader_asset_name.as_deref() {
        Some(name) => {
            if let Some(stem) = embedded_default_stem_for_shader_asset_name(name) {
                RasterPipelineKind::EmbeddedStem(Arc::from(stem))
            } else {
                RasterPipelineKind::Null
            }
        }
        None => RasterPipelineKind::Null,
    };
    ResolvedShaderUpload {
        shader_asset_name,
        pipeline,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::materials::RasterPipelineKind;

    #[test]
    fn missing_file_uses_null_pipeline() {
        let u = ShaderUpload {
            asset_id: 1,
            file: None,
        };
        let r = resolve_shader_upload(&u);
        assert_eq!(r.shader_asset_name, None);
        assert_eq!(r.pipeline, RasterPipelineKind::Null);
    }

    #[test]
    fn inline_shader_lab_text_is_not_a_routing_source() {
        let u = ShaderUpload {
            asset_id: 2,
            file: Some("Shader \"Unlit\"\n{\n".to_string()),
        };
        let r = resolve_shader_upload(&u);
        assert_eq!(r.shader_asset_name, None);
        assert_eq!(r.pipeline, RasterPipelineKind::Null);
    }

    #[test]
    fn non_assetbundle_file_uses_null_pipeline() {
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("unlit.shader");
        std::fs::write(&path, "Shader \"Unlit\" { }").expect("write test shader text");
        let u = ShaderUpload {
            asset_id: 3,
            file: Some(path.to_string_lossy().to_string()),
        };
        let r = resolve_shader_upload(&u);
        assert_eq!(r.shader_asset_name, None);
        assert_eq!(r.pipeline, RasterPipelineKind::Null);
    }
}
