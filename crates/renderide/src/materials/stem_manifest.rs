//! Unity logical shader names → composed WGSL stems under `shaders/target/` (embedded at build time).
//!
//! Resolution uses [`crate::assets::util::normalize_unity_shader_lookup_key`] and a direct
//! `{normalized_key}_default` probe. Material **source** filenames under `shaders/source/materials/`
//! use stems that match normalized Unity shader **asset** names (see crate `build.rs`).

use crate::assets::util::normalize_unity_shader_lookup_key;
use crate::embedded_shaders;

/// Returns `{normalized_key}_default` when that composed target exists in the embedded table.
pub fn embedded_default_stem_for_unity_name(name: &str) -> Option<String> {
    let key = normalize_unity_shader_lookup_key(name);
    let stem = format!("{key}_default");
    embedded_shaders::embedded_target_wgsl(&stem).map(|_| stem)
}

/// Returns the composed WGSL stem for `name` when an embedded `{key}_default` target exists (routing).
pub fn manifest_stem_for_unity_name(name: &str) -> Option<String> {
    embedded_default_stem_for_unity_name(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_asset_style_stem_directly() {
        assert_eq!(
            embedded_default_stem_for_unity_name("ui_textunlit").as_deref(),
            Some("ui_textunlit_default")
        );
    }

    #[test]
    fn resolves_ui_textunlit_from_unity_asset_token() {
        assert_eq!(
            embedded_default_stem_for_unity_name("UI_TextUnlit").as_deref(),
            Some("ui_textunlit_default")
        );
    }

    #[test]
    fn shader_lab_path_form_does_not_match_asset_stem_file() {
        assert_eq!(embedded_default_stem_for_unity_name("UI/Text/Unlit"), None);
    }
}
