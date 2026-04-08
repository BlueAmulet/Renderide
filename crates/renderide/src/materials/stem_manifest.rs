//! Unity logical shader names → composed WGSL stems under `shaders/target/` (embedded at build time).
//!
//! Resolution uses [`crate::assets::util::normalize_unity_shader_lookup_key`] and checks that
//! `{key}_default` exists in [`crate::embedded_shaders::embedded_target_wgsl`] — no JSON manifest.

use crate::assets::util::normalize_unity_shader_lookup_key;
use crate::embedded_shaders;

/// Returns `{normalized_key}_default` when that composed target was built into the embedded table.
///
/// Plain host stems such as `UI_TextUnlit` normalize to `ui_textunlit`; that key aliases to the same
/// targets as ShaderLab `UI/Text/Unlit` → `ui_text_unlit` so one WGSL source suffices.
pub fn embedded_default_stem_for_unity_name(name: &str) -> Option<String> {
    let key = normalize_unity_shader_lookup_key(name);
    let effective_key = match key.as_str() {
        "ui_textunlit" => "ui_text_unlit",
        _ => key.as_str(),
    };
    let stem = format!("{effective_key}_default");
    embedded_shaders::embedded_target_wgsl(&stem).map(|_| stem)
}

/// Returns the composed WGSL stem for `name` when an embedded `{key}_default` target exists (routing).
pub fn manifest_stem_for_unity_name(name: &str) -> Option<String> {
    embedded_default_stem_for_unity_name(name)
}
