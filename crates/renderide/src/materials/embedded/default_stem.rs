//! Unity shader asset names mapped to composed WGSL stems under `shaders/target/` (embedded at build time).
//!
//! Resolution uses [`crate::assets::util::normalize_unity_shader_lookup_key`] and probes
//! `{normalized_key}_default`, matching material source stems under `shaders/materials/*.wgsl`
//! (see crate `build.rs`).

use crate::assets::util::normalize_unity_shader_lookup_key;
use crate::embedded_shaders;

#[cfg(test)]
mod tests;

/// Returns `{normalized_key}_default` when that composed target exists in the embedded table.
pub fn embedded_default_stem_for_shader_asset_name(name: &str) -> Option<String> {
    let key = normalize_unity_shader_lookup_key(name);
    let stem = format!("{key}_default");
    if embedded_shaders::embedded_target_wgsl(&stem).is_some() {
        return Some(stem);
    }
    None
}
