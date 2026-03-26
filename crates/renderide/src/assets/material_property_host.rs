//! Host `MaterialPropertyIdRequest` / `MaterialPropertyIdResult` support.
//!
//! FrooxEngine’s render material manager sends [`MaterialPropertyIdRequest`](crate::shared::MaterialPropertyIdRequest)
//! with shader property **names** (e.g. `_MainTex`). The renderer must reply with one integer per name;
//! those integers are stored on host `MaterialProperty` values and used in `MaterialsUpdateBatch` as
//! `property_id`. Unity’s Renderite driver uses `Shader.PropertyToID` for the same request; this
//! crate interns names to stable integers for the process lifetime so repeated names share one id.

use std::collections::HashMap;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{LazyLock, Mutex};

use crate::config::RenderConfig;

static NEXT_PROPERTY_ID: AtomicI32 = AtomicI32::new(1);
static NAME_TO_ID: LazyLock<Mutex<HashMap<String, i32>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Returns a stable integer id for `name`, allocating a new id the first time this name appears.
///
/// Empty names map to `0` and are not inserted (matches “no property” semantics).
pub fn intern_host_material_property_id(name: &str) -> i32 {
    if name.is_empty() {
        return 0;
    }
    let mut map = NAME_TO_ID
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(&id) = map.get(name) {
        return id;
    }
    let id = NEXT_PROPERTY_ID.fetch_add(1, Ordering::Relaxed);
    map.insert(name.to_string(), id);
    id
}

/// Copies known FrooxEngine / Unity property names into [`RenderConfig`] native UI id maps when
/// [`RenderConfig::use_native_ui_wgsl`] is enabled.
///
/// See the FrooxEngine-to-field table in [`crate::assets::ui_material_contract`] module documentation.
pub fn apply_froox_material_property_name_to_native_ui_config(
    config: &mut RenderConfig,
    name: &str,
    id: i32,
) {
    if !config.use_native_ui_wgsl || name.is_empty() {
        return;
    }
    let u = &mut config.ui_unlit_property_ids;
    let t = &mut config.ui_text_unlit_property_ids;
    match name {
        "_MainTex" => u.main_tex = id,
        "_MaskTex" => u.mask_tex = id,
        "_MainTex_ST" => u.main_tex_st = id,
        "_MaskTex_ST" => u.mask_tex_st = id,
        "_Tint" => u.tint = id,
        "_OverlayTint" => {
            u.overlay_tint = id;
            t.overlay_tint = id;
        }
        "_Cutoff" => u.cutoff = id,
        "_Rect" => {
            u.rect = id;
            t.rect = id;
        }
        "_FontAtlas" => t.font_atlas = id,
        "_TintColor" => t.tint_color = id,
        "_OutlineColor" => t.outline_color = id,
        "_BackgroundColor" => t.background_color = id,
        "_Range" => t.range = id,
        "_FaceDilate" => t.face_dilate = id,
        "_FaceSoftness" => t.face_softness = id,
        "_OutlineSize" => t.outline_size = id,
        "_SrcBlend" => {
            u.src_blend = id;
            t.src_blend = id;
        }
        "_DstBlend" => {
            u.dst_blend = id;
            t.dst_blend = id;
        }
        // FrooxEngine / Unity shader keywords (see `ui_material_contract` module docs).
        "ALPHACLIP" | "alphaclip" => u.alphaclip = id,
        "RECTCLIP" | "rectclip" => {
            u.rectclip = id;
            t.rectclip = id;
        }
        "OVERLAY" | "overlay" => {
            u.overlay = id;
            t.overlay = id;
        }
        "TEXTURE_NORMALMAP" | "texture_normalmap" => u.texture_normalmap = id,
        "TEXTURE_LERPCOLOR" | "texture_lerpcolor" => u.texture_lerpcolor = id,
        "_MASK_TEXTURE_MUL" | "mask_texture_mul" => u.mask_texture_mul = id,
        "_MASK_TEXTURE_CLIP" | "mask_texture_clip" => u.mask_texture_clip = id,
        "RASTER" | "raster" => t.raster = id,
        "SDF" | "sdf" => t.sdf = id,
        "MSDF" | "msdf" => t.msdf = id,
        "OUTLINE" | "outline" => t.outline = id,
        _ => {}
    }
}

#[cfg(test)]
pub(crate) fn reset_material_property_intern_table_for_tests() {
    NEXT_PROPERTY_ID.store(1, Ordering::Relaxed);
    let mut map = NAME_TO_ID
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    map.clear();
}

#[cfg(test)]
mod tests {
    use super::{
        apply_froox_material_property_name_to_native_ui_config, intern_host_material_property_id,
        reset_material_property_intern_table_for_tests,
    };
    use crate::config::RenderConfig;

    #[test]
    fn intern_returns_same_id_for_same_name() {
        reset_material_property_intern_table_for_tests();
        let a = intern_host_material_property_id("_MainTex");
        let b = intern_host_material_property_id("_MainTex");
        assert_eq!(a, b);
        assert_ne!(a, intern_host_material_property_id("_Tint"));
    }

    #[test]
    fn apply_maps_froox_names_to_ui_unlit_and_text_ids() {
        reset_material_property_intern_table_for_tests();
        let mut c = RenderConfig {
            use_native_ui_wgsl: true,
            ..Default::default()
        };
        let m = intern_host_material_property_id("_MainTex");
        let f = intern_host_material_property_id("_FontAtlas");
        apply_froox_material_property_name_to_native_ui_config(&mut c, "_MainTex", m);
        apply_froox_material_property_name_to_native_ui_config(&mut c, "_FontAtlas", f);
        assert_eq!(c.ui_unlit_property_ids.main_tex, m);
        assert_eq!(c.ui_text_unlit_property_ids.font_atlas, f);
    }

    #[test]
    fn apply_skips_when_native_ui_wgsl_disabled() {
        reset_material_property_intern_table_for_tests();
        let mut c = RenderConfig {
            use_native_ui_wgsl: false,
            ..Default::default()
        };
        let id = intern_host_material_property_id("_MainTex");
        apply_froox_material_property_name_to_native_ui_config(&mut c, "_MainTex", id);
        assert_eq!(c.ui_unlit_property_ids.main_tex, -1);
    }

    #[test]
    fn apply_maps_shader_keyword_alphaclip() {
        reset_material_property_intern_table_for_tests();
        let mut c = RenderConfig {
            use_native_ui_wgsl: true,
            ..Default::default()
        };
        let id = intern_host_material_property_id("ALPHACLIP");
        apply_froox_material_property_name_to_native_ui_config(&mut c, "ALPHACLIP", id);
        assert_eq!(c.ui_unlit_property_ids.alphaclip, id);
    }
}
