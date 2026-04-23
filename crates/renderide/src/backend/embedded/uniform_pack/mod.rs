//! Uniform byte packing for embedded `@group(1)` material blocks (reflection-driven defaults and keywords).

use crate::assets::material::{
    MaterialPropertyLookupIds, MaterialPropertyStore, MaterialPropertyValue,
};
use crate::materials::{ReflectedRasterLayout, ReflectedUniformField, ReflectedUniformScalarKind};

use super::layout::{EmbeddedSharedKeywordIds, StemEmbeddedPropertyIds};

mod helpers;
mod tables;

use helpers::{default_vec4_for_field, first_float_by_pids, keyword_float_enabled_by_pid};
use tables::inferred_keyword_float_f32;

/// Packs `UI_TextUnlit`-style `_TextMode`: explicit host value, else `0` (MSDF default).
///
/// FrooxEngine routes text-mode selection through `ShaderKeywords.SetKeyword("MSDF"/"SDF"/"RASTER", …)`
/// which writes to the variant bitmask, not to any float property in the material store. The
/// renderer never receives that bitmask, so the former `MSDF`/`SDF`/`RASTER` keyword probes
/// always failed; they were removed.
fn packed_text_mode_f32(
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    kw: &EmbeddedSharedKeywordIds,
) -> f32 {
    if let Some(MaterialPropertyValue::Float(f)) = store.get_merged(lookup, kw.text_mode) {
        return *f;
    }
    0.0
}

/// Packs `_RectClip`: explicit host value, else `0`.
///
/// FrooxEngine sets rect-clipping via `ShaderKeywords.SetKeyword("RECTCLIP", …)` (variant
/// bitmask); the renderer never sees that signal as a float property, so the former `RECTCLIP` /
/// `rectclip` keyword probes always failed. Removed.
fn packed_rect_clip_f32(
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    kw: &EmbeddedSharedKeywordIds,
) -> f32 {
    if let Some(MaterialPropertyValue::Float(f)) = store.get_merged(lookup, kw.rect_clip) {
        return *f;
    }
    0.0
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PackedFlagsLayout {
    Unlit,
    UiUnlit,
}

fn packed_flags_layout_for_ids(ids: &StemEmbeddedPropertyIds) -> PackedFlagsLayout {
    if ids.uniform_field_ids.contains_key("_Rect")
        && ids.uniform_field_ids.contains_key("_OverlayTint")
        && !ids.uniform_field_ids.contains_key("_OffsetTex_ST")
    {
        PackedFlagsLayout::UiUnlit
    } else {
        PackedFlagsLayout::Unlit
    }
}

/// Packs `flags` from `_Flags` when present; otherwise derives bits from texture presence,
/// `_Cutoff`, and Unity `#pragma multi_compile` keyword floats.
fn pack_flags_u32(
    field_name: &str,
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    kw: &EmbeddedSharedKeywordIds,
    primary_texture_any_kind_present: bool,
    cutoff: f32,
    layout: PackedFlagsLayout,
) -> u32 {
    if field_name != "flags" {
        return 0;
    }
    if let Some(v) = store.get_merged(lookup, kw.flags) {
        match v {
            MaterialPropertyValue::Float(f) => return (*f).max(0.0) as u32,
            MaterialPropertyValue::Float4(a) => return a[0].max(0.0) as u32,
            _ => {}
        }
    }

    let mut flags = 0u32;

    if primary_texture_any_kind_present {
        flags |= 0x01;
    }
    if cutoff > 0.0 && cutoff < 1.0 {
        flags |= 0x02;
    }

    let (mask_mul_bit, mask_clip_bit) = match layout {
        PackedFlagsLayout::Unlit => (0x08, 0x10),
        PackedFlagsLayout::UiUnlit => (0x10, 0x20),
    };

    match layout {
        PackedFlagsLayout::Unlit => {
            if keyword_float_enabled_by_pid(store, lookup, kw.offset_texture) {
                flags |= 0x04;
            }
        }
        PackedFlagsLayout::UiUnlit => {
            if packed_rect_clip_f32(store, lookup, kw) > 0.5 {
                flags |= 0x04;
            }
        }
    }
    if keyword_float_enabled_by_pid(store, lookup, kw.mask_texture_mul) {
        flags |= mask_mul_bit;
    }
    if keyword_float_enabled_by_pid(store, lookup, kw.mask_texture_clip) {
        flags |= mask_clip_bit;
    }
    if let Some(mask_mode) = first_float_by_pids(store, lookup, &[kw.mask_mode]) {
        match mask_mode.round() as i32 {
            // Resonite MaskTextureMode: MultiplyAlpha = 0, Cutoff = 1.
            0 => flags |= mask_mul_bit,
            1 => flags |= mask_clip_bit,
            // Tolerate future/combined modes by enabling both paths.
            v if v > 1 => flags |= mask_mul_bit | mask_clip_bit,
            _ => {}
        }
    }
    if let Some(blend_mode) = first_float_by_pids(store, lookup, &[kw.blend_mode]) {
        // Resonite BlendMode.Cutout = 1.
        if blend_mode.round() as i32 == 1 {
            flags |= 0x02;
        }
    }
    if keyword_float_enabled_by_pid(store, lookup, kw.mul_rgb_by_alpha)
        && layout == PackedFlagsLayout::Unlit
    {
        flags |= 0x20;
    }
    if keyword_float_enabled_by_pid(store, lookup, kw.mul_alpha_intensity)
        && layout == PackedFlagsLayout::Unlit
    {
        flags |= 0x40;
    }

    flags
}

fn write_f32_at(buf: &mut [u8], field: &ReflectedUniformField, v: f32) {
    let off = field.offset as usize;
    if off + 4 <= buf.len() && field.size >= 4 {
        buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
    }
}

fn write_f32x4_at(buf: &mut [u8], field: &ReflectedUniformField, v: &[f32; 4]) {
    let off = field.offset as usize;
    if off + 16 <= buf.len() && field.size >= 16 {
        for (i, c) in v.iter().enumerate() {
            let o = off + i * 4;
            buf[o..o + 4].copy_from_slice(&c.to_le_bytes());
        }
    }
}

fn write_u32_at(buf: &mut [u8], field: &ReflectedUniformField, v: u32) {
    let off = field.offset as usize;
    if off + 4 <= buf.len() && field.size >= 4 {
        buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
    }
}

/// Builds CPU bytes for the reflected material uniform block.
pub(crate) fn build_embedded_uniform_bytes(
    reflected: &ReflectedRasterLayout,
    ids: &StemEmbeddedPropertyIds,
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    primary_texture_any_kind_present: bool,
) -> Option<Vec<u8>> {
    let u = reflected.material_uniform.as_ref()?;
    let mut buf = vec![0u8; u.total_size as usize];

    let mut cutoff = 0.5f32;
    for (field_name, field) in &u.fields {
        let pid = *ids.uniform_field_ids.get(field_name)?;
        match field.kind {
            ReflectedUniformScalarKind::Vec4 => {
                let default =
                    default_vec4_for_field(helpers::shader_writer_unescaped_field_name(field_name));
                let mut v = default;
                if let Some(MaterialPropertyValue::Float4(c)) = store.get_merged(lookup, pid) {
                    v = *c;
                }
                write_f32x4_at(&mut buf, field, &v);
            }
            ReflectedUniformScalarKind::F32 => {
                let v = if field_name == "_TextMode" {
                    packed_text_mode_f32(store, lookup, ids.shared.as_ref())
                } else if field_name == "_RectClip" {
                    packed_rect_clip_f32(store, lookup, ids.shared.as_ref())
                } else if let Some(MaterialPropertyValue::Float(f)) = store.get_merged(lookup, pid)
                {
                    *f
                } else if field_name == "_Cutoff" {
                    // Cutoff materials always have the host writer push `_Cutoff` on first batch;
                    // 0.5 is the Unity-convention fallback for the unobservable pre-first-batch window.
                    first_float_by_pids(store, lookup, &[ids.shared.alpha_cutoff]).unwrap_or(0.5)
                } else {
                    // Multi-compile keyword fields (`_NORMALMAP`, `_ALPHATEST_ON`, …) are inferred
                    // from texture presence + blend reconstruction. Other unknown fields fall
                    // through to uniform-buffer zero-init parity; after the orphan-field cleanup
                    // that branch should be unreachable for any field the renderer actually packs.
                    inferred_keyword_float_f32(
                        helpers::shader_writer_unescaped_field_name(field_name),
                        store,
                        lookup,
                        ids,
                    )
                    .unwrap_or(0.0)
                };
                if field_name == "_Cutoff" {
                    cutoff = v;
                }
                write_f32_at(&mut buf, field, v);
            }
            ReflectedUniformScalarKind::U32 => {
                if field_name == "flags" {
                    let flags = pack_flags_u32(
                        field_name,
                        store,
                        lookup,
                        ids.shared.as_ref(),
                        primary_texture_any_kind_present,
                        cutoff,
                        packed_flags_layout_for_ids(ids),
                    );
                    write_u32_at(&mut buf, field, flags);
                }
            }
            ReflectedUniformScalarKind::Unsupported => {}
        }
    }

    Some(buf)
}

#[cfg(test)]
mod text_uniform_packing_tests {
    use super::tables::inferred_keyword_float_f32;
    use super::*;
    use crate::assets::material::PropertyIdRegistry;
    use crate::assets::material::{MaterialPropertyLookupIds, MaterialPropertyStore};
    use crate::backend::embedded::layout::{EmbeddedSharedKeywordIds, StemEmbeddedPropertyIds};

    fn lookup(material_id: i32) -> MaterialPropertyLookupIds {
        MaterialPropertyLookupIds {
            material_asset_id: material_id,
            mesh_property_block_slot0: None,
        }
    }

    #[test]
    fn packed_text_mode_defaults_to_msdf_when_empty() {
        let store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let kw = EmbeddedSharedKeywordIds::new(&reg);
        assert_eq!(
            packed_text_mode_f32(&store, lookup(1), &kw),
            0.0,
            "FrooxEngine default GlyphRenderMethod is MSDF"
        );
    }

    #[test]
    fn packed_text_mode_explicit_host_value() {
        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let kw = EmbeddedSharedKeywordIds::new(&reg);
        let pid_tm = reg.intern("_TextMode");
        store.set_material(1, pid_tm, MaterialPropertyValue::Float(1.0));
        assert_eq!(packed_text_mode_f32(&store, lookup(1), &kw), 1.0);
    }

    #[test]
    fn cutout_blend_mode_infers_alpha_clip_from_canonical_blend_mode() {
        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
        let pid = reg.intern("_BlendMode");
        store.set_material(12, pid, MaterialPropertyValue::Float(1.0));

        for field_name in ["_ALPHATEST_ON", "_ALPHATEST", "_ALPHACLIP"] {
            assert_eq!(
                inferred_keyword_float_f32(field_name, &store, lookup(12), &ids),
                Some(1.0),
                "{field_name} should enable for cutout _BlendMode"
            );
        }
        assert_eq!(
            inferred_keyword_float_f32("_ALPHABLEND_ON", &store, lookup(12), &ids),
            Some(0.0)
        );
    }

    #[test]
    fn inferred_pbs_keyword_enables_from_texture_presence() {
        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
        let pid = reg.intern("_SpecularMap");
        store.set_material(4, pid, MaterialPropertyValue::Texture(123));
        assert_eq!(
            inferred_keyword_float_f32("_SPECULARMAP", &store, lookup(4), &ids),
            Some(1.0)
        );
        assert_eq!(
            inferred_keyword_float_f32("_ALBEDOTEX", &store, lookup(4), &ids),
            Some(0.0)
        );
    }

    #[test]
    fn vec4_defaults_match_documented_unity_conventions() {
        // Spot-check a few entries in the generic vec4 default table that DO need a non-zero
        // value because the relevant WGSL shaders rely on them prior to host writes.
        assert_eq!(
            default_vec4_for_field("_EmissionColor"),
            [0.0, 0.0, 0.0, 0.0]
        );
        assert_eq!(
            default_vec4_for_field("_SpecularColor"),
            [1.0, 1.0, 1.0, 0.5]
        );
        assert_eq!(default_vec4_for_field("_Rect"), [0.0, 0.0, 1.0, 1.0]);
        assert_eq!(default_vec4_for_field("_Point"), [0.0, 0.0, 0.0, 0.0]);
        assert_eq!(default_vec4_for_field("_OverlayTint"), [1.0, 1.0, 1.0, 0.5]);
        assert_eq!(
            default_vec4_for_field("_BehindFarColor"),
            [0.0, 0.0, 0.0, 1.0]
        );
        assert_eq!(default_vec4_for_field("_Tint0_"), [1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn only_main_texture_bindings_fallback_to_primary_texture() {
        use crate::backend::embedded::texture_resolve::should_fallback_to_primary_texture;
        assert!(should_fallback_to_primary_texture("_MainTex"));
        assert!(!should_fallback_to_primary_texture("_MainTex1"));
        assert!(!should_fallback_to_primary_texture("_SpecularMap"));
    }

    /// `_ALBEDOTEX` keyword inference must treat a packed [`HostTextureAssetKind::RenderTexture`] like a
    /// bound texture (parity with 2D-only `texture_property_asset_id_by_pid`).
    #[test]
    fn albedo_keyword_infers_from_render_texture_packed_id() {
        use crate::assets::texture::{unpack_host_texture_packed, HostTextureAssetKind};

        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
        let main_tex = reg.intern("_MainTex");
        let type_bits = 3u32;
        let pack_type_shift = 32u32.saturating_sub(type_bits);
        let asset_id = 7i32;
        let packed = asset_id | ((HostTextureAssetKind::RenderTexture as i32) << pack_type_shift);
        assert_eq!(
            unpack_host_texture_packed(packed),
            Some((asset_id, HostTextureAssetKind::RenderTexture))
        );
        store.set_material(6, main_tex, MaterialPropertyValue::Texture(packed));
        assert_eq!(
            inferred_keyword_float_f32("_ALBEDOTEX", &store, lookup(6), &ids),
            Some(1.0)
        );
    }
}
