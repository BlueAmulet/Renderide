//! Name-driven keyword inference and scalar default tables for embedded uniform packing.

use crate::materials::host_data::{MaterialPropertyLookupIds, MaterialPropertyStore};

use super::super::layout::{EmbeddedSharedKeywordIds, StemEmbeddedPropertyIds};
use super::helpers::{
    first_float_by_pids, is_keyword_like_field, keyword_float_enabled_any_pids,
    shader_writer_unescaped_field_name, texture_property_present_pids,
};

/// Renderer-reserved material uniform field carrying the raw shader-specific Froox variant bitmask.
const RENDERIDE_VARIANT_BITS_FIELD: &str = "_RenderideVariantBits";

/// Returns the raw renderer-reserved shader variant bitfield, when the reflected field requests it.
pub(super) fn inferred_shader_variant_bits_u32(
    field_name: &str,
    shader_variant_bits: Option<u32>,
    _store: &MaterialPropertyStore,
    _lookup: MaterialPropertyLookupIds,
    _ids: &StemEmbeddedPropertyIds,
) -> Option<u32> {
    if field_name != RENDERIDE_VARIANT_BITS_FIELD {
        return None;
    }
    shader_variant_bits
}

/// Infers a scalar keyword uniform from host-visible material state.
pub(super) fn inferred_keyword_float_f32(
    field_name: &str,
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    ids: &StemEmbeddedPropertyIds,
) -> Option<f32> {
    let field_name = shader_writer_unescaped_field_name(field_name);
    if let Some(probes) = ids.keyword_field_probe_ids.get(field_name)
        && keyword_float_enabled_any_pids(store, lookup, probes)
    {
        return Some(1.0);
    }

    if let Some(value) = ui_unlit_alpha_clip_inferred(field_name, store, lookup, ids) {
        return Some(value);
    }

    let kw = ids.shared.as_ref();
    if let Some(value) = blend_keyword_inferred(field_name, store, lookup, kw) {
        return Some(value);
    }
    if let Some(value) = xiexe_keyword_inferred(field_name, store, lookup, kw) {
        return Some(value);
    }
    let inferred = match texture_keyword_pids(field_name, kw) {
        Some(pids) => texture_property_present_pids(store, lookup, &pids),
        None if is_keyword_like_field(field_name) => false,
        None => return None,
    };
    Some(if inferred { 1.0 } else { 0.0 })
}

/// Converts Unity-style float keyword values into the renderer's packed scalar convention.
fn keyword_float_value(value: f32) -> f32 {
    if value >= 0.5 { 1.0 } else { 0.0 }
}

/// Reads a direct keyword probe value, including explicit false values.
fn keyword_probe_float(
    field_name: &str,
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    ids: &StemEmbeddedPropertyIds,
) -> Option<f32> {
    let probes = ids.keyword_field_probe_ids.get(field_name)?;
    first_float_by_pids(store, lookup, probes)
}

/// Infers the default UI alpha-clip keyword for the `UI/Unlit` material stem.
fn ui_unlit_alpha_clip_inferred(
    field_name: &str,
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    ids: &StemEmbeddedPropertyIds,
) -> Option<f32> {
    if field_name != "_ALPHACLIP" || !ids.ui_unlit_alpha_clip_default_on {
        return None;
    }
    if let Some(value) = keyword_probe_float(field_name, store, lookup, ids) {
        return Some(keyword_float_value(value));
    }
    Some(1.0)
}

/// Resolves alpha-test/alpha-blend/alpha-premultiply/`_MUL_RGB_BY_ALPHA` keywords from host blend
/// state. Returns `None` for unrelated keyword names.
fn blend_keyword_inferred(
    field_name: &str,
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    kw: &EmbeddedSharedKeywordIds,
) -> Option<f32> {
    let value = match field_name {
        "_ALPHATEST_ON" | "_ALPHATEST" | "_ALPHACLIP" => alpha_test_on_inferred(store, lookup, kw),
        "_ALPHABLEND_ON" => alpha_blend_on_inferred(store, lookup, kw),
        "_ALPHAPREMULTIPLY_ON" => alpha_premultiply_on_inferred(store, lookup, kw),
        "_MUL_RGB_BY_ALPHA" => mul_rgb_by_alpha_inferred(store, lookup, kw),
        _ => return None,
    };
    Some(if value { 1.0 } else { 0.0 })
}

/// Returns the host property ids whose presence drives the texture-presence keyword for
/// `field_name`, or `None` for keywords not driven by texture presence.
fn texture_keyword_pids(field_name: &str, kw: &EmbeddedSharedKeywordIds) -> Option<Vec<i32>> {
    Some(match field_name {
        "_LERPTEX" => vec![kw.lerp_tex],
        "_TEXTURE" => vec![
            kw.tex,
            kw.main_tex,
            kw.far_tex,
            kw.near_tex,
            kw.far_tex0,
            kw.near_tex0,
            kw.far_tex1,
            kw.near_tex1,
        ],
        "GRADIENT" => vec![kw.gradient],
        "_ALBEDOTEX" => vec![kw.main_tex, kw.main_tex1],
        "_EMISSION" | "_EMISSIONTEX" => vec![
            kw.emission_map,
            kw.emission_map1,
            kw.emission_map2,
            kw.emission_map3,
        ],
        "_NORMALMAP" => vec![kw.normal_map, kw.normal_map0, kw.normal_map1, kw.bump_map],
        "_SPECULARMAP" => vec![
            kw.specular_map,
            kw.specular_map1,
            kw.specular_map2,
            kw.specular_map3,
            kw.spec_gloss_map,
        ],
        "_SPECGLOSSMAP" => vec![kw.spec_gloss_map],
        "_METALLICGLOSSMAP" => vec![kw.metallic_gloss_map],
        "_METALLICMAP" => vec![
            kw.metallic_map,
            kw.metallic_map1,
            kw.metallic_gloss_map,
            kw.metallic_gloss01,
            kw.metallic_gloss23,
        ],
        "MATCAP" => vec![kw.matcap],
        "_DETAIL_MULX2" => vec![kw.detail_albedo_map, kw.detail_normal_map, kw.detail_mask],
        "_PARALLAXMAP" => vec![kw.parallax_map],
        "_OCCLUSION" => vec![kw.occlusion, kw.occlusion1, kw.occlusion_map],
        "_HEIGHTMAP" => vec![kw.packed_height_map],
        "_PACKED_NORMALMAP" => vec![kw.packed_normal_map01, kw.packed_normal_map23],
        "_PACKED_EMISSIONTEX" => vec![kw.packed_emission_map],
        _ => return None,
    })
}

/// Discriminant of [`crate::shared::MaterialRenderType::TransparentCutout`] on the wire.
/// Captured under the synthetic `_RenderType` property by
/// [`crate::materials::host_data::parse_materials_update_batch_into_store`].
const RENDER_TYPE_TRANSPARENT_CUTOUT: i32 = 1;
/// Discriminant of [`crate::shared::MaterialRenderType::Transparent`] on the wire.
const RENDER_TYPE_TRANSPARENT: i32 = 2;
/// FrooxEngine `BlendMode.Cutout` discriminant (matches Unity Standard `_Mode = 1`).
const BLEND_MODE_CUTOUT: i32 = 1;
/// FrooxEngine `BlendMode.Alpha` discriminant -- Unity Standard `_Mode = 2` (alpha-blend / fade).
const BLEND_MODE_ALPHA: i32 = 2;
/// FrooxEngine `BlendMode.Transparent` discriminant -- Unity Standard `_Mode = 3` (premultiplied).
const BLEND_MODE_TRANSPARENT_PREMULTIPLY: i32 = 3;
/// `UnityEngine.Rendering.BlendMode.One`.
const UNITY_BLEND_FACTOR_ONE: i32 = 1;
/// `UnityEngine.Rendering.BlendMode.SrcAlpha`.
const UNITY_BLEND_FACTOR_SRC_ALPHA: i32 = 5;
/// `UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha`.
const UNITY_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA: i32 = 10;
/// Inclusive lower bound of Unity's AlphaTest queue range (FrooxEngine writes 2450 for
/// `AlphaHandling.AlphaClip` / `BlendMode.Cutout`).
const RENDER_QUEUE_ALPHA_TEST_MIN: i32 = 2450;
/// Inclusive lower bound of Unity's Transparent queue range (FrooxEngine writes 3000 for
/// `AlphaHandling.AlphaBlend` / `BlendMode.Alpha` / `BlendMode.Transparent`). Also the
/// exclusive upper bound of the AlphaTest range.
const RENDER_QUEUE_TRANSPARENT_MIN: i32 = 3000;

/// Reads a float-valued material property as the integer enum/discriminant it represents.
fn read_int_property(
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    kw_pid: i32,
) -> Option<i32> {
    first_float_by_pids(store, lookup, &[kw_pid]).map(|v| v.round() as i32)
}

/// Returns whether either render-type or older mode properties match the requested values.
fn render_type_or_legacy_mode_is(
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    kw: &EmbeddedSharedKeywordIds,
    render_type_value: i32,
    legacy_mode_value: i32,
) -> bool {
    if read_int_property(store, lookup, kw.render_type) == Some(render_type_value) {
        return true;
    }
    let legacy_mode = read_int_property(store, lookup, kw.mode);
    let legacy_blend = read_int_property(store, lookup, kw.blend_mode);
    legacy_mode == Some(legacy_mode_value) || legacy_blend == Some(legacy_mode_value)
}

/// Returns whether one complete blend-factor property pair matches `src_factor` and `dst_factor`.
fn blend_factor_pair_is(
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    src_pid: i32,
    dst_pid: i32,
    src_factor: i32,
    dst_factor: i32,
) -> bool {
    let src = read_int_property(store, lookup, src_pid);
    let dst = read_int_property(store, lookup, dst_pid);
    src == Some(src_factor) && dst == Some(dst_factor)
}

/// Returns whether the host blend factors match `src_factor` and `dst_factor`.
fn blend_factors_are(
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    kw: &EmbeddedSharedKeywordIds,
    src_factor: i32,
    dst_factor: i32,
) -> bool {
    blend_factor_pair_is(
        store,
        lookup,
        kw.src_blend,
        kw.dst_blend,
        src_factor,
        dst_factor,
    ) || blend_factor_pair_is(
        store,
        lookup,
        kw.src_blend_base,
        kw.dst_blend_base,
        src_factor,
        dst_factor,
    )
}

/// Returns whether blend factors describe Unity/FrooxEngine straight alpha blending.
fn straight_alpha_blend_factors(
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    kw: &EmbeddedSharedKeywordIds,
) -> bool {
    blend_factors_are(
        store,
        lookup,
        kw,
        UNITY_BLEND_FACTOR_SRC_ALPHA,
        UNITY_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
    )
}

/// Returns whether blend factors describe Unity/FrooxEngine premultiplied alpha blending.
fn premultiplied_blend_factors(
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    kw: &EmbeddedSharedKeywordIds,
) -> bool {
    blend_factors_are(
        store,
        lookup,
        kw,
        UNITY_BLEND_FACTOR_ONE,
        UNITY_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
    )
}

/// Returns whether blend factors describe Unity/FrooxEngine additive blending.
fn additive_blend_factors(
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    kw: &EmbeddedSharedKeywordIds,
) -> bool {
    blend_factors_are(
        store,
        lookup,
        kw,
        UNITY_BLEND_FACTOR_ONE,
        UNITY_BLEND_FACTOR_ONE,
    )
}

/// Classification of an inferred render queue value.
///
/// Mirrors Unity's standard queue ranges and the values FrooxEngine writes from both
/// `MaterialProvider.SetBlendMode` (Opaque=2000/2550, Cutout=2450/2750, Transparent=3000)
/// and the PBS `AlphaHandling` family (Opaque=2000, AlphaClip=2450, AlphaBlend=3000).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum InferredQueueRange {
    /// Below the AlphaTest threshold (Background / Geometry).
    Opaque,
    /// `[2450, 3000)` -- Unity AlphaTest range.
    AlphaTest,
    /// `>= 3000` -- Unity Transparent range and beyond.
    Transparent,
}

/// Classifies the host render queue into the alpha range implied by Unity's queue constants.
fn render_queue_range(
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    kw: &EmbeddedSharedKeywordIds,
) -> Option<InferredQueueRange> {
    let queue = read_int_property(store, lookup, kw.render_queue)?;
    if queue >= RENDER_QUEUE_TRANSPARENT_MIN {
        Some(InferredQueueRange::Transparent)
    } else if queue >= RENDER_QUEUE_ALPHA_TEST_MIN {
        Some(InferredQueueRange::AlphaTest)
    } else {
        Some(InferredQueueRange::Opaque)
    }
}

/// Returns whether host-visible state implies an alpha-test/cutout shader keyword.
fn alpha_test_on_inferred(
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    kw: &EmbeddedSharedKeywordIds,
) -> bool {
    if render_queue_range(store, lookup, kw) == Some(InferredQueueRange::AlphaTest) {
        return true;
    }
    render_type_or_legacy_mode_is(
        store,
        lookup,
        kw,
        RENDER_TYPE_TRANSPARENT_CUTOUT,
        BLEND_MODE_CUTOUT,
    )
}

/// Returns whether host-visible state implies straight alpha blending.
fn alpha_blend_on_inferred(
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    kw: &EmbeddedSharedKeywordIds,
) -> bool {
    let render_type = read_int_property(store, lookup, kw.render_type);
    if render_type == Some(RENDER_TYPE_TRANSPARENT) {
        return !premultiplied_blend_factors(store, lookup, kw);
    }
    if render_queue_range(store, lookup, kw) == Some(InferredQueueRange::Transparent) {
        return !premultiplied_blend_factors(store, lookup, kw);
    }
    let legacy_mode = read_int_property(store, lookup, kw.mode);
    let legacy_blend = read_int_property(store, lookup, kw.blend_mode);
    legacy_mode == Some(BLEND_MODE_ALPHA) || legacy_blend == Some(BLEND_MODE_ALPHA)
}

/// Returns whether host-visible state implies premultiplied alpha blending.
fn alpha_premultiply_on_inferred(
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    kw: &EmbeddedSharedKeywordIds,
) -> bool {
    let render_type = read_int_property(store, lookup, kw.render_type);
    if render_type == Some(RENDER_TYPE_TRANSPARENT)
        && premultiplied_blend_factors(store, lookup, kw)
    {
        return true;
    }
    if render_queue_range(store, lookup, kw) == Some(InferredQueueRange::Transparent)
        && premultiplied_blend_factors(store, lookup, kw)
    {
        return true;
    }
    let legacy_mode = read_int_property(store, lookup, kw.mode);
    let legacy_blend = read_int_property(store, lookup, kw.blend_mode);
    legacy_mode == Some(BLEND_MODE_TRANSPARENT_PREMULTIPLY)
        || legacy_blend == Some(BLEND_MODE_TRANSPARENT_PREMULTIPLY)
}

/// Infers Xiexe Toon keyword fields from the properties its material provider writes.
fn xiexe_keyword_inferred(
    field_name: &str,
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    kw: &EmbeddedSharedKeywordIds,
) -> Option<f32> {
    let enabled = match field_name {
        "Cutout" => alpha_test_on_inferred(store, lookup, kw),
        "AlphaBlend" => xiexe_alpha_blend_on_inferred(store, lookup, kw),
        "Transparent" => xiexe_transparent_on_inferred(store, lookup, kw),
        _ => return None,
    };
    Some(if enabled { 1.0 } else { 0.0 })
}

/// Returns whether host-visible state enables Xiexe's `AlphaBlend` shader variant.
fn xiexe_alpha_blend_on_inferred(
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    kw: &EmbeddedSharedKeywordIds,
) -> bool {
    let legacy_mode = read_int_property(store, lookup, kw.mode);
    let legacy_blend = read_int_property(store, lookup, kw.blend_mode);
    legacy_mode == Some(BLEND_MODE_ALPHA)
        || legacy_blend == Some(BLEND_MODE_ALPHA)
        || straight_alpha_blend_factors(store, lookup, kw)
}

/// Returns whether host-visible state enables Xiexe's `Transparent` shader variant.
fn xiexe_transparent_on_inferred(
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    kw: &EmbeddedSharedKeywordIds,
) -> bool {
    let legacy_mode = read_int_property(store, lookup, kw.mode);
    let legacy_blend = read_int_property(store, lookup, kw.blend_mode);
    legacy_mode == Some(BLEND_MODE_TRANSPARENT_PREMULTIPLY)
        || legacy_blend == Some(BLEND_MODE_TRANSPARENT_PREMULTIPLY)
        || premultiplied_blend_factors(store, lookup, kw)
}

/// Returns whether host-visible state implies additive RGB-by-alpha multiplication.
fn mul_rgb_by_alpha_inferred(
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    kw: &EmbeddedSharedKeywordIds,
) -> bool {
    let render_type = read_int_property(store, lookup, kw.render_type);
    if render_type == Some(RENDER_TYPE_TRANSPARENT) && additive_blend_factors(store, lookup, kw) {
        return true;
    }
    if render_queue_range(store, lookup, kw) == Some(InferredQueueRange::Transparent)
        && additive_blend_factors(store, lookup, kw)
    {
        return true;
    }
    false
}

// Every uniform field reaching `build_embedded_uniform_bytes` is one of:
//   1. A host-declared property -- `MaterialPropertyStore` always has a value by the time the
//      renderer reads (first material batch pushes every `Sync<X>` via `MaterialUpdateWriter` per
//      `MaterialProviderBase.cs:48-51`).
//   2. A multi-compile keyword field (`_NORMALMAP`, `_ALPHATEST_ON`, etc.) -- inferred by
//      [`inferred_keyword_float_f32`] from texture presence / blend factor reconstruction.
//   3. `_TextMode` font-atlas profile inference, `_RectClip` / `_OVERLAY` explicit-zero defaults,
//      and `_Cutoff` -- handled by special-case probes in the caller.
//
// Previously-held Unity-Properties{} fallback values are irrelevant: FrooxEngine supplies its own
// initial values (from each `MaterialProvider.OnAwake()`), not Unity's. See the audit for detail.
