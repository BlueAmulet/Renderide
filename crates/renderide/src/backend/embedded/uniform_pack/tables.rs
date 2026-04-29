//! Name-driven keyword inference and scalar default tables for embedded uniform packing.

use crate::assets::material::{
    MaterialPropertyLookupIds, MaterialPropertyStore, MaterialPropertyValue,
};

use super::super::layout::{EmbeddedSharedKeywordIds, StemEmbeddedPropertyIds};
use super::helpers::{
    first_float_by_pids, is_keyword_like_field, keyword_float_enabled_any_pids,
    shader_writer_unescaped_field_name, texture_property_present_pids,
};

/// Tolerance (in radians) for treating a `Projection360` `_FOV.xy` value as the default
/// full-sphere `(TAU, π)`. Tighter than any FOV the host realistically writes (the host
/// converts whole-degree values through `* π / 180`, so the residual is below `1e-6`).
const PROJECTION360_FULL_SPHERE_EPSILON: f32 = 1e-3;

pub(super) fn inferred_keyword_float_f32(
    field_name: &str,
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    ids: &StemEmbeddedPropertyIds,
) -> Option<f32> {
    let field_name = shader_writer_unescaped_field_name(field_name);
    if let Some(probes) = ids.keyword_field_probe_ids.get(field_name) {
        if keyword_float_enabled_any_pids(store, lookup, probes) {
            return Some(1.0);
        }
    }

    let kw = ids.shared.as_ref();
    match field_name {
        "_ALPHATEST_ON" | "_ALPHATEST" | "_ALPHACLIP" => {
            return Some(if alpha_test_on_inferred(store, lookup, kw) {
                1.0
            } else {
                0.0
            });
        }
        "_ALPHABLEND_ON" => {
            return Some(if alpha_blend_on_inferred(store, lookup, kw) {
                1.0
            } else {
                0.0
            });
        }
        "_ALPHAPREMULTIPLY_ON" => {
            return Some(if alpha_premultiply_on_inferred(store, lookup, kw) {
                1.0
            } else {
                0.0
            });
        }
        "OUTSIDE_CLIP" | "OUTSIDE_COLOR" | "OUTSIDE_CLAMP" => {
            if let Some(value) = projection360_outside_mode_inferred(field_name, store, lookup, ids)
            {
                return Some(value);
            }
        }
        _ => {}
    }

    let inferred = match field_name {
        "_LERPTEX" => texture_property_present_pids(store, lookup, &[kw.lerp_tex]),
        "_ALBEDOTEX" => texture_property_present_pids(store, lookup, &[kw.main_tex, kw.main_tex1]),
        "_EMISSION" | "_EMISSIONTEX" => {
            texture_property_present_pids(store, lookup, &[kw.emission_map, kw.emission_map1])
        }
        "_NORMALMAP" => texture_property_present_pids(
            store,
            lookup,
            &[kw.normal_map, kw.normal_map1, kw.bump_map],
        ),
        "_SPECULARMAP" => texture_property_present_pids(
            store,
            lookup,
            &[kw.specular_map, kw.specular_map1, kw.spec_gloss_map],
        ),
        "_METALLICGLOSSMAP" => {
            texture_property_present_pids(store, lookup, &[kw.metallic_gloss_map])
        }
        "_METALLICMAP" => texture_property_present_pids(
            store,
            lookup,
            &[kw.metallic_map, kw.metallic_map1, kw.metallic_gloss_map],
        ),
        "_DETAIL_MULX2" => texture_property_present_pids(
            store,
            lookup,
            &[kw.detail_albedo_map, kw.detail_normal_map, kw.detail_mask],
        ),
        "_PARALLAXMAP" => texture_property_present_pids(store, lookup, &[kw.parallax_map]),
        "_OCCLUSION" => texture_property_present_pids(
            store,
            lookup,
            &[kw.occlusion, kw.occlusion1, kw.occlusion_map],
        ),
        _ if is_keyword_like_field(field_name) => false,
        _ => return None,
    };
    Some(if inferred { 1.0 } else { 0.0 })
}

/// Discriminant of [`crate::shared::MaterialRenderType::TransparentCutout`] on the wire.
/// Captured under the synthetic `_RenderType` property by
/// [`crate::assets::material::parse_materials_update_batch_into_store`].
const RENDER_TYPE_TRANSPARENT_CUTOUT: i32 = 1;
/// Discriminant of [`crate::shared::MaterialRenderType::Transparent`] on the wire.
const RENDER_TYPE_TRANSPARENT: i32 = 2;
/// FrooxEngine `BlendMode.Cutout` discriminant (matches Unity Standard `_Mode = 1`).
const BLEND_MODE_CUTOUT: i32 = 1;
/// FrooxEngine `BlendMode.Alpha` discriminant — Unity Standard `_Mode = 2` (alpha-blend / fade).
const BLEND_MODE_ALPHA: i32 = 2;
/// FrooxEngine `BlendMode.Transparent` discriminant — Unity Standard `_Mode = 3` (premultiplied).
const BLEND_MODE_TRANSPARENT_PREMULTIPLY: i32 = 3;
/// `UnityEngine.Rendering.BlendMode.One`.
const UNITY_BLEND_FACTOR_ONE: i32 = 1;
/// `UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha`.
const UNITY_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA: i32 = 10;
/// Inclusive lower bound of Unity's AlphaTest queue range (FrooxEngine writes 2450 for
/// `AlphaHandling.AlphaClip` / `BlendMode.Cutout`).
const RENDER_QUEUE_ALPHA_TEST_MIN: i32 = 2450;
/// Inclusive lower bound of Unity's Transparent queue range (FrooxEngine writes 3000 for
/// `AlphaHandling.AlphaBlend` / `BlendMode.Alpha` / `BlendMode.Transparent`). Also the
/// exclusive upper bound of the AlphaTest range.
const RENDER_QUEUE_TRANSPARENT_MIN: i32 = 3000;

fn read_int_property(
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    kw_pid: i32,
) -> Option<i32> {
    first_float_by_pids(store, lookup, &[kw_pid]).map(|v| v.round() as i32)
}

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

fn premultiplied_blend_factors(
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    kw: &EmbeddedSharedKeywordIds,
) -> bool {
    let src = read_int_property(store, lookup, kw.src_blend);
    let dst = read_int_property(store, lookup, kw.dst_blend);
    src == Some(UNITY_BLEND_FACTOR_ONE) && dst == Some(UNITY_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA)
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
    /// `[2450, 3000)` — Unity AlphaTest range.
    AlphaTest,
    /// `>= 3000` — Unity Transparent range and beyond.
    Transparent,
}

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

/// Inferred value for `Projection360`'s `OUTSIDE_CLIP` / `OUTSIDE_COLOR` / `OUTSIDE_CLAMP`
/// keyword fields, gated on the stem actually owning a `_FOV` uniform field.
///
/// FrooxEngine sets exactly one of these via `keywords.SetKeyword(...)` based on
/// `Projection360Material.OutsideMode`, but `ShaderKeywords.Variant` is never serialized over
/// IPC (`MaterialUpdateWriter` exposes no `SetKeyword`; `MaterialPropertyUpdateType` has no
/// keyword opcode). Without a host signal:
///
/// - Full-sphere FOV (`(TAU, π)` within tolerance): every OUTSIDE field is `0`. The
///   fragment shader's existing fallthrough then behaves like Unity's default
///   `OUTSIDE_CLIP`. The choice is moot in this case — every direction is in-FOV, so the
///   outside branch is never hit.
/// - Partial FOV: `OUTSIDE_CLAMP = 1`, the others `0`. Without the keyword channel this is
///   the only choice that lets a continuous range of partial FOVs render anything (the
///   default-clip fallthrough would discard every pixel outside the FOV cone — exactly the
///   `Projection360`-on-narrow-FOV video-player failure).
///
/// Returns `None` when the stem has no `_FOV` uniform field (i.e., not the `Projection360`
/// material family); the caller then falls through to the generic keyword-field default of
/// `0`. Returns `None` when `_FOV` exists but no value has been written to the property
/// store yet — the generic fallthrough handles that case identically to the current
/// behavior.
fn projection360_outside_mode_inferred(
    field_name: &str,
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    ids: &StemEmbeddedPropertyIds,
) -> Option<f32> {
    let fov_pid = *ids.uniform_field_ids.get("_FOV")?;
    let fov_xy = read_float4_xy(store, lookup, fov_pid)?;

    let eps = PROJECTION360_FULL_SPHERE_EPSILON;
    let full_sphere = (fov_xy[0] - std::f32::consts::TAU).abs() <= eps
        && (fov_xy[1] - std::f32::consts::PI).abs() <= eps;

    let value = if !full_sphere && field_name == "OUTSIDE_CLAMP" {
        1.0
    } else {
        0.0
    };
    Some(value)
}

/// Reads the `.xy` of a `Float4` property, ignoring scalar `Float` writes.
///
/// `_FOV` is always packed as a `float4` on the host (`Projection360Material.cs:445` writes
/// `SetFloat4(_FOV, new float4(FieldOfView, AngleOffset) * (π/180))`); a scalar write would
/// be a host-side bug, so we don't paper over it by accepting a bare `Float`.
fn read_float4_xy(
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    property_id: i32,
) -> Option<[f32; 2]> {
    match store.get_merged(lookup, property_id) {
        Some(MaterialPropertyValue::Float4(v)) => Some([v[0], v[1]]),
        _ => None,
    }
}

// Every uniform field reaching `build_embedded_uniform_bytes` is one of:
//   1. A host-declared property — `MaterialPropertyStore` always has a value by the time the
//      renderer reads (first material batch pushes every `Sync<X>` via `MaterialUpdateWriter` per
//      `MaterialProviderBase.cs:48-51`).
//   2. A multi-compile keyword field (`_NORMALMAP`, `_ALPHATEST_ON`, etc.) — inferred by
//      [`inferred_keyword_float_f32`] from texture presence / blend factor reconstruction.
//   3. `_TextMode` / `_RectClip` / `_Cutoff` — handled by special-case probes in the caller.
//
// Previously-held Unity-Properties{} fallback values are irrelevant: FrooxEngine supplies its own
// initial values (from each `MaterialProvider.OnAwake()`), not Unity's. See the audit for detail.
