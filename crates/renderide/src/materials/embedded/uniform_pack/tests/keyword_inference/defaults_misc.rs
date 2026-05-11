//! Keyword inference tests for this behavior family.

use super::*;

#[test]
fn right_eye_variant_keyword_is_not_inferred_from_right_eye_st_presence() {
    let (_reflected, ids, registry) = reflected_with_f32_fields(&[("_RightEye_ST", 0)]);
    let mut store = MaterialPropertyStore::new();
    store.set_material(
        51,
        registry.intern("_RightEye_ST"),
        MaterialPropertyValue::Float4([0.5, 1.0, 0.5, 0.0]),
    );

    assert_eq!(
        inferred_keyword_float_f32("_RIGHT_EYE_ST", &store, lookup(51), &ids),
        Some(0.0)
    );
}

#[test]
fn lut_lerp_keyword_infers_from_lerp_uniform() {
    let (_reflected, ids, reg) = reflected_with_f32_fields(&[("_Lerp", 0), ("LERP", 4)]);
    let mut store = MaterialPropertyStore::new();
    let lerp_pid = reg.intern("_Lerp");

    assert_eq!(
        inferred_keyword_float_f32("LERP", &store, lookup(24), &ids),
        Some(0.0)
    );

    store.set_material(24, lerp_pid, MaterialPropertyValue::Float(0.0));
    assert_eq!(
        inferred_keyword_float_f32("LERP", &store, lookup(24), &ids),
        Some(0.0)
    );

    store.set_material(24, lerp_pid, MaterialPropertyValue::Float(0.25));
    assert_eq!(
        inferred_keyword_float_f32("LERP", &store, lookup(24), &ids),
        Some(1.0)
    );
}

/// Base LUT materials default to the Unity/FrooxEngine `SRGB` variant unless the reflected
/// keyword field is explicitly supplied as false.

#[test]
fn lut_srgb_keyword_defaults_on() {
    let (_reflected, ids, reg) = reflected_with_f32_fields(&[("SRGB", 0)]);
    let mut store = MaterialPropertyStore::new();

    assert_eq!(
        inferred_keyword_float_f32("SRGB", &store, lookup(25), &ids),
        Some(1.0)
    );

    let srgb_pid = reg.intern("SRGB");
    store.set_material(25, srgb_pid, MaterialPropertyValue::Float(0.0));
    assert_eq!(
        inferred_keyword_float_f32("SRGB", &store, lookup(25), &ids),
        Some(0.0)
    );
}

/// Additive blend factors alone are not enough; the material must also be in a transparent
/// render type or queue range.

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
        default_vec4_for_field("_BackgroundColor"),
        [0.0, 0.0, 0.0, 0.0]
    );
    assert_eq!(default_vec4_for_field("_Range"), [0.001, 0.001, 0.0, 0.0]);
    assert_eq!(
        default_vec4_for_field("_BehindFarColor"),
        [0.0, 0.0, 0.0, 1.0]
    );
    assert_eq!(default_vec4_for_field("_Tint0_"), [1.0, 0.0, 0.0, 1.0]);
}
