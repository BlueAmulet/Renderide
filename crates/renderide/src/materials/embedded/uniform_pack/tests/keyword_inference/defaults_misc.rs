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
