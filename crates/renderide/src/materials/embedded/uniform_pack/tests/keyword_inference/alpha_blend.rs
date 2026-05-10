//! Keyword inference tests for this behavior family.

use super::*;

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

/// `MaterialRenderType::TransparentCutout` (1) on the wire enables the alpha-test keyword
/// family even when the host never sends `_Mode` / `_BlendMode` (the FrooxEngine path).

#[test]
fn transparent_cutout_render_type_infers_alpha_test_family() {
    let mut store = MaterialPropertyStore::new();
    let reg = PropertyIdRegistry::new();
    let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
    let render_type_pid = reg.intern("_RenderType");
    store.set_material(7, render_type_pid, MaterialPropertyValue::Float(1.0));

    for field_name in ["_ALPHATEST_ON", "_ALPHATEST", "_ALPHACLIP"] {
        assert_eq!(
            inferred_keyword_float_f32(field_name, &store, lookup(7), &ids),
            Some(1.0),
            "{field_name} should enable for TransparentCutout render type"
        );
    }
    assert_eq!(
        inferred_keyword_float_f32("_ALPHABLEND_ON", &store, lookup(7), &ids),
        Some(0.0)
    );
    assert_eq!(
        inferred_keyword_float_f32("_ALPHAPREMULTIPLY_ON", &store, lookup(7), &ids),
        Some(0.0)
    );
}

/// `MaterialRenderType::Opaque` (0) -- neither alpha-test nor alpha-blend keyword fires.
/// This is the case that previously bit Unlit: default `_Cutoff = 0.98` lit up the
/// `_Cutoff in (0, 1)` heuristic even though the host had selected Opaque.

#[test]
fn opaque_render_type_disables_all_alpha_keywords() {
    let mut store = MaterialPropertyStore::new();
    let reg = PropertyIdRegistry::new();
    let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
    let render_type_pid = reg.intern("_RenderType");
    store.set_material(8, render_type_pid, MaterialPropertyValue::Float(0.0));

    for field_name in [
        "_ALPHATEST_ON",
        "_ALPHATEST",
        "_ALPHACLIP",
        "_ALPHABLEND_ON",
        "_ALPHAPREMULTIPLY_ON",
    ] {
        assert_eq!(
            inferred_keyword_float_f32(field_name, &store, lookup(8), &ids),
            Some(0.0),
            "{field_name} should be disabled for Opaque render type"
        );
    }
}

#[test]
fn transparent_render_type_with_alpha_factors_infers_alpha_blend() {
    let mut store = MaterialPropertyStore::new();
    let reg = PropertyIdRegistry::new();
    let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
    let render_type_pid = reg.intern("_RenderType");
    let src_blend_pid = reg.intern("_SrcBlend");
    let dst_blend_pid = reg.intern("_DstBlend");
    store.set_material(9, render_type_pid, MaterialPropertyValue::Float(2.0));
    store.set_material(9, src_blend_pid, MaterialPropertyValue::Float(5.0));
    store.set_material(9, dst_blend_pid, MaterialPropertyValue::Float(10.0));

    assert_eq!(
        inferred_keyword_float_f32("_ALPHABLEND_ON", &store, lookup(9), &ids),
        Some(1.0)
    );
    assert_eq!(
        inferred_keyword_float_f32("_ALPHAPREMULTIPLY_ON", &store, lookup(9), &ids),
        Some(0.0)
    );
    assert_eq!(
        inferred_keyword_float_f32("_ALPHATEST_ON", &store, lookup(9), &ids),
        Some(0.0)
    );
}

/// `MaterialRenderType::Transparent` (2) with FrooxEngine `BlendMode.Transparent`
/// (premultiplied) factors `_SrcBlend = One (1)`, `_DstBlend = OneMinusSrcAlpha (10)`
/// maps to `_ALPHAPREMULTIPLY_ON`, not `_ALPHABLEND_ON`.

#[test]
fn transparent_render_type_with_premultiplied_factors_infers_premultiply() {
    let mut store = MaterialPropertyStore::new();
    let reg = PropertyIdRegistry::new();
    let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
    let render_type_pid = reg.intern("_RenderType");
    let src_blend_pid = reg.intern("_SrcBlend");
    let dst_blend_pid = reg.intern("_DstBlend");
    store.set_material(11, render_type_pid, MaterialPropertyValue::Float(2.0));
    store.set_material(11, src_blend_pid, MaterialPropertyValue::Float(1.0));
    store.set_material(11, dst_blend_pid, MaterialPropertyValue::Float(10.0));

    assert_eq!(
        inferred_keyword_float_f32("_ALPHAPREMULTIPLY_ON", &store, lookup(11), &ids),
        Some(1.0)
    );
    assert_eq!(
        inferred_keyword_float_f32("_ALPHABLEND_ON", &store, lookup(11), &ids),
        Some(0.0)
    );
}

#[test]
fn xiexe_cutout_keyword_infers_from_transparent_cutout_render_type() {
    let mut store = MaterialPropertyStore::new();
    let reg = PropertyIdRegistry::new();
    let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
    set_float_property(&mut store, &reg, 70, "_RenderType", 1.0);

    assert_xiexe_alpha_keywords(&store, 70, &ids, true, false, false);
}

#[test]
fn xiexe_cutout_keyword_infers_from_alpha_test_render_queue() {
    let mut store = MaterialPropertyStore::new();
    let reg = PropertyIdRegistry::new();
    let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
    set_float_property(&mut store, &reg, 71, "_RenderQueue", 2450.0);

    assert_xiexe_alpha_keywords(&store, 71, &ids, true, false, false);
}

#[test]
fn xiexe_alpha_blend_keyword_infers_from_base_blend_factors() {
    let mut store = MaterialPropertyStore::new();
    let reg = PropertyIdRegistry::new();
    let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
    set_float_property(&mut store, &reg, 72, "_SrcBlendBase", 5.0);
    set_float_property(&mut store, &reg, 72, "_DstBlendBase", 10.0);

    assert_xiexe_alpha_keywords(&store, 72, &ids, false, true, false);
}

#[test]
fn xiexe_transparent_keyword_infers_from_base_blend_factors() {
    let mut store = MaterialPropertyStore::new();
    let reg = PropertyIdRegistry::new();
    let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
    set_float_property(&mut store, &reg, 73, "_SrcBlendBase", 1.0);
    set_float_property(&mut store, &reg, 73, "_DstBlendBase", 10.0);

    assert_xiexe_alpha_keywords(&store, 73, &ids, false, false, true);
}

#[test]
fn xiexe_opaque_base_blend_factors_leave_alpha_keywords_off() {
    let mut store = MaterialPropertyStore::new();
    let reg = PropertyIdRegistry::new();
    let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
    set_float_property(&mut store, &reg, 74, "_SrcBlendBase", 1.0);
    set_float_property(&mut store, &reg, 74, "_DstBlendBase", 0.0);

    assert_xiexe_alpha_keywords(&store, 74, &ids, false, false, false);
}

#[test]
fn xiexe_base_and_default_blend_factor_pairs_do_not_mix() {
    let mut store = MaterialPropertyStore::new();
    let reg = PropertyIdRegistry::new();
    let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
    set_float_property(&mut store, &reg, 75, "_SrcBlendBase", 5.0);
    set_float_property(&mut store, &reg, 75, "_DstBlend", 10.0);

    assert_xiexe_alpha_keywords(&store, 75, &ids, false, false, false);
}

/// `BlendMode.Additive` writes Transparent render type with `_SrcBlend = One` and
/// `_DstBlend = One`; Unlit uses that signal to enable `_MUL_RGB_BY_ALPHA`.

#[test]
fn transparent_render_type_with_additive_factors_infers_mul_rgb_by_alpha() {
    let mut store = MaterialPropertyStore::new();
    let reg = PropertyIdRegistry::new();
    let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
    let render_type_pid = reg.intern("_RenderType");
    let src_blend_pid = reg.intern("_SrcBlend");
    let dst_blend_pid = reg.intern("_DstBlend");
    store.set_material(13, render_type_pid, MaterialPropertyValue::Float(2.0));
    store.set_material(13, src_blend_pid, MaterialPropertyValue::Float(1.0));
    store.set_material(13, dst_blend_pid, MaterialPropertyValue::Float(1.0));

    assert_eq!(
        inferred_keyword_float_f32("_MUL_RGB_BY_ALPHA", &store, lookup(13), &ids),
        Some(1.0)
    );
    assert_eq!(
        inferred_keyword_float_f32("_ALPHAPREMULTIPLY_ON", &store, lookup(13), &ids),
        Some(0.0)
    );
}

#[test]
fn additive_rgb_by_alpha_keyword_packs_into_reflected_uniform() {
    let (reflected, ids, reg) = reflected_with_f32_fields(&[("_MUL_RGB_BY_ALPHA", 0)]);
    let mut store = MaterialPropertyStore::new();
    store.set_material(
        76,
        reg.intern("_RenderType"),
        MaterialPropertyValue::Float(2.0),
    );
    store.set_material(
        76,
        reg.intern("_SrcBlend"),
        MaterialPropertyValue::Float(1.0),
    );
    store.set_material(
        76,
        reg.intern("_DstBlend"),
        MaterialPropertyValue::Float(1.0),
    );

    assert_eq!(pack_first_f32_value(&reflected, &ids, &store, 76), 1.0);
}

/// Additive blend factors alone are not enough; the material must also be in a transparent
/// render type or queue range.
#[test]
fn opaque_render_type_with_additive_factors_does_not_infer_mul_rgb_by_alpha() {
    let mut store = MaterialPropertyStore::new();
    let reg = PropertyIdRegistry::new();
    let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
    let render_type_pid = reg.intern("_RenderType");
    let src_blend_pid = reg.intern("_SrcBlend");
    let dst_blend_pid = reg.intern("_DstBlend");
    store.set_material(14, render_type_pid, MaterialPropertyValue::Float(0.0));
    store.set_material(14, src_blend_pid, MaterialPropertyValue::Float(1.0));
    store.set_material(14, dst_blend_pid, MaterialPropertyValue::Float(1.0));

    assert_eq!(
        inferred_keyword_float_f32("_MUL_RGB_BY_ALPHA", &store, lookup(14), &ids),
        Some(0.0)
    );
}

/// Render queue inference covers materials that signal transparency through queue state rather
/// than `MaterialRenderType`.

#[test]
fn render_queue_transparent_with_additive_factors_infers_mul_rgb_by_alpha() {
    let mut store = MaterialPropertyStore::new();
    let reg = PropertyIdRegistry::new();
    let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
    let render_queue_pid = reg.intern("_RenderQueue");
    let src_blend_pid = reg.intern("_SrcBlend");
    let dst_blend_pid = reg.intern("_DstBlend");
    store.set_material(15, render_queue_pid, MaterialPropertyValue::Float(3000.0));
    store.set_material(15, src_blend_pid, MaterialPropertyValue::Float(1.0));
    store.set_material(15, dst_blend_pid, MaterialPropertyValue::Float(1.0));

    assert_eq!(
        inferred_keyword_float_f32("_MUL_RGB_BY_ALPHA", &store, lookup(15), &ids),
        Some(1.0)
    );
}

/// PBS materials (`PBS_DualSidedMaterial.cs` and friends) bypass `SetBlendMode` and
/// only signal `AlphaHandling.AlphaClip` by writing render queue 2450 plus the
/// `_ALPHACLIP` shader keyword (which is not on the wire). Queue 2450 alone must
/// enable the alpha-test family.

#[test]
fn render_queue_alpha_test_range_enables_alpha_test_family() {
    let mut store = MaterialPropertyStore::new();
    let reg = PropertyIdRegistry::new();
    let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
    let render_queue_pid = reg.intern("_RenderQueue");
    store.set_material(20, render_queue_pid, MaterialPropertyValue::Float(2450.0));

    for field_name in ["_ALPHATEST_ON", "_ALPHATEST", "_ALPHACLIP"] {
        assert_eq!(
            inferred_keyword_float_f32(field_name, &store, lookup(20), &ids),
            Some(1.0),
            "{field_name} should enable for queue 2450 (AlphaTest range)"
        );
    }
    assert_eq!(
        inferred_keyword_float_f32("_ALPHABLEND_ON", &store, lookup(20), &ids),
        Some(0.0)
    );
    assert_eq!(
        inferred_keyword_float_f32("_ALPHAPREMULTIPLY_ON", &store, lookup(20), &ids),
        Some(0.0)
    );
}

/// Queue 2000 (Geometry / Opaque) must leave every alpha keyword off -- this is the
/// PBS `AlphaHandling.Opaque` default.

#[test]
fn render_queue_opaque_range_disables_all_alpha_keywords() {
    let mut store = MaterialPropertyStore::new();
    let reg = PropertyIdRegistry::new();
    let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
    let render_queue_pid = reg.intern("_RenderQueue");
    store.set_material(21, render_queue_pid, MaterialPropertyValue::Float(2000.0));

    for field_name in [
        "_ALPHATEST_ON",
        "_ALPHATEST",
        "_ALPHACLIP",
        "_ALPHABLEND_ON",
        "_ALPHAPREMULTIPLY_ON",
    ] {
        assert_eq!(
            inferred_keyword_float_f32(field_name, &store, lookup(21), &ids),
            Some(0.0),
            "{field_name} should be disabled for queue 2000 (Opaque range)"
        );
    }
}

/// Queue 3000 (Transparent) without premultiplied blend factors enables `_ALPHABLEND_ON`.

#[test]
fn render_queue_transparent_range_enables_alpha_blend() {
    let mut store = MaterialPropertyStore::new();
    let reg = PropertyIdRegistry::new();
    let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
    let render_queue_pid = reg.intern("_RenderQueue");
    store.set_material(22, render_queue_pid, MaterialPropertyValue::Float(3000.0));

    assert_eq!(
        inferred_keyword_float_f32("_ALPHABLEND_ON", &store, lookup(22), &ids),
        Some(1.0)
    );
    assert_eq!(
        inferred_keyword_float_f32("_ALPHAPREMULTIPLY_ON", &store, lookup(22), &ids),
        Some(0.0)
    );
    assert_eq!(
        inferred_keyword_float_f32("_ALPHATEST_ON", &store, lookup(22), &ids),
        Some(0.0)
    );
}

/// Queue 3000 (Transparent) with premultiplied factors `_SrcBlend = 1`,
/// `_DstBlend = 10` is `BlendMode.Transparent` -- enables `_ALPHAPREMULTIPLY_ON`.

#[test]
fn render_queue_transparent_with_premultiplied_factors_infers_premultiply() {
    let mut store = MaterialPropertyStore::new();
    let reg = PropertyIdRegistry::new();
    let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
    let render_queue_pid = reg.intern("_RenderQueue");
    let src_blend_pid = reg.intern("_SrcBlend");
    let dst_blend_pid = reg.intern("_DstBlend");
    store.set_material(23, render_queue_pid, MaterialPropertyValue::Float(3000.0));
    store.set_material(23, src_blend_pid, MaterialPropertyValue::Float(1.0));
    store.set_material(23, dst_blend_pid, MaterialPropertyValue::Float(10.0));

    assert_eq!(
        inferred_keyword_float_f32("_ALPHAPREMULTIPLY_ON", &store, lookup(23), &ids),
        Some(1.0)
    );
    assert_eq!(
        inferred_keyword_float_f32("_ALPHABLEND_ON", &store, lookup(23), &ids),
        Some(0.0)
    );
}
