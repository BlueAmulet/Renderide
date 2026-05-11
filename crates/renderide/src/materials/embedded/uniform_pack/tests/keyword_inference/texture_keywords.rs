//! Keyword inference tests for this behavior family.

use super::*;

#[test]
fn texture_presence_infers_generic_texture_keyword_only() {
    let mut store = MaterialPropertyStore::new();
    let reg = PropertyIdRegistry::new();
    let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
    store.set_material(
        50,
        reg.intern("_Tex"),
        MaterialPropertyValue::Texture(packed_render_texture(1)),
    );
    store.set_material(
        50,
        reg.intern("_MaskTex"),
        MaterialPropertyValue::Texture(packed_render_texture(2)),
    );
    store.set_material(
        50,
        reg.intern("_OffsetTex"),
        MaterialPropertyValue::Texture(packed_render_texture(3)),
    );

    assert_eq!(
        inferred_keyword_float_f32("_TEXTURE", &store, lookup(50), &ids),
        Some(1.0)
    );
    assert_eq!(
        inferred_keyword_float_f32("_MASK_TEXTURE_MUL", &store, lookup(50), &ids),
        Some(0.0)
    );
    assert_eq!(
        inferred_keyword_float_f32("_MASK_TEXTURE_CLIP", &store, lookup(50), &ids),
        Some(0.0)
    );
    assert_eq!(
        inferred_keyword_float_f32("_OFFSET_TEXTURE", &store, lookup(50), &ids),
        Some(0.0)
    );
}

/// Render-texture bindings must not rewrite Unity `_ST` values behind the shader's back.
#[test]
fn render_texture_binding_leaves_st_uniform_unchanged() {
    let mut fields = HashMap::new();
    fields.insert(
        "_MainTex_ST".to_string(),
        ReflectedUniformField {
            offset: 0,
            size: 16,
            kind: ReflectedUniformScalarKind::Vec4,
        },
    );
    let mut material_group1_names = HashMap::new();
    material_group1_names.insert(1, "_MainTex".to_string());
    let reflected = ReflectedRasterLayout {
        layout_fingerprint: 0,
        material_entries: Vec::new(),
        per_draw_entries: Vec::new(),
        material_uniform: Some(ReflectedMaterialUniformBlock {
            binding: 0,
            total_size: 16,
            fields,
        }),
        material_group1_names,
        vs_vertex_inputs: Vec::new(),
        vs_max_vertex_location: None,
        uses_scene_depth_snapshot: false,
        uses_scene_color_snapshot: false,
        requires_intersection_pass: false,
    };

    let mut store = MaterialPropertyStore::new();
    let reg = PropertyIdRegistry::new();
    let mut ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
    let main_tex_st = reg.intern("_MainTex_ST");
    let main_tex = reg.intern("_MainTex");
    ids.uniform_field_ids
        .insert("_MainTex_ST".to_string(), main_tex_st);
    ids.texture_binding_property_ids
        .insert(1, Arc::from(vec![main_tex].into_boxed_slice()));
    store.set_material(
        24,
        main_tex,
        MaterialPropertyValue::Texture(packed_render_texture(9)),
    );
    store.set_material(
        24,
        main_tex_st,
        MaterialPropertyValue::Float4([2.0, 3.0, 0.25, 0.75]),
    );

    let (texture, texture3d, cubemap, render_texture, video_texture) = empty_texture_pools();
    let pools = EmbeddedTexturePools {
        texture: &texture,
        texture3d: &texture3d,
        cubemap: &cubemap,
        render_texture: &render_texture,
        video_texture: &video_texture,
    };
    let tex_ctx = UniformPackTextureContext {
        pools: &pools,
        primary_texture_2d: -1,
    };

    let bytes = build_embedded_uniform_bytes(&reflected, &ids, &store, lookup(24), &tex_ctx)
        .expect("uniform bytes");

    assert_eq!(read_f32x4(&bytes, 0), [2.0, 3.0, 0.25, 0.75]);
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
        inferred_keyword_float_f32("_SPECGLOSSMAP", &store, lookup(4), &ids),
        Some(0.0)
    );
    let spec_gloss_pid = reg.intern("_SpecGlossMap");
    store.set_material(4, spec_gloss_pid, MaterialPropertyValue::Texture(456));
    assert_eq!(
        inferred_keyword_float_f32("_SPECGLOSSMAP", &store, lookup(4), &ids),
        Some(1.0)
    );
    assert_eq!(
        inferred_keyword_float_f32("_ALBEDOTEX", &store, lookup(4), &ids),
        Some(0.0)
    );
}

#[test]
fn fresnel_texture_keyword_infers_from_far_or_near_textures() {
    let reg = PropertyIdRegistry::new();
    let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);

    for (i, property_name) in [
        "_FarTex",
        "_NearTex",
        "_FarTex0",
        "_NearTex0",
        "_FarTex1",
        "_NearTex1",
    ]
    .iter()
    .enumerate()
    {
        let material_id = 50 + i as i32;
        let mut store = MaterialPropertyStore::new();
        store.set_material(
            material_id,
            reg.intern(property_name),
            MaterialPropertyValue::Texture(packed_texture2d(100 + i as i32)),
        );
        assert_eq!(
            inferred_keyword_float_f32("_TEXTURE", &store, lookup(material_id), &ids),
            Some(1.0),
            "{property_name} should enable _TEXTURE"
        );
    }
}

#[test]
fn inferred_pbs_splat_keywords_enable_from_texture_presence() {
    let mut store = MaterialPropertyStore::new();
    let reg = PropertyIdRegistry::new();
    let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
    for property_name in [
        "_PackedHeightMap",
        "_PackedNormalMap23",
        "_PackedEmissionMap",
        "_MetallicGloss23",
        "_SpecularMap3",
    ] {
        store.set_material(
            54,
            reg.intern(property_name),
            MaterialPropertyValue::Texture(packed_texture2d(123)),
        );
    }

    for field_name in [
        "_HEIGHTMAP",
        "_PACKED_NORMALMAP",
        "_PACKED_EMISSIONTEX",
        "_METALLICMAP",
        "_SPECULARMAP",
    ] {
        assert_eq!(
            inferred_keyword_float_f32(field_name, &store, lookup(54), &ids),
            Some(1.0),
            "{field_name} should infer from its selected texture family"
        );
    }
}

#[test]
fn gradient_keyword_infers_from_gradient_texture() {
    let mut store = MaterialPropertyStore::new();
    let reg = PropertyIdRegistry::new();
    let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
    store.set_material(
        60,
        reg.intern("_Gradient"),
        MaterialPropertyValue::Texture(packed_texture2d(14)),
    );

    assert_eq!(
        inferred_keyword_float_f32("GRADIENT", &store, lookup(60), &ids),
        Some(1.0)
    );
}

#[test]
fn normalmap_keyword_infers_from_normal_map_zero() {
    let mut store = MaterialPropertyStore::new();
    let reg = PropertyIdRegistry::new();
    let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
    store.set_material(
        61,
        reg.intern("_NormalMap0"),
        MaterialPropertyValue::Texture(packed_texture2d(15)),
    );

    assert_eq!(
        inferred_keyword_float_f32("_NORMALMAP", &store, lookup(61), &ids),
        Some(1.0)
    );
}

#[test]
fn matcap_keyword_infers_from_matcap_texture() {
    let mut store = MaterialPropertyStore::new();
    let reg = PropertyIdRegistry::new();
    let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
    store.set_material(
        63,
        reg.intern("_Matcap"),
        MaterialPropertyValue::Texture(packed_texture2d(16)),
    );

    assert_eq!(
        inferred_keyword_float_f32("MATCAP", &store, lookup(63), &ids),
        Some(1.0)
    );
}

#[test]
fn matcap_keyword_stays_off_without_matcap_texture() {
    let store = MaterialPropertyStore::new();
    let reg = PropertyIdRegistry::new();
    let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);

    assert_eq!(
        inferred_keyword_float_f32("MATCAP", &store, lookup(64), &ids),
        Some(0.0)
    );
}

#[test]
fn clip_keyword_stays_off_from_clip_range_properties_only() {
    let mut store = MaterialPropertyStore::new();
    let reg = PropertyIdRegistry::new();
    let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
    store.set_material(
        62,
        reg.intern("_ClipMin"),
        MaterialPropertyValue::Float(0.0),
    );
    store.set_material(
        62,
        reg.intern("_ClipMax"),
        MaterialPropertyValue::Float(10.0),
    );

    assert_eq!(
        inferred_keyword_float_f32("CLIP", &store, lookup(62), &ids),
        Some(0.0)
    );
}

#[test]
fn only_main_texture_bindings_fallback_to_primary_texture() {
    use crate::materials::embedded::texture_resolve::should_fallback_to_primary_texture;
    assert!(should_fallback_to_primary_texture("_MainTex"));
    assert!(!should_fallback_to_primary_texture("_MainTex1"));
    assert!(!should_fallback_to_primary_texture("_SpecularMap"));
}

/// `_ALBEDOTEX` keyword inference must treat a packed [`HostTextureAssetKind::RenderTexture`] like a
/// bound texture (parity with 2D-only `texture_property_asset_id_by_pid`).

#[test]
fn albedo_keyword_infers_from_render_texture_packed_id() {
    use crate::assets::texture::{HostTextureAssetKind, unpack_host_texture_packed};

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
