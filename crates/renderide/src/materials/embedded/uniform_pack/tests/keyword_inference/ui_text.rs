//! Keyword inference tests for this behavior family.

use super::*;

#[test]
fn ui_unlit_stems_default_alpha_clip_on() {
    for stem in ["ui_unlit_default", "ui_unlit_multiview"] {
        let (_reflected, ids, _registry) = reflected_with_f32_fields_for_stem(
            stem,
            &[
                ("_Tint", 0),
                ("_MainTex_ST", 4),
                ("_MaskTex_ST", 8),
                ("_ALPHACLIP", 12),
                ("_TEXTURE_LERPCOLOR", 16),
            ],
        );
        let store = MaterialPropertyStore::new();

        assert_eq!(
            inferred_keyword_float_f32("_ALPHACLIP", &store, lookup(30), &ids),
            Some(1.0),
            "{stem} should inherit UI_UnlitMaterial's default AlphaClip=true"
        );
    }
}

#[test]
fn ui_unlit_alpha_clip_explicit_probe_overrides_default() {
    let (_reflected, ids, registry) = reflected_with_f32_fields_for_stem(
        "ui_unlit_default",
        &[
            ("_Tint", 0),
            ("_MainTex_ST", 4),
            ("_MaskTex_ST", 8),
            ("_ALPHACLIP", 12),
            ("_TEXTURE_LERPCOLOR", 16),
        ],
    );
    let mut store = MaterialPropertyStore::new();
    store.set_material(
        31,
        registry.intern("ALPHACLIP"),
        MaterialPropertyValue::Float(0.0),
    );

    assert_eq!(
        inferred_keyword_float_f32("_ALPHACLIP", &store, lookup(31), &ids),
        Some(0.0)
    );
}

#[test]
fn non_ui_alpha_clip_uniform_stays_off_without_cutout_signal() {
    let (_reflected, ids, _registry) = reflected_with_f32_fields(&[("_ALPHACLIP", 0)]);
    let store = MaterialPropertyStore::new();

    assert_eq!(
        inferred_keyword_float_f32("_ALPHACLIP", &store, lookup(32), &ids),
        Some(0.0)
    );
}

/// `MaterialRenderType::Transparent` (2) with FrooxEngine `BlendMode.Alpha` factors
/// (`_SrcBlend = SrcAlpha (5)`, `_DstBlend = OneMinusSrcAlpha (10)`) maps to
/// `_ALPHABLEND_ON`, not `_ALPHAPREMULTIPLY_ON`.

#[test]
fn explicit_ui_text_control_fields_pack_canonical_values() {
    let (reflected, ids, registry) =
        reflected_with_f32_fields(&[("_TextMode", 0), ("_RectClip", 4), ("_OVERLAY", 8)]);
    let mut store = MaterialPropertyStore::new();
    store.set_material(
        25,
        registry.intern("_TextMode"),
        MaterialPropertyValue::Float(2.0),
    );
    store.set_material(
        25,
        registry.intern("_RectClip"),
        MaterialPropertyValue::Float(1.0),
    );
    store.set_material(
        25,
        registry.intern("_OVERLAY"),
        MaterialPropertyValue::Float(1.0),
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

    let bytes = build_embedded_uniform_bytes(&reflected, &ids, &store, lookup(25), &tex_ctx)
        .expect("uniform bytes");

    assert_eq!(read_f32_at(&bytes, 0), 2.0);
    assert_eq!(read_f32_at(&bytes, 4), 1.0);
    assert_eq!(read_f32_at(&bytes, 8), 1.0);
}

#[test]
fn ui_content_stencil_state_infers_rect_clip() {
    let (reflected, ids, registry) = reflected_with_f32_fields(&[("_RectClip", 0)]);
    let mut store = MaterialPropertyStore::new();
    set_float_property(&mut store, &registry, 27, "_Stencil", 1.0);
    set_float_property(&mut store, &registry, 27, "_StencilComp", 3.0);
    set_float_property(&mut store, &registry, 27, "_StencilReadMask", 1.0);
    set_float_property(&mut store, &registry, 27, "_StencilWriteMask", 0.0);

    assert_eq!(pack_rect_clip_value(&reflected, &ids, &store, 27), 1.0);
}

#[test]
fn explicit_rect_clip_false_overrides_content_stencil_inference() {
    let (reflected, ids, registry) = reflected_with_f32_fields(&[("_RectClip", 0)]);
    let mut store = MaterialPropertyStore::new();
    set_float_property(&mut store, &registry, 28, "_RectClip", 0.0);
    set_float_property(&mut store, &registry, 28, "_Stencil", 1.0);
    set_float_property(&mut store, &registry, 28, "_StencilComp", 3.0);
    set_float_property(&mut store, &registry, 28, "_StencilReadMask", 1.0);
    set_float_property(&mut store, &registry, 28, "_StencilWriteMask", 0.0);

    assert_eq!(pack_rect_clip_value(&reflected, &ids, &store, 28), 0.0);
}

#[test]
fn mask_write_and_clear_states_do_not_infer_rect_clip() {
    let (reflected, ids, registry) = reflected_with_f32_fields(&[("_RectClip", 0)]);
    let cases = [
        (29, 3.0, 2.0, 3.0, 1.0, 3.0, Some(15.0)),
        (30, 3.0, 1.0, 3.0, 3.0, 2.0, Some(0.0)),
    ];

    for (material_id, comp, op, stencil, read_mask, write_mask, color_mask) in cases {
        let mut store = MaterialPropertyStore::new();
        set_float_property(&mut store, &registry, material_id, "_Stencil", stencil);
        set_float_property(&mut store, &registry, material_id, "_StencilComp", comp);
        set_float_property(&mut store, &registry, material_id, "_StencilOp", op);
        set_float_property(
            &mut store,
            &registry,
            material_id,
            "_StencilReadMask",
            read_mask,
        );
        set_float_property(
            &mut store,
            &registry,
            material_id,
            "_StencilWriteMask",
            write_mask,
        );
        if let Some(color_mask) = color_mask {
            set_float_property(&mut store, &registry, material_id, "_ColorMask", color_mask);
        }

        assert_eq!(
            pack_rect_clip_value(&reflected, &ids, &store, material_id),
            0.0
        );
    }
}

#[test]
fn font_atlas_profile_metadata_infers_text_mode() {
    let binding = ResolvedTextureBinding::Texture2D { asset_id: 42 };

    assert_eq!(
        binding_text_mode_from_metadata(binding, Some(ColorProfile::Linear)),
        Some(0.0)
    );
    assert_eq!(
        binding_text_mode_from_metadata(binding, Some(ColorProfile::SRGB)),
        Some(1.0)
    );
    assert_eq!(
        binding_text_mode_from_metadata(binding, Some(ColorProfile::SRGBAlpha)),
        Some(1.0)
    );
    assert_eq!(binding_text_mode_from_metadata(binding, None), None);
    assert_eq!(
        binding_text_mode_from_metadata(
            ResolvedTextureBinding::RenderTexture { asset_id: 42 },
            Some(ColorProfile::SRGB)
        ),
        None
    );
}

#[test]
fn explicit_ui_text_control_fields_ignore_keyword_aliases() {
    let (reflected, ids, registry) =
        reflected_with_f32_fields(&[("_TextMode", 0), ("_RectClip", 4), ("_OVERLAY", 8)]);
    let mut store = MaterialPropertyStore::new();
    for property_name in [
        "TextMode", "textmode", "RectClip", "rectclip", "OVERLAY", "overlay",
    ] {
        store.set_material(
            26,
            registry.intern(property_name),
            MaterialPropertyValue::Float(1.0),
        );
    }
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

    let bytes = build_embedded_uniform_bytes(&reflected, &ids, &store, lookup(26), &tex_ctx)
        .expect("uniform bytes");

    assert_eq!(read_f32_at(&bytes, 0), 0.0);
    assert_eq!(read_f32_at(&bytes, 4), 0.0);
    assert_eq!(read_f32_at(&bytes, 8), 0.0);
}
