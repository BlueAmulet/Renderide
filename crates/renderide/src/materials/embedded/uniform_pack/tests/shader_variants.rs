//! Shader variant bitmask uniform packing tests.

use super::super::*;
use super::common::*;

use crate::materials::embedded::layout::StemEmbeddedPropertyIds;
use crate::materials::embedded::texture_pools::EmbeddedTexturePools;
use crate::materials::host_data::{
    MaterialPropertyStore, MaterialPropertyValue, PropertyIdRegistry,
};
use crate::materials::{ReflectedRasterLayout, ReflectedUniformScalarKind};

const RENDERIDE_VARIANT_BITS_FIELD: &str = "_RenderideVariantBits";
const VARIANT_BITS_OFFSET: usize = 32;
const KW_ALPHATEST: u32 = 1u32 << 0;
const KW_COLOR: u32 = 1u32 << 1;
const KW_MASK_TEXTURE_CLIP: u32 = 1u32 << 2;
const KW_MASK_TEXTURE_MUL: u32 = 1u32 << 3;
const KW_MUL_RGB_BY_ALPHA: u32 = 1u32 << 5;
const KW_OFFSET_TEXTURE: u32 = 1u32 << 6;
const KW_POLARUV: u32 = 1u32 << 7;
const KW_RIGHT_EYE_ST: u32 = 1u32 << 8;
const KW_TEXTURE: u32 = 1u32 << 9;
const KW_TEXTURE_NORMALMAP: u32 = 1u32 << 10;
const KW_VERTEX_SRGB_COLOR: u32 = 1u32 << 12;
const KW_VERTEXCOLORS: u32 = 1u32 << 13;

fn empty_tex_ctx<'a>(pools: &'a EmbeddedTexturePools<'a>) -> UniformPackTextureContext<'a> {
    UniformPackTextureContext {
        pools,
        primary_texture_2d: -1,
    }
}

fn empty_pools() -> (
    crate::gpu_pools::TexturePool,
    crate::gpu_pools::Texture3dPool,
    crate::gpu_pools::CubemapPool,
    crate::gpu_pools::RenderTexturePool,
    crate::gpu_pools::VideoTexturePool,
) {
    empty_texture_pools()
}

fn unlit_variant_bits_reflection() -> (
    ReflectedRasterLayout,
    StemEmbeddedPropertyIds,
    PropertyIdRegistry,
) {
    reflected_with_uniform_fields_for_stem(
        "unlit_default",
        &[
            ("_Color", ReflectedUniformScalarKind::Vec4, 16, 0),
            ("_RightEye_ST", ReflectedUniformScalarKind::Vec4, 16, 16),
            (
                RENDERIDE_VARIANT_BITS_FIELD,
                ReflectedUniformScalarKind::U32,
                4,
                VARIANT_BITS_OFFSET as u32,
            ),
        ],
    )
}

#[test]
fn explicit_unlit_variant_bits_pack_reserved_u32_field() {
    let (reflected, ids, _registry) = unlit_variant_bits_reflection();
    let store = MaterialPropertyStore::new();

    let (texture, texture3d, cubemap, render_texture, video_texture) = empty_pools();
    let pools = EmbeddedTexturePools {
        texture: &texture,
        texture3d: &texture3d,
        cubemap: &cubemap,
        render_texture: &render_texture,
        video_texture: &video_texture,
    };
    let variant_bits = KW_COLOR | KW_TEXTURE;

    let bytes = build_embedded_uniform_bytes_with_value_spaces(
        &reflected,
        &ids,
        &MaterialUniformValueSpaces::default(),
        &store,
        lookup(70),
        &empty_tex_ctx(&pools),
        Some(variant_bits),
    )
    .expect("uniform bytes");

    assert_eq!(read_u32_at(&bytes, VARIANT_BITS_OFFSET), variant_bits);
}

#[test]
fn explicit_zero_unlit_variant_overrides_texture_and_tint_fallbacks() {
    let (reflected, ids, registry) = unlit_variant_bits_reflection();
    let mut store = MaterialPropertyStore::new();
    store.set_material(
        71,
        registry.intern("_Tex"),
        MaterialPropertyValue::Texture(123),
    );
    store.set_material(
        71,
        registry.intern("_Color"),
        MaterialPropertyValue::Float4([0.25, 1.0, 1.0, 1.0]),
    );

    let (texture, texture3d, cubemap, render_texture, video_texture) = empty_pools();
    let pools = EmbeddedTexturePools {
        texture: &texture,
        texture3d: &texture3d,
        cubemap: &cubemap,
        render_texture: &render_texture,
        video_texture: &video_texture,
    };

    let bytes = build_embedded_uniform_bytes_with_value_spaces(
        &reflected,
        &ids,
        &MaterialUniformValueSpaces::default(),
        &store,
        lookup(71),
        &empty_tex_ctx(&pools),
        Some(0),
    )
    .expect("uniform bytes");

    assert_eq!(read_u32_at(&bytes, VARIANT_BITS_OFFSET), 0);
}

#[test]
fn missing_unlit_variant_bits_infer_texture_and_tint_bits() {
    let (reflected, ids, registry) = unlit_variant_bits_reflection();
    let mut store = MaterialPropertyStore::new();
    store.set_material(
        72,
        registry.intern("_Tex"),
        MaterialPropertyValue::Texture(123),
    );
    store.set_material(
        72,
        registry.intern("_Color"),
        MaterialPropertyValue::Float4([0.25, 1.0, 1.0, 1.0]),
    );

    let (texture, texture3d, cubemap, render_texture, video_texture) = empty_pools();
    let pools = EmbeddedTexturePools {
        texture: &texture,
        texture3d: &texture3d,
        cubemap: &cubemap,
        render_texture: &render_texture,
        video_texture: &video_texture,
    };

    let bytes = build_embedded_uniform_bytes_with_value_spaces(
        &reflected,
        &ids,
        &MaterialUniformValueSpaces::default(),
        &store,
        lookup(72),
        &empty_tex_ctx(&pools),
        None,
    )
    .expect("uniform bytes");

    let bits = read_u32_at(&bytes, VARIANT_BITS_OFFSET);
    assert_eq!(bits & KW_TEXTURE, KW_TEXTURE);
    assert_eq!(bits & KW_COLOR, KW_COLOR);
    assert_eq!(bits & KW_VERTEXCOLORS, KW_VERTEXCOLORS);
}

#[test]
fn missing_unlit_variant_bits_infer_observable_unlit_controls() {
    let (reflected, ids, registry) = unlit_variant_bits_reflection();
    let mut store = MaterialPropertyStore::new();
    store.set_material(
        73,
        registry.intern("_OffsetTex"),
        MaterialPropertyValue::Texture(124),
    );
    store.set_material(
        73,
        registry.intern("_MaskTex"),
        MaterialPropertyValue::Texture(125),
    );
    store.set_material(
        73,
        registry.intern("_RightEye_ST"),
        MaterialPropertyValue::Float4([1.0, 1.0, 0.5, 0.0]),
    );
    store.set_material(
        73,
        registry.intern("_RenderType"),
        MaterialPropertyValue::Float(2.0),
    );
    store.set_material(
        73,
        registry.intern("_SrcBlend"),
        MaterialPropertyValue::Float(1.0),
    );
    store.set_material(
        73,
        registry.intern("_DstBlend"),
        MaterialPropertyValue::Float(1.0),
    );

    let (texture, texture3d, cubemap, render_texture, video_texture) = empty_pools();
    let pools = EmbeddedTexturePools {
        texture: &texture,
        texture3d: &texture3d,
        cubemap: &cubemap,
        render_texture: &render_texture,
        video_texture: &video_texture,
    };

    let bytes = build_embedded_uniform_bytes_with_value_spaces(
        &reflected,
        &ids,
        &MaterialUniformValueSpaces::default(),
        &store,
        lookup(73),
        &empty_tex_ctx(&pools),
        None,
    )
    .expect("uniform bytes");

    let bits = read_u32_at(&bytes, VARIANT_BITS_OFFSET);
    assert_eq!(bits & KW_OFFSET_TEXTURE, KW_OFFSET_TEXTURE);
    assert_eq!(bits & KW_MASK_TEXTURE_MUL, KW_MASK_TEXTURE_MUL);
    assert_eq!(bits & KW_RIGHT_EYE_ST, KW_RIGHT_EYE_ST);
    assert_eq!(bits & KW_MUL_RGB_BY_ALPHA, KW_MUL_RGB_BY_ALPHA);
}

#[test]
fn missing_unlit_variant_bits_infer_alpha_test_bit() {
    let (reflected, ids, registry) = unlit_variant_bits_reflection();
    let mut store = MaterialPropertyStore::new();
    store.set_material(
        74,
        registry.intern("_RenderQueue"),
        MaterialPropertyValue::Float(2450.0),
    );

    let (texture, texture3d, cubemap, render_texture, video_texture) = empty_pools();
    let pools = EmbeddedTexturePools {
        texture: &texture,
        texture3d: &texture3d,
        cubemap: &cubemap,
        render_texture: &render_texture,
        video_texture: &video_texture,
    };

    let bytes = build_embedded_uniform_bytes_with_value_spaces(
        &reflected,
        &ids,
        &MaterialUniformValueSpaces::default(),
        &store,
        lookup(74),
        &empty_tex_ctx(&pools),
        None,
    )
    .expect("uniform bytes");

    let bits = read_u32_at(&bytes, VARIANT_BITS_OFFSET);
    assert_eq!(bits & KW_ALPHATEST, KW_ALPHATEST);
}

#[test]
fn missing_unlit_variant_bits_do_not_infer_unobservable_keywords() {
    let (reflected, ids, registry) = unlit_variant_bits_reflection();
    let mut store = MaterialPropertyStore::new();
    store.set_material(
        75,
        registry.intern("_Tex"),
        MaterialPropertyValue::Texture(123),
    );
    store.set_material(
        75,
        registry.intern("_PolarPow"),
        MaterialPropertyValue::Float(2.0),
    );

    let (texture, texture3d, cubemap, render_texture, video_texture) = empty_pools();
    let pools = EmbeddedTexturePools {
        texture: &texture,
        texture3d: &texture3d,
        cubemap: &cubemap,
        render_texture: &render_texture,
        video_texture: &video_texture,
    };

    let bytes = build_embedded_uniform_bytes_with_value_spaces(
        &reflected,
        &ids,
        &MaterialUniformValueSpaces::default(),
        &store,
        lookup(75),
        &empty_tex_ctx(&pools),
        None,
    )
    .expect("uniform bytes");

    let bits = read_u32_at(&bytes, VARIANT_BITS_OFFSET);
    assert_eq!(bits & KW_TEXTURE, KW_TEXTURE);
    assert_eq!(bits & KW_POLARUV, 0);
    assert_eq!(bits & KW_TEXTURE_NORMALMAP, 0);
    assert_eq!(bits & KW_MASK_TEXTURE_CLIP, 0);
    assert_eq!(bits & KW_VERTEX_SRGB_COLOR, 0);
}

#[test]
fn unlit_real_color_property_still_packs_as_vec4() {
    let (reflected, ids, registry) = reflected_with_uniform_fields_for_stem(
        "unlit_default",
        &[
            ("_Color", ReflectedUniformScalarKind::Vec4, 16, 0),
            (
                RENDERIDE_VARIANT_BITS_FIELD,
                ReflectedUniformScalarKind::U32,
                4,
                16,
            ),
        ],
    );
    let mut store = MaterialPropertyStore::new();
    let color = [0.25, 0.5, 0.75, 0.8];
    store.set_material(
        76,
        registry.intern("_Color"),
        MaterialPropertyValue::Float4(color),
    );

    let (texture, texture3d, cubemap, render_texture, video_texture) = empty_pools();
    let pools = EmbeddedTexturePools {
        texture: &texture,
        texture3d: &texture3d,
        cubemap: &cubemap,
        render_texture: &render_texture,
        video_texture: &video_texture,
    };

    let bytes = build_embedded_uniform_bytes_with_value_spaces(
        &reflected,
        &ids,
        &MaterialUniformValueSpaces::default(),
        &store,
        lookup(76),
        &empty_tex_ctx(&pools),
        Some(KW_COLOR),
    )
    .expect("uniform bytes");

    assert_eq!(read_f32x4(&bytes, 0), color);
    assert_eq!(read_u32_at(&bytes, 16), KW_COLOR);
}

#[test]
fn unrelated_u32_uniforms_stay_zero() {
    let (reflected, ids, _registry) = reflected_with_uniform_fields_for_stem(
        "unlit_default",
        &[("_OtherFlags", ReflectedUniformScalarKind::U32, 4, 0)],
    );
    let store = MaterialPropertyStore::new();

    let (texture, texture3d, cubemap, render_texture, video_texture) = empty_pools();
    let pools = EmbeddedTexturePools {
        texture: &texture,
        texture3d: &texture3d,
        cubemap: &cubemap,
        render_texture: &render_texture,
        video_texture: &video_texture,
    };

    let bytes = build_embedded_uniform_bytes_with_value_spaces(
        &reflected,
        &ids,
        &MaterialUniformValueSpaces::default(),
        &store,
        lookup(77),
        &empty_tex_ctx(&pools),
        Some(KW_ALPHATEST),
    )
    .expect("uniform bytes");

    assert_eq!(read_u32_at(&bytes, 0), 0);
}
