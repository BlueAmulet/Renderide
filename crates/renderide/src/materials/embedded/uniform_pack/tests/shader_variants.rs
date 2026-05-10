//! Shader variant bitmask uniform packing tests.

use super::super::*;
use super::common::*;

use crate::materials::embedded::texture_pools::EmbeddedTexturePools;
use crate::materials::host_data::{MaterialPropertyStore, MaterialPropertyValue};

fn empty_tex_ctx<'a>(pools: &'a EmbeddedTexturePools<'a>) -> UniformPackTextureContext<'a> {
    UniformPackTextureContext {
        pools,
        primary_texture_2d: -1,
    }
}

#[test]
fn unlit_variant_bits_override_host_keyword_properties() {
    let (reflected, ids, registry) = reflected_with_f32_fields_for_stem(
        "unlit_default",
        &[("_TEXTURE", 0), ("_COLOR", 4), ("_ALPHATEST", 8)],
    );
    let mut store = MaterialPropertyStore::new();
    store.set_material(
        70,
        registry.intern("_TEXTURE"),
        MaterialPropertyValue::Float(0.0),
    );
    store.set_material(
        70,
        registry.intern("_COLOR"),
        MaterialPropertyValue::Float(1.0),
    );
    store.set_material(
        70,
        registry.intern("_ALPHATEST"),
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
    let variant_bits = (1u32 << 1) | (1u32 << 9);

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

    assert_eq!(read_f32_at(&bytes, 0), 1.0);
    assert_eq!(read_f32_at(&bytes, 4), 1.0);
    assert_eq!(read_f32_at(&bytes, 8), 0.0);
}

#[test]
fn unlit_variant_bits_override_texture_presence_inference() {
    let (reflected, ids, registry) =
        reflected_with_f32_fields_for_stem("unlit_default", &[("_TEXTURE", 0)]);
    let mut store = MaterialPropertyStore::new();
    store.set_material(
        71,
        registry.intern("_Tex"),
        MaterialPropertyValue::Texture(123),
    );

    let (texture, texture3d, cubemap, render_texture, video_texture) = empty_texture_pools();
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

    assert_eq!(read_f32_at(&bytes, 0), 0.0);
}

#[test]
fn missing_variant_bits_keep_existing_keyword_fallback() {
    let (reflected, ids, registry) =
        reflected_with_f32_fields_for_stem("unlit_default", &[("_TEXTURE", 0)]);
    let mut store = MaterialPropertyStore::new();
    store.set_material(
        72,
        registry.intern("_Tex"),
        MaterialPropertyValue::Texture(123),
    );

    let (texture, texture3d, cubemap, render_texture, video_texture) = empty_texture_pools();
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

    assert_eq!(read_f32_at(&bytes, 0), 1.0);
}
