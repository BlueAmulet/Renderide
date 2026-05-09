//! ProceduralSkybox uniform defaults and keyword inference tests.

use super::super::*;
use super::common::*;

use crate::materials::embedded::layout::StemEmbeddedPropertyIds;
use crate::materials::embedded::texture_pools::EmbeddedTexturePools;
use crate::materials::host_data::{
    MaterialPropertyStore, MaterialPropertyValue, PropertyIdRegistry,
};
use crate::materials::{ReflectedRasterLayout, ReflectedUniformScalarKind};
use crate::skybox::params::{
    PROCEDURAL_SKY_DEFAULT_ATMOSPHERE_THICKNESS, PROCEDURAL_SKY_DEFAULT_EXPOSURE,
    PROCEDURAL_SKY_DEFAULT_GROUND_COLOR, PROCEDURAL_SKY_DEFAULT_SKY_TINT,
    PROCEDURAL_SKY_DEFAULT_SUN_COLOR, PROCEDURAL_SKY_DEFAULT_SUN_DIRECTION,
    PROCEDURAL_SKY_DEFAULT_SUN_SIZE,
};

fn procedural_reflected() -> (
    ReflectedRasterLayout,
    StemEmbeddedPropertyIds,
    PropertyIdRegistry,
) {
    let (reflected, mut ids, registry) = reflected_with_uniform_fields(&[
        ("_SkyTint", ReflectedUniformScalarKind::Vec4, 16, 0),
        ("_GroundColor", ReflectedUniformScalarKind::Vec4, 16, 16),
        ("_SunColor", ReflectedUniformScalarKind::Vec4, 16, 32),
        ("_SunDirection", ReflectedUniformScalarKind::Vec4, 16, 48),
        ("_Exposure", ReflectedUniformScalarKind::F32, 4, 64),
        ("_SunSize", ReflectedUniformScalarKind::F32, 4, 68),
        (
            "_AtmosphereThickness",
            ReflectedUniformScalarKind::F32,
            4,
            72,
        ),
        ("_SUNDISK_NONE", ReflectedUniformScalarKind::F32, 4, 76),
        ("_SUNDISK_SIMPLE", ReflectedUniformScalarKind::F32, 4, 80),
        (
            "_SUNDISK_HIGH_QUALITY",
            ReflectedUniformScalarKind::F32,
            4,
            84,
        ),
    ]);
    ids.procedural_skybox_defaults = true;
    (reflected, ids, registry)
}

fn empty_tex_ctx<'a>(pools: &'a EmbeddedTexturePools<'a>) -> UniformPackTextureContext<'a> {
    UniformPackTextureContext {
        pools,
        primary_texture_2d: -1,
    }
}

fn assert_f32x4_near(actual: [f32; 4], expected: [f32; 4]) {
    for (a, e) in actual.iter().zip(expected.iter()) {
        assert!(
            (a - e).abs() < 0.000_001,
            "actual={actual:?} expected={expected:?}"
        );
    }
}

#[test]
fn proceduralskybox_uniform_defaults_match_unity_asset() {
    let (reflected, ids, _) = procedural_reflected();
    let (texture, texture3d, cubemap, render_texture, video_texture) = empty_texture_pools();
    let pools = EmbeddedTexturePools {
        texture: &texture,
        texture3d: &texture3d,
        cubemap: &cubemap,
        render_texture: &render_texture,
        video_texture: &video_texture,
    };
    let value_spaces = MaterialUniformValueSpaces::for_stem("proceduralskybox_default", &reflected);

    let bytes = build_embedded_uniform_bytes_with_value_spaces(
        &reflected,
        &ids,
        &value_spaces,
        &MaterialPropertyStore::new(),
        lookup(1),
        &empty_tex_ctx(&pools),
    )
    .expect("uniform bytes");

    assert_f32x4_near(
        read_f32x4(&bytes, 0),
        srgb_vec4_rgb_to_linear(PROCEDURAL_SKY_DEFAULT_SKY_TINT),
    );
    assert_f32x4_near(
        read_f32x4(&bytes, 16),
        srgb_vec4_rgb_to_linear(PROCEDURAL_SKY_DEFAULT_GROUND_COLOR),
    );
    assert_f32x4_near(
        read_f32x4(&bytes, 32),
        srgb_vec4_rgb_to_linear(PROCEDURAL_SKY_DEFAULT_SUN_COLOR),
    );
    assert_eq!(read_f32x4(&bytes, 48), PROCEDURAL_SKY_DEFAULT_SUN_DIRECTION);
    assert_eq!(read_f32_at(&bytes, 64), PROCEDURAL_SKY_DEFAULT_EXPOSURE);
    assert_eq!(read_f32_at(&bytes, 68), PROCEDURAL_SKY_DEFAULT_SUN_SIZE);
    assert_eq!(
        read_f32_at(&bytes, 72),
        PROCEDURAL_SKY_DEFAULT_ATMOSPHERE_THICKNESS
    );
    assert_eq!(read_f32_at(&bytes, 76), 0.0);
    assert_eq!(read_f32_at(&bytes, 80), 0.0);
    assert_eq!(read_f32_at(&bytes, 84), 1.0);
}

#[test]
fn proceduralskybox_sundisk_property_selects_keyword_mode() {
    let (reflected, ids, registry) = procedural_reflected();
    let (texture, texture3d, cubemap, render_texture, video_texture) = empty_texture_pools();
    let pools = EmbeddedTexturePools {
        texture: &texture,
        texture3d: &texture3d,
        cubemap: &cubemap,
        render_texture: &render_texture,
        video_texture: &video_texture,
    };
    let value_spaces = MaterialUniformValueSpaces::for_stem("proceduralskybox_default", &reflected);
    let mut store = MaterialPropertyStore::new();

    for (mat, sun_disk, expected) in [
        (2, 0.0, [1.0, 0.0, 0.0]),
        (3, 1.0, [0.0, 1.0, 0.0]),
        (4, 2.0, [0.0, 0.0, 1.0]),
    ] {
        store.set_material(
            mat,
            registry.intern("_SunDisk"),
            MaterialPropertyValue::Float(sun_disk),
        );
        let bytes = build_embedded_uniform_bytes_with_value_spaces(
            &reflected,
            &ids,
            &value_spaces,
            &store,
            lookup(mat),
            &empty_tex_ctx(&pools),
        )
        .expect("uniform bytes");

        assert_eq!(read_f32_at(&bytes, 76), expected[0]);
        assert_eq!(read_f32_at(&bytes, 80), expected[1]);
        assert_eq!(read_f32_at(&bytes, 84), expected[2]);
    }
}

#[test]
fn proceduralskybox_explicit_sundisk_keyword_overrides_property() {
    let (reflected, ids, registry) = procedural_reflected();
    let (texture, texture3d, cubemap, render_texture, video_texture) = empty_texture_pools();
    let pools = EmbeddedTexturePools {
        texture: &texture,
        texture3d: &texture3d,
        cubemap: &cubemap,
        render_texture: &render_texture,
        video_texture: &video_texture,
    };
    let value_spaces = MaterialUniformValueSpaces::for_stem("proceduralskybox_default", &reflected);
    let mut store = MaterialPropertyStore::new();
    store.set_material(
        5,
        registry.intern("_SunDisk"),
        MaterialPropertyValue::Float(2.0),
    );
    store.set_material(
        5,
        registry.intern("_SUNDISK_NONE"),
        MaterialPropertyValue::Float(1.0),
    );

    let bytes = build_embedded_uniform_bytes_with_value_spaces(
        &reflected,
        &ids,
        &value_spaces,
        &store,
        lookup(5),
        &empty_tex_ctx(&pools),
    )
    .expect("uniform bytes");

    assert_eq!(read_f32_at(&bytes, 76), 1.0);
    assert_eq!(read_f32_at(&bytes, 80), 0.0);
    assert_eq!(read_f32_at(&bytes, 84), 0.0);
}
