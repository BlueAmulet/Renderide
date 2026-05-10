//! Embedded-material keyword inference tests.

use super::super::tables::inferred_keyword_float_f32;
use super::super::*;
use super::common::*;

use std::sync::Arc;

use hashbrown::HashMap;

use crate::materials::embedded::layout::StemEmbeddedPropertyIds;
use crate::materials::embedded::texture_pools::EmbeddedTexturePools;
use crate::materials::host_data::{
    MaterialPropertyStore, MaterialPropertyValue, PropertyIdRegistry,
};
use crate::materials::{
    ReflectedMaterialUniformBlock, ReflectedRasterLayout, ReflectedUniformScalarKind,
};
use crate::shared::ColorProfile;

fn pack_rect_clip_value(
    reflected: &ReflectedRasterLayout,
    ids: &StemEmbeddedPropertyIds,
    store: &MaterialPropertyStore,
    material_id: i32,
) -> f32 {
    pack_first_f32_value(reflected, ids, store, material_id)
}

fn pack_first_f32_value(
    reflected: &ReflectedRasterLayout,
    ids: &StemEmbeddedPropertyIds,
    store: &MaterialPropertyStore,
    material_id: i32,
) -> f32 {
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

    let bytes = build_embedded_uniform_bytes(reflected, ids, store, lookup(material_id), &tex_ctx)
        .expect("uniform bytes");
    read_f32_at(&bytes, 0)
}

fn set_float_property(
    store: &mut MaterialPropertyStore,
    registry: &PropertyIdRegistry,
    material_id: i32,
    property_name: &str,
    value: f32,
) {
    store.set_material(
        material_id,
        registry.intern(property_name),
        MaterialPropertyValue::Float(value),
    );
}

fn assert_xiexe_alpha_keywords(
    store: &MaterialPropertyStore,
    material_id: i32,
    ids: &StemEmbeddedPropertyIds,
    cutout: bool,
    alpha_blend: bool,
    transparent: bool,
) {
    let expected = |enabled| Some(if enabled { 1.0 } else { 0.0 });
    assert_eq!(
        inferred_keyword_float_f32("Cutout", store, lookup(material_id), ids),
        expected(cutout)
    );
    assert_eq!(
        inferred_keyword_float_f32("AlphaBlend", store, lookup(material_id), ids),
        expected(alpha_blend)
    );
    assert_eq!(
        inferred_keyword_float_f32("Transparent", store, lookup(material_id), ids),
        expected(transparent)
    );
}

mod alpha_blend;
mod defaults_misc;
mod texture_keywords;
mod ui_text;
