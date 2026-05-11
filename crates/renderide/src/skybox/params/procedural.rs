//! Parameter packing for ProceduralSkybox materials.

use crate::backend::material_property_reader::{float_property, float4_property};
use crate::color_space::srgb_f32x4_rgb_to_linear;
use crate::materials::host_data::{
    MaterialPropertyLookupIds, MaterialPropertyStore, PropertyIdRegistry,
};

use super::common::{
    SkyboxEvaluatorParams, SkyboxParamMode, optional_float_property, srgb_float4_property,
};

/// Default ProceduralSkybox `_SkyTint` property in sRGB authoring space.
pub(crate) const PROCEDURAL_SKY_DEFAULT_SKY_TINT: [f32; 4] = [0.5, 0.5, 0.5, 1.0];
/// Default ProceduralSkybox `_GroundColor` property in sRGB authoring space.
pub(crate) const PROCEDURAL_SKY_DEFAULT_GROUND_COLOR: [f32; 4] = [0.369, 0.349, 0.341, 1.0];
/// Default ProceduralSkybox `_SunColor` property in sRGB authoring space.
pub(crate) const PROCEDURAL_SKY_DEFAULT_SUN_COLOR: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
/// Default ProceduralSkybox `_SunDirection` vector.
pub(crate) const PROCEDURAL_SKY_DEFAULT_SUN_DIRECTION: [f32; 4] = [0.577, 0.577, 0.577, 0.0];
/// Default ProceduralSkybox `_Exposure` value.
pub(crate) const PROCEDURAL_SKY_DEFAULT_EXPOSURE: f32 = 1.3;
/// Default ProceduralSkybox `_SunSize` value.
pub(crate) const PROCEDURAL_SKY_DEFAULT_SUN_SIZE: f32 = 0.04;
/// Default ProceduralSkybox `_AtmosphereThickness` value.
pub(crate) const PROCEDURAL_SKY_DEFAULT_ATMOSPHERE_THICKNESS: f32 = 1.0;
/// Default ProceduralSkybox `_SunDisk` value (`2` = high quality).
const PROCEDURAL_SKY_DEFAULT_SUN_DISK: f32 = 2.0;

/// Builds parameter payload for a procedural sky material.
pub(crate) fn procedural_sky_params(
    store: &MaterialPropertyStore,
    registry: &PropertyIdRegistry,
    lookup: MaterialPropertyLookupIds,
) -> SkyboxEvaluatorParams {
    let mut params = SkyboxEvaluatorParams::empty(SkyboxParamMode::Procedural);
    params.color0 = srgb_float4_property(
        store,
        registry,
        lookup,
        "_SkyTint",
        PROCEDURAL_SKY_DEFAULT_SKY_TINT,
    );
    params.color1 = float4_property(
        store,
        registry,
        lookup,
        "_GroundColor",
        PROCEDURAL_SKY_DEFAULT_GROUND_COLOR,
    );
    params.color1 = srgb_f32x4_rgb_to_linear(params.color1);
    params.direction = float4_property(
        store,
        registry,
        lookup,
        "_SunDirection",
        PROCEDURAL_SKY_DEFAULT_SUN_DIRECTION,
    );
    let exposure = float_property(
        store,
        registry,
        lookup,
        "_Exposure",
        PROCEDURAL_SKY_DEFAULT_EXPOSURE,
    );
    let sun_size = float_property(
        store,
        registry,
        lookup,
        "_SunSize",
        PROCEDURAL_SKY_DEFAULT_SUN_SIZE,
    );
    let atmosphere = float_property(
        store,
        registry,
        lookup,
        "_AtmosphereThickness",
        PROCEDURAL_SKY_DEFAULT_ATMOSPHERE_THICKNESS,
    );
    let sun_disk_mode = procedural_sun_disk_mode(store, registry, lookup);
    params.scalars = [exposure, sun_size, atmosphere, sun_disk_mode];
    params.gradient_color0[0] = srgb_float4_property(
        store,
        registry,
        lookup,
        "_SunColor",
        PROCEDURAL_SKY_DEFAULT_SUN_COLOR,
    );
    params
}

/// Encodes the procedural sun disk keyword state as a scalar for WGSL.
fn procedural_sun_disk_mode(
    store: &MaterialPropertyStore,
    registry: &PropertyIdRegistry,
    lookup: MaterialPropertyLookupIds,
) -> f32 {
    let none = optional_float_property(store, registry, lookup, "_SUNDISK_NONE");
    let simple = optional_float_property(store, registry, lookup, "_SUNDISK_SIMPLE");
    let high_quality = optional_float_property(store, registry, lookup, "_SUNDISK_HIGH_QUALITY");
    if none.is_some() || simple.is_some() || high_quality.is_some() {
        if none.is_some_and(|v| v.abs() > f32::EPSILON) {
            return 0.0;
        }
        if simple.is_some_and(|v| v.abs() > f32::EPSILON) {
            return 1.0;
        }
        if high_quality.is_some_and(|v| v.abs() > f32::EPSILON) {
            return 2.0;
        }
        return 0.0;
    }

    let sun_disk = float_property(
        store,
        registry,
        lookup,
        "_SunDisk",
        PROCEDURAL_SKY_DEFAULT_SUN_DISK,
    )
    .round();
    if sun_disk <= 0.0 {
        return 0.0;
    }
    if sun_disk == 1.0 {
        return 1.0;
    }
    2.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::materials::host_data::MaterialPropertyValue;

    fn lookup(material_asset_id: i32) -> MaterialPropertyLookupIds {
        MaterialPropertyLookupIds {
            material_asset_id,
            mesh_property_block_slot0: None,
            mesh_renderer_property_block_id: None,
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
    fn procedural_sky_params_use_unity_defaults() {
        let store = MaterialPropertyStore::new();
        let registry = PropertyIdRegistry::new();
        let params = procedural_sky_params(&store, &registry, lookup(1));

        assert_f32x4_near(
            params.color0,
            srgb_f32x4_rgb_to_linear(PROCEDURAL_SKY_DEFAULT_SKY_TINT),
        );
        assert_f32x4_near(
            params.color1,
            srgb_f32x4_rgb_to_linear(PROCEDURAL_SKY_DEFAULT_GROUND_COLOR),
        );
        assert_eq!(params.direction, PROCEDURAL_SKY_DEFAULT_SUN_DIRECTION);
        assert_eq!(
            params.scalars,
            [
                PROCEDURAL_SKY_DEFAULT_EXPOSURE,
                PROCEDURAL_SKY_DEFAULT_SUN_SIZE,
                PROCEDURAL_SKY_DEFAULT_ATMOSPHERE_THICKNESS,
                PROCEDURAL_SKY_DEFAULT_SUN_DISK,
            ]
        );
        assert_f32x4_near(
            params.gradient_color0[0],
            srgb_f32x4_rgb_to_linear(PROCEDURAL_SKY_DEFAULT_SUN_COLOR),
        );
    }

    #[test]
    fn procedural_sky_params_linearize_color_overrides() {
        let mut store = MaterialPropertyStore::new();
        let registry = PropertyIdRegistry::new();
        let sky_tint = [0.5, 0.25, 0.75, 1.0];
        let ground = [0.2, 0.4, 0.6, 1.0];
        let sun = [1.0, 0.8, 0.6, 1.0];
        store.set_material(
            2,
            registry.intern("_SkyTint"),
            MaterialPropertyValue::Float4(sky_tint),
        );
        store.set_material(
            2,
            registry.intern("_GroundColor"),
            MaterialPropertyValue::Float4(ground),
        );
        store.set_material(
            2,
            registry.intern("_SunColor"),
            MaterialPropertyValue::Float4(sun),
        );

        let params = procedural_sky_params(&store, &registry, lookup(2));

        assert_f32x4_near(params.color0, srgb_f32x4_rgb_to_linear(sky_tint));
        assert_f32x4_near(params.color1, srgb_f32x4_rgb_to_linear(ground));
        assert_f32x4_near(params.gradient_color0[0], srgb_f32x4_rgb_to_linear(sun));
    }

    #[test]
    fn procedural_sky_params_resolve_sun_disk_modes() {
        let mut store = MaterialPropertyStore::new();
        let registry = PropertyIdRegistry::new();
        for (mat, mode, expected) in [(3, 0.0, 0.0), (4, 1.0, 1.0), (5, 2.0, 2.0)] {
            store.set_material(
                mat,
                registry.intern("_SunDisk"),
                MaterialPropertyValue::Float(mode),
            );
            let params = procedural_sky_params(&store, &registry, lookup(mat));
            assert_eq!(params.scalars[3], expected);
        }
    }

    #[test]
    fn procedural_sky_params_explicit_keywords_override_sun_disk() {
        let mut store = MaterialPropertyStore::new();
        let registry = PropertyIdRegistry::new();
        store.set_material(
            6,
            registry.intern("_SunDisk"),
            MaterialPropertyValue::Float(2.0),
        );
        store.set_material(
            6,
            registry.intern("_SUNDISK_NONE"),
            MaterialPropertyValue::Float(1.0),
        );

        let params = procedural_sky_params(&store, &registry, lookup(6));

        assert_eq!(params.scalars[3], 0.0);
    }
}
