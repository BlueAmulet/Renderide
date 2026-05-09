//! Shared CPU-side parameter packing for analytic skybox evaluators.

use bytemuck::{Pod, Zeroable};

use crate::backend::material_property_reader::{
    float_property, float4_array16_property, float4_property,
};
use crate::color_space::srgb_f32x4_rgb_to_linear;
use crate::materials::host_data::{
    MaterialPropertyLookupIds, MaterialPropertyStore, MaterialPropertyValue, PropertyIdRegistry,
};

/// Default sky parameter sample grid used by SH2 projection.
pub(crate) const DEFAULT_SKYBOX_SAMPLE_SIZE: u32 = 64;
/// Default `Projection360` field of view used by Unity material defaults.
pub(crate) const PROJECTION360_DEFAULT_FOV: [f32; 4] =
    [std::f32::consts::TAU, std::f32::consts::PI, 0.0, 0.0];
/// Default texture scale/offset used by Unity `_MainTex_ST` properties.
pub(crate) const DEFAULT_MAIN_TEX_ST: [f32; 4] = [1.0, 1.0, 0.0, 0.0];
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
pub(crate) const PROCEDURAL_SKY_DEFAULT_SUN_DISK: f32 = 2.0;

/// Parameter-only sky evaluator mode used by skybox compute shaders.
#[derive(Clone, Copy, Debug)]
pub(crate) enum SkyboxParamMode {
    /// Procedural sky approximation from material scalar/color properties.
    Procedural = 1,
    /// Gradient sky approximation from material array properties.
    Gradient = 2,
}

/// Uniform payload shared by analytic skybox compute kernels.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct SkyboxEvaluatorParams {
    /// Sample grid edge for projection or generated cubemap face edge for baking.
    pub(crate) sample_size: u32,
    /// Evaluator mode from [`SkyboxParamMode`].
    pub(crate) mode: u32,
    /// Number of active gradient lobes.
    pub(crate) gradient_count: u32,
    /// Reserved alignment slot.
    pub(crate) _pad0: u32,
    /// Generic color slot 0.
    pub(crate) color0: [f32; 4],
    /// Generic color slot 1.
    pub(crate) color1: [f32; 4],
    /// Generic direction and scalar slot.
    pub(crate) direction: [f32; 4],
    /// Generic scalar slot.
    pub(crate) scalars: [f32; 4],
    /// Gradient direction/spread rows.
    pub(crate) dirs_spread: [[f32; 4]; 16],
    /// Gradient color rows A.
    pub(crate) gradient_color0: [[f32; 4]; 16],
    /// Gradient color rows B.
    pub(crate) gradient_color1: [[f32; 4]; 16],
    /// Gradient parameter rows.
    pub(crate) gradient_params: [[f32; 4]; 16],
}

impl SkyboxEvaluatorParams {
    /// Creates a parameter block with the default projection sample grid.
    pub(crate) fn empty(mode: SkyboxParamMode) -> Self {
        Self {
            sample_size: DEFAULT_SKYBOX_SAMPLE_SIZE,
            mode: mode as u32,
            gradient_count: 0,
            _pad0: 0,
            color0: [0.0; 4],
            color1: [0.0; 4],
            direction: [0.0, 1.0, 0.0, 0.0],
            scalars: [1.0, 0.0, 0.0, 0.0],
            dirs_spread: [[0.0; 4]; 16],
            gradient_color0: [[0.0; 4]; 16],
            gradient_color1: [[0.0; 4]; 16],
            gradient_params: [[0.0; 4]; 16],
        }
    }

    /// Returns a copy with the sample or face edge set.
    pub(crate) fn with_sample_size(mut self, sample_size: u32) -> Self {
        self.sample_size = sample_size.max(1);
        self
    }
}

/// Converts a storage-orientation boolean to the shader keyword float convention.
pub(crate) fn storage_v_inverted_flag(storage_v_inverted: bool) -> f32 {
    if storage_v_inverted { 1.0 } else { 0.0 }
}

/// Builds the `Projection360` equirectangular sampling payload shared with compute shaders.
pub(crate) fn projection360_equirect_params(
    store: &MaterialPropertyStore,
    registry: &PropertyIdRegistry,
    lookup: MaterialPropertyLookupIds,
    storage_v_inverted: bool,
) -> SkyboxEvaluatorParams {
    let mut params = SkyboxEvaluatorParams::empty(SkyboxParamMode::Procedural);
    params.color0 = float4_property(store, registry, lookup, "_FOV", PROJECTION360_DEFAULT_FOV);
    params.color1 = float4_property(store, registry, lookup, "_MainTex_ST", DEFAULT_MAIN_TEX_ST);
    params.scalars = [storage_v_inverted_flag(storage_v_inverted), 0.0, 0.0, 0.0];
    params
}

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

/// Builds parameter payload for a gradient sky material.
pub(crate) fn gradient_sky_params(
    store: &MaterialPropertyStore,
    registry: &PropertyIdRegistry,
    lookup: MaterialPropertyLookupIds,
) -> SkyboxEvaluatorParams {
    let mut params = SkyboxEvaluatorParams::empty(SkyboxParamMode::Gradient);
    params.color0 = float4_property(store, registry, lookup, "_BaseColor", [0.0, 0.0, 0.0, 1.0]);
    params.dirs_spread = float4_array16_property(store, registry, lookup, "_DirsSpread");
    params.gradient_color0 = float4_array16_property(store, registry, lookup, "_Color0");
    params.gradient_color1 = float4_array16_property(store, registry, lookup, "_Color1");
    params.gradient_params = float4_array16_property(store, registry, lookup, "_Params");
    params.gradient_count = float_property(store, registry, lookup, "_Gradients", 0.0)
        .round()
        .clamp(0.0, 16.0) as u32;
    if params.gradient_count == 0 {
        params.gradient_count = params
            .dirs_spread
            .iter()
            .position(|v| v.iter().all(|c| c.abs() < 1e-6))
            .unwrap_or(16) as u32;
    }
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

/// Reads a scalar float material property by host name when the property was present.
fn optional_float_property(
    store: &MaterialPropertyStore,
    registry: &PropertyIdRegistry,
    lookup: MaterialPropertyLookupIds,
    name: &str,
) -> Option<f32> {
    let pid = registry.intern(name);
    match store.get_merged(lookup, pid) {
        Some(MaterialPropertyValue::Float(v)) => Some(*v),
        _ => None,
    }
}

/// Reads an sRGB-authored color property and converts its RGB channels to linear.
fn srgb_float4_property(
    store: &MaterialPropertyStore,
    registry: &PropertyIdRegistry,
    lookup: MaterialPropertyLookupIds,
    name: &str,
    fallback: [f32; 4],
) -> [f32; 4] {
    srgb_f32x4_rgb_to_linear(float4_property(store, registry, lookup, name, fallback))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::materials::host_data::MaterialPropertyValue;

    fn lookup(material_asset_id: i32) -> MaterialPropertyLookupIds {
        MaterialPropertyLookupIds {
            material_asset_id,
            mesh_property_block_slot0: None,
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
    fn empty_params_use_documented_defaults() {
        let params = SkyboxEvaluatorParams::empty(SkyboxParamMode::Procedural);

        assert_eq!(params.sample_size, DEFAULT_SKYBOX_SAMPLE_SIZE);
        assert_eq!(params.mode, SkyboxParamMode::Procedural as u32);
        assert_eq!(params.gradient_count, 0);
        assert_eq!(params.direction, [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(params.scalars, [1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn with_sample_size_clamps_zero_to_one() {
        let params = SkyboxEvaluatorParams::empty(SkyboxParamMode::Gradient).with_sample_size(0);

        assert_eq!(params.sample_size, 1);
        assert_eq!(params.mode, SkyboxParamMode::Gradient as u32);
    }

    #[test]
    fn storage_v_inverted_flag_matches_shader_float_convention() {
        assert_eq!(storage_v_inverted_flag(false), 0.0);
        assert_eq!(storage_v_inverted_flag(true), 1.0);
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
