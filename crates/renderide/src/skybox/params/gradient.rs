//! Parameter packing for gradient sky materials.

use crate::backend::material_property_reader::{float_property, float4_array16_property};
use crate::materials::host_data::{
    MaterialPropertyLookupIds, MaterialPropertyStore, PropertyIdRegistry,
};

use super::common::{
    SkyboxEvaluatorParams, SkyboxParamMode, srgb_float4_array16_property, srgb_float4_property,
};

/// Builds parameter payload for a gradient sky material.
pub(crate) fn gradient_sky_params(
    store: &MaterialPropertyStore,
    registry: &PropertyIdRegistry,
    lookup: MaterialPropertyLookupIds,
) -> SkyboxEvaluatorParams {
    let mut params = SkyboxEvaluatorParams::empty(SkyboxParamMode::Gradient);
    params.color0 =
        srgb_float4_property(store, registry, lookup, "_BaseColor", [0.0, 0.0, 0.0, 1.0]);
    params.dirs_spread = float4_array16_property(store, registry, lookup, "_DirsSpread");
    params.gradient_color0 = srgb_float4_array16_property(store, registry, lookup, "_Color0");
    params.gradient_color1 = srgb_float4_array16_property(store, registry, lookup, "_Color1");
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color_space::srgb_f32x4_rgb_to_linear;
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
    fn gradient_sky_params_linearize_color_overrides() {
        let mut store = MaterialPropertyStore::new();
        let registry = PropertyIdRegistry::new();
        let base = [0.5, 0.25, 0.75, 0.6];
        let color0 = [[0.04045, 0.5, 1.25, 0.25], [0.8, 0.6, 0.4, 0.5]];
        let color1 = [[0.2, -0.5, 1.0, 0.75], [0.0, 0.1, 0.9, 0.8]];
        store.set_material(
            7,
            registry.intern("_BaseColor"),
            MaterialPropertyValue::Float4(base),
        );
        store.set_material(
            7,
            registry.intern("_Color0"),
            MaterialPropertyValue::Float4Array(color0.to_vec()),
        );
        store.set_material(
            7,
            registry.intern("_Color1"),
            MaterialPropertyValue::Float4Array(color1.to_vec()),
        );

        let params = gradient_sky_params(&store, &registry, lookup(7));

        assert_f32x4_near(params.color0, srgb_f32x4_rgb_to_linear(base));
        assert_f32x4_near(
            params.gradient_color0[0],
            srgb_f32x4_rgb_to_linear(color0[0]),
        );
        assert_f32x4_near(
            params.gradient_color0[1],
            srgb_f32x4_rgb_to_linear(color0[1]),
        );
        assert_f32x4_near(
            params.gradient_color1[0],
            srgb_f32x4_rgb_to_linear(color1[0]),
        );
        assert_f32x4_near(
            params.gradient_color1[1],
            srgb_f32x4_rgb_to_linear(color1[1]),
        );
    }
}
