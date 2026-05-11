//! Parameter packing for Projection360 equirect skybox materials.

use crate::backend::material_property_reader::float4_property;
use crate::materials::host_data::{
    MaterialPropertyLookupIds, MaterialPropertyStore, PropertyIdRegistry,
};

use super::common::{
    DEFAULT_MAIN_TEX_ST, SkyboxEvaluatorParams, SkyboxParamMode, storage_v_inverted_flag,
};

/// Default `Projection360` field of view used by Unity material defaults.
pub(crate) const PROJECTION360_DEFAULT_FOV: [f32; 4] =
    [std::f32::consts::TAU, std::f32::consts::PI, 0.0, 0.0];

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
