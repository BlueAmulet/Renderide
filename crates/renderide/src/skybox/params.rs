//! Shared CPU-side parameter packing for analytic skybox evaluators.
//!
//! Splits the per-evaluator parameter builders across focused submodules and re-exports the
//! cross-module surface (types, helpers, and Unity-matching default constants) consumed by
//! reflection probes, the IBL bake cache, and the embedded uniform-pack tests.

mod common;
mod gradient;
mod procedural;
mod projection360;

pub(crate) use common::{
    DEFAULT_MAIN_TEX_ST, DEFAULT_SKYBOX_SAMPLE_SIZE, SkyboxEvaluatorParams, SkyboxParamMode,
    storage_v_inverted_flag,
};
pub(crate) use gradient::gradient_sky_params;
pub(crate) use procedural::procedural_sky_params;
pub(crate) use projection360::{PROJECTION360_DEFAULT_FOV, projection360_equirect_params};
