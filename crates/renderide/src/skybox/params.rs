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

// Procedural-sky Unity-default constants are exercised only by uniform-pack regression tests
// (`crate::materials::embedded::uniform_pack::tests`, gated by `#[cfg(test)]`), so gating these
// re-exports the same way keeps the non-test build free of unused-import warnings without
// reaching for an `#[allow]` / `#[expect]` suppression.
#[cfg(test)]
pub(crate) use procedural::{
    PROCEDURAL_SKY_DEFAULT_ATMOSPHERE_THICKNESS, PROCEDURAL_SKY_DEFAULT_EXPOSURE,
    PROCEDURAL_SKY_DEFAULT_GROUND_COLOR, PROCEDURAL_SKY_DEFAULT_SKY_TINT,
    PROCEDURAL_SKY_DEFAULT_SUN_COLOR, PROCEDURAL_SKY_DEFAULT_SUN_DIRECTION,
    PROCEDURAL_SKY_DEFAULT_SUN_SIZE,
};
