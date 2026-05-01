//! Skybox rendering: unified IBL prefilter cache, sky evaluator params, and active-main resolution.

pub(crate) mod ibl_cache;
pub(crate) mod params;
mod prepared;
pub(crate) mod specular;

pub(crate) use ibl_cache::SkyboxIblCache;
pub use prepared::{PreparedClearColorSkybox, PreparedMaterialSkybox, PreparedSkybox};
