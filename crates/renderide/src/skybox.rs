//! Skybox rendering: environment cube cache, IBL specular params, and active-main resolution.

mod environment;
pub(crate) mod params;
mod specular;

pub(crate) use environment::SkyboxEnvironmentCache;
pub(crate) use specular::resolve_active_main_skybox_specular_environment;
