//! Reflection-probe IBL source helpers.
//!
//! [`SkyboxIblSource`] feeds [`crate::skybox::ibl_cache::SkyboxIblCache`] with cubemaps,
//! constant colors, and renderer-captured probe cubemaps. Active skybox materials are not an
//! implicit source for IBL.

mod solid_color;
mod source;

pub(crate) use solid_color::{solid_color_ibl_source, solid_color_params};
pub(crate) use source::{CubemapIblSource, RuntimeCubemapIblSource, SkyboxIblSource};
