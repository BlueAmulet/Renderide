//! Resolves a skybox material into a unified IBL bake source.
//!
//! Returns an [`SkyboxIblSource`] that the [`crate::skybox::ibl_cache::SkyboxIblCache`] consumes to
//! schedule a GGX-prefiltered cubemap bake. Source variants today:
//! - [`SkyboxIblSource::Analytic`] for procedural / gradient skyboxes.
//! - [`SkyboxIblSource::Cubemap`] for Projection360 `_MainCube` (or `_MainTex` cubemap fallback).
//! - [`SkyboxIblSource::Equirect`] for Projection360 `_MainTex` Texture2D.
//! - [`SkyboxIblSource::SolidColor`] for constant-color backgrounds.
//! - [`SkyboxIblSource::RuntimeCubemap`] for renderer-captured reflection probe cubemaps.

mod projection360;
mod resolve;
mod solid_color;
mod source;

pub(crate) use resolve::resolve_skybox_material_ibl_source;
pub(crate) use solid_color::{solid_color_ibl_source, solid_color_params};
pub(crate) use source::{
    CubemapIblSource, EquirectIblSource, RuntimeCubemapIblSource, SkyboxIblSource,
};
