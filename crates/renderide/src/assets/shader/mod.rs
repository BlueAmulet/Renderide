//! Host [`ShaderUpload`](crate::shared::ShaderUpload) handling: logical name extraction and material routing.

pub mod logical_name;
pub mod route;
pub mod unity_asset;

pub use route::{resolve_shader_upload, ResolvedShaderUpload};
