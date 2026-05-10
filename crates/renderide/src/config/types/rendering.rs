//! Rendering toggles, MSAA, vsync, and scene-color format. Persisted as
//! `[rendering]`. Each enum lives in its own submodule and is generated through the shared
//! [`crate::labeled_enum`] macro so adding a new mode is a single declaration with the
//! canonical persist string, label, and any aliases.

mod graphics_api;
mod msaa;
mod scene_color;
mod vsync;

pub use graphics_api::GraphicsApiSetting;
pub use msaa::MsaaSampleCount;
pub use scene_color::SceneColorFormat;
pub use vsync::VsyncMode;

use serde::{Deserialize, Serialize};

/// Rendering toggles and scalars. Persisted as `[rendering]`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RenderingSettings {
    /// Swapchain vsync mode ([`VsyncMode::Off`] / [`VsyncMode::On`]); applied live without restart
    /// through [`crate::gpu::GpuContext::set_present_mode`]. Old `vsync = true / false`,
    /// `vsync = "auto"`, and `vsync = "adaptive"` configs still load via aliases in the
    /// `labeled_enum!` deserializer.
    pub vsync: VsyncMode,
    /// Startup graphics API preference. `Auto` preserves wgpu's default backend discovery; an
    /// explicit API constrains the first adapter-selection attempt and falls back to automatic
    /// selection if that API has no compatible adapter. Applied only when the GPU stack is created,
    /// so changes require a renderer restart.
    #[serde(rename = "graphics_api", default)]
    pub graphics_api: GraphicsApiSetting,
    /// Wall-clock budget per frame for cooperative mesh/texture integration
    /// ([`crate::runtime::RendererRuntime::run_asset_integration`]), in milliseconds.
    #[serde(rename = "asset_integration_budget_ms")]
    pub asset_integration_budget_ms: u32,
    /// Extra post-main budget for dynamic buffer / particle integration, in milliseconds.
    #[serde(rename = "asset_particle_integration_budget_ms")]
    pub asset_particle_integration_budget_ms: u32,
    /// Multisample anti-aliasing for forward rendering. Effective sample count is clamped to the
    /// GPU's supported maximum for the target format. Main-window, VR, CameraRenderTask, and host
    /// RenderTexture camera outputs use this tier; reflection-probe utility captures stay 1x.
    pub msaa: MsaaSampleCount,
    /// Format for the **scene-color** HDR target the forward pass renders into before
    /// [`crate::passes::SceneColorComposePass`] writes the displayable target.
    ///
    /// This is intermediate precision/range (e.g. [`SceneColorFormat::Rgba16Float`]), not the OS
    /// swapchain HDR mode.
    #[serde(rename = "scene_color_format")]
    pub scene_color_format: SceneColorFormat,
}

impl Default for RenderingSettings {
    fn default() -> Self {
        Self {
            vsync: VsyncMode::default(),
            graphics_api: GraphicsApiSetting::default(),
            asset_integration_budget_ms: 2,
            asset_particle_integration_budget_ms: 4,
            msaa: MsaaSampleCount::default(),
            scene_color_format: SceneColorFormat::default(),
        }
    }
}
