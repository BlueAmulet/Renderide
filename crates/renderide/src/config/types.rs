//! Serde/TOML schema for renderer settings (`[display]`, `[rendering]`, `[debug]`, `[post_processing]`).
//!
//! `RendererSettings` is the top-level aggregator; per-domain submodules own each section's structs
//! and serde plumbing so each TOML table maps to a focused file.

use serde::{Deserialize, Serialize};

mod debug;
mod display;
mod post_processing;
mod rendering;
mod watchdog;

pub use debug::{
    DebugHudMainTab, DebugHudMainTabVisibility, DebugHudRendererConfigTab,
    DebugHudRendererConfigTabVisibility, DebugHudSettings, DebugSettings, PowerPreferenceSetting,
};
pub use display::DisplaySettings;
#[cfg(test)]
pub(crate) use post_processing::TonemapSettings;
pub use post_processing::{
    AutoExposureSettings, BloomCompositeMode, BloomSettings, GtaoSettings, PostProcessingSettings,
    TonemapMode,
};
pub use rendering::{
    GraphicsApiSetting, MsaaSampleCount, RenderingSettings, SceneColorFormat, VsyncMode,
};
pub use watchdog::{WatchdogAction, WatchdogSettings};

/// Runtime settings for the renderer process: defaults, merged from file, and edited via the debug UI.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RendererSettings {
    /// Version of the renderer config schema that last wrote this file.
    #[serde(default = "current_config_version")]
    pub config_version: String,
    /// Display caps and related options.
    pub display: DisplaySettings,
    /// Rendering options (e.g. vsync).
    pub rendering: RenderingSettings,
    /// Debug-only flags.
    pub debug: DebugSettings,
    /// Post-processing stack toggles and per-effect parameters.
    pub post_processing: PostProcessingSettings,
    /// Cooperative hang/hitch detection ([`crate::diagnostics::Watchdog`]).
    pub watchdog: WatchdogSettings,
}

impl RendererSettings {
    /// Config schema version emitted by this renderer build.
    pub const CURRENT_CONFIG_VERSION: &'static str = env!("CARGO_PKG_VERSION");

    /// Hardcoded defaults only.
    pub fn from_defaults() -> Self {
        Self::default()
    }
}

impl Default for RendererSettings {
    fn default() -> Self {
        Self {
            config_version: current_config_version(),
            display: DisplaySettings::default(),
            rendering: RenderingSettings::default(),
            debug: DebugSettings::default(),
            post_processing: PostProcessingSettings::default(),
            watchdog: WatchdogSettings::default(),
        }
    }
}

fn current_config_version() -> String {
    RendererSettings::CURRENT_CONFIG_VERSION.to_owned()
}

#[cfg(test)]
mod tests {
    use super::RendererSettings;

    #[test]
    fn default_config_version_uses_crate_version() {
        let settings = RendererSettings::default();

        assert_eq!(
            settings.config_version,
            RendererSettings::CURRENT_CONFIG_VERSION
        );
    }

    #[test]
    fn missing_config_version_deserializes_to_current_version() {
        let settings: RendererSettings =
            toml::from_str("[display]\nfocused_fps = 75\n").expect("deserialize");

        assert_eq!(
            settings.config_version,
            RendererSettings::CURRENT_CONFIG_VERSION
        );
        assert_eq!(settings.display.focused_fps_cap, 75);
    }
}
