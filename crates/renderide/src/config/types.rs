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
pub use post_processing::{
    AutoExposureSettings, BloomCompositeMode, BloomSettings, GtaoSettings, PostProcessingSettings,
    TonemapMode, TonemapSettings,
};
pub use rendering::{
    GraphicsApiSetting, MsaaSampleCount, RenderingSettings, SceneColorFormat, VsyncMode,
};
pub use watchdog::{WatchdogAction, WatchdogSettings};

/// Runtime settings for the renderer process: defaults, merged from file, and edited via the debug UI.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RendererSettings {
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
    /// Hardcoded defaults only.
    pub fn from_defaults() -> Self {
        Self::default()
    }
}
