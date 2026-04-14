//! Serde/TOML schema for renderer settings (`[display]`, `[rendering]`, `[debug]`).

use serde::{Deserialize, Serialize};

/// Display-related caps. Persisted as `[display]`.
///
/// Non-zero values cap desktop redraw scheduling via winit (`ControlFlow::WaitUntil`); OpenXR VR
/// sessions ignore these caps so headset frame pacing is unchanged.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DisplaySettings {
    /// Target max FPS when the window is focused (0 = uncapped).
    #[serde(rename = "focused_fps")]
    pub focused_fps_cap: u32,
    /// Target max FPS when unfocused (0 = uncapped).
    #[serde(rename = "unfocused_fps")]
    pub unfocused_fps_cap: u32,
}

impl Default for DisplaySettings {
    fn default() -> Self {
        Self {
            focused_fps_cap: 240,
            unfocused_fps_cap: 60,
        }
    }
}

/// Rendering toggles and scalars. Persisted as `[rendering]`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RenderingSettings {
    /// Vertical sync via swapchain present mode ([`wgpu::PresentMode::AutoVsync`]); applied live
    /// without restart (see [`crate::gpu::GpuContext::set_vsync`]).
    pub vsync: bool,
}

/// Preferred GPU power mode for future adapter selection (stored; changing at runtime may require
/// re-initialization).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PowerPreferenceSetting {
    /// Maps to [`wgpu::PowerPreference::LowPower`].
    LowPower,
    /// Maps to [`wgpu::PowerPreference::HighPerformance`].
    #[default]
    HighPerformance,
}

impl PowerPreferenceSetting {
    /// All variants for ImGui combo / persistence.
    pub const ALL: [Self; 2] = [Self::LowPower, Self::HighPerformance];

    /// Stable string for TOML / UI (`low_power` / `high_performance`).
    pub fn as_persist_str(self) -> &'static str {
        match self {
            Self::LowPower => "low_power",
            Self::HighPerformance => "high_performance",
        }
    }

    /// Parses case-insensitive persisted or UI token.
    pub fn from_persist_str(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "low_power" | "low" => Some(Self::LowPower),
            "high_performance" | "high" | "performance" => Some(Self::HighPerformance),
            _ => None,
        }
    }

    /// Label for developer UI.
    pub fn label(self) -> &'static str {
        match self {
            Self::LowPower => "Low power",
            Self::HighPerformance => "High performance",
        }
    }
}

/// Debug and diagnostics flags. Persisted as `[debug]`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DebugSettings {
    /// When the `-LogLevel` CLI argument is **not** present, selects [`logger::LogLevel::Trace`] if true or
    /// [`logger::LogLevel::Debug`] if false. If `-LogLevel` is present, it always overrides this flag.
    pub log_verbose: bool,
    /// GPU power preference hint for adapter selection (see [`PowerPreferenceSetting`]).
    pub power_preference: PowerPreferenceSetting,
    /// When true, request backend validation (e.g. Vulkan validation layers) via wgpu instance
    /// flags. Significantly slows rendering; use only when debugging GPU API misuse. Default false. Applies to both desktop
    /// wgpu init and the OpenXR Vulkan / wgpu-hal bootstrap. Native **stdout** and **stderr** are
    /// forwarded to the renderer log file after logging starts (see [`crate::app::run`]), so layer
    /// and spirv-val output is captured regardless of this flag.
    /// Applied when the GPU stack is first created, not on later config updates.
    /// [`crate::config::apply_renderide_gpu_validation_env`] and `WGPU_*` environment variables can still adjust
    /// flags at process start.
    pub gpu_validation_layers: bool,
    /// When true, show the **Frame timing** ImGui window (FPS and CPU/GPU submit-interval metrics). Cheap snapshot;
    /// independent of [`Self::debug_hud_enabled`]. Default true.
    #[serde(default = "default_debug_hud_frame_timing")]
    pub debug_hud_frame_timing: bool,
    /// When true, show **Renderide debug** (Stats / Shader routes) and run mesh-draw stats, frame diagnostics, and
    /// renderer info capture. Default false (performance-first; **Renderer config** or `debug_hud_enabled` in config).
    pub debug_hud_enabled: bool,
    /// When true, capture [`crate::diagnostics::SceneTransformsSnapshot`] each frame and show the **Scene transforms**
    /// ImGui window (can be expensive on large scenes). Independent of [`Self::debug_hud_enabled`] so you can enable
    /// transforms inspection without the main debug panels. Default false.
    pub debug_hud_transforms: bool,
}

impl Default for DebugSettings {
    fn default() -> Self {
        Self {
            log_verbose: false,
            power_preference: PowerPreferenceSetting::default(),
            gpu_validation_layers: false,
            debug_hud_frame_timing: true,
            debug_hud_enabled: false,
            debug_hud_transforms: false,
        }
    }
}

fn default_debug_hud_frame_timing() -> bool {
    true
}

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
}

impl RendererSettings {
    /// Hardcoded defaults only.
    pub fn from_defaults() -> Self {
        Self::default()
    }
}
