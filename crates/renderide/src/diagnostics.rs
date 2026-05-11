//! Dear ImGui diagnostics: **Frame timing** ([`crate::config::DebugSettings::debug_hud_frame_timing`]),
//! **Renderide debug** ([`crate::config::DebugSettings::debug_hud_enabled`]: Stats / Shader routes / Draw state / GPU memory),
//! **Scene transforms** ([`crate::config::DebugSettings::debug_hud_transforms`]),
//! and **Textures** ([`crate::config::DebugSettings::debug_hud_textures`]).
//!
//! Also hosts the cooperative renderer hang/hitch detector ([`Watchdog`]).

pub(crate) mod crash_context;
mod ema;
mod encode_error;
mod frame_history;
mod host_metrics;
mod hud;
mod input;
pub(crate) mod log_throttle;
pub mod per_view;
mod snapshots;
mod watchdog;

pub use ema::FrameTimingEma;
pub use encode_error::DebugHudEncodeError;
pub use frame_history::FrameTimeHistory;
pub use host_metrics::HostHudGatherer;
pub use hud::DebugHud;
pub use input::{DebugHudInput, sanitize_input_state_for_imgui_host};
pub use per_view::{PerViewHudConfig, PerViewHudOutputs, PerViewHudOutputsSlot};
pub use snapshots::{
    BackendDiagSnapshot, FrameDiagnosticsIpcQueues, FrameDiagnosticsSnapshot,
    FrameDiagnosticsSnapshotCapture, FrameTimingHudSnapshot, GpuAllocatorHud,
    GpuAllocatorHudRefresh, GpuAllocatorReportHud, RenderSpaceTransformsSnapshot,
    RendererInfoSnapshot, RendererInfoSnapshotCapture, SceneTransformsSnapshot,
    ShaderRouteSnapshot, TextureDebugSnapshot, XrRecoverableFailureCounts,
};
pub use watchdog::{Heartbeat, Watchdog};
