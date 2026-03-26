//! Diagnostic helpers and optional ImGui on-screen HUD.
//!
//! The HUD is enabled by the `debug-hud` Cargo feature (on by default). Disable default features
//! (`cargo build -p renderide --no-default-features`) for lean builds without `imgui` / `imgui-wgpu`.
//!
//! Submodules: [`live_frame`] (per-frame snapshot types), [`hud`] ([`DebugHud`]), [`helpers`]
//! (IPC drop throttling and log-on-change).

pub mod helpers;
pub mod hud;
pub mod live_frame;

pub use helpers::{DropLogEvent, LogOnChange, ThrottledDropLog};
pub use hud::DebugHud;
pub use live_frame::{GpuAllocatorSnapshot, HostCpuMemorySnapshot, LiveFrameDiagnostics};
