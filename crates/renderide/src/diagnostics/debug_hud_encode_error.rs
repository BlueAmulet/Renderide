//! Errors when encoding the Dear ImGui / wgpu debug HUD pass.

use thiserror::Error;

/// Failure during ImGui draw-list submission or related setup for the overlay pass.
#[derive(Debug, Error)]
pub enum DebugHudEncodeError {
    /// The wgpu renderer for ImGui returned an error string.
    #[error("imgui-wgpu render: {0}")]
    ImguiWgpu(String),
}
