//! Mutable per-frame UI state for HUD windows: open flags and tab filters.
//!
//! Lives on [`crate::diagnostics::DebugHud`] so window bodies can borrow exactly the field they
//! need without the host struct exposing seven independent booleans.

/// Per-window open flags and per-tab filter toggles owned by [`crate::diagnostics::DebugHud`].
///
/// Defaults match prior behavior:
/// - All open flags start `true` so the windows appear on first launch.
/// - All filter toggles start `false` (no narrowing applied).
#[derive(Clone, Copy, Debug)]
pub struct HudUiState {
    /// Whether the **Scene transforms** window is open.
    pub scene_transforms_open: bool,
    /// Whether the **Textures** window is open.
    pub texture_debug_open: bool,
    /// Show only textures referenced by the current view in the **Textures** window.
    pub texture_debug_current_view_only: bool,
    /// Show only overlay/UI-ish draws in the **Draw state** tab.
    pub draw_state_ui_only: bool,
    /// Show only material rows with render-state overrides in the **Draw state** tab.
    pub draw_state_only_overrides: bool,
    /// Show only fallback shader routes in the **Shader routes** tab.
    pub shader_routes_only_fallback: bool,
    /// Whether the **Renderer config** window is open.
    pub renderer_config_open: bool,
}

impl Default for HudUiState {
    fn default() -> Self {
        Self {
            scene_transforms_open: true,
            texture_debug_open: true,
            texture_debug_current_view_only: false,
            draw_state_ui_only: false,
            draw_state_only_overrides: false,
            shader_routes_only_fallback: false,
            renderer_config_open: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::HudUiState;

    #[test]
    fn default_opens_every_window_and_disables_every_filter() {
        let s = HudUiState::default();
        assert!(s.scene_transforms_open);
        assert!(s.texture_debug_open);
        assert!(s.renderer_config_open);
        assert!(!s.texture_debug_current_view_only);
        assert!(!s.draw_state_ui_only);
        assert!(!s.draw_state_only_overrides);
        assert!(!s.shader_routes_only_fallback);
    }
}
