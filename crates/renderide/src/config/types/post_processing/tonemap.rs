//! Tonemapping configuration. Persisted as `[post_processing.tonemap]`.

use serde::{Deserialize, Serialize};

/// Tonemapping configuration. Persisted as `[post_processing.tonemap]`.
///
/// Tonemapping converts unbounded HDR scene-referred radiance to a bounded display-referred linear
/// signal. Output values are in `[0, 1]` linear sRGB so the existing sRGB swapchain encodes gamma
/// correctly without a separate gamma pass.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TonemapSettings {
    /// Selected tonemapping curve (see [`TonemapMode`]).
    pub mode: TonemapMode,
}

/// Tonemapping curve selector for [`TonemapSettings::mode`].
///
/// Adding a new variant only requires extending [`Self::ALL`], [`Self::label`] and any new
/// post-processing pass that consumes it; the chain signature in
/// [`crate::render_graph::cache::PostProcessChainSignature`] does not need to change unless the
/// new mode introduces additional render-graph passes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TonemapMode {
    /// No tonemapping (raw HDR is passed through, identical to the master-disabled path but kept
    /// as an explicit option so the master toggle can stay enabled while only other future
    /// effects run).
    None,
    /// Stephen Hill ACES Fitted (sRGB → AP1, RRT+ODT, AP1 → sRGB). High-quality reference curve
    /// used by Bevy and Unity HDRP.
    #[default]
    AcesFitted,
}

impl TonemapMode {
    /// All variants for ImGui combo lists and config round-trip tests.
    pub const ALL: [Self; 2] = [Self::None, Self::AcesFitted];

    /// Short label for the renderer config window.
    pub fn label(self) -> &'static str {
        match self {
            Self::None => "None (HDR pass-through)",
            Self::AcesFitted => "ACES Fitted (Hill)",
        }
    }
}
