//! Physically-based bloom configuration. Persisted as `[post_processing.bloom]`.

use serde::{Deserialize, Serialize};

/// Physically-based bloom configuration.
///
/// Persisted as `[post_processing.bloom]`. Implements the Call of Duty: Advanced Warfare
/// dual-filter technique (13-tap downsample + 3×3 tent upsample) with Karis-average firefly
/// reduction on the first downsample and an energy-conserving composite. Runs **pre-tonemap** so
/// it scatters HDR-linear light; the tonemap pass then compresses the combined value. Defaults
/// match Bevy's `Bloom::NATURAL` preset and are physically grounded — bloom is scatter from
/// optical imperfections, so no brightness threshold is needed for the physically-based path
/// (see `prefilter_threshold`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BloomSettings {
    /// Whether bloom runs in the post-processing chain. Off by default (opt-in).
    pub enabled: bool,
    /// Baseline scattering strength. Sane range roughly `[0.0, 1.0]`; 0.15 is natural-looking.
    /// An intensity of `0.0` gates the pass off even when [`Self::enabled`] is `true`.
    pub intensity: f32,
    /// Extra boost applied to low-frequency (coarse) mips. Valid range `[0.0, 1.0]`. Higher
    /// values produce a more diffused "glow" that spreads further across the image.
    pub low_frequency_boost: f32,
    /// Curvature of the low-frequency boost falloff. Valid range `[0.0, 1.0]`. Higher values
    /// concentrate the boost in the lowest-frequency mips.
    pub low_frequency_boost_curvature: f32,
    /// High-pass cut-off as a fraction of the mip range. `1.0` keeps every mip; smaller values
    /// drop the lowest-frequency (largest) mips entirely, which tightens the scatter radius.
    pub high_pass_frequency: f32,
    /// Soft-knee prefilter threshold applied to the first downsample (in HDR-linear units).
    /// `0.0` disables the prefilter — physically-based bloom scatters all light, so leave this
    /// at 0 for the realistic path and raise it only for stylized looks.
    pub prefilter_threshold: f32,
    /// Softness of the prefilter knee. Valid range `[0.0, 1.0]`. `0.0` is a hard cutoff.
    pub prefilter_threshold_softness: f32,
    /// How the upsample chain composites into the next mip up (and into the scene).
    pub composite_mode: BloomCompositeMode,
    /// Target height (in pixels) of the largest bloom mip. Each subsequent mip halves the
    /// resolution; smaller values are faster but less wide-spread. 512 matches Bevy's default.
    pub max_mip_dimension: u32,
}

impl Default for BloomSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            intensity: 0.5,
            low_frequency_boost: 0.0,
            low_frequency_boost_curvature: 1.0,
            high_pass_frequency: 1.0,
            prefilter_threshold: 0.0,
            prefilter_threshold_softness: 0.0,
            composite_mode: BloomCompositeMode::EnergyConserving,
            max_mip_dimension: 512,
        }
    }
}

/// Blend rule used when upsampling the bloom pyramid and compositing back onto the scene color.
///
/// [`Self::EnergyConserving`] uses `out = src * c + dst * (1 - c)`, so total radiance is
/// preserved — the scattered light is removed from the base image. This is the physically-based
/// path and matches Bevy's default. [`Self::Additive`] uses `out = src * c + dst`, which brightens
/// the scene by adding the scattered contribution on top — louder, stylized look.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BloomCompositeMode {
    /// Energy-conserving blend (physically-based). Default.
    #[default]
    EnergyConserving,
    /// Additive blend (stylized, brightens the scene).
    Additive,
}

impl BloomCompositeMode {
    /// All variants for ImGui combo lists and config round-trip tests.
    pub const ALL: [Self; 2] = [Self::EnergyConserving, Self::Additive];

    /// Short label for the renderer config window.
    pub fn label(self) -> &'static str {
        match self {
            Self::EnergyConserving => "Energy-Conserving (physical)",
            Self::Additive => "Additive (stylized)",
        }
    }
}
