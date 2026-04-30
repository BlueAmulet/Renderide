//! Analog-axis threshold derivation for OpenXR controller state.

use glam::Vec2;

pub(super) fn vec2_nonzero(v: Vec2) -> bool {
    v.length_squared() > 1e-6
}

/// Raw analog axes and boolean touch hints before threshold expansion.
pub(super) struct OpenxrAnalogAxes {
    /// Trigger analog 0..1.
    pub trigger: f32,
    pub trigger_touch: bool,
    pub trigger_click: bool,
    /// Grip / squeeze analog.
    pub squeeze: f32,
    pub squeeze_click: bool,
    pub thumbstick: Vec2,
    pub thumbstick_touch: bool,
    pub trackpad: Vec2,
    pub trackpad_touch: bool,
    pub trackpad_force: f32,
}

/// Host-style booleans inferred from analog axes (Touch / OpenXR conventions).
pub(super) struct OpenxrAxisDerivedButtons {
    pub(super) trigger_touch: bool,
    pub(super) trigger_click: bool,
    pub(super) grip_touch: bool,
    pub(super) grip_click: bool,
    /// Thumbstick deflection or explicit touch bit.
    pub(super) joystick_touch: bool,
    /// Trackpad deflection, touch bit, or force.
    pub(super) touchpad_touch: bool,
}

/// Expands analog thresholds into touch/click flags used across controller profiles.
pub(super) fn derive_openxr_axis_button_flags(
    analog: &OpenxrAnalogAxes,
) -> OpenxrAxisDerivedButtons {
    OpenxrAxisDerivedButtons {
        trigger_touch: analog.trigger_touch || analog.trigger > 0.01,
        trigger_click: analog.trigger_click || analog.trigger > 0.75,
        grip_touch: analog.squeeze_click || analog.squeeze > 0.05,
        grip_click: analog.squeeze_click || analog.squeeze > 0.85,
        joystick_touch: analog.thumbstick_touch || vec2_nonzero(analog.thumbstick),
        touchpad_touch: analog.trackpad_touch
            || vec2_nonzero(analog.trackpad)
            || analog.trackpad_force > 0.01,
    }
}
