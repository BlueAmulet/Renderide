//! Per-frame ImGui IO bridge: feeds [`crate::diagnostics::DebugHudInput`] into the ImGui
//! [`imgui::Io`] before each frame so the HUD's input matches the renderer's logical input.

use imgui::{Io, MouseButton as ImGuiMouseButton};

use crate::diagnostics::DebugHudInput;

/// Wheel notch unit ImGui expects on its mouse-wheel events. Winit reports unscaled deltas;
/// dividing by this constant produces the per-notch step ImGui consumes for its scroll handling.
const WHEEL_UNIT: f32 = 120.0;

/// Feeds winit-derived [`DebugHudInput`] into ImGui `io` before each frame.
///
/// Cursor position is parked off-screen when the host reports the mouse inactive or the window
/// unfocused so that ImGui does not treat stale positions as hovered events.
pub(crate) fn apply_input(io: &mut Io, input: &DebugHudInput) {
    if input.mouse_active && input.window_focused {
        io.add_mouse_pos_event(input.cursor_px);
    } else {
        io.add_mouse_pos_event([-f32::MAX, -f32::MAX]);
    }
    io.add_mouse_button_event(ImGuiMouseButton::Left, input.left);
    io.add_mouse_button_event(ImGuiMouseButton::Right, input.right);
    io.add_mouse_button_event(ImGuiMouseButton::Middle, input.middle);
    io.add_mouse_button_event(ImGuiMouseButton::Extra1, input.extra1);
    io.add_mouse_button_event(ImGuiMouseButton::Extra2, input.extra2);
    io.add_mouse_wheel_event([
        input.mouse_wheel_delta.x / WHEEL_UNIT,
        input.mouse_wheel_delta.y / WHEEL_UNIT,
    ]);
}
