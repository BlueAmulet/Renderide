//! Platform-neutral input accumulation for [`FrameStartData`](crate::shared::FrameStartData).

use nalgebra::Vector2;

use crate::shared::{DragAndDropEvent, InputState, Key, KeyboardState, MouseState, WindowState};

/// Holds window / pointer / keyboard state between winit events and host lock-step [`InputState`] snapshots.
///
/// Mouse and scroll deltas accumulate until [`Self::take_input_state`] (called when sending
/// `frame_start_data`), matching the historical Unity renderer begin-frame timing.
pub struct WindowInputAccumulator {
    /// Accumulated relative motion (including [`DeviceEvent::MouseMotion`](winit::event::DeviceEvent)).
    pub mouse_delta: Vector2<f32>,
    /// Accumulated scroll wheel / trackpad scroll since the last [`Self::take_input_state`].
    pub scroll_delta: Vector2<f32>,
    /// Pointer position in window space (logical pixels) for [`crate::shared::MouseState`].
    pub window_position: Vector2<f32>,
    /// Last known drawable size in physical pixels.
    pub window_resolution: (u32, u32),
    /// Left mouse button held.
    pub left_held: bool,
    /// Right mouse button held.
    pub right_held: bool,
    /// Middle mouse button held.
    pub middle_held: bool,
    /// Fourth mouse button (back) held.
    pub button4_held: bool,
    /// Fifth mouse button (forward) held.
    pub button5_held: bool,
    /// Whether the cursor is inside the client area.
    pub mouse_active: bool,
    /// Whether the window is focused.
    pub window_focused: bool,
    /// Keys currently held, in host [`Key`] form.
    pub held_keys: Vec<Key>,
    /// Text committed by IME since the last IPC snapshot (`KeyboardState.type_delta`).
    ime_commit_buffer: String,
    /// Single-character text from key events (supplements IME for simple typing).
    text_typing_buffer: String,
    /// Paths from [`WindowEvent::DroppedFile`](winit::event::WindowEvent) coalesced until take.
    pending_drop_paths: Vec<String>,
    /// Last cursor position in physical pixels (for drop-point reporting).
    last_cursor_pixel: Vector2<i32>,
}

impl Default for WindowInputAccumulator {
    fn default() -> Self {
        Self {
            mouse_delta: Vector2::zeros(),
            scroll_delta: Vector2::zeros(),
            window_position: Vector2::zeros(),
            window_resolution: (0, 0),
            left_held: false,
            right_held: false,
            middle_held: false,
            button4_held: false,
            button5_held: false,
            mouse_active: false,
            window_focused: true,
            held_keys: Vec::new(),
            ime_commit_buffer: String::new(),
            text_typing_buffer: String::new(),
            pending_drop_paths: Vec::new(),
            last_cursor_pixel: Vector2::zeros(),
        }
    }
}

impl WindowInputAccumulator {
    /// Records IME-composed text committed by the platform.
    pub fn push_ime_commit(&mut self, text: &str) {
        self.ime_commit_buffer.push_str(text);
    }

    /// Records printable text associated with a key press (not repeats).
    pub fn push_key_text(&mut self, text: &str) {
        self.text_typing_buffer.push_str(text);
    }

    /// Records a file dropped onto the window; paths are batched into the next [`InputState`].
    pub fn push_dropped_file_path(&mut self, path_str: String) {
        self.pending_drop_paths.push(path_str);
    }

    /// Updates logical cursor position used for [`MouseState::window_position`] / desktop mirror.
    pub fn set_cursor_logical(&mut self, x: f64, y: f64) {
        self.window_position.x = x as f32;
        self.window_position.y = y as f32;
        self.last_cursor_pixel.x = x.round() as i32;
        self.last_cursor_pixel.y = y.round() as i32;
    }

    /// Consumes accumulated deltas and returns an [`InputState`] for the host.
    ///
    /// `host_requests_cursor_lock`: merged into [`MouseState::is_active`] (Unity / old session parity).
    pub fn take_input_state(&mut self, host_requests_cursor_lock: bool) -> InputState {
        let type_delta = {
            let mut out = String::new();
            out.push_str(&std::mem::take(&mut self.ime_commit_buffer));
            out.push_str(&std::mem::take(&mut self.text_typing_buffer));
            if out.is_empty() {
                None
            } else {
                Some(out)
            }
        };
        let drag_and_drop_event = self.take_drag_and_drop_if_any();

        let mouse = MouseState {
            is_active: self.mouse_active || host_requests_cursor_lock,
            left_button_state: self.left_held,
            right_button_state: self.right_held,
            middle_button_state: self.middle_held,
            button4_state: self.button4_held,
            button5_state: self.button5_held,
            desktop_position: self.window_position,
            window_position: self.window_position,
            direct_delta: std::mem::take(&mut self.mouse_delta),
            scroll_wheel_delta: std::mem::take(&mut self.scroll_delta),
        };
        let window = WindowState {
            is_window_focused: self.window_focused,
            is_fullscreen: false,
            window_resolution: Vector2::new(
                self.window_resolution.0 as i32,
                self.window_resolution.1 as i32,
            ),
            resolution_settings_applied: false,
            drag_and_drop_event,
        };
        let keyboard = Some(KeyboardState {
            type_delta,
            held_keys: self.held_keys.clone(),
        });
        InputState {
            mouse: Some(mouse),
            keyboard,
            window: Some(window),
            vr: None,
            gamepads: Vec::new(),
            touches: Vec::new(),
            displays: Vec::new(),
        }
    }

    fn take_drag_and_drop_if_any(&mut self) -> Option<DragAndDropEvent> {
        if self.pending_drop_paths.is_empty() {
            return None;
        }
        let paths = std::mem::take(&mut self.pending_drop_paths)
            .into_iter()
            .map(Some)
            .collect();
        Some(DragAndDropEvent {
            paths,
            drop_point: self.last_cursor_pixel,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::WindowInputAccumulator;
    use nalgebra::Vector2;

    #[test]
    fn mouse_delta_accumulates_until_take_input_state() {
        let mut w = WindowInputAccumulator::default();
        w.mouse_delta += Vector2::new(1.0, 2.0);
        w.mouse_delta += Vector2::new(3.0, 4.0);
        let first = w.take_input_state(false);
        let mouse = first.mouse.expect("mouse state");
        assert_eq!(mouse.direct_delta.x, 4.0);
        assert_eq!(mouse.direct_delta.y, 6.0);
        let second = w.take_input_state(false);
        let mouse2 = second.mouse.expect("mouse state");
        assert_eq!(mouse2.direct_delta.x, 0.0);
        assert_eq!(mouse2.direct_delta.y, 0.0);
    }

    #[test]
    fn scroll_delta_accumulates_until_take_input_state() {
        let mut w = WindowInputAccumulator::default();
        w.scroll_delta += Vector2::new(0.0, 120.0);
        w.scroll_delta += Vector2::new(0.0, 60.0);
        let taken = w.take_input_state(false);
        let mouse = taken.mouse.expect("mouse state");
        assert_eq!(mouse.scroll_wheel_delta.y, 180.0);
    }

    #[test]
    fn cursor_lock_merges_into_mouse_active() {
        let mut w = WindowInputAccumulator {
            mouse_active: false,
            ..Default::default()
        };
        let s = w.take_input_state(true);
        assert!(s.mouse.expect("mouse").is_active);
    }

    #[test]
    fn ime_and_text_merge_into_type_delta() {
        let mut w = WindowInputAccumulator::default();
        w.push_ime_commit("hello");
        w.push_key_text("!");
        let s = w.take_input_state(false);
        assert_eq!(
            s.keyboard.expect("kb").type_delta.as_deref(),
            Some("hello!")
        );
        let s2 = w.take_input_state(false);
        assert!(s2.keyboard.expect("kb").type_delta.is_none());
    }
}
