//! Adapts winit 0.30 events into [`WindowInputAccumulator`](super::WindowInputAccumulator).

use std::path::Path;

use winit::dpi::PhysicalPosition;
use winit::event::{DeviceEvent, ElementState, Ime, MouseButton, MouseScrollDelta, WindowEvent};
use winit::window::{CursorGrabMode, Window};

use super::accumulator::WindowInputAccumulator;
use super::key_map::winit_key_to_renderite_key;
use crate::shared::OutputState;

/// Applies a [`WindowEvent`] from winit to the accumulator.
pub fn apply_window_event(acc: &mut WindowInputAccumulator, event: &WindowEvent) {
    match event {
        WindowEvent::Resized(size) => {
            acc.window_resolution = (size.width, size.height);
        }
        WindowEvent::CursorMoved { position, .. } => {
            acc.set_cursor_logical(position.x, position.y);
        }
        WindowEvent::CursorEntered { .. } => acc.mouse_active = true,
        WindowEvent::CursorLeft { .. } => acc.mouse_active = false,
        WindowEvent::Focused(focused) => acc.window_focused = *focused,
        WindowEvent::MouseInput { state, button, .. } => {
            let pressed = *state == ElementState::Pressed;
            match button {
                MouseButton::Left => acc.left_held = pressed,
                MouseButton::Right => acc.right_held = pressed,
                MouseButton::Middle => acc.middle_held = pressed,
                MouseButton::Back => acc.button4_held = pressed,
                MouseButton::Forward => acc.button5_held = pressed,
                MouseButton::Other(_) => {}
            }
        }
        WindowEvent::MouseWheel { delta, .. } => {
            const SCROLL_SCALE: f32 = 120.0;
            match delta {
                MouseScrollDelta::LineDelta(x, y) => {
                    acc.scroll_delta.x += *x * SCROLL_SCALE;
                    acc.scroll_delta.y += *y * SCROLL_SCALE;
                }
                MouseScrollDelta::PixelDelta(p) => {
                    acc.scroll_delta.x += p.x as f32;
                    acc.scroll_delta.y += p.y as f32;
                }
            }
        }
        WindowEvent::KeyboardInput {
            event,
            is_synthetic,
            ..
        } => {
            if *is_synthetic {
                return;
            }
            if event.repeat {
                return;
            }
            if let Some(key) = winit_key_to_renderite_key(event.physical_key) {
                match event.state {
                    ElementState::Pressed => {
                        if !acc.held_keys.contains(&key) {
                            acc.held_keys.push(key);
                        }
                        if let Some(text) = event.text.as_ref() {
                            if !text.is_empty() {
                                acc.push_key_text(text.as_str());
                            }
                        }
                    }
                    ElementState::Released => {
                        acc.held_keys.retain(|held| *held != key);
                    }
                }
            } else if event.state == ElementState::Pressed {
                if let Some(text) = event.text.as_ref() {
                    if !text.is_empty() {
                        acc.push_key_text(text.as_str());
                    }
                }
            }
        }
        WindowEvent::Ime(ime) => match ime {
            Ime::Commit(s) => acc.push_ime_commit(s.as_str()),
            Ime::Enabled | Ime::Disabled | Ime::Preedit(_, _) => {}
        },
        WindowEvent::DroppedFile(path) => {
            acc.push_dropped_file_path(path_to_string_lossy(path));
        }
        _ => {}
    }
}

fn path_to_string_lossy(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

/// Applies relative pointer motion when the cursor is captured (locked / confined).
pub fn apply_device_event(acc: &mut WindowInputAccumulator, event: &DeviceEvent) {
    if let DeviceEvent::MouseMotion { delta } = event {
        acc.mouse_delta.x += delta.0 as f32;
        acc.mouse_delta.y -= delta.1 as f32;
    }
}

/// Applies host [`OutputState`] to the winit window (cursor grab, IME, optional warps).
pub fn apply_output_state_to_window(
    window: &Window,
    state: &OutputState,
) -> Result<(), winit::error::ExternalError> {
    window.set_ime_allowed(state.keyboard_input_active);

    if state.lock_cursor {
        window
            .set_cursor_grab(CursorGrabMode::Locked)
            .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined))?;
        window.set_cursor_visible(false);
        if let Some(p) = state.lock_cursor_position {
            let pos = PhysicalPosition::new(p.x, p.y);
            let _ = window.set_cursor_position(pos);
        }
    } else {
        window.set_cursor_grab(CursorGrabMode::None)?;
        window.set_cursor_visible(true);
    }
    Ok(())
}
