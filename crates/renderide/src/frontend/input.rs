//! Window input: accumulate winit events and pack [`InputState`](crate::shared::InputState) for IPC.
//!
//! Layout:
//! - [`accumulator`] -- [`WindowInputAccumulator`] state holder and its per-frame snapshot.
//! - [`cursor`] -- host [`crate::shared::OutputState`] cursor policy and IME/grab helpers.
//! - [`vr_session`] -- VR headset snapshot construction for [`crate::shared::InputState::vr`].
//! - [`winit`] -- winit window/device event adapter and the underlying key-map / transition tables.

mod accumulator;
mod cursor;
mod vr_session;
mod winit;

pub use accumulator::WindowInputAccumulator;
pub use cursor::{
    CursorOutputTracking, apply_output_state_to_window, apply_per_frame_cursor_lock_when_locked,
    enable_ime_on_window,
};
pub use vr_session::vr_inputs_for_session;
pub use winit::{apply_device_event, apply_window_event};
