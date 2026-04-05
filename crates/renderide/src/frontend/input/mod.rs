//! Window input: accumulate winit events and pack [`InputState`](crate::shared::InputState) for IPC.

mod accumulator;
mod key_map;
mod winit;

pub use accumulator::WindowInputAccumulator;
pub use key_map::winit_key_to_renderite_key;
pub use winit::{apply_device_event, apply_output_state_to_window, apply_window_event};
