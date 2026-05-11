//! Rolling frame-time ring buffer feeding the **Frame timing** HUD sparkline.
//!
//! Decoupled from [`crate::diagnostics::FrameTimingHudSnapshot`] so the runtime can keep its own
//! history independently of any specific snapshot type.

use std::collections::VecDeque;

/// Frametime history length used for the sparkline plot (power of two for predictable caps).
pub const FRAME_TIME_HISTORY_LEN: usize = 128;

/// Rolling frametime ring used to feed the HUD sparkline. Samples are milliseconds.
#[derive(Clone, Debug, Default)]
pub struct FrameTimeHistory {
    samples: VecDeque<f32>,
}

impl FrameTimeHistory {
    /// Empty history (next [`Self::push`] starts filling the ring).
    pub fn new() -> Self {
        Self {
            samples: VecDeque::with_capacity(FRAME_TIME_HISTORY_LEN),
        }
    }

    /// Appends a sample in milliseconds, evicting the oldest when capacity is hit.
    pub fn push(&mut self, ms: f32) {
        if self.samples.len() == FRAME_TIME_HISTORY_LEN {
            self.samples.pop_front();
        }
        self.samples.push_back(ms);
    }

    /// Clones the current samples oldest-first for consumers that want a contiguous slice.
    pub fn to_vec(&self) -> Vec<f32> {
        self.samples.iter().copied().collect()
    }

    /// Number of stored samples (`0..=FRAME_TIME_HISTORY_LEN`).
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// `true` when no samples have been pushed yet.
    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::{FRAME_TIME_HISTORY_LEN, FrameTimeHistory};

    #[test]
    fn history_caps_at_configured_length() {
        let mut h = FrameTimeHistory::new();
        assert!(h.is_empty());
        for i in 0..(FRAME_TIME_HISTORY_LEN + 10) {
            h.push(i as f32);
        }
        assert_eq!(h.len(), FRAME_TIME_HISTORY_LEN);
        let v = h.to_vec();
        assert_eq!(v.first().copied(), Some(10.0));
        assert_eq!(v.last().copied(), Some((FRAME_TIME_HISTORY_LEN + 9) as f32));
    }
}
