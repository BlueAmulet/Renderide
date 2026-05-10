//! Per-frame recovery policy for CPU-mapped GPU buffers.

use super::GpuContext;

const MAPPED_BUFFER_RECOVERY_FRAMES: u8 = 2;

/// Result of beginning one mapped-buffer recovery frame.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct MappedBufferRecoveryFrame {
    /// Current invalidation generation after observing the shared health counter.
    pub(crate) generation: u64,
    /// Whether a new invalidation generation was observed at this frame boundary.
    pub(crate) invalidated: bool,
    /// Whether this frame must avoid CPU-mapped staging/readback buffers.
    pub(crate) avoid_mapped_buffers: bool,
}

impl GpuContext {
    /// Records that mapped staging/readback buffers should be discarded before reuse.
    pub(crate) fn mark_mapped_buffers_invalid(&self, reason: impl AsRef<str>) {
        self.mapped_buffer_health.mark_invalid(reason);
    }

    /// Begins mapped-buffer recovery bookkeeping for a render frame.
    pub(crate) fn begin_mapped_buffer_recovery_frame(&mut self) -> MappedBufferRecoveryFrame {
        let generation = self.mapped_buffer_health.generation();
        let invalidated = self.observe_generation(generation);

        self.avoid_mapped_buffers_this_frame = self.mapped_buffer_recovery_frames_remaining > 0;
        if self.mapped_buffer_recovery_frames_remaining > 0 {
            self.mapped_buffer_recovery_frames_remaining = self
                .mapped_buffer_recovery_frames_remaining
                .saturating_sub(1);
        }

        MappedBufferRecoveryFrame {
            generation,
            invalidated,
            avoid_mapped_buffers: self.avoid_mapped_buffers_this_frame,
        }
    }

    /// Observes invalidations reported by wgpu while the current frame is already running.
    pub(crate) fn observe_mapped_buffer_invalidation_during_frame(&mut self) -> bool {
        let generation = self.mapped_buffer_health.generation();
        let invalidated = self.observe_generation(generation);
        if invalidated {
            self.avoid_mapped_buffers_this_frame = true;
        }
        invalidated
    }

    /// Whether this frame should avoid CPU-mapped staging/readback buffers.
    pub(crate) fn avoid_mapped_buffers_this_frame(&self) -> bool {
        self.avoid_mapped_buffers_this_frame
    }

    /// Current mapped-buffer invalidation generation.
    pub(crate) fn mapped_buffer_invalidation_generation(&self) -> u64 {
        self.mapped_buffer_health.generation()
    }

    fn observe_generation(&mut self, generation: u64) -> bool {
        if generation == self.mapped_buffer_invalidation_seen_generation {
            return false;
        }
        self.mapped_buffer_invalidation_seen_generation = generation;
        self.mapped_buffer_recovery_frames_remaining = MAPPED_BUFFER_RECOVERY_FRAMES;
        true
    }
}
