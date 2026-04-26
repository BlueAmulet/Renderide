//! Error returned when a [`super::memory_packer::MemoryPacker`] runs out of buffer space.

use thiserror::Error;

/// Failure encountered while packing into a fixed-size IPC buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum MemoryPackError {
    /// One or more writes were skipped because the destination buffer ran out of space.
    ///
    /// The first `needed`/`remaining` pair is captured at the point of overflow; subsequent
    /// writes silently no-op so the encoder cursor remains coherent.
    #[error(
        "packer buffer too small: needed {needed} byte(s) for {ty}, {remaining} byte(s) remaining"
    )]
    BufferTooSmall {
        /// Short type name of the value whose write first ran out of room.
        ty: &'static str,
        /// Bytes the offending write required.
        needed: usize,
        /// Bytes still free at the moment of the offending write.
        remaining: usize,
    },
}
