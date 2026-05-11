//! Fence-backed staging-buffer reuse for render-graph uploads.
//!
//! The render graph records many small buffer writes through
//! [`super::frame_upload_batch::FrameUploadBatch`]. This arena owns a small set of persistent
//! `MAP_WRITE | COPY_SRC` buffers that those writes can reuse across frames. A slot is unmapped
//! before submit, marked reusable only after `Queue::on_submitted_work_done` fires, and remapped
//! from the main thread during the next maintenance pass.

mod arena;
mod slot;
mod staging;

pub(crate) use arena::PersistentUploadArena;
pub(crate) use staging::{UploadArenaAcquireStats, UploadArenaPressure};
