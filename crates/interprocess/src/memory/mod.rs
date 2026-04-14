//! Shared read/write mapping backing the queue.

#[cfg(unix)]
mod unix;
#[cfg(windows)]
mod windows;

use std::path::PathBuf;

use crate::error::OpenError;
use crate::options::QueueOptions;
use crate::semaphore::Semaphore;

/// Read/write mapping of the queue file (Unix) or named section (Windows), plus a paired semaphore.
///
/// Obtained via [`SharedMapping::open_queue`]; size matches [`QueueOptions::actual_storage_size`].
pub(crate) struct SharedMapping {
    /// Platform-specific mapping implementation.
    #[cfg(unix)]
    inner: unix::UnixMapping,
    /// Platform-specific mapping implementation.
    #[cfg(windows)]
    inner: windows::WindowsMapping,
}

impl SharedMapping {
    /// Creates or opens the backing store for `options` and the companion wakeup semaphore.
    pub(crate) fn open_queue(options: &QueueOptions) -> Result<(Self, Semaphore), OpenError> {
        #[cfg(unix)]
        {
            let (m, s) = unix::open_queue(options)?;
            debug_assert_eq!(m.len(), options.actual_storage_size() as usize);
            Ok((Self { inner: m }, s))
        }
        #[cfg(windows)]
        {
            let (m, s) = windows::open_queue(options)?;
            debug_assert_eq!(m.len(), options.actual_storage_size() as usize);
            Ok((Self { inner: m }, s))
        }
    }

    /// Base pointer to the mapped region (includes [`crate::layout::QueueHeader`] at offset zero).
    pub(crate) fn as_ptr(&self) -> *const u8 {
        self.inner.as_ptr()
    }

    /// Mutable base pointer to the mapped region.
    pub(crate) fn as_mut_ptr(&mut self) -> *mut u8 {
        self.inner.as_mut_ptr()
    }

    /// Path to the backing `.qu` file when file-backed; `None` on Windows named mappings.
    pub(crate) fn backing_file_path(&self) -> Option<&PathBuf> {
        self.inner.backing_file_path()
    }
}
