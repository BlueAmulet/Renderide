//! Shared backing mapping, capacity, semaphore, and Unix `destroy_on_dispose` cleanup for both queue ends.

use std::fs;

use crate::error::OpenError;
use crate::layout::QueueHeader;
use crate::memory::SharedMapping;
use crate::options::QueueOptions;
use crate::semaphore::Semaphore;

/// Common state opened by [`crate::Publisher::new`] and [`crate::Subscriber::new`].
pub(crate) struct QueueResources {
    mapping: SharedMapping,
    /// Ring capacity in bytes (user data only).
    pub(crate) capacity: i64,
    sem: Semaphore,
    destroy_on_dispose: bool,
}

impl QueueResources {
    /// Creates or opens the mapping and paired semaphore described by `options`.
    pub(crate) fn open(options: QueueOptions) -> Result<Self, OpenError> {
        let (mapping, sem) = SharedMapping::open_queue(&options)?;
        Ok(Self {
            mapping,
            capacity: options.capacity,
            sem,
            destroy_on_dispose: options.destroy_on_dispose,
        })
    }

    /// Pointer to the shared [`QueueHeader`] at the start of the mapping.
    pub(crate) fn header_mut(&mut self) -> *mut QueueHeader {
        self.mapping.as_mut_ptr() as *mut QueueHeader
    }

    /// Pointer to the start of the byte ring (after the queue header).
    ///
    /// # Safety
    ///
    /// [`crate::layout::BUFFER_BYTE_OFFSET`] is within the length of the mapping opened for this queue.
    pub(crate) fn buffer_ptr(&self) -> *const u8 {
        unsafe {
            self.mapping
                .as_ptr()
                .byte_add(crate::layout::BUFFER_BYTE_OFFSET)
        }
    }

    /// Mutable pointer to the start of the byte ring (after the queue header).
    ///
    /// # Safety
    ///
    /// Same as [`Self::buffer_ptr`].
    pub(crate) fn buffer_mut(&mut self) -> *mut u8 {
        unsafe {
            self.mapping
                .as_mut_ptr()
                .byte_add(crate::layout::BUFFER_BYTE_OFFSET)
        }
    }

    /// Wakeup primitive paired with the queue (signal after enqueue).
    pub(crate) fn post(&self) {
        self.sem.post();
    }

    /// Blocks up to `timeout` waiting for a post (used by blocking dequeue).
    pub(crate) fn wait_semaphore_timeout(&self, timeout: std::time::Duration) -> bool {
        self.sem.wait_timeout(timeout)
    }
}

impl Drop for QueueResources {
    fn drop(&mut self) {
        if self.destroy_on_dispose {
            if let Some(path) = self.mapping.backing_file_path() {
                let _ = fs::remove_file(path);
            }
        }
    }
}
