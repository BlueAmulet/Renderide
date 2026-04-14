//! Producer side of the shared-memory queue.

use std::fs;
use std::sync::atomic::Ordering;

use crate::circular_buffer;
use crate::layout::{
    padded_message_length, MessageHeader, MESSAGE_BODY_OFFSET, STATE_READY, STATE_WRITING,
};
use crate::memory::SharedMapping;
use crate::options::QueueOptions;
use crate::semaphore::Semaphore;
use crate::QueueHeader;

/// Sends messages into the queue; signals the paired semaphore after each successful enqueue.
pub struct Publisher {
    mapping: SharedMapping,
    capacity: i64,
    sem: Semaphore,
    destroy_on_dispose: bool,
}

impl Publisher {
    /// Opens the backing mapping and semaphore.
    pub fn new(options: QueueOptions) -> Result<Self, crate::OpenError> {
        let (mapping, sem) = SharedMapping::open_queue(&options)?;
        Ok(Self {
            mapping,
            capacity: options.capacity,
            sem,
            destroy_on_dispose: options.destroy_on_dispose,
        })
    }

    /// Pointer to the shared [`crate::QueueHeader`] at the start of the mapping.
    fn header_mut(&mut self) -> *mut QueueHeader {
        self.mapping.as_mut_ptr() as *mut QueueHeader
    }

    /// Pointer to the start of the byte ring (after the queue header).
    fn buffer_mut(&mut self) -> *mut u8 {
        unsafe {
            self.mapping
                .as_mut_ptr()
                .add(crate::layout::BUFFER_BYTE_OFFSET)
        }
    }

    /// Returns `true` if the ring has enough contiguous logical space for `message_len` (padded).
    fn check_capacity(&self, header: &QueueHeader, message_len: i64) -> bool {
        if message_len > self.capacity {
            return false;
        }
        if header.is_empty() {
            return true;
        }
        let read_phys = header.read_offset % self.capacity;
        let write_phys = header.write_offset % self.capacity;
        if read_phys == write_phys {
            return false;
        }
        let available = if read_phys < write_phys {
            self.capacity - write_phys + read_phys
        } else {
            read_phys - write_phys
        };
        message_len <= available
    }

    /// Pushes one message; returns `false` when the ring has insufficient free space.
    pub fn try_enqueue(&mut self, message: &[u8]) -> bool {
        let len = message.len() as i64;
        let padded = padded_message_length(len);
        let header_ptr = self.header_mut();

        loop {
            let header = unsafe { &*header_ptr };
            if !self.check_capacity(header, padded) {
                return false;
            }
            let write_offset = header.write_offset;
            let new_write = (write_offset + padded) % (self.capacity * 2);

            let write_offset_ptr = unsafe {
                &*(&(*header_ptr).write_offset as *const i64 as *const std::sync::atomic::AtomicI64)
            };
            let prev = write_offset_ptr.compare_exchange(
                write_offset,
                new_write,
                Ordering::SeqCst,
                Ordering::SeqCst,
            );
            if prev.is_err() {
                continue;
            }

            let msg_header = MessageHeader {
                state: STATE_WRITING,
                body_length: len as i32,
            };
            circular_buffer::write(
                self.buffer_mut(),
                self.capacity,
                write_offset,
                bytemuck::bytes_of(&msg_header),
            );

            circular_buffer::write(
                self.buffer_mut(),
                self.capacity,
                write_offset + MESSAGE_BODY_OFFSET,
                message,
            );

            let state_bytes = STATE_READY.to_le_bytes();
            circular_buffer::write(self.buffer_mut(), self.capacity, write_offset, &state_bytes);

            self.sem.post();
            return true;
        }
    }
}

impl Drop for Publisher {
    fn drop(&mut self) {
        if self.destroy_on_dispose {
            if let Some(path) = self.mapping.backing_file_path() {
                let _ = fs::remove_file(path);
            }
        }
    }
}

/// Shared-memory queues are process-wide handles; treat ownership as non-`Sync` socket-style.
///
/// # Safety
///
/// The mapping is owned by this process and may be sent to another thread that owns the
/// [`Publisher`]. The same synchronization rules as the managed implementation apply.
unsafe impl Send for Publisher {}
