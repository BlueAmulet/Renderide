//! Producer side of the shared-memory queue.

use std::sync::atomic::Ordering;

use crate::circular_buffer;
use crate::layout::{
    message_header_wire_bytes, padded_message_length, MESSAGE_BODY_OFFSET, STATE_READY,
    STATE_WRITING,
};
use crate::options::QueueOptions;
use crate::queue_resources::QueueResources;
use crate::QueueHeader;

/// Sends messages into the queue; signals the paired semaphore after each successful enqueue.
pub struct Publisher {
    res: QueueResources,
}

impl Publisher {
    /// Opens the backing mapping and semaphore.
    pub fn new(options: QueueOptions) -> Result<Self, crate::OpenError> {
        Ok(Self {
            res: QueueResources::open(options)?,
        })
    }

    /// Pointer to the shared [`crate::QueueHeader`] at the start of the mapping.
    fn header_mut(&mut self) -> *mut QueueHeader {
        self.res.header_mut()
    }

    /// Pointer to the start of the byte ring (after the queue header).
    fn buffer_mut(&mut self) -> *mut u8 {
        self.res.buffer_mut()
    }

    /// Returns `true` if the ring has enough contiguous logical space for `message_len` (padded).
    fn check_capacity(&self, header: &QueueHeader, message_len: i64) -> bool {
        if message_len > self.res.capacity {
            return false;
        }
        if header.is_empty() {
            return true;
        }
        let read_phys = header.read_offset.load(Ordering::SeqCst) % self.res.capacity;
        let write_phys = header.write_offset.load(Ordering::SeqCst) % self.res.capacity;
        if read_phys == write_phys {
            return false;
        }
        let available = if read_phys < write_phys {
            self.res.capacity - write_phys + read_phys
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
            let write_offset = header.write_offset.load(Ordering::SeqCst);
            let new_write = (write_offset + padded) % (self.res.capacity * 2);

            let prev = header.write_offset.compare_exchange(
                write_offset,
                new_write,
                Ordering::SeqCst,
                Ordering::SeqCst,
            );
            if prev.is_err() {
                continue;
            }

            let wire = message_header_wire_bytes(STATE_WRITING, len as i32);
            circular_buffer::write(self.buffer_mut(), self.res.capacity, write_offset, &wire);

            circular_buffer::write(
                self.buffer_mut(),
                self.res.capacity,
                write_offset + MESSAGE_BODY_OFFSET,
                message,
            );

            let state_bytes = STATE_READY.to_le_bytes();
            circular_buffer::write(
                self.buffer_mut(),
                self.res.capacity,
                write_offset,
                &state_bytes,
            );

            self.res.post();
            return true;
        }
    }
}
