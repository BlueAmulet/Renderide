//! Consumer side of the shared-memory queue.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::circular_buffer;
use crate::layout::{
    padded_message_length, MessageHeader, MESSAGE_BODY_OFFSET, STATE_LOCKED, STATE_READY,
    TICKS_FOR_TEN_SECONDS,
};
use crate::options::QueueOptions;
use crate::queue_resources::QueueResources;
use crate::QueueHeader;

/// `DateTime.UtcNow.Ticks` value at the Unix epoch (100 ns ticks since 0001-01-01 UTC).
const DOTNET_TICKS_AT_UNIX_EPOCH: i64 = 621_355_968_000_000_000;

/// Starting value for the contention counter in blocking [`Subscriber::dequeue`] (managed client parity).
const DEQUEUE_BACKOFF_COUNTER_INITIAL: i32 = -5;

/// After this many backoff steps, use a fixed long semaphore wait instead of ramping wait milliseconds from the counter.
const DEQUEUE_BACKOFF_HEAVY_PHASE_AFTER: i32 = 10;

/// Milliseconds for the steady semaphore wait once past [`DEQUEUE_BACKOFF_HEAVY_PHASE_AFTER`].
const DEQUEUE_BACKOFF_HEAVY_WAIT_MS: u64 = 10;

/// Current instant in the same 100 ns tick domain as .NET `DateTime.UtcNow.Ticks`.
fn utc_now_ticks() -> i64 {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(d) => {
            let since_unix_100ns = (d.as_nanos() / 100) as i64;
            since_unix_100ns.saturating_add(DOTNET_TICKS_AT_UNIX_EPOCH)
        }
        Err(_) => DOTNET_TICKS_AT_UNIX_EPOCH,
    }
}

/// Receives messages from the queue using the same contention and backoff pattern as the managed client.
pub struct Subscriber {
    res: QueueResources,
}

impl Subscriber {
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
    fn buffer_ptr(&self) -> *const u8 {
        self.res.buffer_ptr()
    }

    /// Mutable pointer to the start of the byte ring (after the queue header).
    fn buffer_mut(&mut self) -> *mut u8 {
        self.res.buffer_mut()
    }

    /// Blocks until a message arrives or `cancel` is set, using semaphore-backed backoff.
    pub fn dequeue(&mut self, cancel: &AtomicBool) -> Vec<u8> {
        let mut num = DEQUEUE_BACKOFF_COUNTER_INITIAL;
        loop {
            if let Some(msg) = self.try_dequeue() {
                return msg;
            }
            if cancel.load(Ordering::Relaxed) {
                break;
            }
            if num > DEQUEUE_BACKOFF_HEAVY_PHASE_AFTER {
                self.res
                    .wait_semaphore_timeout(Duration::from_millis(DEQUEUE_BACKOFF_HEAVY_WAIT_MS));
            } else {
                let old_num = num;
                num = num.saturating_add(1);
                if old_num > 0 {
                    self.res
                        .wait_semaphore_timeout(Duration::from_millis(num as u64));
                } else {
                    std::thread::yield_now();
                }
            }
        }
        vec![]
    }

    /// Returns the next message if one is ready; non-blocking aside from contender spin windows.
    pub fn try_dequeue(&mut self) -> Option<Vec<u8>> {
        let header_ptr = self.header_mut();
        let header = unsafe { &*header_ptr };

        if header.is_empty() {
            return None;
        }

        let ticks = utc_now_ticks();
        let read_lock = header.read_lock_timestamp.load(Ordering::SeqCst);
        if ticks - read_lock < TICKS_FOR_TEN_SECONDS {
            return None;
        }

        if header
            .read_lock_timestamp
            .compare_exchange(read_lock, ticks, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return None;
        }

        let result = (|| {
            let header = unsafe { &*header_ptr };
            if header.is_empty() {
                return None;
            }
            let read_offset = header.read_offset.load(Ordering::SeqCst);
            let write_offset = header.write_offset.load(Ordering::SeqCst);
            let msg_header_ptr = unsafe {
                self.buffer_ptr()
                    .add((read_offset % self.res.capacity) as usize)
                    as *const MessageHeader
            };

            let msg = unsafe { &*msg_header_ptr };
            let spin_ticks = ticks;
            loop {
                match msg.state.compare_exchange(
                    STATE_READY,
                    STATE_LOCKED,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                ) {
                    Ok(_) => break,
                    Err(_) => {
                        if utc_now_ticks() - spin_ticks > TICKS_FOR_TEN_SECONDS {
                            header.read_offset.store(write_offset, Ordering::SeqCst);
                            return None;
                        }
                        std::hint::spin_loop();
                    }
                }
            }

            let body_len = msg.body_length as i64;
            let padded = padded_message_length(body_len);

            let body_offset = read_offset + MESSAGE_BODY_OFFSET;
            let body_len_usize = body_len as usize;
            let msg_result = circular_buffer::read(
                self.buffer_ptr(),
                self.res.capacity,
                body_offset,
                body_len_usize,
            );

            circular_buffer::clear(
                self.buffer_mut(),
                self.res.capacity,
                read_offset,
                padded as usize,
            );

            let new_read = (read_offset + padded) % (self.res.capacity * 2);
            header.read_offset.store(new_read, Ordering::SeqCst);

            Some(msg_result)
        })();

        unsafe { &*header_ptr }
            .read_lock_timestamp
            .store(0, Ordering::SeqCst);
        result
    }
}
