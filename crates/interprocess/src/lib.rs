//! Cloudtoid.Interprocess-compatible queue for IPC with Resonite host.
//! Uses shared memory and POSIX semaphores on Linux.

mod backend;
mod circular_buffer;
mod publisher;
mod queue;
mod subscriber;

pub use publisher::Publisher;
pub use queue::{QueueFactory, QueueOptions};
pub use subscriber::Subscriber;
