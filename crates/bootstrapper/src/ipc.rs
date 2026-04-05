//! Bootstrapper IPC queue pair (`bootstrapper_in` / `bootstrapper_out`).

use interprocess::{Publisher, QueueFactory, QueueOptions, Subscriber};

use crate::BootstrapError;

/// Cloudtoid queue capacity for user-visible bytes (matches ResoBoot `8192`).
pub const BOOTSTRAP_QUEUE_CAPACITY: i64 = 8192;

/// Subscriber + publisher pair used for Host ↔ bootstrapper messaging.
pub struct BootstrapQueues {
    /// Host → bootstrapper (`*_in` from Host’s perspective).
    pub incoming: Subscriber,
    /// Bootstrapper → Host.
    pub outgoing: Publisher,
}

impl BootstrapQueues {
    /// Opens queues with `destroy_on_dispose` so Wine/Linux backing files can be removed on drop.
    pub fn open(shared_memory_prefix: &str) -> Result<Self, BootstrapError> {
        let incoming_name = format!("{shared_memory_prefix}.bootstrapper_in");
        let outgoing_name = format!("{shared_memory_prefix}.bootstrapper_out");

        let incoming_opts =
            QueueOptions::with_destroy(&incoming_name, BOOTSTRAP_QUEUE_CAPACITY, true).map_err(
                |e| BootstrapError::Interprocess(format!("incoming queue options: {e}")),
            )?;

        let outgoing_opts =
            QueueOptions::with_destroy(&outgoing_name, BOOTSTRAP_QUEUE_CAPACITY, true).map_err(
                |e| BootstrapError::Interprocess(format!("outgoing queue options: {e}")),
            )?;

        let factory = QueueFactory::new();
        let incoming = factory
            .create_subscriber(incoming_opts)
            .map_err(|e| BootstrapError::Interprocess(format!("create_subscriber: {e}")))?;
        let outgoing = factory
            .create_publisher(outgoing_opts)
            .map_err(|e| BootstrapError::Interprocess(format!("create_publisher: {e}")))?;

        Ok(Self { incoming, outgoing })
    }
}
