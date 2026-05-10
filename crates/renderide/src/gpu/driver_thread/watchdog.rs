//! Watchdog for blocking GPU driver-thread companion calls.
//!
//! Used by OpenXR calls that may block on the compositor so a stalled runtime surfaces in
//! `logs/renderer/*.log` instead of silently freezing the frame loop. The watchdog observes but
//! cannot interrupt blocking external calls.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

/// Arms a background thread that logs if [`Self::disarm`] is not called within `timeout`.
pub(crate) struct BlockingCallWatchdog {
    /// Sender whose disconnect signals the worker to exit.
    tx: Option<mpsc::SyncSender<()>>,
    /// Joined during [`Self::disarm`] so the worker observes disconnect before return.
    handle: Option<thread::JoinHandle<()>>,
}

impl BlockingCallWatchdog {
    /// Spawns the watchdog thread. The returned guard must be disarmed within `timeout`.
    pub(crate) fn arm(timeout: Duration, label: &'static str) -> Self {
        Self::arm_inner(timeout, label, None)
    }

    /// Spawns a watchdog that lowers stall severity after cooperative shutdown starts.
    pub(crate) fn arm_shutdown_aware(
        timeout: Duration,
        label: &'static str,
        shutdown_requested: Arc<AtomicBool>,
    ) -> Self {
        Self::arm_inner(timeout, label, Some(shutdown_requested))
    }

    fn arm_inner(
        timeout: Duration,
        label: &'static str,
        shutdown_requested: Option<Arc<AtomicBool>>,
    ) -> Self {
        let (tx, rx) = mpsc::sync_channel::<()>(0);
        let handle = thread::Builder::new()
            .name(format!("gpu-blocking-call-watchdog:{label}"))
            .spawn(move || match rx.recv_timeout(timeout) {
                Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => {}
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    log_watchdog_timeout(label, timeout, shutdown_requested.as_deref());
                    let _ = rx.recv();
                }
            })
            .ok();
        Self {
            tx: Some(tx),
            handle,
        }
    }

    /// Signals the worker to exit and waits for it to observe the disconnect.
    pub(crate) fn disarm(mut self) {
        drop(self.tx.take());
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn log_watchdog_timeout(
    label: &'static str,
    timeout: Duration,
    shutdown_requested: Option<&AtomicBool>,
) {
    if shutdown_requested.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        logger::warn!(
            "{label} exceeded {}ms during shutdown -- external runtime may be stalled",
            timeout.as_millis()
        );
        return;
    }
    logger::error!(
        "{label} exceeded {}ms -- external runtime may be stalled",
        timeout.as_millis()
    );
}

impl Drop for BlockingCallWatchdog {
    fn drop(&mut self) {
        drop(self.tx.take());
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;

    #[test]
    fn disarm_before_timeout_does_not_block() {
        let start = Instant::now();
        let watchdog = BlockingCallWatchdog::arm(Duration::from_secs(5), "test_disarm");
        watchdog.disarm();
        assert!(
            start.elapsed() < Duration::from_millis(500),
            "disarm should return promptly"
        );
    }

    #[test]
    fn drop_without_disarm_does_not_hang() {
        let start = Instant::now();
        {
            let _watchdog = BlockingCallWatchdog::arm(Duration::from_secs(5), "test_drop");
        }
        assert!(
            start.elapsed() < Duration::from_millis(500),
            "drop should disarm promptly"
        );
    }

    #[test]
    fn timeout_fires_then_disarm_still_returns() {
        let watchdog = BlockingCallWatchdog::arm(Duration::from_millis(10), "test_timeout");
        thread::sleep(Duration::from_millis(50));
        watchdog.disarm();
    }
}
