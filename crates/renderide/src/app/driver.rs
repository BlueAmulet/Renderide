//! Winit application driver state and event-loop integration.

mod events;
mod frame;
mod present;
mod target;
mod xr;

use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use logger::LogLevel;
use winit::event_loop::ActiveEventLoop;

use crate::frontend::input::{CursorOutputTracking, WindowInputAccumulator};
use crate::runtime::RendererRuntime;

use self::target::RenderTarget;
use self::xr::XrInputCache;
use super::bootstrap::{
    ExternalShutdownCoordinator, GpuStartupConfig, effective_renderer_log_level,
};
use super::exit::{ExitReason, ExitState, RunExit};
use super::frame_clock::FrameClock;

/// Interval between log flushes when using file logging in the winit handler.
const LOG_FLUSH_INTERVAL: Duration = Duration::from_secs(1);

/// Winit application handler for the renderer process.
pub(crate) struct AppDriver {
    runtime: RendererRuntime,
    startup_gpu: GpuStartupConfig,
    log_level_cli: Option<LogLevel>,
    target: Option<RenderTarget>,
    exit: ExitState,
    log_flush: LogFlushCadence,
    input: WindowInputAccumulator,
    cursor_output_tracking: CursorOutputTracking,
    frame_clock: FrameClock,
    external_shutdown: Option<ExternalShutdownCoordinator>,
    main_heartbeat: Option<crate::diagnostics::Heartbeat>,
    xr_input_cache: XrInputCache,
}

impl AppDriver {
    /// Builds initial app state after process bootstrap; window/GPU target creation is lazy.
    pub(crate) fn new(
        runtime: RendererRuntime,
        startup_gpu: GpuStartupConfig,
        log_level_cli: Option<LogLevel>,
        external_shutdown: Option<ExternalShutdownCoordinator>,
        main_heartbeat: Option<crate::diagnostics::Heartbeat>,
    ) -> Self {
        Self {
            runtime,
            startup_gpu,
            log_level_cli,
            target: None,
            exit: ExitState::default(),
            log_flush: LogFlushCadence::default(),
            input: WindowInputAccumulator::default(),
            cursor_output_tracking: CursorOutputTracking::default(),
            frame_clock: FrameClock::default(),
            external_shutdown,
            main_heartbeat,
            xr_input_cache: XrInputCache::default(),
        }
    }

    /// Returns the normal process exit requested by this app driver.
    pub(crate) fn into_run_exit(self) -> RunExit {
        self.exit.run_exit()
    }

    fn request_exit(&mut self, reason: ExitReason, event_loop: &ActiveEventLoop) {
        self.exit.request(reason);
        event_loop.exit();
    }

    fn check_external_shutdown(&mut self, event_loop: &ActiveEventLoop) -> bool {
        let Some(coord) = self.external_shutdown.as_ref() else {
            return false;
        };
        if !coord.requested.load(Ordering::Relaxed) {
            return false;
        }
        if coord.log_when_checked {
            logger::info!("Graceful shutdown requested; exiting event loop");
        }
        self.request_exit(ExitReason::ExternalShutdown, event_loop);
        true
    }

    fn sync_log_level_from_settings(&self) {
        let log_verbose = self
            .runtime
            .settings()
            .read()
            .map(|s| s.debug.log_verbose)
            .unwrap_or(false);
        logger::set_max_level(effective_renderer_log_level(
            self.log_level_cli,
            log_verbose,
        ));
    }
}

#[derive(Debug)]
struct LogFlushCadence {
    last_log_flush: Option<Instant>,
    interval: Duration,
}

impl Default for LogFlushCadence {
    fn default() -> Self {
        Self {
            last_log_flush: None,
            interval: LOG_FLUSH_INTERVAL,
        }
    }
}

impl LogFlushCadence {
    fn flush_if_due(&mut self) {
        let now = Instant::now();
        let should = self
            .last_log_flush
            .is_none_or(|t| now.duration_since(t) >= self.interval);
        if should {
            logger::flush();
            self.last_log_flush = Some(now);
        }
    }
}
