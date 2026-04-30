//! Process services shared by the windowed and headless app drivers.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(not(windows))]
use std::thread;

use crate::config::WatchdogSettings;
use crate::diagnostics::{Heartbeat, Watchdog};

/// Cooperative exit flag for OS-driven shutdown.
pub(crate) struct ExternalShutdownCoordinator {
    /// Set by the graceful-shutdown path; polled by app drivers.
    pub(crate) requested: Arc<AtomicBool>,
    /// When `true`, emit a log line when the driver first observes the request.
    pub(crate) log_when_checked: bool,
}

/// Long-lived services installed for the renderer process.
pub(crate) struct AppServices {
    /// OS-driven cooperative shutdown observer.
    pub(crate) external_shutdown: Option<ExternalShutdownCoordinator>,
    /// Watchdog guard; dropping joins the watchdog thread.
    pub(crate) watchdog: Option<Watchdog>,
    /// Main-thread heartbeat registered with the watchdog.
    pub(crate) main_heartbeat: Option<Heartbeat>,
}

/// Installs shutdown handling, watchdog, profiling main-thread state, and rayon worker naming.
pub(crate) fn install_app_services(watchdog_settings: WatchdogSettings) -> AppServices {
    let external_shutdown = install_external_shutdown();
    let watchdog = Watchdog::install(watchdog_settings);
    let main_heartbeat = watchdog.as_ref().map(|w| w.register_current_thread("main"));

    crate::profiling::register_main_thread();
    init_rayon_pool();

    AppServices {
        external_shutdown,
        watchdog,
        main_heartbeat,
    }
}

fn init_rayon_pool() {
    if let Err(e) = rayon::ThreadPoolBuilder::new()
        .thread_name(|i| format!("rayon-worker-{i}"))
        .start_handler(crate::profiling::rayon_thread_start_handler())
        .build_global()
    {
        logger::warn!("Rayon global pool already initialized or build_global failed: {e}");
    }
}

#[cfg(unix)]
fn shutdown_signal_display_name(sig: i32) -> &'static str {
    match sig {
        s if s == libc::SIGTERM => "SIGTERM",
        s if s == libc::SIGINT => "SIGINT",
        s if s == libc::SIGHUP => "SIGHUP",
        _ => "unknown",
    }
}

#[cfg(unix)]
fn register_sigterm_fallback(flag: &Arc<AtomicBool>) {
    if let Err(e) = signal_hook::flag::register(libc::SIGTERM, Arc::clone(flag)) {
        logger::warn!("Failed to register SIGTERM fallback: {e}");
    }
}

#[cfg(unix)]
fn install_external_shutdown_unix() -> ExternalShutdownCoordinator {
    use signal_hook::iterator::Signals;

    let flag = Arc::new(AtomicBool::new(false));
    match Signals::new([libc::SIGTERM, libc::SIGINT, libc::SIGHUP]) {
        Ok(mut signals) => {
            let f = Arc::clone(&flag);
            match thread::Builder::new()
                .name("shutdown-signals".to_owned())
                .spawn(move || {
                    for sig in signals.forever() {
                        logger::info!(
                            "Received shutdown signal ({}); cooperative exit",
                            shutdown_signal_display_name(sig)
                        );
                        f.store(true, Ordering::Relaxed);
                    }
                }) {
                Ok(_join) => ExternalShutdownCoordinator {
                    requested: flag,
                    log_when_checked: false,
                },
                Err(e) => {
                    logger::error!("Failed to spawn shutdown-signals thread: {e}");
                    register_sigterm_fallback(&flag);
                    ExternalShutdownCoordinator {
                        requested: flag,
                        log_when_checked: true,
                    }
                }
            }
        }
        Err(e) => {
            logger::warn!(
                "Failed to register graceful shutdown signals ({e}); falling back to SIGTERM only"
            );
            register_sigterm_fallback(&flag);
            ExternalShutdownCoordinator {
                requested: flag,
                log_when_checked: true,
            }
        }
    }
}

#[cfg(windows)]
fn install_external_shutdown_windows() -> ExternalShutdownCoordinator {
    let flag = Arc::new(AtomicBool::new(false));
    let f = Arc::clone(&flag);
    match ctrlc::set_handler(move || {
        logger::info!("Received Ctrl+C (console control); cooperative exit");
        f.store(true, Ordering::Relaxed);
    }) {
        Ok(()) => ExternalShutdownCoordinator {
            requested: flag,
            log_when_checked: false,
        },
        Err(e) => {
            logger::warn!("Failed to register Ctrl+C handler: {e}");
            ExternalShutdownCoordinator {
                requested: flag,
                log_when_checked: false,
            }
        }
    }
}

fn install_external_shutdown() -> Option<ExternalShutdownCoordinator> {
    #[cfg(unix)]
    {
        Some(install_external_shutdown_unix())
    }
    #[cfg(windows)]
    {
        Some(install_external_shutdown_windows())
    }
    #[cfg(not(any(unix, windows)))]
    {
        None
    }
}

#[cfg(all(test, unix))]
mod shutdown_signal_display_name_tests {
    use super::shutdown_signal_display_name;

    #[test]
    fn known_signals_map_to_name() {
        assert_eq!(shutdown_signal_display_name(libc::SIGTERM), "SIGTERM");
        assert_eq!(shutdown_signal_display_name(libc::SIGINT), "SIGINT");
        assert_eq!(shutdown_signal_display_name(libc::SIGHUP), "SIGHUP");
    }

    #[test]
    fn unrecognized_signal_is_unknown() {
        assert_eq!(shutdown_signal_display_name(libc::SIGUSR1), "unknown");
        assert_eq!(shutdown_signal_display_name(-1), "unknown");
    }
}
