//! Process services shared by the windowed and headless app drivers.

use crate::config::WatchdogSettings;
use crate::diagnostics::{Heartbeat, Watchdog};

pub(crate) use super::signals::ExternalShutdownCoordinator;

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
    let external_shutdown = super::signals::install_external_shutdown();
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
