//! Process bootstrap before the winit loop: logging, panic hook, config, IPC connection, and
//! [`super::renderide_app::RenderideApp`] construction.

use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use logger::{LogComponent, LogLevel};
use winit::event_loop::EventLoop;

use crate::config::{load_renderer_settings, log_config_resolve_trace, settings_handle_from};
use crate::connection::{get_connection_parameters, try_claim_renderer_singleton};
use crate::frontend::InitState;
use crate::runtime::RendererRuntime;
use crate::shared::{HeadOutputDevice, RendererInitData};

use super::renderide_app::RenderideApp;

/// Interval between log flushes when using file logging in the winit handler.
pub(super) const LOG_FLUSH_INTERVAL: Duration = Duration::from_secs(1);

/// Max time to wait for [`RendererInitData`] after IPC connect before exiting with an error.
const IPC_INIT_WAIT_TIMEOUT: Duration = Duration::from_secs(60);

/// Chooses the process max log level after [`logger::init_for`].
///
/// Precedence: **`-LogLevel`** (if present) always wins. If absent, [`crate::config::DebugSettings::log_verbose`]
/// selects [`LogLevel::Trace`] when true and [`LogLevel::Debug`] when false.
pub(super) fn effective_renderer_log_level(cli: Option<LogLevel>, log_verbose: bool) -> LogLevel {
    if let Some(level) = cli {
        level
    } else if log_verbose {
        LogLevel::Trace
    } else {
        LogLevel::Debug
    }
}

/// Runs the winit event loop until exit or window close.
pub fn run() -> Option<i32> {
    if let Err(e) = try_claim_renderer_singleton() {
        eprintln!("{e}");
        return Some(1);
    }

    let timestamp = logger::log_filename_timestamp();
    let log_level_cli = logger::parse_log_level_from_args();
    let initial_log_level = log_level_cli.unwrap_or(LogLevel::Debug);
    let log_path =
        match logger::init_for(LogComponent::Renderer, &timestamp, initial_log_level, false) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Failed to initialize logging: {e}");
                return Some(1);
            }
        };

    logger::info!("Logging to {}", log_path.display());

    // Vulkan validation (spirv-val, VK_LAYER_KHRONOS_validation) and other native code often use
    // stdout and/or stderr; forward both so messages reach the log file regardless of wgpu flags
    // or `VK_INSTANCE_LAYERS` (see `native_stdio`).
    crate::native_stdio::ensure_stdio_forwarded_to_logger();

    // Fatal faults (signals / SEH / Mach exceptions) do not run the panic hook; register after
    // stdio forwarding so a duplicate of the preserved terminal fd exists when tee is enabled.
    crate::fatal_crash_log::install(&log_path);

    let config_load = load_renderer_settings();
    logger::set_max_level(effective_renderer_log_level(
        log_level_cli,
        config_load.settings.debug.log_verbose,
    ));
    log_config_resolve_trace(&config_load.resolve);
    let settings_handle = settings_handle_from(&config_load);
    let initial_vsync = config_load.settings.rendering.vsync;
    let initial_gpu_validation = config_load.settings.debug.gpu_validation_layers;

    let log_path_hook = log_path.clone();
    std::panic::set_hook(Box::new(move |info| {
        let report = logger::panic_report(info);
        logger::append_panic_report_to_file(&log_path_hook, &report);
        crate::native_stdio::try_write_preserved_stderr(report.as_bytes());
    }));

    let params = get_connection_parameters();
    let mut runtime = RendererRuntime::new(
        params.clone(),
        settings_handle,
        config_load.save_path.clone(),
    );
    if let Err(e) = runtime.connect_ipc() {
        if params.is_some() {
            logger::error!("IPC connect failed: {e}");
            return Some(1);
        }
    }

    if params.is_some() && runtime.is_ipc_connected() {
        logger::info!("IPC connected (Primary/Background)");
        if wait_for_renderer_init_data(&mut runtime).is_err() {
            return Some(1);
        }
    } else if params.is_some() {
        logger::warn!("IPC params present but connection state unexpected");
    } else {
        logger::info!("Standalone mode (no -QueueName/-QueueCapacity; desktop GPU, no host init)");
    }

    let event_loop = match EventLoop::new() {
        Ok(el) => el,
        Err(e) => {
            logger::error!("EventLoop::new failed: {e}");
            return Some(1);
        }
    };

    let mut app = RenderideApp::new(
        runtime,
        initial_vsync,
        initial_gpu_validation,
        log_level_cli,
    );

    let _ = event_loop.run_app(&mut app);
    app.exit_code
}

/// Blocks until [`RendererInitData`] arrives or IPC fails (non-standalone only).
fn wait_for_renderer_init_data(runtime: &mut RendererRuntime) -> Result<(), ()> {
    let deadline = Instant::now() + IPC_INIT_WAIT_TIMEOUT;
    while runtime.init_state() == InitState::Uninitialized {
        if Instant::now() > deadline {
            logger::error!("Timed out waiting for RendererInitData from host");
            return Err(());
        }
        runtime.poll_ipc();
        if runtime.fatal_error() {
            logger::error!("Fatal IPC error while waiting for RendererInitData");
            return Err(());
        }
        thread::sleep(Duration::from_millis(1));
    }
    Ok(())
}

/// Standalone runs have no host init; IPC runs should have [`RendererInitData`] before the window exists.
pub(super) fn effective_output_device_for_gpu(
    pending: Option<&RendererInitData>,
) -> HeadOutputDevice {
    pending
        .map(|i| i.output_device)
        .unwrap_or(HeadOutputDevice::Screen)
}

pub(super) fn apply_window_title_from_init(
    window: &Arc<winit::window::Window>,
    init: &RendererInitData,
) {
    if let Some(ref title) = init.window_title {
        window.set_title(title);
    }
}

#[cfg(test)]
mod effective_log_level_tests {
    use super::effective_renderer_log_level;
    use logger::LogLevel;

    #[test]
    fn cli_always_overrides_log_verbose() {
        assert_eq!(
            effective_renderer_log_level(Some(LogLevel::Warn), true),
            LogLevel::Warn
        );
    }

    #[test]
    fn no_cli_uses_trace_when_log_verbose() {
        assert_eq!(effective_renderer_log_level(None, true), LogLevel::Trace);
    }

    #[test]
    fn no_cli_uses_debug_when_not_log_verbose() {
        assert_eq!(effective_renderer_log_level(None, false), LogLevel::Debug);
    }
}
