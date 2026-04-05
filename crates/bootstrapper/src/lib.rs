//! ResoBoot-compatible bootstrapper: launches Renderite Host, bridges clipboard/renderer over shared-memory queues.
//!
//! The binary entry point is [`run`]; use [`BootstrapOptions`] to supply Host arguments and logging.

#![warn(missing_docs)]

mod child_lifetime;
mod cleanup;
pub mod config;
mod error;
mod host;
mod ipc;
mod orchestration;
mod paths;
mod protocol;
mod runtime;

pub use error::BootstrapError;

/// Inputs for [`run`]: Host argv, optional verbosity, and log filename timestamp.
#[derive(Debug, Clone)]
pub struct BootstrapOptions {
    /// Arguments forwarded to Renderite Host (before `-Invisible` / `-shmprefix`).
    pub host_args: Vec<String>,
    /// Maximum level written to the bootstrapper log file; also forwarded to Renderide when set.
    pub log_level: Option<logger::LogLevel>,
    /// Filename segment from [`logger::log_filename_timestamp`] (without `.log`).
    pub log_timestamp: String,
}

/// Initializes logging under `logs/bootstrapper/` (or the directory in the `RENDERIDE_LOGS_ROOT`
/// environment variable), installs a panic hook, then runs the bootstrap sequence.
///
/// Panics are logged and swallowed with `Ok(())` to mirror the production ResoBoot behavior.
pub fn run(options: BootstrapOptions) -> Result<(), BootstrapError> {
    let current_dir = std::env::current_dir().map_err(BootstrapError::CurrentDir)?;
    let shared_memory_prefix =
        config::generate_shared_memory_prefix(16).map_err(BootstrapError::Prefix)?;
    let resonite_config = config::ResoBootConfig::new(shared_memory_prefix, options.log_level)
        .map_err(BootstrapError::CurrentDir)?;

    let logs_base = orchestration::logs_base_for_run(&current_dir);
    std::fs::create_dir_all(logs_base.join("bootstrapper")).map_err(BootstrapError::Io)?;
    let log_path = logs_base
        .join("bootstrapper")
        .join(format!("{}.log", options.log_timestamp));

    let max_level = options.log_level.unwrap_or(logger::LogLevel::Trace);
    logger::init(&log_path, max_level, false).map_err(BootstrapError::Logging)?;

    let panic_log = log_path.clone();
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        logger::log_panic(&panic_log, info);
        default_hook(info);
    }));

    let ctx = orchestration::RunContext {
        host_args: options.host_args,
        log_timestamp: options.log_timestamp,
        logs_base,
    };

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        orchestration::run(&resonite_config, ctx)
    }));

    match result {
        Ok(r) => r,
        Err(e) => {
            logger::error!("Exception in bootstrapper:\n{e:?}");
            logger::flush();
            Ok(())
        }
    }
}
