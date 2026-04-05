//! Full bootstrap sequence: IPC, Host spawn, watchdogs, queue loop, Wine cleanup.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::child_lifetime::ChildLifetimeGroup;
use crate::cleanup;
use crate::config::ResoBootConfig;
use crate::host;
use crate::ipc::BootstrapQueues;
use crate::protocol;
use crate::BootstrapError;

/// Paths and argv for a single bootstrap run (owned so a panic boundary can move it).
pub struct RunContext {
    /// Extra Host CLI args (before `-Invisible` / `-shmprefix` are appended).
    pub host_args: Vec<String>,
    /// Shared log file timestamp segment (`HostOutput_{timestamp}.log`).
    pub log_timestamp: String,
    /// Root for `bootstrapper/` and `host/` log files.
    pub logs_base: PathBuf,
}

/// Runs the bootstrapper main loop after logging is initialized.
pub fn run(config: &ResoBootConfig, ctx: RunContext) -> Result<(), BootstrapError> {
    if let Some(ref level) = config.renderide_log_level {
        logger::info!("Renderide log level: {}", level.as_arg());
    }

    logger::info!("Bootstrapper start");
    logger::info!("Shared memory prefix: {}", config.shared_memory_prefix);

    let lifetime = ChildLifetimeGroup::new()?;
    let mut queues = BootstrapQueues::open(&config.shared_memory_prefix)?;

    let incoming_name = format!("{}.bootstrapper_in", config.shared_memory_prefix);
    let outgoing_name = format!("{}.bootstrapper_out", config.shared_memory_prefix);
    logger::info!(
        "Queues: incoming={incoming_name} outgoing={outgoing_name} (capacity {})",
        crate::ipc::BOOTSTRAP_QUEUE_CAPACITY
    );

    let RunContext {
        host_args,
        log_timestamp,
        logs_base,
    } = ctx;

    let mut args: Vec<String> = host_args;
    args.push("-Invisible".to_string());
    args.push("-shmprefix".to_string());
    args.push(config.shared_memory_prefix.clone());
    logger::info!("Host args: {:?}", args);

    let mut child = host::spawn_host(config, &args, &lifetime)?;
    logger::info!("Process started. Id: {}", child.id());

    host::set_host_above_normal_priority(&child);

    let host_log_dir = logs_base.join("host");
    fs::create_dir_all(&host_log_dir).map_err(BootstrapError::Io)?;
    let host_log_path = host_log_dir.join(format!("HostOutput_{log_timestamp}.log"));

    if let Some(stdout) = child.stdout.take() {
        host::spawn_output_drainer(host_log_path.clone(), stdout, "[Host stdout]");
    }
    if let Some(stderr) = child.stderr.take() {
        host::spawn_output_drainer(host_log_path, stderr, "[Host stderr]");
    }

    let cancel = Arc::new(AtomicBool::new(false));

    let heartbeat_deadline = Arc::new(Mutex::new(
        std::time::Instant::now() + Duration::from_secs(120),
    ));
    let cancel_wd = Arc::clone(&cancel);
    let deadline_wd = Arc::clone(&heartbeat_deadline);
    let _heartbeat_watchdog = std::thread::spawn(move || {
        while !cancel_wd.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(250));
            let Ok(deadline) = deadline_wd.lock() else {
                continue;
            };
            if std::time::Instant::now() > *deadline {
                cancel_wd.store(true, Ordering::SeqCst);
                logger::info!("Bootstrapper messaging timeout!");
                break;
            }
        }
    });

    if !config.is_wine {
        logger::info!("Process watcher: cancel when Host process exits");
        let cancel_host = Arc::clone(&cancel);
        let host_out_name = format!("HostOutput_{}.log", log_timestamp);
        std::thread::spawn(move || {
            let exit_status = loop {
                match child.try_wait() {
                    Ok(Some(status)) => break Some(status),
                    Ok(None) => {}
                    Err(e) => {
                        logger::error!("Process watcher try_wait error: {}", e);
                        break None;
                    }
                }
                std::thread::sleep(Duration::from_secs(1));
            };
            let exit_info = exit_status
                .as_ref()
                .map(|s| format!(" (exit code: {s})"))
                .unwrap_or_default();
            let msg = format!(
                "Host process exited{exit_info}. Check logs/host/{host_out_name} for stdout/stderr."
            );
            logger::info!("{msg}");
            eprintln!("{msg}");
            cancel_host.store(true, Ordering::SeqCst);
        });
    } else {
        logger::info!("Wine mode: Host exit watcher disabled (child is shell wrapper)");
    }

    protocol::queue_loop(
        &mut queues.incoming,
        &mut queues.outgoing,
        config,
        &cancel,
        &lifetime,
        &heartbeat_deadline,
    );

    if config.is_wine {
        cleanup::remove_wine_queue_backing_files(&config.shared_memory_prefix);
    }

    logger::info!("Bootstrapper end");
    Ok(())
}

/// Resolves the directory that contains `bootstrapper/` and `host/` log subfolders.
pub fn logs_base_for_run(current_dir: &Path) -> PathBuf {
    if let Ok(root) = std::env::var("RENDERIDE_LOGS_ROOT") {
        PathBuf::from(root)
    } else {
        current_dir.join("logs")
    }
}
