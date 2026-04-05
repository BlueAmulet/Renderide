//! Bootstrapper binary entry point.

#![cfg_attr(windows, windows_subsystem = "windows")]

fn main() {
    let (host_args, log_level) = bootstrapper::config::parse_args();
    let timestamp = logger::log_filename_timestamp();

    if let Err(e) = bootstrapper::run(bootstrapper::BootstrapOptions {
        host_args,
        log_level,
        log_timestamp: timestamp,
    }) {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
