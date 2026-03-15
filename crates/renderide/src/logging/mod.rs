//! Log infrastructure for panic hook. No runtime logging; panic hook writes to logs/Renderide.log.

use std::path::Path;

/// Path to Renderide.log in the logs folder at repo root (two levels up from crates/renderide).
pub fn log_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .unwrap_or_else(|| Path::new("."))
        .join("logs")
        .join("Renderide.log")
}

/// Initialize logging. Ensures logs directory exists. Call once at startup before panic hook.
pub fn init_log() {
    let path = log_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
}
