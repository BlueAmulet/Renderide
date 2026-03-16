//! Tiered logging for the renderer. Supports file output and optional console.
//!
//! Call `init()` once at startup before setting the panic hook. The panic hook
//! uses `log_panic()` to write to logs/Renderide.log.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::sync::Mutex;
use std::sync::OnceLock;

/// Log level for filtering. Lower ordinal = higher priority.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    /// Critical errors.
    Error,
    /// Warnings.
    Warn,
    /// Informational messages.
    Info,
    /// Debug diagnostics.
    Debug,
    /// Verbose trace.
    Trace,
}

/// Logger that writes to file and optionally to stderr.
struct Logger {
    /// File output. Mutex for thread-safe writes.
    file: Mutex<std::fs::File>,
    /// Whether to also write to stderr.
    console: bool,
    /// Maximum level to log. Messages at or below this level are written.
    max_level: LogLevel,
}

/// Global logger instance. Set by `init()`.
static LOGGER: OnceLock<Logger> = OnceLock::new();

/// Path to Renderide.log in the logs folder at repo root (two levels up from crates/renderide).
pub fn log_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .unwrap_or_else(|| Path::new("."))
        .join("logs")
        .join("Renderide.log")
}

/// Initializes logging. Creates logs directory, opens file, optionally enables console.
/// Call once at startup before panic hook. Console output is enabled when
/// `RENDERIDE_LOG_CONSOLE` env var is set.
pub fn init() {
    let path = log_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let file = match OpenOptions::new().create(true).append(true).open(&path) {
        Ok(f) => f,
        Err(_) => return,
    };
    let console = std::env::var("RENDERIDE_LOG_CONSOLE").is_ok();
    let logger = Logger {
        file: Mutex::new(file),
        console,
        max_level: LogLevel::Trace,
    };
    let _ = LOGGER.set(logger);
}

/// Writes panic info and backtrace to the log file. Does not acquire the logger
/// mutex, so it is safe to call from a panic handler. Opens the file directly.
pub fn log_panic(info: &std::panic::PanicInfo) {
    let path = log_path();
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(&path) {
        let _ = writeln!(f, "PANIC: {}", info);
        let _ = writeln!(f, "Backtrace:\n{:?}", std::backtrace::Backtrace::capture());
        let _ = f.flush();
    }
}

/// Internal log writer. Called by the log macros.
pub(crate) fn log(level: LogLevel, args: std::fmt::Arguments<'_>) {
    if let Some(logger) = LOGGER.get()
        && level <= logger.max_level {
            let msg = args.to_string();
            let timestamp = format_timestamp();
            let line = format!("[{}] {:?} {}\n", timestamp, level, msg);
            if let Ok(mut file) = logger.file.lock() {
                let _ = file.write_all(line.as_bytes());
                let _ = file.flush();
            }
            if logger.console {
                let _ = std::io::stderr().write_all(line.as_bytes());
                let _ = std::io::stderr().flush();
            }
        }
}

/// Returns a simple timestamp string. Uses std::time for a minimal format.
fn format_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let Ok(dur) = SystemTime::now().duration_since(UNIX_EPOCH) else {
        return "?".to_string();
    };
    let secs = dur.as_secs();
    let millis = dur.subsec_millis();
    let mins = (secs / 60) % 60;
    let hours = (secs / 3600) % 24;
    let secs = secs % 60;
    format!("{:02}:{:02}:{:02}.{:03}", hours, mins, secs, millis)
}

impl std::fmt::Debug for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LogLevel::Error => write!(f, "ERROR"),
            LogLevel::Warn => write!(f, "WARN"),
            LogLevel::Info => write!(f, "INFO"),
            LogLevel::Debug => write!(f, "DEBUG"),
            LogLevel::Trace => write!(f, "TRACE"),
        }
    }
}

/// Logs at error level.
#[macro_export]
macro_rules! error {
    ($($arg:tt)*) => {
        $crate::logging::log($crate::logging::LogLevel::Error, format_args!($($arg)*))
    };
}

/// Logs at warn level.
#[macro_export]
macro_rules! warn {
    ($($arg:tt)*) => {
        $crate::logging::log($crate::logging::LogLevel::Warn, format_args!($($arg)*))
    };
}

/// Logs at info level.
#[macro_export]
macro_rules! info {
    ($($arg:tt)*) => {
        $crate::logging::log($crate::logging::LogLevel::Info, format_args!($($arg)*))
    };
}

/// Logs at debug level.
#[macro_export]
macro_rules! debug {
    ($($arg:tt)*) => {
        $crate::logging::log($crate::logging::LogLevel::Debug, format_args!($($arg)*))
    };
}

/// Logs at trace level.
#[macro_export]
macro_rules! trace {
    ($($arg:tt)*) => {
        $crate::logging::log($crate::logging::LogLevel::Trace, format_args!($($arg)*))
    };
}
