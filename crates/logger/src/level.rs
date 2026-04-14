//! Log severity ordering and `-LogLevel` argument parsing.

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

impl LogLevel {
    /// Parses a level string (case-insensitive). Returns [`None`] for invalid values.
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "error" | "e" => Some(Self::Error),
            "warn" | "warning" | "w" => Some(Self::Warn),
            "info" | "i" => Some(Self::Info),
            "debug" | "d" => Some(Self::Debug),
            "trace" | "t" => Some(Self::Trace),
            _ => None,
        }
    }

    /// Returns the string to pass as `-LogLevel` value.
    pub fn as_arg(&self) -> &'static str {
        match self {
            Self::Error => "error",
            Self::Warn => "warn",
            Self::Info => "info",
            Self::Debug => "debug",
            Self::Trace => "trace",
        }
    }
}

impl std::fmt::Debug for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Error => write!(f, "ERROR"),
            Self::Warn => write!(f, "WARN"),
            Self::Info => write!(f, "INFO"),
            Self::Debug => write!(f, "DEBUG"),
            Self::Trace => write!(f, "TRACE"),
        }
    }
}

/// Stable `0..=4` tag for [`LogLevel`] (matches [`PartialOrd`] order).
#[inline]
pub(crate) fn level_to_tag(level: LogLevel) -> u8 {
    match level {
        LogLevel::Error => 0,
        LogLevel::Warn => 1,
        LogLevel::Info => 2,
        LogLevel::Debug => 3,
        LogLevel::Trace => 4,
    }
}

/// Maps a stored `0..=4` tag back to [`LogLevel`]. Values above `4` clamp to [`LogLevel::Trace`].
#[inline]
pub(crate) fn tag_to_level(tag: u8) -> LogLevel {
    match tag.min(4) {
        0 => LogLevel::Error,
        1 => LogLevel::Warn,
        2 => LogLevel::Info,
        3 => LogLevel::Debug,
        _ => LogLevel::Trace,
    }
}

/// Parses `-LogLevel` from command line args (case-insensitive).
///
/// Returns [`None`] if not present or invalid; otherwise the parsed level.
pub fn parse_log_level_from_args() -> Option<LogLevel> {
    let args: Vec<String> = std::env::args().collect();
    parse_log_level_from_slice(&args)
}

/// Parses `-LogLevel` / value pairs from a full argv slice (same layout as [`std::env::args`], including the executable).
pub(crate) fn parse_log_level_from_slice(args: &[String]) -> Option<LogLevel> {
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        if arg.eq_ignore_ascii_case("-LogLevel") && i + 1 < args.len() {
            return LogLevel::parse(&args[i + 1]);
        }
        i += 1;
    }
    None
}

/// Roundtrip tests for [`level_to_tag`] and [`tag_to_level`], and `-LogLevel` argv parsing.
#[cfg(test)]
mod tag_tests {
    use super::*;

    fn tokens(args: &[&str]) -> Vec<String> {
        args.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn parse_log_level_from_slice_finds_flag() {
        assert_eq!(
            parse_log_level_from_slice(&tokens(&["prog", "-LogLevel", "debug"])),
            Some(LogLevel::Debug)
        );
    }

    #[test]
    fn parse_log_level_from_slice_case_insensitive_flag() {
        assert_eq!(
            parse_log_level_from_slice(&tokens(&["prog", "-loglevel", "INFO"])),
            Some(LogLevel::Info)
        );
    }

    #[test]
    fn parse_log_level_from_slice_ignores_other_tokens() {
        assert_eq!(
            parse_log_level_from_slice(&tokens(&["prog", "-x", "-LogLevel", "warn", "y"])),
            Some(LogLevel::Warn)
        );
    }

    #[test]
    fn parse_log_level_from_slice_missing_value() {
        assert!(parse_log_level_from_slice(&tokens(&["prog", "-LogLevel"])).is_none());
    }

    #[test]
    fn parse_log_level_from_slice_absent() {
        assert!(parse_log_level_from_slice(&tokens(&["prog", "a", "b"])).is_none());
    }

    #[test]
    fn level_tag_roundtrip() {
        for l in [
            LogLevel::Error,
            LogLevel::Warn,
            LogLevel::Info,
            LogLevel::Debug,
            LogLevel::Trace,
        ] {
            assert_eq!(tag_to_level(level_to_tag(l)), l);
        }
    }
}
