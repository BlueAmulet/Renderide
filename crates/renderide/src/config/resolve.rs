//! Locate `config.ini`: `RENDERIDE_CONFIG`, then standard search paths.

use std::path::{Path, PathBuf};

/// How the INI path was chosen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConfigSource {
    /// `RENDERIDE_CONFIG` pointed at an existing file.
    Env,
    /// First hit among exe-adjacent / cwd searches.
    Search,
    /// No file found; caller uses defaults only.
    None,
}

/// Result of resolving a config path (whether or not a file was read).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConfigResolveOutcome {
    /// Every path checked, in order (`RENDERIDE_CONFIG` first when set, then search candidates).
    pub attempted_paths: Vec<PathBuf>,
    /// First existing regular file used for INI content.
    pub loaded_path: Option<PathBuf>,
    pub source: ConfigSource,
}

const FILE_NAME: &str = "config.ini";
const ENV_OVERRIDE: &str = "RENDERIDE_CONFIG";

fn push_unique(out: &mut Vec<PathBuf>, p: PathBuf) {
    if !out.iter().any(|x| x == &p) {
        out.push(p);
    }
}

fn search_candidates() -> Vec<PathBuf> {
    let mut v = Vec::new();

    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            v.push(dir.join(FILE_NAME));
            if let Some(parent) = dir.parent() {
                v.push(parent.join(FILE_NAME));
            }
        }
    }

    if let Ok(cwd) = std::env::current_dir() {
        v.push(cwd.join(FILE_NAME));
        if let Some(p1) = cwd.parent() {
            if let Some(p2) = p1.parent() {
                v.push(p2.join(FILE_NAME));
            }
        }
    }

    v
}

/// Resolves the config file path. If `RENDERIDE_CONFIG` is set but missing, logs a warning and
/// continues with the search list.
pub fn resolve_config_path() -> ConfigResolveOutcome {
    let mut attempted_paths = Vec::new();

    if let Ok(raw) = std::env::var(ENV_OVERRIDE) {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            let p = PathBuf::from(trimmed);
            push_unique(&mut attempted_paths, p.clone());
            if p.is_file() {
                return ConfigResolveOutcome {
                    attempted_paths,
                    loaded_path: Some(p),
                    source: ConfigSource::Env,
                };
            }
            logger::warn!(
                "{ENV_OVERRIDE}={} does not exist or is not a file; trying default locations",
                p.display()
            );
        }
    }

    for p in search_candidates() {
        push_unique(&mut attempted_paths, p.clone());
        if p.is_file() {
            return ConfigResolveOutcome {
                attempted_paths,
                loaded_path: Some(p),
                source: ConfigSource::Search,
            };
        }
    }

    ConfigResolveOutcome {
        attempted_paths,
        loaded_path: None,
        source: ConfigSource::None,
    }
}

/// Reads the file at `path` if it exists.
pub fn read_config_file(path: &Path) -> std::io::Result<String> {
    std::fs::read_to_string(path)
}

/// Picks the path used when persisting settings from the UI or [`crate::config::save_renderer_settings`].
///
/// - If a file was loaded ([`ConfigResolveOutcome::loaded_path`]), that path is used.
/// - Otherwise: prefer `current_dir()/config.ini` when the directory exists and is writable; else
///   the first path in the same search order as [`resolve_config_path`] whose parent exists and is
///   writable (typically the executable directory).
pub fn resolve_save_path(resolve: &ConfigResolveOutcome) -> PathBuf {
    if let Some(p) = resolve.loaded_path.clone() {
        return p;
    }

    if let Ok(cwd) = std::env::current_dir() {
        let p = cwd.join(FILE_NAME);
        if is_dir_writable(cwd.as_path()) {
            return p;
        }
    }

    for p in search_candidates() {
        if let Some(parent) = p.parent() {
            if parent.as_os_str().is_empty() {
                continue;
            }
            if is_dir_writable(parent) {
                return p;
            }
        }
    }

    // Last resort: cwd join even if we could not verify writability (save may fail at runtime).
    std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join(FILE_NAME)
}

fn is_dir_writable(dir: &Path) -> bool {
    if !dir.is_dir() {
        return false;
    }
    // Best-effort: create a probe file (same approach as `access` is not fully portable for ACLs).
    let probe = dir.join(".renderide_write_probe");
    match std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&probe)
    {
        Ok(_) => {
            let _ = std::fs::remove_file(&probe);
            true
        }
        Err(_) => false,
    }
}
