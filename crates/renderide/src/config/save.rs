//! Atomic TOML persistence for [`super::types::RendererSettings`].
//!
//! Splits the file IO out of [`super::load`] so the load and save sides can be reasoned about
//! independently. The atomic-write path uses a `.<file>.tmp` sibling and `rename`, which is
//! atomic on every supported OS.

use std::io;
use std::path::Path;

use super::resolve::FILE_NAME_TOML;
use super::types::RendererSettings;

/// Writes `settings` to `path` as TOML atomically (temp file in the same directory, then
/// `rename`).
pub fn save_renderer_settings(path: &Path, settings: &RendererSettings) -> io::Result<()> {
    let contents = toml::to_string_pretty(settings).map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("TOML serialization failed: {e}"),
        )
    })?;

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let file_name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(FILE_NAME_TOML);
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let tmp = parent.join(format!(".{file_name}.tmp"));
    std::fs::write(&tmp, contents.as_bytes())?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::save_renderer_settings;
    use crate::config::types::RendererSettings;

    #[test]
    fn atomic_save_roundtrip() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.toml");
        let s = RendererSettings::from_defaults();
        save_renderer_settings(&path, &s).expect("save");
        let text = std::fs::read_to_string(&path).expect("read");
        let s2: RendererSettings = toml::from_str(&text).expect("toml");
        assert_eq!(s, s2);
    }

    #[test]
    fn toml_roundtrip_string() {
        let s = RendererSettings::from_defaults();
        let text = toml::to_string_pretty(&s).expect("ser");
        let s2: RendererSettings = toml::from_str(&text).expect("de");
        assert_eq!(s, s2);
    }
}
