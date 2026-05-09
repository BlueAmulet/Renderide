//! Runtime-owned renderer configuration handles.

use std::path::PathBuf;

use crate::config::{RendererSettingsHandle, save_renderer_settings};

/// Settings handle, persistence path, and config-write suppression state.
pub(super) struct RuntimeConfigState {
    /// Process-wide renderer settings shared with the HUD and frame loop.
    pub(super) settings: RendererSettingsHandle,
    /// Target path for persisting renderer settings from the ImGui config window.
    config_save_path: PathBuf,
    /// When true, ImGui and config save helpers must not overwrite `config.toml`.
    suppress_renderer_config_disk_writes: bool,
}

impl RuntimeConfigState {
    /// Creates runtime config state from the loaded settings handle and save path.
    pub(super) fn new(settings: RendererSettingsHandle, config_save_path: PathBuf) -> Self {
        Self {
            settings,
            config_save_path,
            suppress_renderer_config_disk_writes: false,
        }
    }

    /// Cloned config save path for backend HUD attach.
    pub(super) fn cloned_config_save_path(&self) -> PathBuf {
        self.config_save_path.clone()
    }

    /// Sets whether renderer config disk writes are blocked.
    pub(super) fn set_suppress_renderer_config_disk_writes(&mut self, value: bool) {
        self.suppress_renderer_config_disk_writes = value;
    }

    /// Whether renderer config disk writes are blocked.
    pub(super) fn suppress_renderer_config_disk_writes(&self) -> bool {
        self.suppress_renderer_config_disk_writes
    }

    /// Toggles the master ImGui overlay visibility setting and persists it when allowed.
    pub(super) fn toggle_imgui_visibility(&self) -> Option<bool> {
        let Ok(mut settings) = self.settings.write() else {
            logger::warn!(
                "Failed to toggle ImGui visibility: renderer settings store is unavailable"
            );
            return None;
        };

        settings.debug.hud.imgui_visible = !settings.debug.hud.imgui_visible;
        let visible = settings.debug.hud.imgui_visible;

        if self.suppress_renderer_config_disk_writes {
            logger::error!(
                "Refusing to save renderer config to {}: disk writes suppressed after startup extract failure",
                self.config_save_path.display()
            );
            return Some(visible);
        }

        if let Err(e) = save_renderer_settings(&self.config_save_path, &settings) {
            logger::warn!(
                "Failed to save renderer config to {}: {e}",
                self.config_save_path.display()
            );
        }

        Some(visible)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, RwLock};

    use crate::config::RendererSettings;

    use super::RuntimeConfigState;

    #[test]
    fn toggle_imgui_visibility_updates_memory_and_disk() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.toml");
        let settings = Arc::new(RwLock::new(RendererSettings::default()));
        let state = RuntimeConfigState::new(Arc::clone(&settings), path.clone());

        assert_eq!(state.toggle_imgui_visibility(), Some(false));
        assert!(
            !settings
                .read()
                .expect("settings read")
                .debug
                .hud
                .imgui_visible
        );

        let text = std::fs::read_to_string(path).expect("read saved config");
        let saved: RendererSettings = toml::from_str(&text).expect("decode saved config");
        assert!(!saved.debug.hud.imgui_visible);
    }

    #[test]
    fn toggle_imgui_visibility_respects_disk_write_suppression() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.toml");
        let settings = Arc::new(RwLock::new(RendererSettings::default()));
        let mut state = RuntimeConfigState::new(Arc::clone(&settings), path.clone());
        state.set_suppress_renderer_config_disk_writes(true);

        assert_eq!(state.toggle_imgui_visibility(), Some(false));
        assert!(
            !settings
                .read()
                .expect("settings read")
                .debug
                .hud
                .imgui_visible
        );
        assert!(!path.exists());
    }

    #[test]
    fn toggle_imgui_visibility_keeps_memory_change_when_save_fails() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config-as-dir");
        std::fs::create_dir_all(&path).expect("create config dir");
        let settings = Arc::new(RwLock::new(RendererSettings::default()));
        let state = RuntimeConfigState::new(Arc::clone(&settings), path);

        assert_eq!(state.toggle_imgui_visibility(), Some(false));
        assert!(
            !settings
                .read()
                .expect("settings read")
                .debug
                .hud
                .imgui_visible
        );
    }
}
