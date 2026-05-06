//! Runtime-owned renderer configuration handles.

use std::path::PathBuf;

use crate::config::RendererSettingsHandle;

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

    /// Path written by the renderer config HUD.
    pub(super) fn config_save_path(&self) -> &PathBuf {
        &self.config_save_path
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
}
