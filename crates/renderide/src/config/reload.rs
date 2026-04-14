//! Optional debounced file watching for the renderer config path.
//!
//! [`RendererSettings`](super::RendererSettings) already implements [`PartialEq`]; after detecting a
//! change with [`ConfigFileWatcher::poll_changed`], call [`super::persist::load_renderer_settings`] and
//! compare to the in-memory copy to decide whether to apply updates.

use std::path::Path;
use std::sync::mpsc::Receiver;
use std::time::Duration;

use notify::RecursiveMode;
use notify_debouncer_mini::{new_debouncer, DebounceEventResult, Debouncer};

/// Returns `true` if `a` and `b` differ.
pub fn renderer_settings_changed(a: &super::RendererSettings, b: &super::RendererSettings) -> bool {
    a != b
}

/// Debounced watcher for edits to a config file (for example [`super::resolve::FILE_NAME_TOML`]).
pub struct ConfigFileWatcher {
    _debouncer: Debouncer<notify::RecommendedWatcher>,
    notify: Receiver<()>,
}

impl ConfigFileWatcher {
    /// Creates a watcher with a 400ms debounce timeout.
    ///
    /// Watches the parent directory of `config_path` and reports when the file at `config_path`
    /// receives debounced filesystem events.
    pub fn new(config_path: &Path) -> notify::Result<Self> {
        let (tx, rx) = std::sync::mpsc::channel();
        let target = config_path.to_path_buf();
        let mut debouncer = new_debouncer(
            Duration::from_millis(400),
            move |res: DebounceEventResult| {
                if let Ok(events) = res {
                    for ev in events {
                        if ev.path == target {
                            let _ = tx.send(());
                            break;
                        }
                    }
                }
            },
        )?;
        let parent = config_path.parent().unwrap_or_else(|| Path::new("."));
        debouncer
            .watcher()
            .watch(parent, RecursiveMode::NonRecursive)?;
        Ok(Self {
            _debouncer: debouncer,
            notify: rx,
        })
    }

    /// Returns `true` if at least one debounced change was detected since the last call.
    pub fn poll_changed(&self) -> bool {
        let mut any = false;
        while self.notify.try_recv().is_ok() {
            any = true;
        }
        any
    }
}
