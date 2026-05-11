//! On-disk lifecycle for renderer settings.
//!
//! Three sibling concerns:
//! - [`resolve`]: pick a `config.toml` path from `RENDERIDE_CONFIG`, the binary directory,
//!   workspace roots, and the cwd; also computes the save path.
//! - [`load`]: run the layered defaults -> TOML -> `RENDERIDE_*` env -> post-extract pipeline,
//!   tolerate unknown keys for forward/backward compatibility, and apply versioned migrations.
//! - [`save`]: write `config.toml` atomically, merging unknown keys back so downgrades do not
//!   drop newer settings.

pub(crate) mod load;
pub(crate) mod resolve;
pub(super) mod save;

pub use load::{
    ConfigFilePolicy, ConfigLoadResult, load_renderer_settings, log_config_resolve_trace,
};
pub use resolve::find_renderide_workspace_root;
pub use save::{save_renderer_settings, save_renderer_settings_pruned};

#[cfg(test)]
pub(crate) use resolve::{ConfigResolveOutcome, ConfigSource};
