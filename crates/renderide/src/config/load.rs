//! Layered loader for [`super::types::RendererSettings`].
//!
//! The load pipeline is expressed as an explicit ordered [`Vec<ConfigLayer>`] so the precedence
//! chain (defaults -> TOML file -> `RENDERIDE_*` env -> post-extract overrides like
//! `RENDERIDE_GPU_VALIDATION`) is visible in one place. Each layer is one of the variants of
//! [`ConfigLayer`]; pre-extract layers feed the figment merge, post-extract layers run as
//! mutators on the extracted [`super::types::RendererSettings`].
//!
//! [`load_renderer_settings`] is the entry point used by the bootstrap; it builds the canonical
//! pipeline for the requested [`ConfigFilePolicy`] and runs it.

use std::path::PathBuf;

use figment::Figment;
use figment::providers::{Env, Format, Serialized, Toml};
use toml_edit::{DocumentMut, Item, Table, value};

use super::resolve::{
    ConfigResolveOutcome, ConfigSource, apply_generated_config, is_dir_writable, read_config_file,
    renderide_config_env_nonempty, resolve_config_path, resolve_save_path,
};
use super::save::{save_migrated_renderer_config, save_renderer_settings_pruned};
use super::types::{AutoExposureSettings, RendererSettings};

const MAX_COMPATIBILITY_DROPS: usize = 64;
const LEGACY_AUTO_EXPOSURE_DEFAULT_TARGET_EV: f64 = -3.0;

/// Controls whether the TOML config file is consulted during startup.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ConfigFilePolicy {
    /// Normal: discover, load, and (if absent) auto-create `config.toml`.
    #[default]
    Load,
    /// Skip all file I/O; use struct defaults plus `RENDERIDE_*` env vars only.
    /// Forces `suppress_config_disk_writes = true`.
    Ignore,
}

/// Full load result: resolved path and save path for persistence.
#[derive(Clone, Debug)]
pub struct ConfigLoadResult {
    /// Effective settings after merge.
    pub settings: RendererSettings,
    /// Path resolution diagnostics.
    pub resolve: ConfigResolveOutcome,
    /// Target file for [`super::save::save_renderer_settings`] and the ImGui config window.
    pub save_path: PathBuf,
    /// When `true`, disk persistence is disabled until restart because startup config extraction
    /// failed in a way that could not be repaired by ignoring an incompatible TOML key.
    pub suppress_config_disk_writes: bool,
}

/// One step in the [`LoadPipeline`].
///
/// The pre-extract variants ([`Self::Defaults`], [`Self::Toml`], [`Self::EnvPrefixed`]) feed the
/// figment merge; [`Self::PostExtract`] runs as a mutator on the extracted
/// [`RendererSettings`] after extraction. Layers are applied in the order they appear in the
/// pipeline.
pub enum ConfigLayer {
    /// Insert struct defaults via [`Serialized::defaults`].
    Defaults,
    /// Merge an in-memory TOML string. Use this when the resolver located a file on disk and
    /// loaded its contents.
    Toml(String),
    /// Merge `RENDERIDE_*`-style environment variables. `prefix` is the env-var prefix
    /// (typically `"RENDERIDE_"`), `separator` is the nested-key separator
    /// (typically `"__"`).
    EnvPrefixed {
        /// Environment variable prefix (e.g. `"RENDERIDE_"`).
        prefix: &'static str,
        /// Nested-key separator (e.g. `"__"` to map `RENDERIDE_DEBUG__GPU_VALIDATION_LAYERS`
        /// to `debug.gpu_validation_layers`).
        separator: &'static str,
    },
    /// Post-extract mutator. Runs after [`figment::Figment::extract`] succeeds; useful for
    /// special-case env overrides that don't fit the structured `RENDERIDE_*` namespace
    /// (currently only `RENDERIDE_GPU_VALIDATION`).
    PostExtract(fn(&mut RendererSettings)),
}

/// An ordered chain of [`ConfigLayer`] entries. Construct with [`LoadPipeline::new`] then push
/// layers, or build the canonical chain via [`canonical_layers`] / [`load_renderer_settings`].
#[derive(Default)]
pub struct LoadPipeline {
    layers: Vec<ConfigLayer>,
}

impl LoadPipeline {
    /// Empty pipeline (no defaults inserted yet -- the canonical chain always starts with
    /// [`ConfigLayer::Defaults`]).
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a layer to the pipeline.
    pub fn push(&mut self, layer: ConfigLayer) -> &mut Self {
        self.layers.push(layer);
        self
    }

    /// Runs the pipeline: builds the figment from pre-extract layers, extracts
    /// [`RendererSettings`], then runs post-extract mutators in order.
    pub fn extract(self) -> Result<RendererSettings, Box<figment::Error>> {
        let mut figment = Figment::new();
        let mut mutators: Vec<fn(&mut RendererSettings)> = Vec::new();
        for layer in self.layers {
            match layer {
                ConfigLayer::Defaults => {
                    figment = figment.merge(Serialized::defaults(RendererSettings::default()));
                }
                ConfigLayer::Toml(content) => {
                    figment = figment.merge(Toml::string(&content));
                }
                ConfigLayer::EnvPrefixed { prefix, separator } => {
                    figment = figment.merge(Env::prefixed(prefix).split(separator));
                }
                ConfigLayer::PostExtract(f) => {
                    mutators.push(f);
                }
            }
        }
        let mut settings = figment.extract::<RendererSettings>().map_err(Box::new)?;
        for f in mutators {
            f(&mut settings);
        }
        Ok(settings)
    }
}

/// Builds the canonical `RENDERIDE_*` env layering with post-extract mutators, optionally
/// including a TOML layer when `toml_content` is provided.
pub fn canonical_layers(toml_content: Option<String>) -> Vec<ConfigLayer> {
    let mut v = Vec::with_capacity(5);
    v.push(ConfigLayer::Defaults);
    if let Some(content) = toml_content {
        v.push(ConfigLayer::Toml(content));
    }
    v.push(ConfigLayer::EnvPrefixed {
        prefix: "RENDERIDE_",
        separator: "__",
    });
    v.push(ConfigLayer::PostExtract(apply_renderide_gpu_validation_env));
    v.push(ConfigLayer::PostExtract(apply_current_config_version));
    v
}

/// Overrides [`super::types::DebugSettings::gpu_validation_layers`] when
/// `RENDERIDE_GPU_VALIDATION` is set.
///
/// Truthy values (`1`, `true`, `yes`) force validation on; falsey (`0`, `false`, `no`) force
/// off. If unset, the value from config or defaults is unchanged. Wired into the canonical
/// pipeline as a [`ConfigLayer::PostExtract`] entry so the precedence rule lives next to the
/// other layers.
pub fn apply_renderide_gpu_validation_env(settings: &mut RendererSettings) {
    match std::env::var("RENDERIDE_GPU_VALIDATION").as_deref() {
        Ok("1" | "true" | "yes") => settings.debug.gpu_validation_layers = true,
        Ok("0" | "false" | "no") => settings.debug.gpu_validation_layers = false,
        _ => {}
    }
}

/// Pins runtime settings to the config schema version emitted by this renderer build.
pub fn apply_current_config_version(settings: &mut RendererSettings) {
    RendererSettings::CURRENT_CONFIG_VERSION.clone_into(&mut settings.config_version);
}

/// Resolves `config.toml`, runs the canonical [`LoadPipeline`], and produces a
/// [`ConfigLoadResult`].
///
/// Precedence (top wins): post-extract mutators (`RENDERIDE_GPU_VALIDATION`) -> `RENDERIDE_*`
/// env -> TOML file (skipped under [`ConfigFilePolicy::Ignore`]) -> struct defaults.
///
/// When no file exists and [`renderide_config_env_nonempty`] is false, writes defaults to the
/// save path (see [`super::resolve::resolve_save_path`]) and loads that file. This
/// auto-creation is skipped when `policy` is [`ConfigFilePolicy::Ignore`].
pub fn load_renderer_settings(policy: ConfigFilePolicy) -> ConfigLoadResult {
    if policy == ConfigFilePolicy::Ignore {
        return load_with_ignore_policy();
    }

    let mut resolve = resolve_config_path();
    let mut suppress_config_disk_writes = false;
    let mut settings = initial_settings_from_resolve(&mut suppress_config_disk_writes, &resolve);

    if resolve.loaded_path.is_none() && !renderide_config_env_nonempty() {
        maybe_create_default_config_and_reload(
            &mut resolve,
            &mut settings,
            &mut suppress_config_disk_writes,
        );
    }

    let save_path = resolve_save_path(&resolve);
    logger::trace!("Renderer config will persist to {}", save_path.display());

    ConfigLoadResult {
        settings,
        resolve,
        save_path,
        suppress_config_disk_writes,
    }
}

/// Builds the [`ConfigFilePolicy::Ignore`] result: skip TOML, run defaults+env+overrides only,
/// and force `suppress_config_disk_writes`.
fn load_with_ignore_policy() -> ConfigLoadResult {
    if renderide_config_env_nonempty() {
        logger::warn!(
            "--ignore-config is active; RENDERIDE_CONFIG is also set but the file will be skipped"
        );
    }
    let settings = match run_pipeline(None) {
        Ok(s) => s,
        Err(e) => {
            logger::error!(
                "Renderer config Figment extract failed (--ignore-config, defaults+env): {e:#}"
            );
            RendererSettings::default()
        }
    };
    let resolve = ConfigResolveOutcome {
        attempted_paths: vec![],
        loaded_path: None,
        source: ConfigSource::None,
    };
    let save_path = resolve_save_path(&resolve);
    logger::info!("--ignore-config: skipping TOML file; using struct defaults + RENDERIDE_* env");
    ConfigLoadResult {
        settings,
        resolve,
        save_path,
        suppress_config_disk_writes: true,
    }
}

/// Runs the canonical pipeline with optional TOML content.
fn run_pipeline(toml_content: Option<String>) -> Result<RendererSettings, Box<figment::Error>> {
    let mut pipeline = LoadPipeline::new();
    for layer in canonical_layers(toml_content) {
        pipeline.push(layer);
    }
    pipeline.extract()
}

#[derive(Debug)]
struct ConfigCompatibilityDrop {
    path: String,
    value: String,
    error: String,
}

#[derive(Debug)]
struct ToleratedTomlLoad {
    settings: RendererSettings,
    drops: Vec<ConfigCompatibilityDrop>,
    migrated_toml: Option<String>,
}

fn run_pipeline_tolerating_toml(
    toml_content: &str,
) -> Result<ToleratedTomlLoad, Box<figment::Error>> {
    let Ok(mut document) = toml_content.parse::<DocumentMut>() else {
        return run_pipeline(Some(toml_content.to_string())).map(|settings| ToleratedTomlLoad {
            settings,
            drops: vec![],
            migrated_toml: None,
        });
    };

    let migrated_toml = migrate_unversioned_config(&mut document).then(|| document.to_string());

    let mut drops = Vec::new();
    for _ in 0..MAX_COMPATIBILITY_DROPS {
        match run_pipeline(Some(document.to_string())) {
            Ok(settings) => {
                return Ok(ToleratedTomlLoad {
                    settings,
                    drops,
                    migrated_toml,
                });
            }
            Err(e) => {
                let Some(path) = compatibility_error_path(&e) else {
                    return Err(e);
                };
                let Some(removed) = remove_document_path(&mut document, &path) else {
                    return Err(e);
                };
                drops.push(ConfigCompatibilityDrop {
                    path: path.join("."),
                    value: summarize_removed_item(&removed),
                    error: e.to_string(),
                });
            }
        }
    }

    run_pipeline(Some(document.to_string())).map(|settings| ToleratedTomlLoad {
        settings,
        drops,
        migrated_toml,
    })
}

fn migrate_unversioned_config(document: &mut DocumentMut) -> bool {
    if document.get("config_version").is_some() {
        return false;
    }

    migrate_unversioned_auto_exposure_compensation(document);
    document.as_table_mut().insert(
        "config_version",
        value(RendererSettings::CURRENT_CONFIG_VERSION),
    );
    true
}

fn migrate_unversioned_auto_exposure_compensation(document: &mut DocumentMut) {
    // TODO(2026-05-24): Remove this one-time migration after the 2026-05-10 introduction has aged out.
    let target_ev = document
        .get("post_processing")
        .and_then(Item::as_table)
        .and_then(|table| table.get("auto_exposure"))
        .and_then(Item::as_table)
        .and_then(|table| table.get("compensation_ev"))
        .and_then(item_to_f64)
        .unwrap_or(LEGACY_AUTO_EXPOSURE_DEFAULT_TARGET_EV);
    let compensation_ev = target_ev - f64::from(AutoExposureSettings::MIDDLE_GRAY_EV);

    let Some(post_processing) = get_or_create_table(document.as_table_mut(), "post_processing")
    else {
        return;
    };
    let Some(auto_exposure) = get_or_create_table(post_processing, "auto_exposure") else {
        return;
    };
    auto_exposure.insert("compensation_ev", value(compensation_ev));
}

fn get_or_create_table<'a>(table: &'a mut Table, key: &str) -> Option<&'a mut Table> {
    table
        .entry(key)
        .or_insert_with(|| Item::Table(Table::new()))
        .as_table_mut()
}

fn item_to_f64(item: &Item) -> Option<f64> {
    item.as_value().and_then(|value| {
        value
            .as_float()
            .or_else(|| value.as_integer().map(|v| v as f64))
    })
}

fn compatibility_error_path(error: &figment::Error) -> Option<Vec<String>> {
    let path = error
        .path
        .iter()
        .filter(|segment| segment.as_str() != "default")
        .cloned()
        .collect::<Vec<_>>();
    if path.is_empty() { None } else { Some(path) }
}

fn remove_document_path(document: &mut DocumentMut, path: &[String]) -> Option<Item> {
    let (last, parents) = path.split_last()?;
    let mut table = document.as_table_mut();
    for segment in parents {
        table = table.get_mut(segment)?.as_table_mut()?;
    }
    table.remove(last)
}

fn summarize_removed_item(item: &Item) -> String {
    const MAX_LEN: usize = 160;
    let mut text = item.to_string().replace(['\n', '\r'], " ");
    text = text.trim().to_string();
    if text.len() > MAX_LEN {
        text.truncate(MAX_LEN);
        text.push_str("...");
    }
    text
}

fn log_compatibility_drops(path: &std::path::Path, drops: &[ConfigCompatibilityDrop]) {
    for drop in drops {
        logger::warn!(
            "Ignoring incompatible renderer config key {} in {}: {} ({})",
            drop.path,
            path.display(),
            drop.value,
            drop.error
        );
    }
}

/// Loads settings from a resolved config path, or defaults plus env when the file is missing or
/// unreadable.
fn initial_settings_from_resolve(
    suppress_config_disk_writes: &mut bool,
    resolve: &ConfigResolveOutcome,
) -> RendererSettings {
    if let Some(path) = resolve.loaded_path.as_ref() {
        logger::info!("Loading renderer config from {}", path.display());
        match read_config_file(path) {
            Ok(content) => match run_pipeline_tolerating_toml(&content) {
                Ok(load) => {
                    log_compatibility_drops(path, &load.drops);
                    persist_migrated_toml(path, load.migrated_toml.as_deref());
                    load.settings
                }
                Err(e) => {
                    logger::error!(
                        "Renderer config Figment extract failed for {}: {e:#}",
                        path.display()
                    );
                    *suppress_config_disk_writes = true;
                    RendererSettings::default()
                }
            },
            Err(e) => {
                logger::warn!("Failed to read {}: {e}; using defaults", path.display());
                fallback_to_defaults_plus_env(suppress_config_disk_writes)
            }
        }
    } else {
        logger::info!("Renderer config file not found; using built-in defaults");
        logger::trace!(
            "config search tried {} path(s)",
            resolve.attempted_paths.len()
        );
        fallback_to_defaults_plus_env(suppress_config_disk_writes)
    }
}

/// Runs the pipeline without a TOML layer (defaults + env + post-extract overrides) and falls
/// back to [`RendererSettings::default`] on Figment failure.
fn fallback_to_defaults_plus_env(suppress_config_disk_writes: &mut bool) -> RendererSettings {
    match run_pipeline(None) {
        Ok(s) => s,
        Err(e) => {
            logger::error!("Renderer config Figment extract failed (defaults+env): {e:#}");
            *suppress_config_disk_writes = true;
            RendererSettings::default()
        }
    }
}

/// When no config was loaded and env overrides are empty, writes default `config.toml` and
/// reloads from disk.
fn maybe_create_default_config_and_reload(
    resolve: &mut ConfigResolveOutcome,
    settings: &mut RendererSettings,
    suppress_config_disk_writes: &mut bool,
) {
    let path = resolve_save_path(resolve);
    if path.exists() {
        return;
    }
    let Some(parent) = path.parent() else {
        return;
    };
    if !is_dir_writable(parent) {
        logger::trace!(
            "Not creating default config at {} (directory not writable)",
            path.display()
        );
        return;
    }
    match save_renderer_settings_pruned(&path, &RendererSettings::from_defaults()) {
        Ok(()) => {
            logger::info!("Created default renderer config at {}", path.display());
            apply_generated_config(resolve, path.clone());
            match read_config_file(&path) {
                Ok(content) => match run_pipeline_tolerating_toml(&content) {
                    Ok(load) => {
                        log_compatibility_drops(&path, &load.drops);
                        persist_migrated_toml(&path, load.migrated_toml.as_deref());
                        *settings = load.settings;
                    }
                    Err(e) => {
                        logger::error!(
                            "Figment extract failed for newly created {}: {e:#}",
                            path.display()
                        );
                        *suppress_config_disk_writes = true;
                    }
                },
                Err(e) => {
                    logger::warn!(
                        "Failed to read newly created {}: {e}; using defaults",
                        path.display()
                    );
                }
            }
        }
        Err(e) => {
            logger::warn!("Failed to create default config at {}: {e}", path.display());
        }
    }
}

fn persist_migrated_toml(path: &std::path::Path, migrated_toml: Option<&str>) {
    let Some(contents) = migrated_toml else {
        return;
    };

    match save_migrated_renderer_config(path, contents) {
        Ok(()) => logger::info!(
            "Migrated renderer config {} to config_version {}",
            path.display(),
            RendererSettings::CURRENT_CONFIG_VERSION
        ),
        Err(e) => logger::warn!(
            "Failed to persist migrated renderer config {}: {e}",
            path.display()
        ),
    }
}

/// Logs [`ConfigLoadResult::resolve`] at trace level for troubleshooting.
pub fn log_config_resolve_trace(resolve: &ConfigResolveOutcome) {
    if resolve.source == ConfigSource::None && !resolve.attempted_paths.is_empty() {
        for p in &resolve.attempted_paths {
            let exists = p.as_path().is_file();
            logger::trace!("  config candidate {} [{}]", p.display(), exists);
        }
    }
}

#[cfg(test)]
mod tests;
