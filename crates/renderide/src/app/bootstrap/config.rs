//! App bootstrap configuration loading and GPU startup settings.

use logger::LogLevel;

use crate::config::{
    ConfigFilePolicy, ConfigLoadResult, GraphicsApiSetting, VsyncMode, load_renderer_settings,
    log_config_resolve_trace,
};
use crate::ipc::get_ignore_config;

/// Fixed swapchain frame latency used for every GPU startup path.
pub(crate) const MAX_FRAME_LATENCY: u32 = 2;

/// Initial GPU/swapchain knobs read once during process bootstrap.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GpuStartupConfig {
    /// Initial vsync preference resolved against surface capabilities by `GpuContext`.
    pub(crate) vsync: VsyncMode,
    /// Initial maximum swapchain frame latency.
    pub(crate) max_frame_latency: u32,
    /// Whether to enable wgpu/Vulkan validation layers at startup.
    pub(crate) gpu_validation_layers: bool,
    /// Adapter ranking preference for desktop/headless GPU selection.
    pub(crate) power_preference: wgpu::PowerPreference,
    /// Startup graphics API preference for desktop/headless GPU selection.
    pub(crate) graphics_api: GraphicsApiSetting,
}

/// App configuration bundle consumed by bootstrap dispatch.
pub(crate) struct AppConfig {
    /// Full renderer config load result.
    pub(crate) load: ConfigLoadResult,
    /// Initial GPU settings distilled from the renderer config.
    pub(crate) gpu: GpuStartupConfig,
}

/// Chooses the process max log level after file logging is initialized.
pub(crate) fn effective_renderer_log_level(cli: Option<LogLevel>, log_verbose: bool) -> LogLevel {
    if let Some(level) = cli {
        level
    } else if log_verbose {
        LogLevel::Trace
    } else {
        LogLevel::Debug
    }
}

/// Loads renderer config, applies log verbosity, and extracts GPU startup settings.
pub(crate) fn load_app_config(log_level_cli: Option<LogLevel>) -> AppConfig {
    let config_file_policy = if get_ignore_config() {
        ConfigFilePolicy::Ignore
    } else {
        ConfigFilePolicy::Load
    };
    let load = load_renderer_settings(config_file_policy);
    logger::set_max_level(effective_renderer_log_level(
        log_level_cli,
        load.settings.debug.log_verbose,
    ));
    log_config_resolve_trace(&load.resolve);

    let gpu = GpuStartupConfig {
        vsync: load.settings.rendering.vsync,
        max_frame_latency: MAX_FRAME_LATENCY,
        gpu_validation_layers: load.settings.debug.gpu_validation_layers,
        power_preference: load.settings.debug.power_preference.to_wgpu(),
        graphics_api: load.settings.rendering.graphics_api,
    };

    AppConfig { load, gpu }
}

#[cfg(test)]
mod tests {
    use super::effective_renderer_log_level;
    use logger::LogLevel;

    #[test]
    fn cli_always_overrides_log_verbose() {
        assert_eq!(
            effective_renderer_log_level(Some(LogLevel::Warn), true),
            LogLevel::Warn
        );
    }

    #[test]
    fn no_cli_uses_trace_when_log_verbose() {
        assert_eq!(effective_renderer_log_level(None, true), LogLevel::Trace);
    }

    #[test]
    fn no_cli_uses_debug_when_not_log_verbose() {
        assert_eq!(effective_renderer_log_level(None, false), LogLevel::Debug);
    }
}
