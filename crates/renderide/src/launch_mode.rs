//! Resolved render launch mode: desktop vs OpenXR (see plan Phase 5).

use std::env;

/// How the renderer presents: desktop window only, or OpenXR session with mirror window.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RenderMode {
    /// Standard winit + wgpu surface (default).
    #[default]
    Desktop,
    /// Vulkan device shared with OpenXR; mirror window + headset submission when active.
    OpenXr,
}

impl RenderMode {
    /// Environment variable overriding CLI and config: `desktop` or `openxr`.
    pub const ENV_RENDER_MODE: &'static str = "RENDERIDE_RENDER_MODE";

    /// When set to `1`, show a desktop vs VR dialog if mode is not already chosen (see [`Self::resolve_with_optional_dialog`]).
    pub const ENV_PROMPT_RENDER_MODE: &'static str = "RENDERIDE_PROMPT_RENDER_MODE";

    /// Skip the optional `rfd` dialog (CI / headless / automation).
    pub const ENV_SKIP_RENDER_MODE_DIALOG: &'static str = "RENDERIDE_SKIP_RENDER_MODE_DIALOG";

    /// CLI token for [`Self::Desktop`].
    pub const ARG_DESKTOP: &'static str = "desktop";
    /// CLI token for [`Self::OpenXr`].
    pub const ARG_OPENXR: &'static str = "openxr";

    /// Parses a single token (case-insensitive).
    pub fn parse_token(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            Self::ARG_DESKTOP => Some(Self::Desktop),
            Self::ARG_OPENXR => Some(Self::OpenXr),
            _ => None,
        }
    }

    /// Resolves mode: `RENDERIDE_RENDER_MODE`, then `--render-mode` / `-RenderMode`, then optional
    /// `config.ini` value ([`crate::config::LaunchSettings::render_mode`]), then `default`.
    pub fn resolve_from_env_args_and_ini(ini: Option<Self>, default: Self) -> Self {
        if let Ok(v) = env::var(Self::ENV_RENDER_MODE) {
            if let Some(m) = Self::parse_token(&v) {
                return m;
            }
        }
        if let Some(m) = Self::parse_from_cli() {
            return m;
        }
        if let Some(m) = ini {
            return m;
        }
        default
    }

    /// Like [`Self::resolve_from_env_args_and_ini`], but may show an [`rfd`] dialog when nothing else fixed the mode.
    ///
    /// The dialog runs only if **all** of the following hold:
    /// - `RENDERIDE_RENDER_MODE` is unset
    /// - no `--render-mode` / `-RenderMode` argument with a valid token
    /// - `ini_mode` is [`None`]
    /// - `RENDERIDE_PROMPT_RENDER_MODE=1`
    /// - `CI` is unset, and `RENDERIDE_SKIP_RENDER_MODE_DIALOG` is unset
    ///
    /// **Yes** → [`Self::OpenXr`], **No** → [`Self::Desktop`].
    pub fn resolve_with_optional_dialog(ini_mode: Option<Self>, default: Self) -> Self {
        let resolved = Self::resolve_from_env_args_and_ini(ini_mode, default);
        if env::var(Self::ENV_RENDER_MODE).is_ok() {
            return resolved;
        }
        if Self::parse_from_cli().is_some() {
            return resolved;
        }
        if ini_mode.is_some() {
            return resolved;
        }
        if env::var(Self::ENV_PROMPT_RENDER_MODE).ok().as_deref() != Some("1") {
            return resolved;
        }
        if env::var("CI").is_ok() || env::var(Self::ENV_SKIP_RENDER_MODE_DIALOG).is_ok() {
            return resolved;
        }
        Self::prompt_desktop_or_openxr().unwrap_or(resolved)
    }

    fn parse_from_cli() -> Option<Self> {
        let args: Vec<String> = env::args().collect();
        let mut i = 0usize;
        while i < args.len() {
            let a = args[i].as_str();
            if a.eq_ignore_ascii_case("--render-mode") || a.eq_ignore_ascii_case("-rendermode") {
                if let Some(next) = args.get(i + 1).and_then(|s| Self::parse_token(s)) {
                    return Some(next);
                }
            }
            i += 1;
        }
        None
    }

    fn prompt_desktop_or_openxr() -> Option<Self> {
        let yes = rfd::MessageDialog::new()
            .set_title("Renderide")
            .set_description("Launch using OpenXR (VR headset) instead of desktop?\n\nYes = OpenXR\nNo = Desktop")
            .set_buttons(rfd::MessageButtons::YesNo)
            .show();
        match yes {
            rfd::MessageDialogResult::Yes => Some(Self::OpenXr),
            rfd::MessageDialogResult::No => Some(Self::Desktop),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_tokens() {
        assert_eq!(
            RenderMode::parse_token("desktop"),
            Some(RenderMode::Desktop)
        );
        assert_eq!(RenderMode::parse_token("OPENXR"), Some(RenderMode::OpenXr));
        assert_eq!(RenderMode::parse_token("nope"), None);
    }
}
