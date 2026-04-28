//! Shader route listing (embedded vs fallback).

use crate::diagnostics::FrameDiagnosticsSnapshot;

/// Host shader asset id, AssetBundle shader asset name (or `<none>`), and material family per line.
pub(super) fn shader_mappings_tab(
    ui: &imgui::Ui,
    frame: Option<&FrameDiagnosticsSnapshot>,
    only_fallback: &mut bool,
) {
    let Some(d) = frame else {
        ui.text("Waiting for frame diagnostics…");
        return;
    };
    ui.checkbox("Only fallback routes", only_fallback);
    if d.shader_routes.is_empty() {
        ui.text("No shader route data");
    } else {
        for route in &d.shader_routes {
            if *only_fallback && route.implemented {
                continue;
            }
            ui.text_wrapped(format!(
                "{}  {}  {}  {}",
                route.shader_asset_id,
                route.shader_asset_name.as_deref().unwrap_or("<none>"),
                route.pipeline_label,
                if route.implemented {
                    "implemented"
                } else {
                    "fallback"
                },
            ));
        }
    }
}
