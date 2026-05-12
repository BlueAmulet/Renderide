//! Scene transforms overlay: per-render-space world TRS tables.

use imgui::{ListClipper, TabItem, TabItemFlags, TableFlags, TreeNodeFlags};

use crate::diagnostics::SceneTransformsSnapshot;
use crate::diagnostics::snapshots::RenderSpaceTransformsSnapshot;
use crate::shared::LayerType;

const TAG_OVERLAY: [f32; 4] = [0.40, 1.00, 0.55, 1.00];
const TAG_HIDDEN: [f32; 4] = [1.00, 0.55, 1.00, 1.00];
const DIM: [f32; 4] = [0.55, 0.55, 0.55, 1.00];

use super::super::layout::{self, Viewport, WindowSlot};
use super::super::state::HudUiState;
use super::super::view::HudWindow;

const SCENE_W: f32 = 720.0;
const SCENE_H: f32 = 420.0;

/// **Scene transforms** HUD window -- one tab per render space, clipped TRS table per tab.
///
/// First-use Y prefers the bottom of the viewport minus [`SCENE_H`] but never crosses
/// [`layout::scene_transforms_min_y`] (avoids covering the **Renderer config** + **Frame timing**
/// stack on first use).
pub struct SceneTransformsWindow;

impl HudWindow for SceneTransformsWindow {
    type Data<'a> = &'a SceneTransformsSnapshot;
    type State = HudUiState;

    fn title(&self) -> &str {
        "Scene transforms"
    }

    fn anchor(&self, viewport: Viewport) -> WindowSlot {
        let y = layout::scene_transforms_y(viewport.height as f32, SCENE_H);
        WindowSlot {
            position: [layout::MARGIN, y],
            size: [SCENE_W, SCENE_H],
            size_min: [360.0, 220.0],
            size_max: [f32::INFINITY, f32::INFINITY],
        }
    }

    fn bg_alpha(&self) -> f32 {
        0.85
    }

    fn read_open_flag(&self, state: &Self::State) -> Option<bool> {
        Some(state.scene_transforms_open)
    }

    fn write_open_flag(&self, state: &mut Self::State, value: bool) {
        state.scene_transforms_open = value;
    }

    fn body(&self, ui: &imgui::Ui, snapshot: Self::Data<'_>, state: &mut Self::State) {
        // Self-diagnostic banner so "no spaces" is actionable: distinguishes "capture never fired"
        // from "capture ran but had nothing".
        if !snapshot.captured {
            ui.text_colored(
                [1.00, 0.55, 0.55, 1.00],
                "Snapshot has never been captured.",
            );
            ui.text_disabled(
                "Check Renderer config -> Scene transforms HUD is on AND F7 shows ImGui.",
            );
            return;
        }
        if snapshot.spaces.is_empty() {
            ui.text_colored(
                [1.00, 0.90, 0.40, 1.00],
                format!(
                    "Capture ran but produced 0 spaces (scene.render_space_count() = {}).",
                    snapshot.scene_space_count
                ),
            );
            if snapshot.scene_space_count > 0 {
                ui.text_disabled(
                    "Scene has spaces but iteration filtered them out -- this is a bug.",
                );
            } else {
                ui.text_disabled("Scene genuinely empty -- waiting for host FrameSubmitData.");
            }
            return;
        }
        ui.text_disabled(format!(
            "{} space(s) shown  (scene reports {})",
            snapshot.spaces.len(),
            snapshot.scene_space_count
        ));
        if let Some(_bar) = ui.tab_bar("scene_transform_tabs") {
            for space in &snapshot.spaces {
                let tab_label = format!("Space {}##tab_space_{}", space.space_id, space.space_id);
                let flags = if state.scene_transforms_space_restore_pending
                    && state.scene_transforms_space_id == Some(space.space_id)
                {
                    TabItemFlags::SET_SELECTED
                } else {
                    TabItemFlags::empty()
                };
                if let Some(_tab) = TabItem::new(tab_label).flags(flags).begin(ui) {
                    state.scene_transforms_space_id = Some(space.space_id);
                    state.scene_transforms_space_restore_pending = false;
                    scene_transform_space_tab(ui, space);
                }
            }
        }
    }
}

/// Renders space header fields, the layer-assignments roll-up, and the transform table.
fn scene_transform_space_tab(ui: &imgui::Ui, space: &RenderSpaceTransformsSnapshot) {
    ui.text(format!(
        "active={}  overlay={}  private={}",
        space.is_active, space.is_overlay, space.is_private
    ));

    let overlay_count = space
        .layer_assignments
        .iter()
        .filter(|e| e.layer == LayerType::Overlay)
        .count();
    let hidden_count = space
        .layer_assignments
        .iter()
        .filter(|e| e.layer == LayerType::Hidden)
        .count();
    let header = format!(
        "Layer assignments  ({} total, {} overlay, {} hidden)##layers_{}",
        space.layer_assignments.len(),
        overlay_count,
        hidden_count,
        space.space_id,
    );
    if ui.collapsing_header(&header, TreeNodeFlags::DEFAULT_OPEN) {
        if space.layer_assignments.is_empty() {
            ui.indent_by(8.0);
            ui.text_disabled("(no LayerComponent registered against this space)");
            ui.unindent_by(8.0);
        } else {
            let table_id = format!("layer_assignments##space_{}", space.space_id);
            let flags = TableFlags::BORDERS | TableFlags::ROW_BG | TableFlags::SIZING_STRETCH_PROP;
            if let Some(_t) = ui.begin_table_with_sizing(&table_id, 2, flags, [0.0, 0.0], 0.0) {
                ui.table_setup_column("Node");
                ui.table_setup_column("Layer");
                ui.table_headers_row();
                for entry in &space.layer_assignments {
                    ui.table_next_row();
                    ui.table_next_column();
                    ui.text(format!("{}", entry.node_id));
                    ui.table_next_column();
                    layer_text(ui, Some(entry.layer));
                }
            }
        }
        ui.spacing();
    }

    let rows = &space.rows;
    let n = rows.len();
    let table_id = format!("transforms##space_{}", space.space_id);
    let table_flags = TableFlags::BORDERS
        | TableFlags::ROW_BG
        | TableFlags::SCROLL_Y
        | TableFlags::RESIZABLE
        | TableFlags::SIZING_STRETCH_PROP;
    if let Some(_table) = ui.begin_table_with_sizing(&table_id, 6, table_flags, [0.0, 320.0], 0.0) {
        ui.table_setup_column("ID");
        ui.table_setup_column("Parent");
        ui.table_setup_column("Layer");
        ui.table_setup_column("Translation (world)");
        ui.table_setup_column("Rotation (xyzw)");
        ui.table_setup_column("Scale (world)");
        ui.table_headers_row();

        let clip = ListClipper::new(n as i32);
        let tok = clip.begin(ui);
        for row_i in tok.iter() {
            let row = &rows[row_i as usize];
            ui.table_next_row();
            ui.table_next_column();
            ui.text(format!("{}", row.transform_id));
            ui.table_next_column();
            ui.text(format!("{}", row.parent_id));
            ui.table_next_column();
            layer_text(ui, row.resolved_layer);
            match &row.world {
                None => {
                    ui.table_next_column();
                    ui.text_disabled("--");
                    ui.table_next_column();
                    ui.text_disabled("--");
                    ui.table_next_column();
                    ui.text_disabled("--");
                }
                Some(w) => {
                    ui.table_next_column();
                    ui.text(format!(
                        "{:.4}  {:.4}  {:.4}",
                        w.translation.x, w.translation.y, w.translation.z
                    ));
                    ui.table_next_column();
                    ui.text(format!(
                        "{:.4}  {:.4}  {:.4}  {:.4}",
                        w.rotation.x, w.rotation.y, w.rotation.z, w.rotation.w
                    ));
                    ui.table_next_column();
                    ui.text(format!(
                        "{:.4}  {:.4}  {:.4}",
                        w.scale.x, w.scale.y, w.scale.z
                    ));
                }
            }
        }
    }
}

fn layer_text(ui: &imgui::Ui, layer: Option<LayerType>) {
    match layer {
        Some(LayerType::Overlay) => ui.text_colored(TAG_OVERLAY, "Overlay"),
        Some(LayerType::Hidden) => ui.text_colored(TAG_HIDDEN, "Hidden"),
        None => ui.text_colored(DIM, "--"),
    }
}
