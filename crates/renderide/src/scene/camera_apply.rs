//! [`CameraRenderablesUpdate`] ingestion from shared memory (FrooxEngine `CamerasManager` parity).

use crate::ipc::SharedMemoryAccessor;
use crate::shared::{CameraRenderablesUpdate, CameraState, CAMERA_STATE_HOST_ROW_BYTES};

use super::error::SceneError;
use super::render_space::RenderSpaceState;

/// One host camera renderable in a render space (dense table; `renderable_index` ↔ row in host state buffer).
#[derive(Debug, Clone)]
pub struct CameraRenderableEntry {
    /// Dense index in [`RenderSpaceState::cameras`] (matches [`CameraState::renderable_index`]).
    pub renderable_index: i32,
    /// Node / transform index for the camera component.
    pub transform_id: i32,
    /// Latest packed state from shared memory.
    pub state: CameraState,
    /// When non-empty, only these transform indices are drawn (Unity selective list).
    pub selective_transform_ids: Vec<i32>,
    /// Transform indices excluded from drawing when selective is empty.
    pub exclude_transform_ids: Vec<i32>,
}

/// Applies [`CameraRenderablesUpdate`]: removals → additions → state rows + transform id lists.
pub(crate) fn apply_camera_renderables_update(
    space: &mut RenderSpaceState,
    shm: &mut SharedMemoryAccessor,
    update: &CameraRenderablesUpdate,
    scene_id: i32,
) -> Result<(), SceneError> {
    if update.removals.length > 0 {
        let ctx = format!("camera removals scene_id={scene_id}");
        let removals = shm
            .access_copy_diagnostic_with_context::<i32>(&update.removals, Some(&ctx))
            .map_err(SceneError::SharedMemoryAccess)?;
        for &raw in removals.iter().take_while(|&&i| i >= 0) {
            let idx = raw as usize;
            if idx < space.cameras.len() {
                space.cameras.swap_remove(idx);
            }
        }
    }
    if update.additions.length > 0 {
        let ctx = format!("camera additions scene_id={scene_id}");
        let additions = shm
            .access_copy_diagnostic_with_context::<i32>(&update.additions, Some(&ctx))
            .map_err(SceneError::SharedMemoryAccess)?;
        let added_node_ids: Vec<i32> = additions.iter().take_while(|&&i| i >= 0).copied().collect();
        for &node_id in &added_node_ids {
            space.cameras.push(CameraRenderableEntry {
                renderable_index: -1,
                transform_id: node_id,
                state: CameraState::default(),
                selective_transform_ids: Vec::new(),
                exclude_transform_ids: Vec::new(),
            });
        }
    }
    if update.states.length > 0 {
        let ctx = format!("camera states scene_id={scene_id}");
        let states = shm
            .access_copy_memory_packable_rows::<CameraState>(
                &update.states,
                CAMERA_STATE_HOST_ROW_BYTES,
                Some(&ctx),
            )
            .map_err(SceneError::SharedMemoryAccess)?;
        let transform_ids = if update.transform_ids.length > 0 {
            let ctx_t = format!("camera transform_ids scene_id={scene_id}");
            Some(
                shm.access_copy_diagnostic_with_context::<i32>(&update.transform_ids, Some(&ctx_t))
                    .map_err(SceneError::SharedMemoryAccess)?,
            )
        } else {
            None
        };
        let mut tid_cursor = 0usize;
        for state in states {
            if state.renderable_index < 0 {
                break;
            }
            let idx = state.renderable_index as usize;
            let Some(entry) = space.cameras.get_mut(idx) else {
                continue;
            };
            entry.renderable_index = state.renderable_index;
            entry.state = state;
            let sel = state.selective_render_count.max(0) as usize;
            let excl = state.exclude_render_count.max(0) as usize;
            let need = sel.saturating_add(excl);
            let tids = transform_ids.as_deref();
            if let Some(slice) = tids {
                if tid_cursor.saturating_add(need) <= slice.len() {
                    if sel > 0 {
                        entry.selective_transform_ids =
                            slice[tid_cursor..tid_cursor + sel].to_vec();
                        tid_cursor += sel;
                    } else {
                        entry.selective_transform_ids.clear();
                    }
                    if excl > 0 {
                        entry.exclude_transform_ids = slice[tid_cursor..tid_cursor + excl].to_vec();
                        tid_cursor += excl;
                    } else {
                        entry.exclude_transform_ids.clear();
                    }
                } else {
                    logger::warn!(
                        "camera state renderable_index={}: transform_ids buffer too short (need {need} after {tid_cursor}, len {})",
                        state.renderable_index,
                        slice.len()
                    );
                    entry.selective_transform_ids.clear();
                    entry.exclude_transform_ids.clear();
                }
            } else {
                entry.selective_transform_ids.clear();
                entry.exclude_transform_ids.clear();
            }
        }
    }
    Ok(())
}
