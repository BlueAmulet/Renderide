//! Host layer assignment ingestion and inherited mesh layer resolution.

use crate::ipc::SharedMemoryAccessor;
use crate::shared::{LayerType, LayerUpdate};

use super::error::SceneError;
use super::render_space::{LayerAssignmentEntry, RenderSpaceState};
use super::transforms_apply::TransformRemovalEvent;
use super::world::fixup_transform_id;

pub(crate) fn apply_layer_update(
    space: &mut RenderSpaceState,
    shm: &mut SharedMemoryAccessor,
    update: &LayerUpdate,
    scene_id: i32,
) -> Result<(), SceneError> {
    if update.removals.length > 0 {
        let ctx = format!("layer removals scene_id={scene_id}");
        let removals = shm
            .access_copy_diagnostic_with_context::<i32>(&update.removals, Some(&ctx))
            .map_err(SceneError::SharedMemoryAccess)?;
        for &raw in removals.iter().take_while(|&&idx| idx >= 0) {
            let idx = raw as usize;
            if idx < space.layer_assignments.len() {
                space.layer_assignments.swap_remove(idx);
            }
        }
    }

    if update.additions.length > 0 {
        let ctx = format!("layer additions scene_id={scene_id}");
        let additions = shm
            .access_copy_diagnostic_with_context::<i32>(&update.additions, Some(&ctx))
            .map_err(SceneError::SharedMemoryAccess)?;
        for &node_id in additions.iter().take_while(|&&id| id >= 0) {
            space.layer_assignments.push(LayerAssignmentEntry {
                node_id,
                layer: LayerType::Hidden,
            });
        }
    }

    if update.layer_assignments.length > 0 {
        let ctx = format!("layer assignments scene_id={scene_id}");
        let assignments = shm
            .access_copy_diagnostic_with_context::<LayerType>(&update.layer_assignments, Some(&ctx))
            .map_err(SceneError::SharedMemoryAccess)?;
        for (idx, layer) in assignments.into_iter().enumerate() {
            let Some(entry) = space.layer_assignments.get_mut(idx) else {
                continue;
            };
            entry.layer = layer;
        }
    }

    Ok(())
}

pub(crate) fn resolve_mesh_layers_from_assignments(space: &mut RenderSpaceState) {
    let node_parents = &space.node_parents;
    let layer_assignments = &space.layer_assignments;

    for renderer in &mut space.static_mesh_renderers {
        renderer.layer = resolve_layer_for_node(node_parents, layer_assignments, renderer.node_id)
            .unwrap_or_default();
    }

    for renderer in &mut space.skinned_mesh_renderers {
        renderer.base.layer =
            resolve_layer_for_node(node_parents, layer_assignments, renderer.base.node_id)
                .unwrap_or_default();
    }
}

fn resolve_layer_for_node(
    node_parents: &[i32],
    layer_assignments: &[LayerAssignmentEntry],
    node_id: i32,
) -> Option<LayerType> {
    if node_id < 0 {
        return None;
    }

    let mut cursor = node_id;
    for _ in 0..node_parents.len() {
        if let Some(entry) = layer_assignments
            .iter()
            .rev()
            .find(|entry| entry.node_id == cursor)
        {
            return Some(entry.layer);
        }
        let parent = *node_parents.get(cursor as usize)?;
        if parent < 0 || parent == cursor || parent as usize >= node_parents.len() {
            return None;
        }
        cursor = parent;
    }
    None
}

pub(crate) fn fixup_layer_assignments_for_transform_removals(
    space: &mut RenderSpaceState,
    removals: &[TransformRemovalEvent],
) {
    for removal in removals {
        for entry in &mut space.layer_assignments {
            entry.node_id = fixup_transform_id(
                entry.node_id,
                removal.removed_index,
                removal.last_index_before_swap,
            );
        }
        space.layer_assignments.retain(|entry| entry.node_id >= 0);
    }
}

#[cfg(test)]
mod tests {
    use crate::scene::{LayerAssignmentEntry, RenderSpaceState, StaticMeshRenderer};
    use crate::shared::LayerType;

    use super::resolve_mesh_layers_from_assignments;

    #[test]
    fn resolves_layer_from_nearest_ancestor_assignment() {
        let mut space = RenderSpaceState {
            node_parents: vec![-1, 0, 1],
            layer_assignments: vec![
                LayerAssignmentEntry {
                    node_id: 0,
                    layer: LayerType::Overlay,
                },
                LayerAssignmentEntry {
                    node_id: 1,
                    layer: LayerType::Hidden,
                },
            ],
            static_mesh_renderers: vec![StaticMeshRenderer {
                node_id: 2,
                layer: LayerType::Overlay,
                ..Default::default()
            }],
            ..Default::default()
        };

        resolve_mesh_layers_from_assignments(&mut space);

        assert_eq!(space.static_mesh_renderers[0].layer, LayerType::Hidden);
    }

    #[test]
    fn inherited_overlay_applies_to_static_and_skinned_children() {
        let mut space = RenderSpaceState {
            node_parents: vec![-1, 0, 1],
            layer_assignments: vec![LayerAssignmentEntry {
                node_id: 1,
                layer: LayerType::Overlay,
            }],
            static_mesh_renderers: vec![StaticMeshRenderer {
                node_id: 2,
                ..Default::default()
            }],
            skinned_mesh_renderers: vec![crate::scene::SkinnedMeshRenderer {
                base: StaticMeshRenderer {
                    node_id: 2,
                    ..Default::default()
                },
                ..Default::default()
            }],
            ..Default::default()
        };

        resolve_mesh_layers_from_assignments(&mut space);

        assert_eq!(space.static_mesh_renderers[0].layer, LayerType::Overlay);
        assert_eq!(
            space.skinned_mesh_renderers[0].base.layer,
            LayerType::Overlay
        );
    }
}
