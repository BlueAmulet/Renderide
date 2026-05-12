//! Read-only snapshot of per-render-space world transforms for the debug HUD (no ImGui types).

use glam::{Quat, Vec3};

use crate::scene::{RenderSpaceId, SceneCoordinator};
use crate::shared::LayerType;

/// One frame's world-space transform samples for every tracked render space.
#[derive(Clone, Debug, Default)]
pub struct SceneTransformsSnapshot {
    /// Sorted by [`RenderSpaceTransformsSnapshot::space_id`] for stable UI tab order.
    pub spaces: Vec<RenderSpaceTransformsSnapshot>,
    /// `true` once [`Self::capture`] has populated this snapshot. Distinguishes "capture never
    /// fired" (gating bug) from "capture ran but scene has no spaces" in the HUD.
    pub captured: bool,
    /// `SceneCoordinator::render_space_count()` at capture time. Compared against `spaces.len()`
    /// to surface "scene has spaces but iteration skipped them" pathologies.
    pub scene_space_count: usize,
}

/// Host render-space id, flags, and one row per dense transform index.
#[derive(Clone, Debug)]
pub struct RenderSpaceTransformsSnapshot {
    /// Host dictionary key for this space.
    pub space_id: i32,
    /// Mirrors the render-space active flag.
    pub is_active: bool,
    /// Mirrors the render-space overlay flag.
    pub is_overlay: bool,
    /// Mirrors the render-space private flag.
    pub is_private: bool,
    /// One row per dense transform node in this space.
    pub rows: Vec<TransformRow>,
    /// Raw `LayerComponent` assignments registered for this space, in host append order. Each
    /// entry is a `(node_id, LayerType)` pair recording "this node and its descendants inherit
    /// this layer". Diagnostic-only -- read via [`crate::scene::RenderSpaceView::layer_assignments`].
    pub layer_assignments: Vec<LayerAssignmentRow>,
}

/// Dense transform index, parent link, and hierarchy world TRS when available.
#[derive(Clone, Debug)]
pub struct TransformRow {
    /// Dense index in the host transform arena (`0..node_count`).
    pub transform_id: usize,
    /// Parent dense index, or `-1` for a hierarchy root under the space root.
    pub parent_id: i32,
    /// Resolved special layer for this transform (inherited from the nearest ancestor with an
    /// `Overlay` / `Hidden` assignment). `None` means no special layer applies.
    pub resolved_layer: Option<LayerType>,
    /// World matrix from the parent chain; `None` if the cache had no valid entry.
    pub world: Option<WorldTransformSample>,
}

/// One row in a render space's [`RenderSpaceTransformsSnapshot::layer_assignments`] list.
#[derive(Clone, Copy, Debug)]
pub struct LayerAssignmentRow {
    /// Dense transform index the assignment is anchored to.
    pub node_id: i32,
    /// The layer that the host registered for this node.
    pub layer: LayerType,
}

/// Decomposed world TRS from [`SceneCoordinator::world_matrix`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldTransformSample {
    /// World-space translation.
    pub translation: Vec3,
    /// World-space rotation.
    pub rotation: Quat,
    /// World-space scale (lossy; from matrix decomposition).
    pub scale: Vec3,
}

impl SceneTransformsSnapshot {
    /// Builds a snapshot from the scene coordinator (call after [`SceneCoordinator::flush_world_caches`]
    /// when matrices must match the latest host submit).
    pub fn capture(scene: &SceneCoordinator) -> Self {
        let scene_space_count = scene.render_space_count();
        let mut spaces: Vec<RenderSpaceTransformsSnapshot> = scene
            .render_space_ids()
            .filter_map(|id| RenderSpaceTransformsSnapshot::capture_one(scene, id))
            .collect();
        spaces.sort_by_key(|s| s.space_id);
        Self {
            spaces,
            captured: true,
            scene_space_count,
        }
    }
}

impl RenderSpaceTransformsSnapshot {
    fn capture_one(scene: &SceneCoordinator, id: RenderSpaceId) -> Option<Self> {
        let space = scene.space(id)?;
        let space_id = id.0;
        let transforms = space.local_transforms();
        let parents = space.node_parents();
        let mut rows = Vec::with_capacity(transforms.len());
        for transform_id in 0..transforms.len() {
            let parent_id = parents.get(transform_id).copied().unwrap_or(-1);
            let resolved_layer = scene.transform_special_layer(id, transform_id);
            let world = scene
                .world_matrix(id, transform_id)
                .and_then(world_sample_from_mat4);
            rows.push(TransformRow {
                transform_id,
                parent_id,
                resolved_layer,
                world,
            });
        }
        let layer_assignments = space
            .layer_assignments()
            .iter()
            .map(|entry| LayerAssignmentRow {
                node_id: entry.node_id,
                layer: entry.layer,
            })
            .collect();
        Some(Self {
            space_id,
            is_active: space.is_active(),
            is_overlay: space.is_overlay(),
            is_private: space.is_private(),
            rows,
            layer_assignments,
        })
    }
}

fn world_sample_from_mat4(m: glam::Mat4) -> Option<WorldTransformSample> {
    let (scale, rotation, translation) = m.to_scale_rotation_translation();
    if !translation.is_finite() || !rotation.is_finite() || !scale.is_finite() {
        return None;
    }
    Some(WorldTransformSample {
        translation,
        rotation,
        scale,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_empty_scene() {
        let scene = SceneCoordinator::new();
        let snap = SceneTransformsSnapshot::capture(&scene);
        assert!(snap.spaces.is_empty());
    }

    #[test]
    fn capture_single_space_two_nodes_parent_child() {
        let mut scene = SceneCoordinator::new();
        let id = RenderSpaceId(7);
        scene.test_seed_space_identity_worlds(
            id,
            vec![
                crate::shared::RenderTransform::default(),
                crate::shared::RenderTransform::default(),
            ],
            vec![-1, 0],
        );

        let snap = SceneTransformsSnapshot::capture(&scene);
        assert_eq!(snap.spaces.len(), 1);
        assert_eq!(snap.spaces[0].space_id, 7);
        assert_eq!(snap.spaces[0].rows.len(), 2);
        assert_eq!(snap.spaces[0].rows[0].transform_id, 0);
        assert_eq!(snap.spaces[0].rows[0].parent_id, -1);
        assert_eq!(snap.spaces[0].rows[1].transform_id, 1);
        assert_eq!(snap.spaces[0].rows[1].parent_id, 0);
        assert!(snap.spaces[0].rows[0].resolved_layer.is_none());
        assert!(
            snap.spaces[0].rows[0].world.is_some(),
            "identity world matrix should decompose"
        );
        assert!(snap.spaces[0].layer_assignments.is_empty());
    }
}
