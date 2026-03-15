//! Per-space draw batch for rendering.

use nalgebra::Matrix4;

use crate::shared::RenderTransform;

/// Per-space draw batch for rendering.
#[derive(Clone)]
pub struct SpaceDrawBatch {
    /// Scene/space identifier.
    pub space_id: i32,
    /// Whether this is an overlay.
    pub is_overlay: bool,
    /// View transform for this space.
    pub view_transform: RenderTransform,
    /// Draws: (model_matrix, mesh_asset_id, is_skinned, material_id, bone_transform_ids for skinned).
    pub draws: Vec<(Matrix4<f32>, i32, bool, i32, Option<Vec<i32>>)>,
}
