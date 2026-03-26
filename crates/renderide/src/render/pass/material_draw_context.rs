//! Per-draw material identity for [`MaterialPropertyStore`] merged lookup (material + mesh property block).

use crate::assets::MaterialPropertyLookupIds;

/// Resolved property lookup for one mesh draw (native UI WGSL and similar paths).
#[derive(Clone, Copy, Debug)]
pub(super) struct MaterialDrawContext {
    /// Material asset id and optional slot-0 `MaterialPropertyBlock` from the mesh renderer update.
    pub(super) property_lookup: MaterialPropertyLookupIds,
}

impl MaterialDrawContext {
    /// Builds merged lookup from the pipeline’s material id and drawable slot metadata.
    pub(super) fn for_non_skinned_draw(
        material_asset_id: i32,
        mesh_renderer_property_block_slot0_id: Option<i32>,
    ) -> Self {
        Self {
            property_lookup: MaterialPropertyLookupIds {
                material_asset_id,
                mesh_property_block_slot0: mesh_renderer_property_block_slot0_id,
            },
        }
    }
}
