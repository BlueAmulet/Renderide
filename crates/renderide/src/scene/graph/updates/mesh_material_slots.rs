//! Slot material and `MaterialPropertyBlock` asset ids from `mesh_materials_and_property_blocks`.
//!
//! Matches Renderite `MeshRendererManager.ApplyUpdate` consumption order:
//! for each [`MeshRendererStatePod`](super::super::pods::MeshRendererStatePod), read `materialCount`
//! ints (material asset ids), then when `materialPropertyBlockCount >= 0`, read that many property
//! block ids.

use crate::scene::Drawable;
use crate::shared::ShadowCastMode;
use crate::shared::enum_repr::EnumRepr;

use super::super::pods::MeshRendererStatePod;

/// Applies mesh renderer state to an optional drawable and advances `cursor` through `packed_ids`.
///
/// When `drawable` is `None` (e.g. invalid renderable index), mesh fields are not written but
/// packed ids are still consumed so the stream stays aligned with the host.
pub(super) fn apply_mesh_renderer_state_row(
    mut drawable: Option<&mut Drawable>,
    state: &MeshRendererStatePod,
    packed_ids: Option<&[i32]>,
    cursor: &mut usize,
) {
    if let Some(d) = drawable.as_mut() {
        d.mesh_handle = state.mesh_asset_id;
        d.sort_key = state.sorting_order;
        d.shadow_cast_mode = ShadowCastMode::from_i32(state.shadow_cast_mode as i32);
    }

    if state.material_count < 0 {
        return;
    }

    let packed = packed_ids.unwrap_or(&[]);
    let mc = state.material_count.max(0) as usize;

    let slot0_material = if mc > 0 {
        if *cursor + mc <= packed.len() {
            let m0 = packed[*cursor];
            *cursor += mc;
            Some(m0)
        } else {
            *cursor = packed.len();
            None
        }
    } else {
        None
    };

    let slot0_pb = if state.material_property_block_count >= 0 {
        let pbc = state.material_property_block_count.max(0) as usize;
        if pbc > 0 {
            if *cursor + pbc <= packed.len() {
                let p0 = packed[*cursor];
                *cursor += pbc;
                Some(p0)
            } else {
                *cursor = packed.len();
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    if let Some(d) = drawable.as_mut() {
        if mc > 0 {
            d.material_handle = slot0_material;
        } else {
            d.material_handle = None;
        }
        if state.material_property_block_count >= 0 {
            d.mesh_renderer_property_block_slot0_id = slot0_pb;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::LayerType;

    fn state(
        renderable_index: i32,
        mesh_id: i32,
        material_count: i32,
        property_block_count: i32,
    ) -> MeshRendererStatePod {
        MeshRendererStatePod {
            renderable_index,
            mesh_asset_id: mesh_id,
            material_count,
            material_property_block_count: property_block_count,
            sorting_order: 0,
            shadow_cast_mode: 1,
            _motion_vector_mode: 0,
            _pad: [0; 2],
        }
    }

    #[test]
    fn material_and_property_block_slot0_from_packed() {
        let packed = [10, 20, 30, 40];
        let mut d = Drawable {
            node_id: 0,
            layer: LayerType::overlay,
            mesh_handle: -1,
            material_handle: None,
            sort_key: 0,
            is_skinned: false,
            bone_transform_ids: None,
            root_bone_transform_id: None,
            blend_shape_weights: None,
            stencil_state: None,
            material_override_block_id: None,
            mesh_renderer_property_block_slot0_id: None,
            render_transform_override: None,
            shadow_cast_mode: ShadowCastMode::on,
        };
        let mut c = 0usize;
        apply_mesh_renderer_state_row(Some(&mut d), &state(0, 100, 2, 2), Some(&packed), &mut c);
        assert_eq!(d.mesh_handle, 100);
        assert_eq!(d.material_handle, Some(10));
        assert_eq!(d.mesh_renderer_property_block_slot0_id, Some(30));
        assert_eq!(c, 4);
    }

    #[test]
    fn invalid_index_still_advances_cursor() {
        let packed = [1, 2];
        let mut c = 0usize;
        apply_mesh_renderer_state_row(None, &state(99, 0, 1, -1), Some(&packed), &mut c);
        assert_eq!(c, 1);
    }
}
