//! Coverage for bone-assignment rows that describe skinned renderers without bones.

use crate::scene::meshes::types::SkinnedMeshRenderer;
use crate::scene::render_space::RenderSpaceState;
use crate::shared::BoneAssignment;

use super::{ExtractedSkinnedMeshRenderablesUpdate, apply_skinned_bone_index_buffers_extracted};

fn space_with_existing_bones() -> RenderSpaceState {
    let mut space = RenderSpaceState::default();
    let renderer = SkinnedMeshRenderer {
        bone_transform_indices: vec![3, 4, 5],
        root_bone_transform_id: Some(9),
        ..Default::default()
    };
    space.skinned_mesh_renderers.push(renderer);
    space
}

#[test]
fn zero_bone_assignment_applies_without_index_slab() {
    let mut space = space_with_existing_bones();
    let extracted = ExtractedSkinnedMeshRenderablesUpdate {
        bone_assignments: vec![
            BoneAssignment {
                renderable_index: 0,
                root_bone_transform_id: 12,
                bone_count: 0,
            },
            BoneAssignment {
                renderable_index: -1,
                root_bone_transform_id: -1,
                bone_count: 0,
            },
        ],
        bone_transform_indexes: Vec::new(),
        ..Default::default()
    };

    apply_skinned_bone_index_buffers_extracted(&mut space, &extracted, 0);

    assert!(
        space.skinned_mesh_renderers[0]
            .bone_transform_indices
            .is_empty()
    );
    assert_eq!(
        space.skinned_mesh_renderers[0].root_bone_transform_id,
        Some(12)
    );
}

#[test]
fn positive_bone_assignment_still_requires_index_slab() {
    let mut space = space_with_existing_bones();
    let extracted = ExtractedSkinnedMeshRenderablesUpdate {
        bone_assignments: vec![
            BoneAssignment {
                renderable_index: 0,
                root_bone_transform_id: 12,
                bone_count: 2,
            },
            BoneAssignment {
                renderable_index: -1,
                root_bone_transform_id: -1,
                bone_count: 0,
            },
        ],
        bone_transform_indexes: Vec::new(),
        ..Default::default()
    };

    apply_skinned_bone_index_buffers_extracted(&mut space, &extracted, 1);

    assert_eq!(
        space.skinned_mesh_renderers[0].bone_transform_indices,
        vec![3, 4, 5]
    );
    assert_eq!(
        space.skinned_mesh_renderers[0].root_bone_transform_id,
        Some(9)
    );
}
