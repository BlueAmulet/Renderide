//! Specular IBL reflection-probe baking, binding, and CPU-side selection.

mod atlas;
mod selection;
mod source;
mod system;

pub use selection::{ReflectionProbeDrawSelection, ReflectionProbeFrameSelection};
pub use system::ReflectionProbeSpecularSystem;

#[cfg(test)]
use selection::{
    ReflectionProbeSpatialIndex, SpatialProbe, aabb_volume, intersection_volume_vec3a,
};
#[cfg(test)]
use source::{
    metadata_for_spatial, resolve_baked_probe_source, resolve_space_skybox_fallback_source,
    skybox_fallback_metadata,
};

#[cfg(test)]
mod tests {
    use super::*;

    use crate::backend::AssetTransferQueue;
    use crate::backend::frame_gpu::{
        REFLECTION_PROBE_METADATA_SH2_SOURCE_LOCAL, REFLECTION_PROBE_METADATA_SH2_SOURCE_SKYBOX,
    };
    use crate::materials::MaterialSystem;
    use crate::scene::RenderSpaceId;
    use crate::shared::{
        ReflectionProbeClear, ReflectionProbeState, ReflectionProbeType, RenderSH2,
    };
    use glam::{Vec3, Vec3A};

    fn probe(index: i32, atlas: u16, importance: i32, min: Vec3, max: Vec3) -> SpatialProbe {
        SpatialProbe {
            renderable_index: index,
            atlas_index: atlas,
            importance,
            aabb_min: Vec3A::from(min),
            aabb_max: Vec3A::from(max),
            center: Vec3A::from((min + max) * 0.5),
            volume: aabb_volume(min, max),
        }
    }

    #[test]
    fn higher_priority_overrides_lower_priority() {
        let index = ReflectionProbeSpatialIndex::build(vec![
            probe(0, 1, 0, Vec3::splat(-100.0), Vec3::splat(100.0)),
            probe(1, 2, 1, Vec3::splat(-1.0), Vec3::splat(1.0)),
        ]);

        let selection = index.select((Vec3::splat(-0.25), Vec3::splat(0.25)));

        assert_eq!(selection, ReflectionProbeDrawSelection::one(2));
    }

    #[test]
    fn missing_baked_cubemap_is_not_a_probe_source() {
        let assets = AssetTransferQueue::new();
        let state = ReflectionProbeState {
            intensity: 1.0,
            cubemap_asset_id: 42,
            r#type: ReflectionProbeType::Baked,
            ..ReflectionProbeState::default()
        };

        assert!(resolve_baked_probe_source(state, &assets).is_none());
    }

    #[test]
    fn frame_selection_uses_skybox_fallback_when_no_probe_hits() {
        let mut selection = ReflectionProbeFrameSelection::default();
        let space_id = RenderSpaceId(7);
        selection.rebuild_spatial(Vec::new(), [(space_id, 9)]);

        let draw = selection.select(space_id, (Vec3::splat(-1.0), Vec3::splat(1.0)));

        assert_eq!(
            draw,
            ReflectionProbeDrawSelection {
                first_atlas_index: 9,
                second_atlas_index: 0,
                second_weight: 0.0,
                hit_count: 0,
            }
        );
    }

    #[test]
    fn frame_selection_prefers_probe_hit_over_skybox_fallback() {
        let mut selection = ReflectionProbeFrameSelection::default();
        let space_id = RenderSpaceId(7);
        selection.rebuild_spatial(
            [(
                space_id,
                probe(0, 3, 1, Vec3::splat(-1.0), Vec3::splat(1.0)),
            )],
            [(space_id, 9)],
        );

        let draw = selection.select(space_id, (Vec3::splat(-0.5), Vec3::splat(0.5)));

        assert_eq!(draw, ReflectionProbeDrawSelection::one(3));
    }

    #[test]
    fn missing_skybox_material_is_not_skybox_fallback_source() {
        let materials = MaterialSystem::new();
        let assets = AssetTransferQueue::new();

        assert!(resolve_space_skybox_fallback_source(-1, &materials, &assets).is_none());
    }

    #[test]
    fn skybox_fallback_metadata_allows_specular_while_sh2_is_pending() {
        let metadata = skybox_fallback_metadata(5, None);

        assert_eq!(metadata.params, [1.0, 4.0, 0.0, 0.0]);
        assert_eq!(metadata.sh2, [[0.0; 4]; 9]);
    }

    #[test]
    fn skybox_fallback_metadata_marks_completed_sh2_as_skybox_source() {
        let sh = RenderSH2 {
            sh0: Vec3::new(1.0, 2.0, 3.0),
            sh8: Vec3::new(4.0, 5.0, 6.0),
            ..RenderSH2::default()
        };

        let metadata = skybox_fallback_metadata(5, Some(&sh));

        assert_eq!(
            metadata.params,
            [1.0, 4.0, 0.0, REFLECTION_PROBE_METADATA_SH2_SOURCE_SKYBOX]
        );
        assert_eq!(metadata.sh2[0], [1.0, 2.0, 3.0, 0.0]);
        assert_eq!(metadata.sh2[8], [4.0, 5.0, 6.0, 0.0]);
    }

    #[test]
    fn skybox_only_spatial_probe_metadata_marks_local_sh2_source() {
        let spatial = probe(0, 3, 0, Vec3::splat(-1.0), Vec3::splat(1.0));
        let sh = RenderSH2 {
            sh0: Vec3::ONE,
            ..RenderSH2::default()
        };
        let state = ReflectionProbeState {
            flags: 0b001,
            r#type: ReflectionProbeType::OnChanges,
            intensity: 1.0,
            ..ReflectionProbeState::default()
        };

        let metadata = metadata_for_spatial(&spatial, state, &sh);

        assert_eq!(
            metadata.params[3],
            REFLECTION_PROBE_METADATA_SH2_SOURCE_LOCAL
        );
    }

    #[test]
    fn non_skybox_spatial_probe_metadata_marks_local_sh2_source() {
        let spatial = probe(0, 3, 0, Vec3::splat(-1.0), Vec3::splat(1.0));
        let sh = RenderSH2 {
            sh0: Vec3::ONE,
            ..RenderSH2::default()
        };
        let state = ReflectionProbeState {
            flags: 0b001,
            clear_flags: ReflectionProbeClear::Color,
            intensity: 1.0,
            ..ReflectionProbeState::default()
        };

        let metadata = metadata_for_spatial(&spatial, state, &sh);

        assert_eq!(
            metadata.params[3],
            REFLECTION_PROBE_METADATA_SH2_SOURCE_LOCAL
        );
    }

    #[test]
    fn same_priority_selects_two_by_intersection_volume() {
        let index = ReflectionProbeSpatialIndex::build(vec![
            probe(
                0,
                1,
                1,
                Vec3::new(-1.0, -1.0, -1.0),
                Vec3::new(1.0, 1.0, 1.0),
            ),
            probe(
                1,
                2,
                1,
                Vec3::new(0.0, -1.0, -1.0),
                Vec3::new(2.0, 1.0, 1.0),
            ),
            probe(
                2,
                3,
                1,
                Vec3::new(0.75, -1.0, -1.0),
                Vec3::new(2.0, 1.0, 1.0),
            ),
        ]);

        let selection = index.select((Vec3::new(-0.5, -0.5, -0.5), Vec3::new(1.5, 0.5, 0.5)));

        assert_eq!(selection.hit_count, 2);
        assert_eq!(selection.first_atlas_index, 1);
        assert_eq!(selection.second_atlas_index, 2);
        assert!(selection.second_weight > 0.0 && selection.second_weight < 1.0);
    }

    #[test]
    fn contained_same_priority_probes_still_blend() {
        let index = ReflectionProbeSpatialIndex::build(vec![
            probe(0, 1, 1, Vec3::splat(-10.0), Vec3::splat(10.0)),
            probe(1, 2, 1, Vec3::splat(-1.0), Vec3::splat(1.0)),
        ]);

        let selection = index.select((Vec3::splat(-0.5), Vec3::splat(0.5)));

        assert_eq!(selection.hit_count, 2);
        assert_eq!(selection.first_atlas_index, 2);
        assert_eq!(selection.second_atlas_index, 1);
        assert!(selection.second_weight > 0.0 && selection.second_weight < 1.0);
    }

    #[test]
    fn bvh_matches_bruteforce_candidates() {
        let probes: Vec<_> = (0..32)
            .map(|i| {
                let x = i as f32 * 0.5;
                probe(
                    i,
                    (i + 1) as u16,
                    1,
                    Vec3::new(x, -1.0, -1.0),
                    Vec3::new(x + 1.0, 1.0, 1.0),
                )
            })
            .collect();
        let index = ReflectionProbeSpatialIndex::build(probes.clone());
        let object = (Vec3::new(4.2, -0.25, -0.25), Vec3::new(6.1, 0.25, 0.25));
        let selection = index.select(object);

        let mut brute: Vec<_> = probes
            .iter()
            .filter_map(|probe| {
                let v = intersection_volume_vec3a(
                    probe.aabb_min,
                    probe.aabb_max,
                    Vec3A::from(object.0),
                    Vec3A::from(object.1),
                );
                (v > 0.0).then_some((probe.atlas_index, v))
            })
            .collect();
        brute.sort_by(|a, b| b.1.total_cmp(&a.1));

        assert_eq!(selection.first_atlas_index, brute[0].0);
        assert_eq!(selection.second_atlas_index, brute[1].0);
    }
}
