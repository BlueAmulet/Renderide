//! Reflection-probe render-task planning tests.

use glam::Vec3;
use hashbrown::HashSet;

use crate::camera::HostCameraFrame;
use crate::shared::{
    ReflectionProbeRenderTask, ReflectionProbeState, ReflectionProbeTimeSlicingMode,
};

use super::*;

#[test]
fn reflection_probe_captures_use_single_sample_policy() {
    assert_eq!(
        REFLECTION_PROBE_SAMPLE_COUNT_POLICY,
        OffscreenSampleCountPolicy::SingleSample
    );
}

fn matrix_direction_for_uv(face: ProbeCubeFace, u: f32, v: f32) -> Vec3 {
    let x = 2.0 * u - 1.0;
    let y = 1.0 - 2.0 * v;
    probe_face_world_matrix(Vec3::ZERO, face)
        .transform_vector3(Vec3::new(x, y, 1.0))
        .normalize()
}

#[test]
fn cubemap_face_directions_match_bitmap_cube_order() {
    let samples = [
        (ProbeCubeFace::PosX, Vec3::new(1.0, 1.0, 1.0)),
        (ProbeCubeFace::NegX, Vec3::new(-1.0, 1.0, -1.0)),
        (ProbeCubeFace::PosY, Vec3::new(-1.0, 1.0, -1.0)),
        (ProbeCubeFace::NegY, Vec3::new(-1.0, -1.0, 1.0)),
        (ProbeCubeFace::PosZ, Vec3::new(-1.0, 1.0, 1.0)),
        (ProbeCubeFace::NegZ, Vec3::new(1.0, 1.0, -1.0)),
    ];
    for (face, expected) in samples {
        let actual = face.direction_for_uv(0.0, 0.0);
        assert!((actual - expected.normalize()).length() < 1e-6);
    }
}

#[test]
fn probe_face_world_matrices_match_bitmap_cube_directions() {
    for face in ProbeCubeFace::ALL {
        assert!((matrix_direction_for_uv(face, 0.5, 0.5) - face.basis().forward).length() < 1e-6);
        for (u, v) in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)] {
            let actual = matrix_direction_for_uv(face, u, v);
            let expected = face.direction_for_uv(u, v);
            assert!(
                (actual - expected).length() < 1e-6,
                "{face:?} uv=({u}, {v}) actual={actual:?} expected={expected:?}"
            );
        }
    }
}

#[test]
fn onchanges_individual_faces_steps_one_remaining_face() {
    let faces = onchanges::onchanges_faces_for_step(
        ReflectionProbeTimeSlicingMode::IndividualFaces,
        0b0000_0011,
    );

    assert_eq!(faces, vec![ProbeCubeFace::PosY]);
}

#[test]
fn onchanges_all_faces_at_once_steps_all_remaining_faces() {
    let faces = onchanges::onchanges_faces_for_step(
        ReflectionProbeTimeSlicingMode::AllFacesAtOnce,
        0b0011_0001,
    );

    assert_eq!(
        faces,
        vec![
            ProbeCubeFace::NegX,
            ProbeCubeFace::PosY,
            ProbeCubeFace::NegY
        ]
    );
}

#[test]
fn probe_face_projection_is_square_ninety_degrees() {
    let frame = host_camera_frame_for_probe_face(
        HostCameraFrame::default(),
        ReflectionProbeState {
            near_clip: 0.1,
            far_clip: 100.0,
            ..Default::default()
        },
        (256, 256),
        Vec3::ZERO,
        ProbeCubeFace::PosZ,
    );
    let view = frame
        .explicit_view
        .expect("probe face should use explicit camera view");

    assert!((view.proj.x_axis.x - 1.0).abs() < 1e-6);
    assert!((view.proj.y_axis.y - 1.0).abs() < 1e-6);
}

#[test]
fn reflection_probe_bake_views_disable_post_processing() {
    let policy = reflection_probe_bake_post_processing();

    assert!(!policy.is_enabled());
    assert!(!policy.screen_space_reflections);
    assert!(!policy.motion_blur);
}

#[test]
fn skybox_only_probe_uses_empty_selective_filter() {
    let task = ReflectionProbeRenderTask {
        exclude_transform_ids: vec![1, 2],
        ..Default::default()
    };
    let state = ReflectionProbeState {
        flags: 0b001,
        ..Default::default()
    };

    let filter = draw_filter_from_reflection_probe_task(&task, &state);

    assert!(filter.only.as_ref().is_some_and(HashSet::is_empty));
    assert!(filter.exclude.is_empty());
}
