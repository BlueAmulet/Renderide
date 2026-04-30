//! Secondary render-texture camera algebra.

use glam::Mat4;

use crate::scene::SceneCoordinator;
use crate::shared::{CameraProjection, CameraState, HeadOutputDevice};

use super::{
    CameraClipPlanes, CameraPose, EyeView, HostCameraFrame, OrthographicProjectionSpec, Viewport,
    clamp_desktop_fov_degrees, effective_head_output_clip_planes,
};

/// Returns `true` when [`CameraState::flags`] bit 0 is set.
#[inline]
pub fn camera_state_enabled(flags: u16) -> bool {
    flags & 1 != 0
}

/// Builds a [`HostCameraFrame`] for rendering through a secondary camera to a render texture.
pub fn host_camera_frame_for_render_texture(
    base: &HostCameraFrame,
    state: &CameraState,
    viewport_px: (u32, u32),
    camera_world_matrix: Mat4,
    scene: &SceneCoordinator,
) -> HostCameraFrame {
    let viewport = Viewport::from_tuple(viewport_px);
    let root_scale = scene
        .active_main_space()
        .map(|space| space.root_transform.scale);
    let (near_clip, far_clip) = effective_head_output_clip_planes(
        state.near_clip,
        state.far_clip,
        HeadOutputDevice::Screen,
        root_scale,
    );
    let clip = CameraClipPlanes::new(near_clip, far_clip);
    let pose = CameraPose::from_world_matrix(camera_world_matrix);
    let fov_degrees = clamp_desktop_fov_degrees(state.field_of_view);
    let explicit_view = match state.projection {
        CameraProjection::Orthographic => {
            let spec = OrthographicProjectionSpec::new(state.orthographic_size, clip);
            EyeView::from_pose_projection(pose, spec.projection(viewport))
        }
        CameraProjection::Perspective | CameraProjection::Panoramic => {
            EyeView::perspective_from_pose(pose, viewport, fov_degrees, clip)
        }
    };
    let primary_ortho_task = match state.projection {
        CameraProjection::Orthographic => Some(OrthographicProjectionSpec::new(
            state.orthographic_size,
            clip,
        )),
        CameraProjection::Perspective | CameraProjection::Panoramic => None,
    };

    HostCameraFrame {
        frame_index: base.frame_index,
        clip,
        desktop_fov_degrees: fov_degrees,
        vr_active: false,
        output_device: base.output_device,
        primary_ortho_task,
        stereo: None,
        head_output_transform: base.head_output_transform,
        explicit_view: Some(explicit_view),
        eye_world_position: base.eye_world_position,
        suppress_occlusion_temporal: false,
    }
}

#[cfg(test)]
mod tests {
    use glam::{Mat4, Vec3};

    use crate::scene::{RenderSpaceId, SceneCoordinator};
    use crate::shared::{CameraProjection, CameraState, HeadOutputDevice};

    use super::{camera_state_enabled, host_camera_frame_for_render_texture};
    use crate::camera::{HostCameraFrame, apply_view_handedness_fix};

    #[test]
    fn camera_state_enabled_reads_bit_zero() {
        assert!(!camera_state_enabled(0));
        assert!(camera_state_enabled(1));
        assert!(camera_state_enabled(0xffff));
        assert!(!camera_state_enabled(2));
    }

    #[test]
    fn host_camera_frame_secondary_sets_pose_and_projection_override() {
        let scene = SceneCoordinator::new();
        let base = HostCameraFrame {
            output_device: HeadOutputDevice::Screen,
            ..Default::default()
        };
        let cam_world = Mat4::from_translation(Vec3::new(4.0, 5.0, 6.0));
        let state = CameraState {
            projection: CameraProjection::Perspective,
            field_of_view: 55.0,
            near_clip: 0.05,
            far_clip: 2000.0,
            ..Default::default()
        };

        let out =
            host_camera_frame_for_render_texture(&base, &state, (1280, 720), cam_world, &scene);

        let expected_w2v = apply_view_handedness_fix(cam_world.inverse());
        let explicit = out.explicit_view.expect("secondary camera explicit view");
        assert_eq!(explicit.view, expected_w2v);
        assert!(explicit.proj.is_finite());
        assert_eq!(out.primary_ortho_task, None);
        assert_eq!(out.desktop_fov_degrees, state.field_of_view);
        assert!(!out.vr_active);
    }

    #[test]
    fn host_camera_frame_secondary_orthographic_sets_primary_ortho_task() {
        let mut scene = SceneCoordinator::new();
        scene.test_seed_space_identity_worlds(
            RenderSpaceId(1),
            vec![crate::shared::RenderTransform::default()],
            vec![-1],
        );
        let base = HostCameraFrame::default();
        let cam_world = Mat4::IDENTITY;
        let state = CameraState {
            projection: CameraProjection::Orthographic,
            orthographic_size: 8.0,
            near_clip: 0.1,
            far_clip: 500.0,
            ..Default::default()
        };

        let out =
            host_camera_frame_for_render_texture(&base, &state, (640, 480), cam_world, &scene);

        let ortho = out.primary_ortho_task.expect("orthographic task");
        assert_eq!(ortho.half_height, 8.0);
        assert_eq!(ortho.clip, out.clip);
        assert!(out.explicit_view.expect("explicit view").proj.is_finite());
    }
}
