//! Secondary render-texture camera algebra.

use glam::{Mat4, Vec3};

use crate::shared::{CameraProjection, CameraState, HeadOutputDevice};

use super::{
    CameraClipPlanes, CameraPose, CameraProjectionKind, EyeView, HostCameraFrame,
    OrthographicProjectionSpec, Viewport, clamp_desktop_fov_degrees,
    effective_head_output_clip_planes,
};

/// Returns `true` when [`CameraState::flags`] bit 0 is set.
#[inline]
pub fn camera_state_enabled(flags: u16) -> bool {
    flags & 1 != 0
}

/// Returns `true` when [`CameraState::flags`] bit 1 is set.
#[inline]
pub fn camera_state_use_transform_scale(flags: u16) -> bool {
    flags & (1 << 1) != 0
}

/// Returns `true` when [`CameraState::flags`] bit 3 is set.
#[inline]
pub fn camera_state_render_private_ui(flags: u16) -> bool {
    flags & (1 << 3) != 0
}

/// Builds a [`HostCameraFrame`] for rendering through a secondary camera to a render texture.
pub fn host_camera_frame_for_render_texture(
    base: &HostCameraFrame,
    state: &CameraState,
    viewport_px: (u32, u32),
    camera_world_matrix: Mat4,
) -> HostCameraFrame {
    let viewport = Viewport::from_tuple(viewport_px);
    let transform_scale = if camera_state_use_transform_scale(state.flags) {
        uniform_scale_from_matrix(camera_world_matrix)
    } else {
        1.0
    };
    let (near_clip, far_clip) = effective_head_output_clip_planes(
        state.near_clip,
        state.far_clip,
        HeadOutputDevice::Screen,
        Some(Vec3::splat(transform_scale)),
    );
    let clip = CameraClipPlanes::new(near_clip, far_clip);
    let pose = CameraPose::from_world_matrix(camera_world_matrix);
    let fov_degrees = clamp_desktop_fov_degrees(state.field_of_view);
    let explicit_view = match state.projection {
        CameraProjection::Orthographic => {
            let spec =
                OrthographicProjectionSpec::new(state.orthographic_size * transform_scale, clip);
            EyeView::from_pose_projection(pose, spec.projection(viewport))
        }
        CameraProjection::Perspective | CameraProjection::Panoramic => {
            EyeView::perspective_from_pose(pose, viewport, fov_degrees, clip)
        }
    };
    let primary_ortho_task = match state.projection {
        CameraProjection::Orthographic => Some(OrthographicProjectionSpec::new(
            state.orthographic_size * transform_scale,
            clip,
        )),
        CameraProjection::Perspective | CameraProjection::Panoramic => None,
    };
    let projection_kind = match state.projection {
        CameraProjection::Orthographic => CameraProjectionKind::Orthographic,
        CameraProjection::Perspective | CameraProjection::Panoramic => {
            CameraProjectionKind::Perspective
        }
    };

    HostCameraFrame {
        frame_index: base.frame_index,
        clip,
        desktop_fov_degrees: fov_degrees,
        vr_active: false,
        output_device: base.output_device,
        projection_kind,
        primary_ortho_task,
        stereo: None,
        head_output_transform: base.head_output_transform,
        explicit_view: Some(explicit_view),
        eye_world_position: Some(pose.world_position),
        suppress_occlusion_temporal: false,
    }
}

fn uniform_scale_from_matrix(matrix: Mat4) -> f32 {
    let (scale, _, _) = matrix.to_scale_rotation_translation();
    let avg = (scale.x.abs() + scale.y.abs() + scale.z.abs()) / 3.0;
    if avg.is_finite() && avg > 1e-8 {
        avg
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use glam::{Mat4, Vec3};

    use crate::shared::{CameraProjection, CameraState, HeadOutputDevice};

    use super::{
        camera_state_enabled, camera_state_render_private_ui, camera_state_use_transform_scale,
        host_camera_frame_for_render_texture,
    };
    use crate::camera::{
        CameraProjectionKind, HostCameraFrame, OrthographicProjectionSpec, Viewport,
        WorldProjectionSet, apply_view_handedness_fix,
    };
    use crate::scene::SceneCoordinator;

    #[test]
    fn camera_state_enabled_reads_bit_zero() {
        assert!(!camera_state_enabled(0));
        assert!(camera_state_enabled(1));
        assert!(camera_state_enabled(0xffff));
        assert!(!camera_state_enabled(2));
    }

    #[test]
    fn camera_state_flags_decode_scale_and_private_ui_bits() {
        assert!(!camera_state_use_transform_scale(0));
        assert!(camera_state_use_transform_scale(1 << 1));
        assert!(!camera_state_render_private_ui(0));
        assert!(camera_state_render_private_ui(1 << 3));
    }

    #[test]
    fn host_camera_frame_secondary_sets_pose_and_projection_override() {
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

        let out = host_camera_frame_for_render_texture(&base, &state, (1280, 720), cam_world);

        let expected_w2v = apply_view_handedness_fix(cam_world.inverse());
        let explicit = out.explicit_view.expect("secondary camera explicit view");
        assert_eq!(explicit.view, expected_w2v);
        assert!(explicit.proj.is_finite());
        assert_eq!(out.primary_ortho_task, None);
        assert_eq!(out.projection_kind, CameraProjectionKind::Perspective);
        assert_eq!(out.desktop_fov_degrees, state.field_of_view);
        assert!(!out.vr_active);
    }

    #[test]
    fn host_camera_frame_secondary_orthographic_sets_primary_ortho_task() {
        let base = HostCameraFrame::default();
        let cam_world = Mat4::IDENTITY;
        let state = CameraState {
            projection: CameraProjection::Orthographic,
            orthographic_size: 8.0,
            near_clip: 0.1,
            far_clip: 500.0,
            ..Default::default()
        };

        let out = host_camera_frame_for_render_texture(&base, &state, (640, 480), cam_world);

        let ortho = out.primary_ortho_task.expect("orthographic task");
        assert_eq!(ortho.half_height, 8.0);
        assert_eq!(ortho.clip, out.clip);
        assert_eq!(out.projection_kind, CameraProjectionKind::Orthographic);
        assert!(out.explicit_view.expect("explicit view").proj.is_finite());
    }

    #[test]
    fn host_camera_frame_secondary_orthographic_uses_state_projection_matrix() {
        let base = HostCameraFrame::default();
        let cam_world = Mat4::IDENTITY;
        let viewport = (640, 480);
        let clip = crate::camera::CameraClipPlanes::new(0.1, 500.0);
        let perspective_state = CameraState {
            projection: CameraProjection::Perspective,
            field_of_view: 60.0,
            near_clip: clip.near,
            far_clip: clip.far,
            ..Default::default()
        };
        let orthographic_state = CameraState {
            projection: CameraProjection::Orthographic,
            orthographic_size: 8.0,
            near_clip: clip.near,
            far_clip: clip.far,
            ..Default::default()
        };

        let perspective =
            host_camera_frame_for_render_texture(&base, &perspective_state, viewport, cam_world);
        let orthographic =
            host_camera_frame_for_render_texture(&base, &orthographic_state, viewport, cam_world);
        let expected_ortho =
            OrthographicProjectionSpec::new(8.0, clip).projection(Viewport::from_tuple(viewport));

        let perspective_proj = perspective.explicit_view.expect("perspective view").proj;
        let orthographic_proj = orthographic.explicit_view.expect("orthographic view").proj;
        assert_eq!(orthographic_proj, expected_ortho);
        assert_ne!(orthographic_proj, perspective_proj);
    }

    #[test]
    fn orthographic_camera_state_reaches_world_projection_set() {
        let base = HostCameraFrame::default();
        let cam_world = Mat4::IDENTITY;
        let viewport = (640, 480);
        let clip = crate::camera::CameraClipPlanes::new(0.1, 500.0);
        let state = CameraState {
            projection: CameraProjection::Orthographic,
            orthographic_size: 8.0,
            near_clip: clip.near,
            far_clip: clip.far,
            ..Default::default()
        };

        let host_camera = host_camera_frame_for_render_texture(&base, &state, viewport, cam_world);
        let expected_ortho =
            OrthographicProjectionSpec::new(8.0, clip).projection(Viewport::from_tuple(viewport));
        let projections =
            WorldProjectionSet::from_scene_host(&SceneCoordinator::new(), viewport, &host_camera);

        assert_eq!(projections.world_proj, expected_ortho);
        assert_eq!(projections.overlay_proj, expected_ortho);
    }

    #[test]
    fn host_camera_frame_secondary_use_transform_scale_scales_ortho_and_clip() {
        let base = HostCameraFrame::default();
        let cam_world = Mat4::from_scale(Vec3::splat(2.0));
        let state = CameraState {
            projection: CameraProjection::Orthographic,
            flags: 1 << 1,
            orthographic_size: 4.0,
            near_clip: 0.1,
            far_clip: 500.0,
            ..Default::default()
        };

        let out = host_camera_frame_for_render_texture(&base, &state, (640, 480), cam_world);

        let ortho = out.primary_ortho_task.expect("orthographic task");
        assert_eq!(ortho.half_height, 8.0);
        assert!((out.near_clip() - 0.2).abs() < 1e-6);
        assert!((out.far_clip() - 1000.0).abs() < 1e-4);
    }
}
