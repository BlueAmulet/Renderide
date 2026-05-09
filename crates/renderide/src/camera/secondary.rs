//! Secondary render-texture camera algebra.

use glam::{Mat4, Vec3};

use crate::shared::{CameraProjection, CameraState};

use super::{
    CameraClipPlanes, CameraPose, CameraProjectionKind, EyeView, HostCameraFrame,
    OrthographicProjectionSpec, Viewport, clamp_desktop_fov_degrees,
};

const CAMERA_CONTROLLER_SCALE_MIN: f32 = 1e-5;
const CAMERA_CONTROLLER_SCALE_MAX: f32 = 1e6;
const CAMERA_CONTROLLER_ORTHOGRAPHIC_SIZE_MIN: f32 = 1e-6;
const CAMERA_CONTROLLER_ORTHOGRAPHIC_SIZE_MAX: f32 = 1e6;
const CAMERA_CONTROLLER_NEAR_CLIP_MIN: f32 = 1e-4;
const CAMERA_CONTROLLER_CLIP_MAX: f32 = 1e6;
const CAMERA_CONTROLLER_FAR_ABOVE_NEAR_MIN: f32 = 1e-4;

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

/// Returns `true` when [`CameraState::flags`] bit 6 is set.
#[inline]
pub fn camera_state_post_processing(flags: u16) -> bool {
    flags & (1 << 6) != 0
}

/// Returns `true` when [`CameraState::flags`] bit 7 is set.
#[inline]
pub fn camera_state_screen_space_reflections(flags: u16) -> bool {
    flags & (1 << 7) != 0
}

/// Returns `true` when [`CameraState::flags`] bit 8 is set.
#[inline]
pub fn camera_state_motion_blur(flags: u16) -> bool {
    flags & (1 << 8) != 0
}

/// Returns `true` when [`CameraState::flags`] bit 3 is set.
#[inline]
#[cfg(test)]
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
    let transform_scale = camera_controller_transform_scale(state.flags, camera_world_matrix);
    let clip = camera_controller_clip_planes(state, transform_scale);
    let pose = CameraPose::from_world_matrix(camera_world_matrix);
    let fov_degrees = clamp_desktop_fov_degrees(state.field_of_view);
    let orthographic_size = camera_controller_orthographic_size(state, transform_scale);
    let explicit_view = match state.projection {
        CameraProjection::Orthographic => {
            let spec = OrthographicProjectionSpec::new(orthographic_size, clip);
            EyeView::from_pose_projection(pose, spec.projection(viewport))
        }
        CameraProjection::Perspective | CameraProjection::Panoramic => {
            EyeView::perspective_from_pose(pose, viewport, fov_degrees, clip)
        }
    };
    let primary_ortho_task = match state.projection {
        CameraProjection::Orthographic => {
            Some(OrthographicProjectionSpec::new(orthographic_size, clip))
        }
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

fn camera_controller_transform_scale(flags: u16, matrix: Mat4) -> f32 {
    if !camera_state_use_transform_scale(flags) {
        return 1.0;
    }
    let (scale, _, _) = matrix.to_scale_rotation_translation();
    camera_controller_transform_scale_from_lossy(scale)
}

fn camera_controller_transform_scale_from_lossy(lossy_scale: Vec3) -> f32 {
    let mut scale = (lossy_scale.x + lossy_scale.y + lossy_scale.z) * 0.333_333_34;
    if scale.is_nan() {
        scale = 0.0;
    }
    camera_controller_clamp(
        scale,
        CAMERA_CONTROLLER_SCALE_MIN,
        CAMERA_CONTROLLER_SCALE_MAX,
    )
}

fn camera_controller_clip_planes(state: &CameraState, transform_scale: f32) -> CameraClipPlanes {
    let near_clip = camera_controller_clamp(
        state.near_clip * transform_scale,
        CAMERA_CONTROLLER_NEAR_CLIP_MIN,
        CAMERA_CONTROLLER_CLIP_MAX,
    );
    let far_min =
        CAMERA_CONTROLLER_NEAR_CLIP_MIN.max(near_clip + CAMERA_CONTROLLER_FAR_ABOVE_NEAR_MIN);
    let far_clip = camera_controller_clamp(
        state.far_clip * transform_scale,
        far_min,
        CAMERA_CONTROLLER_CLIP_MAX,
    );
    CameraClipPlanes::new(near_clip, far_clip)
}

fn camera_controller_orthographic_size(state: &CameraState, transform_scale: f32) -> f32 {
    camera_controller_clamp(
        state.orthographic_size * transform_scale,
        CAMERA_CONTROLLER_ORTHOGRAPHIC_SIZE_MIN,
        CAMERA_CONTROLLER_ORTHOGRAPHIC_SIZE_MAX,
    )
}

fn camera_controller_clamp(value: f32, min: f32, max: f32) -> f32 {
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}

#[cfg(test)]
mod tests {
    use glam::{Mat4, Vec3};

    use crate::shared::{CameraProjection, CameraState, HeadOutputDevice};

    use super::{
        camera_controller_transform_scale_from_lossy, camera_state_enabled,
        camera_state_motion_blur, camera_state_post_processing, camera_state_render_private_ui,
        camera_state_screen_space_reflections, camera_state_use_transform_scale,
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
    fn camera_state_flags_decode_post_processing_bits() {
        assert!(!camera_state_post_processing(0));
        assert!(camera_state_post_processing(1 << 6));
        assert!(!camera_state_screen_space_reflections(0));
        assert!(camera_state_screen_space_reflections(1 << 7));
        assert!(!camera_state_motion_blur(0));
        assert!(camera_state_motion_blur(1 << 8));
    }

    #[test]
    fn host_camera_frame_secondary_without_transform_scale_uses_camera_controller_clamps() {
        let state = CameraState {
            projection: CameraProjection::Orthographic,
            orthographic_size: 0.0,
            near_clip: 0.0,
            far_clip: 0.0,
            ..Default::default()
        };

        let out = host_camera_frame_for_render_texture(
            &HostCameraFrame::default(),
            &state,
            (640, 480),
            Mat4::from_scale(Vec3::splat(9.0)),
        );

        let ortho = out.primary_ortho_task.expect("orthographic task");
        assert_eq!(ortho.half_height, 1e-6);
        assert!((out.near_clip() - 1e-4).abs() < 1e-8);
        assert!((out.far_clip() - 2e-4).abs() < 1e-8);
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
    fn host_camera_frame_secondary_use_transform_scale_averages_non_uniform_scale() {
        let base = HostCameraFrame::default();
        let cam_world = Mat4::from_scale(Vec3::new(1.0, 2.0, 6.0));
        let state = CameraState {
            projection: CameraProjection::Orthographic,
            flags: 1 << 1,
            orthographic_size: 2.0,
            near_clip: 0.1,
            far_clip: 50.0,
            ..Default::default()
        };

        let out = host_camera_frame_for_render_texture(&base, &state, (640, 480), cam_world);

        let ortho = out.primary_ortho_task.expect("orthographic task");
        assert!((ortho.half_height - 6.0).abs() < 1e-6);
        assert!((out.near_clip() - 0.3).abs() < 1e-6);
        assert!((out.far_clip() - 150.0).abs() < 1e-4);
    }

    #[test]
    fn host_camera_frame_secondary_perspective_use_transform_scale_scales_clip_not_fov() {
        let base = HostCameraFrame::default();
        let cam_world = Mat4::from_scale(Vec3::splat(4.0));
        let state = CameraState {
            projection: CameraProjection::Perspective,
            flags: 1 << 1,
            field_of_view: 45.0,
            near_clip: 0.2,
            far_clip: 30.0,
            ..Default::default()
        };

        let out = host_camera_frame_for_render_texture(&base, &state, (640, 480), cam_world);

        assert_eq!(out.primary_ortho_task, None);
        assert_eq!(out.desktop_fov_degrees, 45.0);
        assert!((out.near_clip() - 0.8).abs() < 1e-6);
        assert!((out.far_clip() - 120.0).abs() < 1e-4);
    }

    #[test]
    fn camera_controller_transform_scale_matches_clamp_edges() {
        assert_eq!(
            camera_controller_transform_scale_from_lossy(Vec3::splat(f32::NAN)),
            1e-5
        );
        assert_eq!(
            camera_controller_transform_scale_from_lossy(Vec3::splat(-2.0)),
            1e-5
        );
        assert_eq!(
            camera_controller_transform_scale_from_lossy(Vec3::splat(2.0e6)),
            1e6
        );
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
