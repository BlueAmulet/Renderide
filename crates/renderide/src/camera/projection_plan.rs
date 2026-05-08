//! Shared projection plans for world, overlay, stereo, and secondary camera paths.

use glam::Mat4;

use crate::scene::SceneCoordinator;

use super::{
    CameraClipPlanes, HostCameraFrame, OrthographicProjectionSpec, Viewport,
    clamp_desktop_fov_degrees, effective_head_output_clip_planes, reverse_z_perspective,
};

/// Half-height of the screen-overlay orthographic frustum.
///
/// Mirrors the host `RadiantDash` desktop layout in `UpdateProjection`: the dash visuals are
/// scaled (`VisualsRoot.LocalScale = 1/num2 * num5`) so the stacked Top/Screen/Buttons panels span
/// exactly one unit vertically. A `half_height = 0.5` ortho frustum maps that unit-height range to
/// the swapchain edges, with the viewport aspect ratio extending the horizontal range.
const SCREEN_OVERLAY_HALF_HEIGHT: f32 = 0.5;

/// Projection matrices shared by world-mesh culling and forward rendering.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldProjectionSet {
    /// Effective clip planes after host output-device and root-scale adjustment.
    pub clip: CameraClipPlanes,
    /// Viewport this projection was built for.
    pub viewport: Viewport,
    /// Projection for world draws.
    pub world_proj: Mat4,
    /// Projection for overlay draws.
    pub overlay_proj: Mat4,
    /// Stereo view-projection pair when the host camera is actively stereo.
    pub stereo_view_proj: Option<(Mat4, Mat4)>,
}

impl WorldProjectionSet {
    /// Builds world/overlay projection data from scene root scale, viewport, and host camera.
    pub fn from_scene_host(
        scene: &SceneCoordinator,
        viewport_px: (u32, u32),
        host_camera: &HostCameraFrame,
    ) -> Self {
        let viewport = Viewport::from_tuple(viewport_px);
        let explicit_proj = host_camera.explicit_view_projection().map(|(_, proj)| proj);
        let root_scale = explicit_proj.is_none().then(|| {
            scene
                .active_main_space()
                .map(|space| space.root_transform().scale)
        });
        let (near, far) = effective_head_output_clip_planes(
            host_camera.clip.near,
            host_camera.clip.far,
            host_camera.output_device,
            root_scale.flatten(),
        );
        let clip = CameraClipPlanes::new(near, far);
        let fov_rad = clamp_desktop_fov_degrees(host_camera.desktop_fov_degrees).to_radians();
        let world_proj = explicit_proj.unwrap_or_else(|| {
            reverse_z_perspective(viewport.aspect(), fov_rad, clip.near, clip.far)
        });
        // Screen overlay (host `LayerType.Overlay` per-mesh layer) uses a dedicated unit-height
        // orthographic projection sized to the swapchain. Intentionally NOT taken from
        // `host_camera.overlay_projection`, which falls back to `primary_ortho_task` -- the host
        // sets that from the first orthographic camera in `data.render_tasks`, typically the
        // dash camera's `OrthographicSize = 0.5f` task that is meant for dash-RT rendering, not
        // for the screen overlay. Sharing it produced a tiny half-meter overlay frustum and
        // pushed the dash off-screen on desktop.
        let overlay_proj = explicit_proj.unwrap_or_else(|| {
            OrthographicProjectionSpec::new(SCREEN_OVERLAY_HALF_HEIGHT, clip).projection(viewport)
        });
        let stereo_view_proj = host_camera
            .active_stereo()
            .map(|stereo| stereo.view_proj_pair());
        Self {
            clip,
            viewport,
            world_proj,
            overlay_proj,
            stereo_view_proj,
        }
    }
}

#[cfg(test)]
mod tests {
    use glam::{Mat4, Vec3};

    use super::WorldProjectionSet;
    use crate::camera::{EyeView, HostCameraFrame};
    use crate::scene::SceneCoordinator;

    #[test]
    fn explicit_secondary_projection_replaces_world_and_overlay_projection() {
        let scene = SceneCoordinator::new();
        let explicit_proj = Mat4::from_scale(Vec3::new(2.0, 3.0, 1.0));
        let host_camera = HostCameraFrame {
            explicit_view: Some(EyeView::new(
                Mat4::IDENTITY,
                explicit_proj,
                Mat4::IDENTITY,
                Vec3::ZERO,
            )),
            ..Default::default()
        };

        let set = WorldProjectionSet::from_scene_host(&scene, (1280, 720), &host_camera);

        assert_eq!(set.world_proj, explicit_proj);
        assert_eq!(set.overlay_proj, explicit_proj);
    }
}
