//! Shared projection plans for world, overlay, stereo, and secondary camera paths.

use glam::Mat4;

use crate::scene::SceneCoordinator;

use super::{
    CameraClipPlanes, HostCameraFrame, Viewport, clamp_desktop_fov_degrees,
    effective_head_output_clip_planes, reverse_z_perspective,
};

/// Projection matrices shared by world-mesh culling and forward rendering.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldProjectionSet {
    /// Effective clip planes after host output-device and root-scale adjustment.
    pub clip: CameraClipPlanes,
    /// Viewport this projection was built for.
    pub viewport: Viewport,
    /// Reverse-Z perspective for world draws.
    pub world_proj: Mat4,
    /// Orthographic projection for overlay draws.
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
        let (near, far) = effective_head_output_clip_planes(
            host_camera.clip.near,
            host_camera.clip.far,
            host_camera.output_device,
            scene
                .active_main_space()
                .map(|space| space.root_transform.scale),
        );
        let clip = CameraClipPlanes::new(near, far);
        let fov_rad = clamp_desktop_fov_degrees(host_camera.desktop_fov_degrees).to_radians();
        let world_proj = reverse_z_perspective(viewport.aspect(), fov_rad, clip.near, clip.far);
        let overlay_proj = host_camera.overlay_projection(viewport, clip);
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
