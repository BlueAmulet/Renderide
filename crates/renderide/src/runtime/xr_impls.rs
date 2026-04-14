//! [`crate::xr::XrHostCameraSync`] and [`crate::xr::XrMultiviewFrameRenderer`] for [`super::RendererRuntime`].

use glam::{Mat4, Quat, Vec3};

use crate::gpu::GpuContext;
use crate::render_graph::ExternalFrameTargets;
use crate::render_graph::GraphExecuteError;
use crate::shared::HeadOutputDevice;

use super::RendererRuntime;
use winit::window::Window;

impl crate::xr::XrHostCameraSync for RendererRuntime {
    fn near_clip(&self) -> f32 {
        self.host_camera.near_clip
    }

    fn far_clip(&self) -> f32 {
        self.host_camera.far_clip
    }

    fn output_device(&self) -> HeadOutputDevice {
        self.host_camera.output_device
    }

    fn vr_active(&self) -> bool {
        self.host_camera.vr_active
    }

    fn scene_root_scale_for_clip(&self) -> Option<Vec3> {
        self.scene
            .active_main_space()
            .map(|space| space.root_transform.scale)
    }

    fn world_from_tracking(&self, center_pose_tracking: Option<(Vec3, Quat)>) -> Mat4 {
        self.scene
            .active_main_space()
            .map(|space| {
                crate::xr::tracking_space_to_world_matrix(
                    &space.root_transform,
                    &space.view_transform,
                    space.override_view_position,
                    center_pose_tracking,
                )
            })
            .unwrap_or(Mat4::IDENTITY)
    }

    fn set_head_output_transform(&mut self, transform: Mat4) {
        self.host_camera.head_output_transform = transform;
    }

    fn set_stereo_view_proj(&mut self, vp: Option<(Mat4, Mat4)>) {
        self.host_camera.stereo_view_proj = vp;
    }

    fn set_stereo_views(&mut self, views: Option<(Mat4, Mat4)>) {
        self.host_camera.stereo_views = views;
    }
}

impl crate::xr::XrMultiviewFrameRenderer for RendererRuntime {
    fn execute_frame_graph_external_multiview(
        &mut self,
        gpu: &mut GpuContext,
        window: &Window,
        external: ExternalFrameTargets<'_>,
    ) -> Result<(), GraphExecuteError> {
        self.run_frame_graph_external_multiview(gpu, window, external)
    }
}
