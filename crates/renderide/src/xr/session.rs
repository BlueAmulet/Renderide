//! OpenXR session frame loop: wait, begin, locate views, end.

use glam::{Mat4, Quat, Vec3};
use openxr as xr;
use openxr::{CompositionLayerProjection, CompositionLayerProjectionView, SwapchainSubImage};

use crate::render_graph::{apply_view_handedness_fix, reverse_z_perspective};

/// Per-eye view–projection from OpenXR [`xr::View`] (reverse-Z, engine handedness).
pub fn view_projection_from_xr_view(view: &xr::View, near: f32, far: f32) -> Mat4 {
    let pose = view.pose;
    let (xr_translation, xr_rotation) = openxr_pose_to_glam(&pose);
    let eye = xr_translation;
    let forward = xr_rotation * Vec3::Z;
    let up = xr_rotation * Vec3::Y;
    let view_mat = Mat4::look_at_rh(eye, eye + forward, up);
    let view_mat = apply_view_handedness_fix(view_mat);

    let tan_left = view.fov.angle_left.tan();
    let tan_right = view.fov.angle_right.tan();
    let tan_down = view.fov.angle_down.tan();
    let tan_up = view.fov.angle_up.tan();
    let tan_width = tan_right - tan_left;
    let tan_height = tan_up - tan_down;
    let aspect = tan_width / tan_height.max(1e-6);
    let vertical_fov = (tan_down + tan_up).atan() * 2.0;
    let proj = reverse_z_perspective(aspect, vertical_fov, near, far);
    proj * view_mat
}

fn openxr_pose_to_glam(pose: &xr::Posef) -> (Vec3, Quat) {
    let rotation = {
        let o = pose.orientation;
        Quat::from_rotation_x(180.0f32.to_radians()) * glam::quat(o.w, o.z, o.y, o.x)
    };
    let translation = glam::vec3(-pose.position.x, pose.position.y, -pose.position.z);
    (translation, rotation)
}

/// Owns OpenXR session objects (constructed in [`super::bootstrap::init_wgpu_openxr`]).
pub struct XrSessionState {
    pub(super) xr_instance: xr::Instance,
    pub(super) environment_blend_mode: xr::EnvironmentBlendMode,
    pub(super) session: xr::Session<xr::Vulkan>,
    pub(super) session_running: bool,
    pub(super) frame_wait: xr::FrameWaiter,
    pub(super) frame_stream: xr::FrameStream<xr::Vulkan>,
    pub(super) stage: xr::Space,
    pub(super) event_storage: xr::EventDataBuffer,
}

impl XrSessionState {
    pub(super) fn new(
        xr_instance: xr::Instance,
        environment_blend_mode: xr::EnvironmentBlendMode,
        session: xr::Session<xr::Vulkan>,
        frame_wait: xr::FrameWaiter,
        frame_stream: xr::FrameStream<xr::Vulkan>,
        stage: xr::Space,
    ) -> Self {
        Self {
            xr_instance,
            environment_blend_mode,
            session,
            session_running: false,
            frame_wait,
            frame_stream,
            stage,
            event_storage: xr::EventDataBuffer::new(),
        }
    }

    /// Poll events and return `false` if the session should exit.
    pub fn poll_events(&mut self) -> Result<bool, xr::sys::Result> {
        while let Some(event) = self.xr_instance.poll_event(&mut self.event_storage)? {
            use xr::Event::*;
            match event {
                SessionStateChanged(e) => match e.state() {
                    xr::SessionState::READY => {
                        self.session
                            .begin(xr::ViewConfigurationType::PRIMARY_STEREO)?;
                        self.session_running = true;
                    }
                    xr::SessionState::STOPPING => {
                        self.session.end()?;
                        self.session_running = false;
                    }
                    xr::SessionState::EXITING | xr::SessionState::LOSS_PENDING => {
                        return Ok(false);
                    }
                    _ => {}
                },
                InstanceLossPending(_) => return Ok(false),
                _ => {}
            }
        }
        Ok(true)
    }

    /// Whether the OpenXR session is running.
    pub fn session_running(&self) -> bool {
        self.session_running
    }

    /// OpenXR instance handle (swapchain creation, view enumeration).
    pub fn xr_instance(&self) -> &xr::Instance {
        &self.xr_instance
    }

    /// Underlying Vulkan session (swapchain lifetime).
    pub fn xr_vulkan_session(&self) -> &xr::Session<xr::Vulkan> {
        &self.session
    }

    /// Blocks until the next frame, begins the frame stream. Returns `None` if not ready or idle.
    pub fn wait_frame(&mut self) -> Result<Option<xr::FrameState>, xr::sys::Result> {
        if !self.session_running {
            std::thread::sleep(std::time::Duration::from_millis(10));
            return Ok(None);
        }
        let state = self.frame_wait.wait()?;
        self.frame_stream.begin()?;
        Ok(Some(state))
    }

    /// Ends the frame with no composition layers (mirror path until swapchain submission is wired).
    pub fn end_frame_empty(
        &mut self,
        predicted_display_time: xr::Time,
    ) -> Result<(), xr::sys::Result> {
        self.frame_stream
            .end(predicted_display_time, self.environment_blend_mode, &[])
    }

    /// Submits a stereo projection layer referencing the acquired swapchain image (`image_rect` in pixels).
    pub fn end_frame_projection(
        &mut self,
        predicted_display_time: xr::Time,
        swapchain: &xr::Swapchain<xr::Vulkan>,
        views: &[xr::View],
        image_rect: xr::Rect2Di,
    ) -> Result<(), xr::sys::Result> {
        if views.len() < 2 {
            return self.end_frame_empty(predicted_display_time);
        }
        let projection_views = [
            CompositionLayerProjectionView::new()
                .pose(views[0].pose)
                .fov(views[0].fov)
                .sub_image(
                    SwapchainSubImage::new()
                        .swapchain(swapchain)
                        .image_array_index(0)
                        .image_rect(image_rect),
                ),
            CompositionLayerProjectionView::new()
                .pose(views[1].pose)
                .fov(views[1].fov)
                .sub_image(
                    SwapchainSubImage::new()
                        .swapchain(swapchain)
                        .image_array_index(1)
                        .image_rect(image_rect),
                ),
        ];
        let layer = CompositionLayerProjection::new()
            .space(&self.stage)
            .views(&projection_views);
        self.frame_stream.end(
            predicted_display_time,
            self.environment_blend_mode,
            &[&layer],
        )
    }

    /// Locates stereo views for the predicted display time.
    pub fn locate_views(
        &self,
        predicted_display_time: xr::Time,
    ) -> Result<Vec<xr::View>, xr::sys::Result> {
        let (_, views) = self.session.locate_views(
            xr::ViewConfigurationType::PRIMARY_STEREO,
            predicted_display_time,
            &self.stage,
        )?;
        Ok(views)
    }
}
