//! OpenXR session frame loop: wait, begin, locate views, end.
//!
//! OpenXR [`xr::Posef`] is documented as transforming **from the space’s local frame to the
//! reference frame** (e.g. view pose in stage space). The camera **view** matrix is therefore
//! `inverse(T_ref_from_view)` after mapping the pose into engine space, not a reconstructed
//! `look_at` from forward/up vectors.

use glam::{Mat3, Mat4, Quat, Vec3};
use openxr as xr;
use openxr::{CompositionLayerProjection, CompositionLayerProjectionView, SwapchainSubImage};

use crate::render_graph::{apply_view_handedness_fix, reverse_z_perspective_openxr_fov};

/// Basis that maps OpenXR stage axes (X right, Y up, −Z forward) into engine space (X left, Y up,
/// −Z forward): `p_eng = S * p_xr` with `S = diag(−1, 1, −1)`.
///
/// Rotation uses the same transform: `R_eng = S * R_xr * S`, so translation and orientation stay
/// consistent (partial X/Z flips on position alone would skew yaw vs forward).
#[inline]
fn openxr_to_engine_basis() -> Mat3 {
    Mat3::from_diagonal(Vec3::new(-1.0, 1.0, -1.0))
}

/// `T_ref_from_view`: maps view-local points into the reference (stage) frame.
#[inline]
pub(crate) fn ref_from_view_matrix(pose: &xr::Posef) -> Mat4 {
    let (translation, rotation) = openxr_pose_to_engine(pose);
    Mat4::from_rotation_translation(rotation, translation)
}

/// Per-eye view–projection from OpenXR [`xr::View`] (reverse-Z, engine handedness).
pub fn view_projection_from_xr_view(view: &xr::View, near: f32, far: f32) -> Mat4 {
    let ref_from_view = ref_from_view_matrix(&view.pose);
    let view_mat = apply_view_handedness_fix(ref_from_view.inverse());
    let proj = reverse_z_perspective_openxr_fov(&view.fov, near, far);
    proj * view_mat
}

/// Maps an OpenXR [`xr::Posef`] to engine translation + rotation (same basis as [`view_projection_from_xr_view`]).
pub fn openxr_pose_to_engine(pose: &xr::Posef) -> (Vec3, Quat) {
    let o = pose.orientation;
    let quat_xr = Quat::from_xyzw(o.x, o.y, o.z, o.w);
    let s = openxr_to_engine_basis();
    let r_xr = Mat3::from_quat(quat_xr);
    let r_eng = s * r_xr * s;
    let quat_eng = Quat::from_mat3(&r_eng).normalize();
    let p_xr = Vec3::new(pose.position.x, pose.position.y, pose.position.z);
    let p_eng = s * p_xr;
    (p_eng, quat_eng)
}

/// Headset position and rotation in engine space (same basis as [`view_projection_from_xr_view`]).
pub fn headset_pose_from_xr_view(view: &xr::View) -> (Vec3, Quat) {
    openxr_pose_to_engine(&view.pose)
}

/// OpenXR requires a unit quaternion; some runtimes briefly report `(0,0,0,0)`, which makes
/// `xrEndFrame` fail with `XR_ERROR_POSE_INVALID`.
fn sanitize_pose_for_end_frame(pose: xr::Posef) -> xr::Posef {
    let o = pose.orientation;
    let len_sq = o.x * o.x + o.y * o.y + o.z * o.z + o.w * o.w;
    if len_sq.is_finite() && len_sq >= 1e-10 {
        pose
    } else {
        xr::Posef {
            orientation: xr::Quaternionf {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 1.0,
            },
            position: pose.position,
        }
    }
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

    /// Stage reference space used for [`Self::locate_views`] and controller [`xr::Space`] location.
    pub fn stage_space(&self) -> &xr::Space {
        &self.stage
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
    ///
    /// Layer 0 uses [`views`]\[1] (pose + FOV) and layer 1 uses [`views`]\[0], matching the stereo
    /// view–projection assignment (`left` from `views[1]`, `right` from `views[0]`) so multiview
    /// `view_index` 0/1 aligns with the submitted layers and stereo parallax matches the compositor.
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
        let v0 = &views[1];
        let v1 = &views[0];
        let pose0 = sanitize_pose_for_end_frame(v0.pose);
        let pose1 = sanitize_pose_for_end_frame(v1.pose);
        let projection_views = [
            CompositionLayerProjectionView::new()
                .pose(pose0)
                .fov(v0.fov)
                .sub_image(
                    SwapchainSubImage::new()
                        .swapchain(swapchain)
                        .image_array_index(0)
                        .image_rect(image_rect),
                ),
            CompositionLayerProjectionView::new()
                .pose(pose1)
                .fov(v1.fov)
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

#[cfg(test)]
mod tests {
    use super::*;
    use openxr as xr;

    fn pose_identity() -> xr::Posef {
        xr::Posef {
            orientation: xr::Quaternionf {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 1.0,
            },
            position: xr::Vector3f {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
        }
    }

    #[test]
    fn identity_pose_maps_to_identity_ref_from_view() {
        let m = ref_from_view_matrix(&pose_identity());
        assert!(
            m.abs_diff_eq(Mat4::IDENTITY, 1e-4),
            "expected identity ref_from_view, got {m:?}"
        );
    }

    #[test]
    fn identity_openxr_pose_maps_to_identity_engine_quat() {
        let (_p, q) = openxr_pose_to_engine(&pose_identity());
        assert!(
            q.abs_diff_eq(Quat::IDENTITY, 1e-4),
            "expected identity engine orientation, got {q:?}"
        );
    }

    #[test]
    fn ref_from_view_forward_matches_basis_rotated_neg_z() {
        let angle = std::f32::consts::FRAC_PI_4;
        let q_xr = Quat::from_rotation_y(angle);
        let pose = xr::Posef {
            orientation: xr::Quaternionf {
                x: q_xr.x,
                y: q_xr.y,
                z: q_xr.z,
                w: q_xr.w,
            },
            position: xr::Vector3f {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
        };
        let ref_from_view = ref_from_view_matrix(&pose);
        let forward_ref = ref_from_view.transform_vector3(-Vec3::Z);
        let (_p, q_eng) = openxr_pose_to_engine(&pose);
        let r_eng = Mat3::from_quat(q_eng);
        let expected = r_eng * (-Vec3::Z);
        assert!(
            forward_ref.abs_diff_eq(expected, 1e-3),
            "forward_ref={forward_ref:?} expected={expected:?}"
        );
    }
}
