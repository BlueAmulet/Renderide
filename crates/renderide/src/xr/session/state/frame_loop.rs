//! OpenXR frame wait, view location, and frame-end submission.

use openxr as xr;
use openxr::{CompositionLayerProjection, CompositionLayerProjectionView, SwapchainSubImage};
use std::time::Duration;

use super::super::end_frame_watchdog::EndFrameWatchdog;
use super::XrSessionState;

/// Deadline for a single `xrEndFrame` call before the watchdog logs a compositor stall.
///
/// 500 ms is an order of magnitude above normal VR frame budgets (<= ~16 ms at 60 Hz, ~11 ms at
/// 90 Hz) while short enough that a true freeze surfaces within one log-visible interval.
const END_FRAME_WATCHDOG_TIMEOUT: Duration = Duration::from_millis(500);

impl XrSessionState {
    /// Blocks until the next frame, begins the frame stream. Returns `None` if not ready or idle.
    ///
    /// On a successful `frame_stream.begin()` sets [`Self::frame_open`] so the outer loop knows a
    /// matching `end_frame_*` must be called.
    pub fn wait_frame(
        &mut self,
        gpu_queue_access_gate: &crate::gpu::GpuQueueAccessGate,
    ) -> Result<Option<xr::FrameState>, xr::sys::Result> {
        if !self.session_running {
            std::thread::sleep(Duration::from_millis(10));
            return Ok(None);
        }
        let state = self.frame_wait.wait()?;
        {
            profiling::scope!("xr::frame_stream_begin");
            let _gate = gpu_queue_access_gate.lock();
            self.frame_stream.begin()?;
        };
        self.frame_open = true;
        Ok(Some(state))
    }

    /// Ends the frame with no composition layers (mirror path, or visibility fallback).
    pub fn end_frame_empty(
        &mut self,
        predicted_display_time: xr::Time,
        gpu_queue_access_gate: &crate::gpu::GpuQueueAccessGate,
    ) -> Result<(), xr::sys::Result> {
        profiling::scope!("xr::end_frame_empty");
        let wd = EndFrameWatchdog::arm(END_FRAME_WATCHDOG_TIMEOUT, "end_frame_empty");
        let res = {
            let _gate = gpu_queue_access_gate.lock();
            self.frame_stream
                .end(predicted_display_time, self.environment_blend_mode, &[])
        };
        self.frame_open = false;
        wd.disarm();
        res
    }

    /// Ends the frame via [`Self::end_frame_empty`] only if a frame scope is currently open; a
    /// no-op otherwise. Error paths in `xr::app_integration` call this after bailing out of HMD
    /// submit so the begin/end frame contract is honoured regardless of where submission failed.
    pub fn end_frame_if_open(
        &mut self,
        predicted_display_time: xr::Time,
        gpu_queue_access_gate: &crate::gpu::GpuQueueAccessGate,
    ) -> Result<(), xr::sys::Result> {
        if !self.frame_open {
            return Ok(());
        }
        self.end_frame_empty(predicted_display_time, gpu_queue_access_gate)
    }

    /// Submits a stereo projection layer referencing the acquired swapchain image.
    ///
    /// For the primary stereo view configuration (`PRIMARY_STEREO`), `views[0]` is the left eye and
    /// `views[1]` the right eye. Composition layer 0 / `image_array_index` 0 is the left eye, layer
    /// 1 / index 1 the right eye, matching multiview `view_index` in the stereo path.
    pub fn end_frame_projection(
        &mut self,
        predicted_display_time: xr::Time,
        swapchain: &xr::Swapchain<xr::Vulkan>,
        views: &[xr::View],
        image_rect: xr::Rect2Di,
        gpu_queue_access_gate: &crate::gpu::GpuQueueAccessGate,
    ) -> Result<(), xr::sys::Result> {
        profiling::scope!("xr::end_frame");
        if views.len() < 2 {
            return self.end_frame_empty(predicted_display_time, gpu_queue_access_gate);
        }
        let v0 = &views[0]; // left eye
        let v1 = &views[1]; // right eye
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
        let wd = EndFrameWatchdog::arm(END_FRAME_WATCHDOG_TIMEOUT, "end_frame_projection");
        let res = {
            let _gate = gpu_queue_access_gate.lock();
            self.frame_stream.end(
                predicted_display_time,
                self.environment_blend_mode,
                &[&layer],
            )
        };
        self.frame_open = false;
        wd.disarm();
        res
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

/// OpenXR requires a unit quaternion; some runtimes briefly report `(0,0,0,0)`, which makes
/// `xrEndFrame` fail with `XR_ERROR_POSE_INVALID`.
fn sanitize_pose_for_end_frame(pose: xr::Posef) -> xr::Posef {
    let o = pose.orientation;
    let len_sq =
        o.w.mul_add(o.w, o.z.mul_add(o.z, o.x.mul_add(o.x, o.y * o.y)));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_pose_replaces_invalid_orientation_with_identity() {
        let pose = xr::Posef {
            orientation: xr::Quaternionf {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 0.0,
            },
            position: xr::Vector3f {
                x: 1.0,
                y: 2.0,
                z: 3.0,
            },
        };
        let sanitized = sanitize_pose_for_end_frame(pose);
        assert_eq!(sanitized.orientation.x, 0.0);
        assert_eq!(sanitized.orientation.y, 0.0);
        assert_eq!(sanitized.orientation.z, 0.0);
        assert_eq!(sanitized.orientation.w, 1.0);
        assert_eq!(sanitized.position, pose.position);
    }
}
