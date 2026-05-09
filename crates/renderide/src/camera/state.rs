//! Host camera state types: per-frame camera fields, stereo bundle, and view identity.
//!
//! These types are populated by the host frame submit and consumed by world-mesh culling,
//! cluster lighting, world-mesh forward draw prep, the render graph's per-view planning, and
//! diagnostics. They live in `crate::camera` (and not in `render_graph/`) so non-graph
//! modules can talk about cameras and views without depending on the graph framework.

use glam::{Mat4, Vec3};

use crate::scene::RenderSpaceId;
use crate::shared::HeadOutputDevice;

use super::geometry::{CameraClipPlanes, EyeView, OrthographicProjectionSpec, Viewport};
use super::stereo::StereoViewMatrices;

/// Stable logical identity for one secondary camera view.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SecondaryCameraId {
    /// Render space containing the camera.
    pub render_space_id: RenderSpaceId,
    /// Dense host camera renderable index within the render space.
    pub renderable_index: i32,
}

impl SecondaryCameraId {
    /// Builds a secondary-camera id from the host render-space and dense camera row.
    pub const fn new(render_space_id: RenderSpaceId, renderable_index: i32) -> Self {
        Self {
            render_space_id,
            renderable_index,
        }
    }
}

/// Stable logical identity for one host camera readback task view.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CameraRenderTaskViewId {
    /// Render space requested by the host task.
    pub render_space_id: RenderSpaceId,
    /// Dense index within the drained host task batch.
    pub task_index: i32,
}

impl CameraRenderTaskViewId {
    /// Builds a camera readback view id from the host render-space and task batch index.
    pub const fn new(render_space_id: RenderSpaceId, task_index: i32) -> Self {
        Self {
            render_space_id,
            task_index,
        }
    }
}

/// Stable logical identity for one reflection-probe cubemap bake face.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ReflectionProbeRenderTaskViewId {
    /// Render space requested by the host task.
    pub render_space_id: RenderSpaceId,
    /// Host reflection-probe bake task id.
    pub render_task_id: i32,
    /// Cubemap face index in host `BitmapCube` order.
    pub face_index: u8,
}

impl ReflectionProbeRenderTaskViewId {
    /// Builds a reflection-probe bake face view id.
    pub const fn new(render_space_id: RenderSpaceId, render_task_id: i32, face_index: u8) -> Self {
        Self {
            render_space_id,
            render_task_id,
            face_index,
        }
    }
}

/// Identifies one logical render view for view-scoped resources and temporal state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ViewId {
    /// Main window or OpenXR multiview (shared primary-view state).
    Main,
    /// Secondary camera, tracked independently from the render target asset it writes.
    SecondaryCamera(SecondaryCameraId),
    /// One-shot host camera readback task view.
    CameraRenderTask(CameraRenderTaskViewId),
    /// One-shot reflection-probe cubemap bake face view.
    ReflectionProbeRenderTask(ReflectionProbeRenderTaskViewId),
}

impl ViewId {
    /// Builds the stable logical identity for one secondary camera view.
    pub const fn secondary_camera(render_space_id: RenderSpaceId, renderable_index: i32) -> Self {
        Self::SecondaryCamera(SecondaryCameraId::new(render_space_id, renderable_index))
    }

    /// Builds the stable logical identity for one camera readback task view.
    pub const fn camera_render_task(render_space_id: RenderSpaceId, task_index: i32) -> Self {
        Self::CameraRenderTask(CameraRenderTaskViewId::new(render_space_id, task_index))
    }

    /// Builds the stable logical identity for one reflection-probe bake face view.
    pub const fn reflection_probe_render_task(
        render_space_id: RenderSpaceId,
        render_task_id: i32,
        face_index: u8,
    ) -> Self {
        Self::ReflectionProbeRenderTask(ReflectionProbeRenderTaskViewId::new(
            render_space_id,
            render_task_id,
            face_index,
        ))
    }
}

/// Projection family used by shader helpers that need to distinguish perspective from orthographic math.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum CameraProjectionKind {
    /// Perspective projection with rays converging on the camera position.
    #[default]
    Perspective,
    /// Orthographic projection with parallel camera rays.
    Orthographic,
}

/// Latest camera-related fields from host [`crate::shared::FrameSubmitData`], updated each `frame_submit`.
#[derive(Clone, Copy, Debug)]
pub struct HostCameraFrame {
    /// Host lock-step frame index (`-1` before the first submit in standalone).
    pub frame_index: i32,
    /// Clip distances from the host frame submission.
    pub clip: CameraClipPlanes,
    /// Vertical field of view in **degrees** (matches host `desktopFOV`).
    pub desktop_fov_degrees: f32,
    /// Whether the host reported VR output as active for this frame.
    pub vr_active: bool,
    /// Init-time head output device selected by the host.
    pub output_device: HeadOutputDevice,
    /// Active projection family for the view represented by this frame.
    pub projection_kind: CameraProjectionKind,
    /// First orthographic render-task projection (overlay main-camera ortho override).
    pub primary_ortho_task: Option<OrthographicProjectionSpec>,
    /// Per-eye stereo matrices when this frame renders the OpenXR multiview view; [`None`] on
    /// desktop or secondary-RT views. Set together via [`StereoViewMatrices`] so the view-projection,
    /// view-only matrices, and per-eye camera positions cannot drift out of sync.
    pub stereo: Option<StereoViewMatrices>,
    /// Legacy Unity `HeadOutput.transform` in renderer world space.
    pub head_output_transform: Mat4,
    /// Explicit per-view camera data (e.g. secondary render-texture cameras).
    pub explicit_view: Option<EyeView>,
    /// Eye/camera world position derived from the active main render space's `view_transform`.
    pub eye_world_position: Option<Vec3>,
    /// Skips Hi-Z temporal state and uses uncull or frustum-only paths for this view.
    pub suppress_occlusion_temporal: bool,
}

impl Default for HostCameraFrame {
    fn default() -> Self {
        Self {
            frame_index: -1,
            clip: CameraClipPlanes::default(),
            desktop_fov_degrees: 60.0,
            vr_active: false,
            output_device: HeadOutputDevice::Screen,
            projection_kind: CameraProjectionKind::Perspective,
            primary_ortho_task: None,
            stereo: None,
            head_output_transform: Mat4::IDENTITY,
            explicit_view: None,
            eye_world_position: None,
            suppress_occlusion_temporal: false,
        }
    }
}

impl HostCameraFrame {
    /// Returns the near clip distance.
    #[cfg(test)]
    pub const fn near_clip(self) -> f32 {
        self.clip.near
    }

    /// Returns the far clip distance.
    #[cfg(test)]
    pub const fn far_clip(self) -> f32 {
        self.clip.far
    }

    /// Returns the explicit world-to-view override when present.
    pub fn explicit_world_to_view(self) -> Option<Mat4> {
        self.explicit_view.map(|view| view.view)
    }

    /// Returns the explicit camera world position when present.
    pub fn explicit_world_position(self) -> Option<Vec3> {
        self.explicit_view.map(|view| view.world_position)
    }

    /// Returns the explicit view and projection override when present.
    pub fn explicit_view_projection(self) -> Option<(Mat4, Mat4)> {
        self.explicit_view.map(|view| (view.view, view.proj))
    }

    /// Returns active stereo only when the host frame is currently VR-active.
    pub fn active_stereo(self) -> Option<StereoViewMatrices> {
        if self.vr_active { self.stereo } else { None }
    }

    /// Returns the primary orthographic projection, or a unit fallback.
    ///
    /// Historically used for screen-overlay (`LayerType.Overlay`) draws but the screen overlay now
    /// builds its own unit-half-height ortho in [`super::WorldProjectionSet::from_scene_host`] so
    /// it does not get hijacked by the host's dash camera ortho task. Kept for parity with secondary
    /// orthographic camera paths that still want the host's `primary_ortho_task` value.
    #[allow(dead_code)]
    pub fn overlay_projection(self, viewport: Viewport, fallback_clip: CameraClipPlanes) -> Mat4 {
        self.primary_ortho_task
            .unwrap_or_else(|| OrthographicProjectionSpec::new(1.0, fallback_clip))
            .projection(viewport)
    }

    /// Resolves the world-space origin used for view-distance sorting.
    pub fn view_origin_world(self) -> Vec3 {
        self.explicit_world_position()
            .or(self.eye_world_position)
            .unwrap_or_else(|| self.head_output_transform.col(3).truncate())
    }

    /// Resolves left/right world camera positions for frame globals.
    pub fn camera_world_pair(self) -> (Vec3, Vec3) {
        if let Some(camera_world) = self.explicit_world_position() {
            return (camera_world, camera_world);
        }
        if let Some(stereo) = self.stereo {
            return stereo.world_position_pair();
        }
        let camera_world = self
            .eye_world_position
            .unwrap_or_else(|| self.head_output_transform.col(3).truncate());
        (camera_world, camera_world)
    }
}

#[cfg(test)]
mod tests {
    use glam::{Mat4, Vec3};

    use super::{CameraProjectionKind, EyeView, HostCameraFrame, StereoViewMatrices};

    fn eye_at(position: Vec3) -> EyeView {
        EyeView::new(Mat4::IDENTITY, Mat4::IDENTITY, Mat4::IDENTITY, position)
    }

    #[test]
    fn view_origin_prefers_explicit_then_eye_then_head_output() {
        let mut camera = HostCameraFrame {
            head_output_transform: Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0)),
            ..Default::default()
        };
        assert_eq!(camera.view_origin_world(), Vec3::new(1.0, 2.0, 3.0));

        camera.eye_world_position = Some(Vec3::new(4.0, 5.0, 6.0));
        assert_eq!(camera.view_origin_world(), Vec3::new(4.0, 5.0, 6.0));

        camera.explicit_view = Some(eye_at(Vec3::new(7.0, 8.0, 9.0)));
        assert_eq!(camera.view_origin_world(), Vec3::new(7.0, 8.0, 9.0));
    }

    #[test]
    fn camera_world_pair_prefers_explicit_then_stereo_then_eye() {
        let mut camera = HostCameraFrame {
            eye_world_position: Some(Vec3::new(1.0, 0.0, 0.0)),
            ..Default::default()
        };
        assert_eq!(
            camera.camera_world_pair(),
            (Vec3::new(1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0))
        );

        camera.stereo = Some(StereoViewMatrices::new(
            eye_at(Vec3::new(2.0, 0.0, 0.0)),
            eye_at(Vec3::new(3.0, 0.0, 0.0)),
        ));
        assert_eq!(
            camera.camera_world_pair(),
            (Vec3::new(2.0, 0.0, 0.0), Vec3::new(3.0, 0.0, 0.0))
        );

        camera.explicit_view = Some(eye_at(Vec3::new(4.0, 0.0, 0.0)));
        assert_eq!(
            camera.camera_world_pair(),
            (Vec3::new(4.0, 0.0, 0.0), Vec3::new(4.0, 0.0, 0.0))
        );
    }

    #[test]
    fn host_camera_defaults_to_perspective_projection_kind() {
        assert_eq!(
            HostCameraFrame::default().projection_kind,
            CameraProjectionKind::Perspective
        );
    }
}
