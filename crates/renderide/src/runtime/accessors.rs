//! Thin [`super::RendererRuntime`] accessors and forwards to the frontend, backend, and settings.

use crate::config::RendererSettingsHandle;
use crate::connection::InitError;
use crate::diagnostics::DebugHudInput;
use crate::frontend::InitState;
use crate::gpu::GpuContext;
use crate::shared::RendererInitData;

use super::RendererRuntime;

impl RendererRuntime {
    /// Updates the latest [`crate::shared::FrameSubmitData::render_tasks`] count for the HUD.
    pub(crate) fn set_last_submit_render_task_count(&mut self, n: usize) {
        self.diagnostics.set_last_submit_render_task_count(n);
    }

    /// Increments the cumulative scene-apply failure counter surfaced on the diagnostics HUD.
    pub(crate) fn note_frame_submit_apply_failure(&mut self) {
        self.diagnostics.note_frame_submit_apply_failure();
    }

    /// Disables writing `config.toml` from the HUD when load-time Figment extraction failed.
    pub fn set_suppress_renderer_config_disk_writes(&mut self, value: bool) {
        self.config.set_suppress_renderer_config_disk_writes(value);
    }

    /// Shared settings store ([`crate::config::RendererSettings`]).
    pub fn settings(&self) -> &RendererSettingsHandle {
        &self.config.settings
    }

    /// Opens Primary/Background queues when [`Self::new`] was given connection parameters.
    pub fn connect_ipc(&mut self) -> Result<(), InitError> {
        self.frontend.connect_ipc()
    }

    /// Whether IPC queues are open.
    pub fn is_ipc_connected(&self) -> bool {
        self.frontend.is_ipc_connected()
    }

    /// Host/renderer init handshake phase (see [`crate::frontend::RendererFrontend::init_state`]).
    pub fn init_state(&self) -> InitState {
        self.frontend.init_state()
    }

    /// After a successful [`FrameSubmitData`] application, host may expect another begin-frame.
    #[cfg(test)]
    pub fn last_frame_data_processed(&self) -> bool {
        self.frontend.last_frame_data_processed()
    }

    /// Current lock-step frame index echoed to the host.
    pub fn last_frame_index(&self) -> i32 {
        self.frontend.last_frame_index()
    }

    /// Host requested an orderly renderer shutdown over IPC.
    pub fn shutdown_requested(&self) -> bool {
        self.frontend.shutdown_requested()
    }

    /// Unrecoverable IPC/init error; begin-frame is suppressed until reset.
    pub fn fatal_error(&self) -> bool {
        self.frontend.fatal_error()
    }

    /// Whether the host last reported VR mode as active (see [`crate::camera::HostCameraFrame::vr_active`]).
    pub fn vr_active(&self) -> bool {
        self.host_camera.vr_active
    }

    /// Host [`RendererInitData`] after connect, before [`Self::take_pending_init`] consumes it.
    pub fn pending_init(&self) -> Option<&RendererInitData> {
        self.frontend.pending_init()
    }

    /// Applies pending init once a GPU/window stack exists (e.g. window title).
    pub fn take_pending_init(&mut self) -> Option<RendererInitData> {
        self.frontend.take_pending_init()
    }

    /// Call after [`crate::gpu::GpuContext`] is created so mesh/texture uploads can use the GPU.
    ///
    /// On attach failure, an error is logged; CPU-side work may continue but GPU rendering paths remain
    /// unconfigured until a successful attach.
    pub fn attach_gpu(&mut self, gpu: &GpuContext) {
        use std::sync::Arc;

        let device = gpu.device().clone();
        let queue = Arc::clone(gpu.queue());
        let gpu_queue_access_gate = gpu.gpu_queue_access_gate().clone();
        let renderer_settings = Arc::clone(&self.config.settings);
        let config_save_path = self.config.cloned_config_save_path();
        let suppress_renderer_config_disk_writes =
            self.config.suppress_renderer_config_disk_writes();
        let shm = self.frontend.shared_memory_mut();
        if let Err(e) = self.backend.attach(
            crate::backend::RenderBackendAttachDesc {
                device,
                queue,
                gpu_queue_access_gate,
                gpu_limits: Arc::clone(gpu.limits()),
                surface_format: gpu.config_format(),
                renderer_settings,
                config_save_path,
                suppress_renderer_config_disk_writes,
            },
            shm,
        ) {
            logger::error!("GPU attach failed: {e}; CPU work continues, GPU draws disabled");
        }
    }

    /// Per-frame pointer state for the ImGui overlay ([`diagnostics::DebugHud`]).
    pub fn set_debug_hud_input(&mut self, input: DebugHudInput) {
        self.backend.set_debug_hud_input(input);
    }

    /// Wall-clock roundtrip (ms) between consecutive `tick_frame` starts; drives the HUD's
    /// FPS / Frame readout. Set in the tick prologue so the value cleanly reflects the
    /// roundtrip period.
    pub fn set_debug_hud_wall_frame_time_ms(&mut self, frame_time_ms: f64) {
        self.backend.set_debug_hud_wall_frame_time_ms(frame_time_ms);
    }

    /// Last ImGui `want_capture_mouse` after the previous successful HUD encode; used when filtering [`InputState`] for the host.
    pub fn debug_hud_last_want_capture_mouse(&self) -> bool {
        self.backend.debug_hud_last_want_capture_mouse()
    }

    /// Last ImGui `want_capture_keyboard` after the previous successful HUD encode; used when filtering [`InputState`] for the host.
    pub fn debug_hud_last_want_capture_keyboard(&self) -> bool {
        self.backend.debug_hud_last_want_capture_keyboard()
    }
}

#[cfg(test)]
impl RendererRuntime {
    /// Installs a shared-memory accessor for tests that apply [`crate::shared::FrameSubmitData`].
    pub(crate) fn test_set_shared_memory(&mut self, prefix: impl Into<String>) {
        use crate::ipc::SharedMemoryAccessor;
        self.frontend
            .set_shared_memory(SharedMemoryAccessor::new(prefix.into()));
    }
}
