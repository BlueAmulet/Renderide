//! Renderer façade: orchestrates **frontend** (IPC / shared memory / lock-step), **scene** (host
//! logical state), and **backend** (GPU pools, material store, uploads).
//!
//! Phase order aligns with the historical session loop: drain incoming commands first, then emit
//! [`FrameStartData`](crate::shared::FrameStartData) when lock-step allows (see `app` `tick_frame`).
//! Asset integration between begin-frame and frame processing remains a stub here.
//!
//! Lock-step is driven by the `last_frame_index` field of [`FrameStartData`](crate::shared::FrameStartData)
//! on the **outgoing** `frame_start_data` the renderer sends from [`RendererRuntime::pre_frame`].
//! If the host sends [`RendererCommand::FrameStartData`](crate::shared::RendererCommand::FrameStartData),
//! optional payloads are trace-logged until consumers exist.

mod accessors;
mod commands;
mod debug_hud_frame;
mod frame_submit;
mod host_camera_apply;
mod ipc_init_dispatch;
mod lights_ipc;
mod lockstep;
mod secondary_cameras;
mod shader_material_ipc;
mod xr_impls;

use std::path::PathBuf;
use std::time::{Duration, Instant};

use crate::backend::RenderBackend;
use crate::config::RendererSettingsHandle;
use crate::connection::ConnectionParams;
use crate::frontend::RendererFrontend;
use crate::gpu::GpuContext;

use crate::render_graph::{ExternalFrameTargets, GraphExecuteError, HostCameraFrame};

pub use crate::frontend::InitState;
use crate::ipc::SharedMemoryAccessor;
use crate::scene::SceneCoordinator;
use crate::shared::{
    FrameSubmitData, InputState, LightsBufferRendererSubmission, MaterialsUpdateBatch, OutputState,
    RendererCommand, RendererInitData, ShaderUnload, ShaderUpload,
};
use winit::window::Window;

/// Facade: [`RendererFrontend`] + [`SceneCoordinator`] + [`RenderBackend`] + ingestion helpers.
pub struct RendererRuntime {
    frontend: RendererFrontend,
    backend: RenderBackend,
    /// Render spaces and dense transform / mesh state from [`FrameSubmitData`](crate::shared::FrameSubmitData).
    pub scene: SceneCoordinator,
    /// Last host clip / FOV / VR / ortho task state for [`crate::render_graph::FrameRenderParams`].
    pub host_camera: HostCameraFrame,
    /// Process-wide renderer settings (shared with the debug HUD and the frame loop).
    settings: RendererSettingsHandle,
    /// Target path for persisting [`Self::settings`] from the ImGui config window.
    config_save_path: PathBuf,
    /// Throttled host CPU/RAM sampling for the debug HUD.
    host_hud: crate::diagnostics::HostHudGatherer,
    /// [`FrameSubmitData::render_tasks`] length from the last applied frame submit (HUD).
    last_submit_render_task_count: usize,
}

impl RendererRuntime {
    /// Builds a runtime; does not open IPC yet (see [`Self::connect_ipc`]).
    pub fn new(
        params: Option<ConnectionParams>,
        settings: RendererSettingsHandle,
        config_save_path: PathBuf,
    ) -> Self {
        Self {
            frontend: RendererFrontend::new(params),
            backend: RenderBackend::new(),
            scene: SceneCoordinator::new(),
            host_camera: HostCameraFrame::default(),
            settings,
            config_save_path,
            host_hud: crate::diagnostics::HostHudGatherer::default(),
            last_submit_render_task_count: 0,
        }
    }

    /// Records and presents one frame via the backend’s compiled render graph.
    pub fn execute_frame_graph(
        &mut self,
        gpu: &mut GpuContext,
        window: &Window,
    ) -> Result<(), GraphExecuteError> {
        self.backend
            .frame_resources
            .prepare_lights_from_scene(&self.scene);
        self.sync_debug_hud_diagnostics_from_settings();
        let scene_ref: &SceneCoordinator = &self.scene;
        self.backend
            .execute_frame_graph(gpu, window, scene_ref, self.host_camera)
    }

    /// Renders to OpenXR multiview array targets (see [`RenderBackend::execute_frame_graph_external_multiview`]).
    pub fn execute_frame_graph_external_multiview(
        &mut self,
        gpu: &mut GpuContext,
        window: &Window,
        external: ExternalFrameTargets<'_>,
    ) -> Result<(), GraphExecuteError> {
        self.run_frame_graph_external_multiview(gpu, window, external)
    }

    pub(super) fn run_frame_graph_external_multiview(
        &mut self,
        gpu: &mut GpuContext,
        window: &Window,
        external: ExternalFrameTargets<'_>,
    ) -> Result<(), GraphExecuteError> {
        self.backend
            .frame_resources
            .prepare_lights_from_scene(&self.scene);
        self.sync_debug_hud_diagnostics_from_settings();
        let scene_ref: &SceneCoordinator = &self.scene;
        self.backend.execute_frame_graph_external_multiview(
            gpu,
            window,
            scene_ref,
            self.host_camera,
            external,
        )
    }

    /// Whether the next tick should build [`InputState`] and call [`Self::pre_frame`].
    pub fn should_send_begin_frame(&self) -> bool {
        self.frontend.should_send_begin_frame()
    }

    /// Records wall-clock spacing for host FPS metrics. Call at the very start of each winit tick,
    /// before [`Self::poll_ipc`], OpenXR, and [`Self::pre_frame`].
    pub fn tick_frame_wall_clock_begin(&mut self, now: Instant) {
        self.backend.frame_resources.reset_light_prep_for_tick();
        self.frontend.on_tick_frame_wall_clock(now);
    }

    /// Completes wall-clock timing for the tick: stores elapsed from `frame_start` for the next
    /// [`crate::shared::PerformanceState::render_time`]. Call once before every return from
    /// [`crate::app::RenderideApp::tick_frame`].
    pub fn tick_frame_wall_clock_end(&mut self, frame_start: Instant) {
        self.frontend
            .set_perf_last_total_us(frame_start.elapsed().as_micros() as u64);
    }

    /// Host [`OutputState::lock_cursor`] bit merged into packed mouse state.
    pub fn host_cursor_lock_requested(&self) -> bool {
        self.frontend.host_cursor_lock_requested()
    }

    /// If connected and init is complete, sends [`FrameStartData`](crate::shared::FrameStartData) when we are ready for the next host frame.
    pub fn pre_frame(&mut self, inputs: InputState) {
        self.frontend.pre_frame(inputs);
    }

    /// Drains pending host window policy after [`Self::poll_ipc`].
    pub fn take_pending_output_state(&mut self) -> Option<OutputState> {
        self.frontend.take_pending_output_state()
    }

    /// Last [`OutputState`] from the host (for per-frame cursor lock / warp).
    pub fn last_output_state(&self) -> Option<&OutputState> {
        self.frontend.last_output_state()
    }

    /// Placeholder for bounded asset integration between begin-frame and frame processing (Unity:
    /// `RunAssetIntegration`). Upload work is driven from [`Self::poll_ipc`] via
    /// [`crate::backend::RenderBackend`].
    pub fn run_asset_integration_stub(&mut self, _budget: Duration) {}

    /// Drains IPC and dispatches commands. Each poll batch is ordered so `renderer_init_data` runs
    /// first, then frame submits, then the rest (see [`RendererFrontend::poll_commands`]).
    pub fn poll_ipc(&mut self) {
        self.backend.begin_ipc_poll_mesh_upload_budget();
        let batch = self.frontend.poll_commands();
        for cmd in batch {
            ipc_init_dispatch::dispatch_ipc_command(self, cmd);
        }
        let (shm, ipc) = self.frontend.transport_pair_mut();
        if let Some(shm) = shm {
            self.backend.finish_ipc_poll_mesh_upload_deferred(shm, ipc);
        }
    }

    pub(super) fn on_init_data(&mut self, d: RendererInitData) {
        self.host_camera.output_device = d.output_device;
        if let Some(ref prefix) = d.shared_memory_prefix {
            self.frontend
                .set_shared_memory(SharedMemoryAccessor::new(prefix.clone()));
            logger::info!("Shared memory prefix: {}", prefix);
            let (shm, ipc) = self.frontend.transport_pair_mut();
            if let (Some(shm), Some(ipc)) = (shm, ipc) {
                self.backend.flush_pending_material_batches(shm, ipc);
            }
        }
        self.frontend.set_pending_init(d.clone());
        if let Some(ref mut ipc) = self.frontend.ipc_mut() {
            ipc_init_dispatch::send_renderer_init_result(ipc, d.output_device);
        }
        self.frontend.on_init_received();
    }

    pub(super) fn handle_running_command(&mut self, cmd: RendererCommand) {
        commands::handle_running_command(self, cmd);
    }

    fn on_shader_upload(&mut self, upload: ShaderUpload) {
        shader_material_ipc::on_shader_upload(&mut self.frontend, &mut self.backend, upload);
    }

    fn on_shader_unload(&mut self, unload: ShaderUnload) {
        shader_material_ipc::on_shader_unload(&mut self.backend, unload);
    }

    fn on_materials_update_batch(&mut self, batch: MaterialsUpdateBatch) {
        shader_material_ipc::on_materials_update_batch(
            &mut self.frontend,
            &mut self.backend,
            batch,
        );
    }

    fn on_lights_buffer_renderer_submission(&mut self, sub: LightsBufferRendererSubmission) {
        let buffer_id = sub.lights_buffer_unique_id;
        let (shm, ipc) = self.frontend.transport_pair_mut();
        let Some(shm) = shm else {
            logger::warn!("lights_buffer_renderer_submission: no shared memory (id={buffer_id})");
            return;
        };
        lights_ipc::apply_lights_buffer_submission(&mut self.scene, shm, ipc, sub);
    }

    fn on_frame_submit(&mut self, data: FrameSubmitData) {
        let prev_frame_index = self.host_camera.frame_index;
        lockstep::trace_duplicate_frame_index_if_interesting(data.frame_index, prev_frame_index);
        frame_submit::process_frame_submit(self, data);
    }
}
