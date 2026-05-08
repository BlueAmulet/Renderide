//! Per-tick wiring from [`super::RendererRuntime`] to the backend [`crate::backend::RenderBackend`] debug HUD.

use std::time::Instant;

use crate::diagnostics::DebugHudEncodeError;
use crate::gpu::GpuContext;

use super::RendererRuntime;

impl RendererRuntime {
    /// Fills renderer info, frame diagnostics, and related main-tab HUD snapshots when the main HUD is on.
    fn capture_main_debug_hud_panels(&mut self, gpu: &GpuContext, now: Instant) {
        let host = self.diagnostics.host_hud.snapshot();
        self.diagnostics.refresh_gpu_allocator_report_hud(gpu, now);
        let next_refresh_in_secs = self.diagnostics.allocator_report_next_refresh_in_secs(now);
        let (ipc_pri_str, ipc_bg_str) = self.frontend.ipc_consecutive_outbound_drop_streaks();
        let backend_diag = self.backend.snapshot_for_diagnostics();
        let frame_diag = crate::diagnostics::FrameDiagnosticsSnapshot::capture(
            crate::diagnostics::FrameDiagnosticsSnapshotCapture {
                host,
                last_submit_render_task_count: self.diagnostics.last_submit_render_task_count,
                pending_camera_readbacks: self.diagnostics.pending_camera_readbacks,
                completed_camera_readbacks: self.diagnostics.completed_camera_readbacks,
                failed_camera_readbacks: self.diagnostics.failed_camera_readbacks,
                backend: &backend_diag,
                ipc: crate::diagnostics::FrameDiagnosticsIpcQueues {
                    ipc_primary_outbound_drop_this_tick: self
                        .frontend
                        .ipc_outbound_primary_drop_this_tick(),
                    ipc_background_outbound_drop_this_tick: self
                        .frontend
                        .ipc_outbound_background_drop_this_tick(),
                    ipc_primary_consecutive_fail_streak: ipc_pri_str,
                    ipc_background_consecutive_fail_streak: ipc_bg_str,
                },
                xr: crate::diagnostics::XrRecoverableFailureCounts {
                    xr_wait_frame_failures: self.xr_stats.wait_frame_failures,
                    xr_locate_views_failures: self.xr_stats.locate_views_failures,
                },
                allocator: crate::diagnostics::GpuAllocatorHudRefresh {
                    gpu_allocator_totals: self.diagnostics.allocator_report_totals,
                    gpu_allocator_report: self.diagnostics.allocator_report_hud.clone(),
                    gpu_allocator_report_next_refresh_in_secs: next_refresh_in_secs,
                },
                frame_submit_apply_failures: self.diagnostics.frame_submit_apply_failures,
                unhandled_ipc_command_event_total: self.unhandled_ipc_command_event_total(),
            },
        );
        let msaa_requested = self
            .config
            .settings
            .read()
            .map(|s| s.rendering.msaa.as_count())
            .unwrap_or(1);
        let snapshot = crate::diagnostics::RendererInfoSnapshot::capture(
            crate::diagnostics::RendererInfoSnapshotCapture {
                ipc_connected: self.is_ipc_connected(),
                init_state: self.init_state(),
                last_frame_index: self.last_frame_index(),
                adapter_info: gpu.adapter_info(),
                gpu_limits: gpu.limits().as_ref(),
                surface_format: gpu.config_format(),
                viewport_px: gpu.surface_extent_px(),
                present_mode: gpu.present_mode(),
                scene: &self.scene,
                backend: &backend_diag,
                gpu,
                msaa_requested_samples: msaa_requested,
            },
        );
        self.backend.set_debug_hud_snapshot(snapshot);
        self.backend.set_debug_hud_frame_diagnostics(frame_diag);
    }

    /// Copies debug HUD capture flags into the backend before the render graph runs.
    pub(super) fn sync_debug_hud_diagnostics_from_settings(&mut self) {
        let (main, textures) = self
            .config
            .settings
            .read()
            .map(|s| (s.debug.debug_hud_enabled, s.debug.debug_hud_textures))
            .unwrap_or((false, false));
        self.backend.set_debug_hud_main_enabled(main);
        self.backend.set_debug_hud_textures_enabled(textures);
        self.backend
            .clear_debug_hud_current_view_texture_2d_asset_ids();
    }

    /// Updates debug HUD snapshots after [`crate::gpu::GpuContext::end_frame_timing`] for the winit tick.
    pub fn capture_debug_hud_after_frame_end(&mut self, gpu: &GpuContext) {
        profiling::scope!("hud::capture_snapshot");
        let wall_ms = self.backend.debug_frame_time_ms();
        self.diagnostics.frame_time_history.push(wall_ms as f32);
        // Host CPU / RAM / process RAM are sampled every tick so the Frame timing overlay can show
        // them without requiring the full debug HUD (heavier panels are still gated below).
        let host = self.diagnostics.host_hud.snapshot();
        let frame_timing = crate::diagnostics::FrameTimingHudSnapshot::capture(
            gpu,
            wall_ms,
            &host,
            &self.diagnostics.frame_time_history,
            &mut self.diagnostics.frame_timing_ema,
        );
        self.backend.set_debug_hud_frame_timing(frame_timing);
        let gpu_pass_timings = gpu
            .latest_gpu_pass_timings_handle()
            .lock()
            .map(|guard| guard.clone())
            .unwrap_or_default();
        self.backend
            .set_debug_hud_gpu_pass_timings(gpu_pass_timings);

        let (main_hud, transforms_hud, textures_hud) = self
            .config
            .settings
            .read()
            .map(|s| {
                (
                    s.debug.debug_hud_enabled,
                    s.debug.debug_hud_transforms,
                    s.debug.debug_hud_textures,
                )
            })
            .unwrap_or((false, false, false));

        if main_hud {
            let now = Instant::now();
            self.capture_main_debug_hud_panels(gpu, now);
        } else {
            self.backend.clear_debug_hud_stats_snapshots();
            self.diagnostics.clear_allocator_report();
        }

        if transforms_hud {
            let scene_transforms =
                crate::diagnostics::SceneTransformsSnapshot::capture(&self.scene);
            self.backend
                .set_debug_hud_scene_transforms_snapshot(scene_transforms);
        } else {
            self.backend.clear_debug_hud_scene_transforms_snapshot();
        }

        if textures_hud {
            let textures = crate::diagnostics::TextureDebugSnapshot::capture(
                self.backend.texture_pool(),
                self.backend.debug_hud_current_view_texture_2d_asset_ids(),
            );
            self.backend.set_debug_hud_texture_debug_snapshot(textures);
        } else {
            self.backend.clear_debug_hud_texture_debug_snapshot();
        }
    }

    /// Encodes the Dear ImGui debug overlay onto an acquired swapchain view (e.g. after the VR mirror blit).
    ///
    /// Uses the same composite path as the desktop render graph (`LoadOp::Load`). Caller must keep
    /// [`Self::set_debug_hud_frame_data`] in sync for this tick before encoding.
    pub(crate) fn encode_debug_hud_overlay_on_surface(
        &mut self,
        gpu: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        backbuffer: &wgpu::TextureView,
    ) -> Result<(), DebugHudEncodeError> {
        let device = gpu.device().as_ref();
        let extent = gpu.surface_extent_px();
        let q = gpu.queue().as_ref();
        self.backend
            .encode_debug_hud_overlay(device, q, encoder, backbuffer, extent)
    }
}
