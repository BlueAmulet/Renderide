//! Per-tick wiring from [`super::RendererRuntime`] to the backend [`crate::backend::RenderBackend`] debug HUD.

use crate::gpu::GpuContext;

use super::RendererRuntime;

impl RendererRuntime {
    /// Copies debug HUD capture flags into the backend before the render graph runs.
    pub(super) fn sync_debug_hud_diagnostics_from_settings(&mut self) {
        let (main, textures) = self
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
        let frame_timing = crate::diagnostics::FrameTimingHudSnapshot::capture(
            gpu,
            self.backend.debug_frame_time_ms(),
        );
        self.backend.set_debug_hud_frame_timing(frame_timing);

        let (main_hud, transforms_hud, textures_hud) = self
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
            let host = self.host_hud.snapshot();
            let frame_diag = crate::diagnostics::FrameDiagnosticsSnapshot::capture(
                gpu,
                self.backend.debug_frame_time_ms(),
                host,
                self.last_submit_render_task_count,
                &self.backend,
            );
            let snapshot = crate::diagnostics::RendererInfoSnapshot::capture(
                self.is_ipc_connected(),
                self.init_state(),
                self.last_frame_index(),
                gpu.adapter_info(),
                gpu.limits().as_ref(),
                gpu.config_format(),
                gpu.surface_extent_px(),
                gpu.present_mode(),
                self.backend.debug_frame_time_ms(),
                &self.scene,
                &self.backend,
            );
            self.backend.set_debug_hud_snapshot(snapshot);
            self.backend.set_debug_hud_frame_diagnostics(frame_diag);
        } else {
            self.backend.clear_debug_hud_stats_snapshots();
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
    ) -> Result<(), String> {
        let device = gpu.device().as_ref();
        let extent = gpu.surface_extent_px();
        let q = gpu
            .queue()
            .lock()
            .map_err(|e| format!("queue mutex poisoned: {e}"))?;
        self.backend
            .encode_debug_hud_overlay(device, &q, encoder, backbuffer, extent)
    }
}
