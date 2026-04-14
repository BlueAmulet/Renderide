//! Offscreen passes for scene cameras targeting host [`crate::resources::GpuRenderTexture`] assets.

use crate::gpu::GpuContext;
use crate::render_graph::{
    camera_state_enabled, draw_filter_from_camera_entry, host_camera_frame_for_render_texture,
    ExternalOffscreenTargets, GraphExecuteError,
};
use crate::scene::{RenderSpaceId, SceneCoordinator};

use super::RendererRuntime;
use winit::window::Window;

impl RendererRuntime {
    /// Renders secondary cameras to host render textures before the main swapchain pass.
    pub fn render_secondary_cameras_to_render_textures(
        &mut self,
        gpu: &mut GpuContext,
        window: &Window,
    ) -> Result<(), GraphExecuteError> {
        self.backend
            .frame_resources
            .prepare_lights_from_scene(&self.scene);
        self.sync_debug_hud_diagnostics_from_settings();

        let mut tasks: Vec<(RenderSpaceId, f32, usize)> = Vec::new();
        for sid in self.scene.render_space_ids() {
            let Some(space) = self.scene.space(sid) else {
                continue;
            };
            if !space.is_active {
                continue;
            }
            for (idx, cam) in space.cameras.iter().enumerate() {
                if !camera_state_enabled(cam.state.flags) {
                    continue;
                }
                if cam.state.render_texture_asset_id < 0 {
                    continue;
                }
                tasks.push((sid, cam.state.depth, idx));
            }
        }
        tasks.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        for (sid, _, cam_idx) in tasks {
            let Some(space) = self.scene.space(sid) else {
                continue;
            };
            let Some(entry) = space.cameras.get(cam_idx) else {
                continue;
            };
            if !camera_state_enabled(entry.state.flags) {
                continue;
            }
            let rt_id = entry.state.render_texture_asset_id;
            let (color_view, depth_texture, depth_view, viewport, color_format) = {
                let Some(rt) = self.backend.render_texture_pool().get(rt_id) else {
                    logger::trace!(
                        "secondary camera: render texture asset {rt_id} not resident; skipping"
                    );
                    continue;
                };
                let Some(dt) = rt.depth_texture.clone() else {
                    logger::warn!("secondary camera: render texture {rt_id} missing depth");
                    continue;
                };
                let Some(dv) = rt.depth_view.clone() else {
                    logger::warn!("secondary camera: render texture {rt_id} missing depth view");
                    continue;
                };
                (
                    rt.color_view.clone(),
                    dt,
                    dv,
                    (rt.width, rt.height),
                    rt.wgpu_color_format,
                )
            };
            let Some(world_m) = self.scene.world_matrix(sid, entry.transform_id as usize) else {
                continue;
            };
            let hc = host_camera_frame_for_render_texture(
                &self.host_camera,
                &entry.state,
                viewport,
                world_m,
                &self.scene,
            );
            let filter = draw_filter_from_camera_entry(entry);
            let ext = ExternalOffscreenTargets {
                render_texture_asset_id: rt_id,
                color_view: color_view.as_ref(),
                depth_texture: depth_texture.as_ref(),
                depth_view: depth_view.as_ref(),
                extent_px: viewport,
                color_format,
            };
            let scene_ref: &SceneCoordinator = &self.scene;
            self.backend.execute_frame_graph_offscreen_single_view(
                gpu,
                window,
                scene_ref,
                hc,
                ext,
                Some(filter),
            )?;
        }
        Ok(())
    }
}
