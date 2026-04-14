//! Offscreen passes for scene cameras targeting host [`crate::resources::GpuRenderTexture`] assets.

use std::sync::Arc;

use rayon::prelude::*;

use crate::assets::material::MaterialDictionary;
use crate::gpu::GpuContext;
use crate::materials::{MaterialRouter, RasterPipelineKind};
use crate::pipelines::ShaderPermutation;
use crate::render_graph::{
    camera_state_enabled, collect_and_sort_world_mesh_draws, draw_filter_from_camera_entry,
    host_camera_frame_for_render_texture, CameraTransformDrawFilter, ExternalOffscreenTargets,
    GraphExecuteError, HostCameraFrame, WorldMeshDrawCollection,
};
use crate::scene::{RenderSpaceId, SceneCoordinator};

use super::RendererRuntime;
use winit::window::Window;

/// Resolved secondary camera target and host frame data (one entry per RT draw).
struct SecondaryRtPrepared {
    host_camera: HostCameraFrame,
    filter: CameraTransformDrawFilter,
    rt_id: i32,
    color_view: Arc<wgpu::TextureView>,
    depth_texture: Arc<wgpu::Texture>,
    depth_view: Arc<wgpu::TextureView>,
    viewport: (u32, u32),
    color_format: wgpu::TextureFormat,
}

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

        let mut prepared: Vec<SecondaryRtPrepared> = Vec::new();
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
            prepared.push(SecondaryRtPrepared {
                host_camera: hc,
                filter,
                rt_id,
                color_view,
                depth_texture,
                depth_view,
                viewport,
                color_format,
            });
        }

        let render_context = self.scene.active_main_render_context();
        let scene_ref: &SceneCoordinator = &self.scene;
        let property_store = self.backend.material_property_store();
        let mesh_pool = self.backend.mesh_pool();
        let fallback_router = MaterialRouter::new(RasterPipelineKind::DebugWorldNormals);
        let router_ref = self
            .backend
            .materials
            .material_registry
            .as_ref()
            .map(|r| &r.router)
            .unwrap_or(&fallback_router);

        let prefetched: Vec<WorldMeshDrawCollection> = prepared
            .par_iter()
            .map(|prep| {
                let dict = MaterialDictionary::new(property_store);
                collect_and_sort_world_mesh_draws(
                    scene_ref,
                    mesh_pool,
                    &dict,
                    router_ref,
                    ShaderPermutation(0),
                    render_context,
                    prep.host_camera.head_output_transform,
                    None,
                    Some(&prep.filter),
                )
            })
            .collect();

        for (prep, collection) in prepared.into_iter().zip(prefetched) {
            let ext = ExternalOffscreenTargets {
                render_texture_asset_id: prep.rt_id,
                color_view: prep.color_view.as_ref(),
                depth_texture: prep.depth_texture.as_ref(),
                depth_view: prep.depth_view.as_ref(),
                extent_px: prep.viewport,
                color_format: prep.color_format,
            };
            self.backend.execute_frame_graph_offscreen_single_view(
                gpu,
                window,
                scene_ref,
                prep.host_camera,
                ext,
                Some(prep.filter),
                Some(collection),
            )?;
        }
        Ok(())
    }
}
