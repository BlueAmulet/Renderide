//! [`RenderBackend`] -- thin facade for frame execution and IPC-facing GPU work.
//!
//! Core subsystems live in [`super::MaterialSystem`], [`crate::backend::AssetTransferQueue`],
//! [`super::FrameResourceManager`], and [`crate::occlusion::OcclusionSystem`]; this type wires attach,
//! the compiled render graph, mesh deform preprocess, and debug HUD.
//!
//! Graph execution lives in the `execute` submodule; IPC-facing asset handlers in `asset_ipc`.

mod asset_ipc;
mod diagnostics;
mod draw_preparation;
mod execute;
mod frame_packet;
mod frame_services;
mod graph_access;
mod graph_cache;
mod graph_state;
mod reflection_services;

use std::path::PathBuf;
use std::sync::Arc;

use thiserror::Error;

use crate::backend::AssetTransferQueue;
use crate::backend::asset_transfers as asset_uploads;
use crate::config::{PostProcessingSettings, RendererSettingsHandle, SceneColorFormat};
use crate::diagnostics::{DebugHudEncodeError, DebugHudInput, SceneTransformsSnapshot};
use crate::gpu::GpuLimits;
use crate::gpu_pools::{MeshPool, RenderTexturePool, TexturePool};
use crate::materials::host_data::MaterialPropertyStore;
use crate::render_graph::TransientPool;
use crate::world_mesh::{WorldMeshDrawStateRow, WorldMeshDrawStats};

use super::{FrameGpuBindingsError, FrameResourceManager};
use crate::materials::MaterialSystem;
use crate::materials::embedded::EmbeddedMaterialBindError;
use crate::occlusion::OcclusionSystem;
use diagnostics::BackendDiagnostics;
use draw_preparation::BackendDrawPreparation;
use frame_services::BackendFrameServices;
pub(crate) use graph_access::BackendGraphAccess;
use graph_state::RenderGraphState;
use reflection_services::ReflectionProbeServices;

pub(crate) use frame_packet::ExtractedFrameShared;

/// GPU attach failed for frame binds (`@group(0/1/2)`) or embedded materials (`@group(1)`).
#[derive(Debug, Error)]
pub enum RenderBackendAttachError {
    /// Frame / empty material / per-draw allocation failed atomically.
    #[error(transparent)]
    FrameGpuBindings(#[from] FrameGpuBindingsError),
    /// Embedded raster `@group(1)` bind resources could not be created.
    #[error(transparent)]
    EmbeddedMaterialBind(#[from] EmbeddedMaterialBindError),
}

/// Device, queue, and settings passed to [`RenderBackend::attach`] (shared-memory flush is passed separately for borrow reasons).
pub struct RenderBackendAttachDesc {
    /// Logical device for uploads and graph encoding.
    pub device: Arc<wgpu::Device>,
    /// Queue used for submits and GPU writes.
    pub queue: Arc<wgpu::Queue>,
    /// Shared GPU queue access gate cloned from [`crate::gpu::GpuContext`]; acquired by
    /// upload, submit, and OpenXR queue-access paths. See [`crate::gpu::GpuQueueAccessGate`].
    pub gpu_queue_access_gate: crate::gpu::GpuQueueAccessGate,
    /// Capabilities for buffer sizing and MSAA.
    pub gpu_limits: Arc<GpuLimits>,
    /// Swapchain / main surface format for HUD and pipelines.
    pub surface_format: wgpu::TextureFormat,
    /// Live renderer settings (HUD, VR budgets, etc.).
    pub renderer_settings: RendererSettingsHandle,
    /// Path for persisting HUD/config from the debug overlay.
    pub config_save_path: PathBuf,
    /// When `true`, the ImGui config window must not write `config.toml` (startup extract failed).
    pub suppress_renderer_config_disk_writes: bool,
}

fn scene_color_usage_supported(format: wgpu::TextureFormat, limits: &GpuLimits) -> bool {
    limits.texture_usage_supported(
        format,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
    )
}

fn scene_color_format_supports_signed_rgb(format: wgpu::TextureFormat) -> bool {
    matches!(
        format,
        wgpu::TextureFormat::Rgba16Float | wgpu::TextureFormat::Rgba32Float
    )
}

fn effective_scene_color_format(
    requested: wgpu::TextureFormat,
    limits: &GpuLimits,
    signed_rgb_required: bool,
) -> wgpu::TextureFormat {
    if signed_rgb_required && !scene_color_format_supports_signed_rgb(requested) {
        let signed_default = SceneColorFormat::Rgba16Float.wgpu_format();
        if scene_color_usage_supported(signed_default, limits) {
            return signed_default;
        }
    }
    if scene_color_usage_supported(requested, limits) {
        return requested;
    }
    let default = SceneColorFormat::default().wgpu_format();
    if scene_color_usage_supported(default, limits) {
        return default;
    }
    wgpu::TextureFormat::Rgba8Unorm
}

/// Coordinates materials, asset uploads, per-frame GPU binds, occlusion, optional deform + ImGui HUD, and the render graph.
pub struct RenderBackend {
    /// Material property store, shader routes, pipeline registry, embedded `@group(1)` binds.
    pub(crate) materials: MaterialSystem,
    /// Mesh/texture upload queues, budgets, format tables, pools, and GPU device/queue for uploads.
    pub(crate) asset_transfers: AssetTransferQueue,
    /// Per-frame bind groups, mesh deformation services, skin cache, and MSAA depth resolve resources.
    frame_services: BackendFrameServices,
    /// CPU draw-preparation caches and material-batch caches.
    draw_preparation: BackendDrawPreparation,
    /// Backend-owned world-mesh forward frame planning caches.
    world_mesh_frame_planner: super::BackendWorldMeshFramePlanner,
    /// Dear ImGui overlay and diagnostics snapshot state.
    diagnostics: BackendDiagnostics,
    /// Nonblocking reflection-probe projection, bake, cache, and selection services.
    reflection_probes: ReflectionProbeServices,
    /// Render-graph cache, transient pool, history registry, and view-scoped graph resource ownership.
    graph_state: RenderGraphState,
    /// Hierarchical depth pyramid, CPU readback, and temporal cull state for occlusion culling.
    pub(crate) occlusion: OcclusionSystem,
    /// Swapchain or primary output color format used for frame-graph cache identity.
    surface_format: Option<wgpu::TextureFormat>,
    /// Live settings for per-frame graph parameters (scene HDR format, etc.); set in [`Self::attach`].
    renderer_settings: Option<RendererSettingsHandle>,
}

impl Default for RenderBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderBackend {
    /// Empty pools and material store; no GPU until [`Self::attach`].
    pub fn new() -> Self {
        Self {
            materials: MaterialSystem::new(),
            asset_transfers: AssetTransferQueue::new(),
            frame_services: BackendFrameServices::new(),
            draw_preparation: BackendDrawPreparation::new(),
            world_mesh_frame_planner: super::BackendWorldMeshFramePlanner::new(),
            diagnostics: BackendDiagnostics::new(),
            reflection_probes: ReflectionProbeServices::new(),
            graph_state: RenderGraphState::new(),
            occlusion: OcclusionSystem::new(),
            surface_format: None,
            renderer_settings: None,
        }
    }

    /// Requested HDR scene-color [`wgpu::TextureFormat`] from [`crate::config::RenderingSettings`].
    ///
    /// Falls back to [`SceneColorFormat::default`] when settings are unavailable (pre-attach).
    fn requested_scene_color_format_wgpu(&self) -> wgpu::TextureFormat {
        self.renderer_settings
            .as_ref()
            .and_then(|h| h.read().ok())
            .map_or_else(
                || SceneColorFormat::default().wgpu_format(),
                |s| s.rendering.scene_color_format.wgpu_format(),
            )
    }

    /// Effective HDR scene-color [`wgpu::TextureFormat`] supported by the active device.
    pub(crate) fn scene_color_format_wgpu(&self) -> wgpu::TextureFormat {
        let signed_rgb_required = self
            .frame_services
            .frame_resources
            .signed_scene_color_required();
        let requested = match self.requested_scene_color_format_wgpu() {
            format if signed_rgb_required && !scene_color_format_supports_signed_rgb(format) => {
                SceneColorFormat::Rgba16Float.wgpu_format()
            }
            format => format,
        };
        self.gpu_limits().map_or(requested, |limits| {
            effective_scene_color_format(requested, limits, signed_rgb_required)
        })
    }

    /// Returns true when negative lights force signed scene-color HDR for the current frame.
    pub(crate) fn signed_scene_color_active(&self) -> bool {
        self.frame_services
            .frame_resources
            .signed_scene_color_required()
            && scene_color_format_supports_signed_rgb(self.scene_color_format_wgpu())
    }

    /// Snapshot of the live GTAO settings for the current frame.
    ///
    /// Seeded into each view's blackboard as [`crate::passes::post_processing::settings_slot::GtaoSettingsSlot`]
    /// so the shader UBO reflects slider changes without rebuilding the compiled render graph
    /// (the chain signature only tracks enable booleans, so parameter edits wouldn't otherwise
    /// reach the pass).
    pub(crate) fn live_gtao_settings(&self) -> crate::config::GtaoSettings {
        self.renderer_settings
            .as_ref()
            .and_then(|h| h.read().ok())
            .map(|s| s.post_processing.gtao)
            .unwrap_or_default()
    }

    /// Snapshot of the live bloom settings for the current frame.
    ///
    /// Seeded into each view's blackboard as [`crate::passes::post_processing::settings_slot::BloomSettingsSlot`]
    /// so the first downsample's params UBO and the upsample blend constants reflect slider
    /// changes without rebuilding the compiled render graph. The effective `max_mip_dimension`
    /// is the one exception -- it drives mip-chain texture sizes, so it lives on the chain
    /// signature and triggers a rebuild instead.
    pub(crate) fn live_bloom_settings(&self) -> crate::config::BloomSettings {
        self.renderer_settings
            .as_ref()
            .and_then(|h| h.read().ok())
            .map(|s| s.post_processing.bloom)
            .unwrap_or_default()
    }

    /// Snapshot of the live auto-exposure settings for the current frame.
    ///
    /// Seeded into each view's blackboard as
    /// [`crate::passes::post_processing::settings_slot::AutoExposureSettingsSlot`] so histogram
    /// settings and adaptation speed edits take effect without rebuilding the compiled graph.
    pub(crate) fn live_auto_exposure_settings(&self) -> crate::config::AutoExposureSettings {
        self.renderer_settings
            .as_ref()
            .and_then(|h| h.read().ok())
            .map(|s| s.post_processing.auto_exposure)
            .unwrap_or_default()
    }

    /// Count of host Texture2D asset ids that have received a [`crate::shared::SetTexture2DFormat`] (CPU-side table).
    pub fn texture_format_registration_count(&self) -> usize {
        self.asset_transfers.texture_format_registration_count()
    }

    /// Count of GPU-resident textures with `mip_levels_resident > 0` (at least mip0 uploaded).
    pub fn texture_mip0_ready_count(&self) -> usize {
        self.asset_transfers
            .texture_pool()
            .iter()
            .filter(|t| t.mip_levels_resident > 0)
            .count()
    }

    /// Resets per-tick light prep flags, mesh deform coalescing, and advances the skin cache frame counter.
    ///
    /// Call once per winit tick before IPC and frame work (see [`crate::runtime::RendererRuntime::tick_frame_wall_clock_begin`]).
    pub fn reset_light_prep_for_tick(&mut self) {
        self.frame_services.reset_for_tick();
    }

    /// GPU limits snapshot after [`Self::attach`], if attach succeeded.
    pub fn gpu_limits(&self) -> Option<&Arc<GpuLimits>> {
        self.asset_transfers.gpu_limits()
    }

    /// Mutable frame resources for runtime draw-preparation handoffs.
    pub(crate) fn frame_resources_mut(&mut self) -> &mut FrameResourceManager {
        &mut self.frame_services.frame_resources
    }

    /// Drains latest video clock-error samples produced by asset integration.
    pub(crate) fn take_pending_video_clock_errors(
        &mut self,
    ) -> Vec<crate::shared::VideoTextureClockErrorState> {
        self.asset_transfers.take_pending_video_clock_errors()
    }

    /// Mesh pool and VRAM accounting (draw prep, debugging).
    pub fn mesh_pool(&self) -> &MeshPool {
        self.asset_transfers.mesh_pool()
    }

    /// Resident Texture2D table (bind-group prep).
    pub fn texture_pool(&self) -> &TexturePool {
        self.asset_transfers.texture_pool()
    }

    /// Host render texture targets (secondary cameras, material sampling).
    pub fn render_texture_pool(&self) -> &RenderTexturePool {
        self.asset_transfers.render_texture_pool()
    }

    /// Answers host SH2 task rows for the latest frame submit without blocking GPU readback.
    pub(crate) fn answer_reflection_probe_sh2_tasks(
        &mut self,
        shm: &mut crate::ipc::SharedMemoryAccessor,
        scene: &crate::scene::SceneCoordinator,
        data: &crate::shared::FrameSubmitData,
    ) {
        self.reflection_probes.answer_sh2_frame_submit_tasks(
            shm,
            scene,
            &self.materials,
            &self.asset_transfers,
            data,
        );
    }

    /// Advances nonblocking SH2 GPU jobs and schedules queued projection work.
    pub(crate) fn maintain_reflection_probe_sh2_jobs(&mut self, gpu: &mut crate::gpu::GpuContext) {
        self.reflection_probes
            .maintain_sh2_jobs(gpu, &self.asset_transfers);
    }

    /// Advances reflection-probe specular IBL jobs and syncs frame-global probe bindings.
    pub(crate) fn maintain_reflection_probe_specular_jobs(
        &mut self,
        gpu: &mut crate::gpu::GpuContext,
        scene: &crate::scene::SceneCoordinator,
        render_context: crate::shared::RenderingContext,
    ) {
        let resources = self.reflection_probes.maintain_specular_jobs(
            gpu,
            scene,
            &self.materials,
            &self.asset_transfers,
            render_context,
        );
        let _ = self
            .frame_services
            .frame_resources
            .sync_reflection_probe_specular_resources(gpu.device(), resources);
    }

    /// Material property store (host uniforms, textures, shader asset bindings).
    pub fn material_property_store(&self) -> &MaterialPropertyStore {
        self.materials.material_property_store()
    }

    /// Property name interning for material batches.
    pub fn property_id_registry(&self) -> &crate::materials::host_data::PropertyIdRegistry {
        self.materials.property_id_registry()
    }

    /// Registered material families and pipeline cache (after GPU attach).
    pub fn material_registry(&self) -> Option<&crate::materials::MaterialRegistry> {
        self.materials.material_registry()
    }

    /// Number of schedules passes in the compiled frame graph, or `0` if none.
    pub fn frame_graph_pass_count(&self) -> usize {
        self.graph_state.frame_graph_cache.pass_count()
    }

    /// Compile-time topological wave count for the cached frame graph, or `0` if none has been built yet.
    pub fn frame_graph_topo_levels(&self) -> usize {
        self.graph_state.frame_graph_cache.topo_levels()
    }

    /// Call after [`crate::gpu::GpuContext`] is created so mesh/texture uploads can use the GPU.
    ///
    /// Wires device/queue into uploads, allocates frame binds and materials, and builds the default graph.
    /// `shm` flushes pending mesh/texture payloads that require shared-memory reads; omit when none is
    /// available yet (uploads stay queued).
    ///
    /// On error, CPU-side asset queues may already be partially configured; GPU draws must not run until
    /// a successful attach.
    pub fn attach(
        &mut self,
        desc: RenderBackendAttachDesc,
        shm: Option<&mut crate::ipc::SharedMemoryAccessor>,
    ) -> Result<(), RenderBackendAttachError> {
        let RenderBackendAttachDesc {
            device,
            queue,
            gpu_queue_access_gate,
            gpu_limits,
            surface_format,
            renderer_settings,
            config_save_path,
            suppress_renderer_config_disk_writes,
        } = desc;
        self.renderer_settings = Some(renderer_settings.clone());
        self.surface_format = Some(surface_format);
        self.asset_transfers.attach_gpu_runtime(
            device.clone(),
            queue.clone(),
            gpu_queue_access_gate,
            Arc::clone(&gpu_limits),
        );
        self.frame_services
            .attach(device.as_ref(), queue.as_ref(), Arc::clone(&gpu_limits))?;
        self.diagnostics.attach(
            device.as_ref(),
            queue.as_ref(),
            surface_format,
            renderer_settings,
            config_save_path,
            suppress_renderer_config_disk_writes,
        );
        self.materials
            .try_attach_gpu(device.clone(), &queue, Arc::clone(&gpu_limits))?;
        asset_uploads::attach_flush_pending_asset_uploads(&mut self.asset_transfers, &device, shm);

        let (post_processing_settings, msaa_sample_count) = self
            .renderer_settings
            .as_ref()
            .and_then(|h| {
                h.read()
                    .ok()
                    .map(|g| (g.post_processing.clone(), g.rendering.msaa.as_count() as u8))
            })
            .unwrap_or_else(|| (PostProcessingSettings::default(), 1));
        let graph_post_processing =
            self.effective_post_processing_settings_for_graph(&post_processing_settings);
        let shape = self.frame_graph_shape_for(&graph_post_processing, msaa_sample_count, false);
        self.sync_frame_graph_cache(&graph_post_processing, shape);
        logger::info!(
            "backend attached: surface_format={:?} scene_color_format={:?} msaa_sample_count={} mesh_preprocess={} msaa_depth_resolve={} frame_graph_passes={} frame_graph_topo_levels={}",
            surface_format,
            self.scene_color_format_wgpu(),
            msaa_sample_count,
            self.frame_services.mesh_preprocess_enabled(),
            self.frame_services.msaa_depth_resolve_enabled(),
            self.frame_graph_pass_count(),
            self.frame_graph_topo_levels(),
        );
        Ok(())
    }

    /// Updates whether main HUD diagnostics run (mirrors [`crate::config::DebugSettings::debug_hud_enabled`]).
    pub fn set_debug_hud_main_enabled(&mut self, enabled: bool) {
        self.diagnostics.set_main_enabled(enabled);
    }

    /// Updates whether texture HUD diagnostics run.
    pub(crate) fn set_debug_hud_textures_enabled(&mut self, enabled: bool) {
        self.diagnostics.set_textures_enabled(enabled);
    }

    /// Clears the current-view Texture2D set before collecting this frame's submitted draws.
    pub(crate) fn clear_debug_hud_current_view_texture_2d_asset_ids(&mut self) {
        self.diagnostics.clear_current_view_texture_2d_asset_ids();
    }

    /// Texture2D ids used by submitted world draws for the current view.
    pub(crate) fn debug_hud_current_view_texture_2d_asset_ids(
        &self,
    ) -> &std::collections::BTreeSet<i32> {
        self.diagnostics.current_view_texture_2d_asset_ids()
    }

    /// Updates pointer state for the ImGui overlay (called once per render_views).
    pub fn set_debug_hud_input(&mut self, input: DebugHudInput) {
        self.diagnostics.set_input(input);
    }

    /// Updates the wall-clock roundtrip (ms) for the HUD's FPS / Frame readout.
    pub fn set_debug_hud_wall_frame_time_ms(&mut self, frame_time_ms: f64) {
        self.diagnostics.set_wall_frame_time_ms(frame_time_ms);
    }

    /// Last inter-frame time in milliseconds supplied by the app for HUD FPS.
    pub(crate) fn debug_frame_time_ms(&self) -> f64 {
        self.diagnostics.frame_time_ms()
    }

    /// [`imgui::Io::want_capture_mouse`] from the last successful HUD encode (used to filter host IPC on the next tick).
    pub(crate) fn debug_hud_last_want_capture_mouse(&self) -> bool {
        self.diagnostics.last_want_capture_mouse()
    }

    /// [`imgui::Io::want_capture_keyboard`] from the last successful HUD encode (used to filter host IPC on the next tick).
    pub(crate) fn debug_hud_last_want_capture_keyboard(&self) -> bool {
        self.diagnostics.last_want_capture_keyboard()
    }

    /// Whether the HUD will draw visible content this frame.
    pub(crate) fn debug_hud_has_visible_content(&self) -> bool {
        self.diagnostics.has_visible_content()
    }

    /// Clears cached input-capture state when HUD encoding is skipped.
    pub(crate) fn clear_debug_hud_input_capture(&mut self) {
        self.diagnostics.clear_input_capture();
    }

    /// Stores [`crate::diagnostics::RendererInfoSnapshot`] for the next HUD frame.
    pub(crate) fn set_debug_hud_snapshot(
        &mut self,
        snapshot: crate::diagnostics::RendererInfoSnapshot,
    ) {
        self.diagnostics.set_snapshot(snapshot);
    }

    pub(crate) fn set_debug_hud_frame_diagnostics(
        &mut self,
        snapshot: crate::diagnostics::FrameDiagnosticsSnapshot,
    ) {
        self.diagnostics.set_frame_diagnostics(snapshot);
    }

    pub(crate) fn set_debug_hud_frame_timing(
        &mut self,
        snapshot: crate::diagnostics::FrameTimingHudSnapshot,
    ) {
        self.diagnostics.set_frame_timing(snapshot);
    }

    /// Pushes the latest flattened GPU pass timings into the debug HUD's **GPU passes** tab.
    pub(crate) fn set_debug_hud_gpu_pass_timings(
        &mut self,
        timings: Vec<crate::profiling::GpuPassEntry>,
    ) {
        self.diagnostics.set_gpu_pass_timings(timings);
    }

    /// Clears Stats / Shader routes payloads only (not frame timing or scene transforms).
    pub(crate) fn clear_debug_hud_stats_snapshots(&mut self) {
        self.diagnostics.clear_stats_snapshots();
    }

    /// Clears the **Scene transforms** HUD payload.
    pub(crate) fn clear_debug_hud_scene_transforms_snapshot(&mut self) {
        self.diagnostics.clear_scene_transforms_snapshot();
    }

    pub(crate) fn last_world_mesh_draw_stats(&self) -> WorldMeshDrawStats {
        self.diagnostics.last_world_mesh_draw_stats()
    }

    pub(crate) fn last_world_mesh_draw_state_rows(&self) -> Vec<WorldMeshDrawStateRow> {
        self.diagnostics.last_world_mesh_draw_state_rows()
    }

    /// Plain-data backend snapshot consumed by the diagnostics HUD.
    ///
    /// Returns a [`crate::diagnostics::BackendDiagSnapshot`] capturing the fields
    /// `FrameDiagnosticsSnapshot::capture` and `RendererInfoSnapshot::capture` need, so the
    /// diagnostics layer never borrows `&RenderBackend` directly.
    pub fn snapshot_for_diagnostics(&self) -> crate::diagnostics::BackendDiagSnapshot {
        let store = self.material_property_store();
        let shader_routes = self
            .material_registry()
            .map(|reg| {
                reg.shader_routes_for_hud()
                    .into_iter()
                    .map(
                        |(id, pipeline, name)| crate::diagnostics::ShaderRouteSnapshot {
                            shader_asset_id: id,
                            pipeline,
                            shader_asset_name: name,
                        },
                    )
                    .collect()
            })
            .unwrap_or_default();
        crate::diagnostics::BackendDiagSnapshot {
            texture_format_registration_count: self.texture_format_registration_count(),
            texture_mip0_ready_count: self.texture_mip0_ready_count(),
            texture_pool_resident_count: self.texture_pool().len(),
            render_texture_pool_len: self.render_texture_pool().len(),
            mesh_pool_entry_count: self.mesh_pool().len(),
            shader_routes,
            last_world_mesh_draw_stats: self.last_world_mesh_draw_stats(),
            last_world_mesh_draw_state_rows: self.last_world_mesh_draw_state_rows(),
            material_property_slots: store.material_property_slot_count(),
            property_block_slots: store.property_block_slot_count(),
            material_shader_bindings: store.material_shader_binding_count(),
            frame_graph_pass_count: self.frame_graph_pass_count(),
            frame_graph_topo_levels: self.frame_graph_topo_levels(),
            gpu_light_count: self.frame_services.frame_resources.frame_lights().len(),
            signed_scene_color_active: self.signed_scene_color_active(),
        }
    }

    /// Updates the **Scene transforms** Dear ImGui window payload for the next composite pass.
    pub(crate) fn set_debug_hud_scene_transforms_snapshot(
        &mut self,
        snapshot: SceneTransformsSnapshot,
    ) {
        self.diagnostics.set_scene_transforms_snapshot(snapshot);
    }

    /// Updates the **Textures** Dear ImGui window payload for the next composite pass.
    pub(crate) fn set_debug_hud_texture_debug_snapshot(
        &mut self,
        snapshot: crate::diagnostics::TextureDebugSnapshot,
    ) {
        self.diagnostics.set_texture_debug_snapshot(snapshot);
    }

    /// Clears the **Textures** HUD payload.
    pub(crate) fn clear_debug_hud_texture_debug_snapshot(&mut self) {
        self.diagnostics.clear_texture_debug_snapshot();
    }

    /// Composites the debug HUD with `LoadOp::Load` onto the swapchain in `encoder`.
    pub(crate) fn encode_debug_hud_overlay(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        backbuffer: &wgpu::TextureView,
        extent: (u32, u32),
    ) -> Result<(), DebugHudEncodeError> {
        self.diagnostics
            .encode_overlay(device, queue, encoder, backbuffer, extent)
    }

    /// Mutable render-graph transient resource pool.
    pub(crate) fn transient_pool_mut(&mut self) -> &mut TransientPool {
        self.graph_state.transient_pool_mut()
    }

    /// Synchronizes backend view-scoped resource ownership against the runtime's active view list.
    pub(crate) fn sync_active_views<I>(&mut self, active_views: I)
    where
        I: IntoIterator<Item = crate::camera::ViewId>,
    {
        let retired = self.graph_state.sync_active_views(active_views);
        if retired.is_empty() {
            return;
        }
        logger::debug!(
            "retiring {} inactive view-scoped resource sets",
            retired.len()
        );
        self.world_mesh_frame_planner
            .release_view_resources(&retired);
        for view_id in retired {
            self.frame_services.frame_resources.retire_view(view_id);
            self.graph_state.history_registry_mut().retire_view(view_id);
            let _ = self.occlusion.retire_view(view_id);
        }
    }

    /// Releases resources for one-shot views that were never part of the active-view registry.
    pub(crate) fn retire_one_shot_views(&mut self, retired: &[crate::camera::ViewId]) {
        if retired.is_empty() {
            return;
        }
        logger::debug!(
            "retiring {} one-shot view-scoped resource sets",
            retired.len()
        );
        self.graph_state.release_view_resources(retired);
        self.world_mesh_frame_planner
            .release_view_resources(retired);
        for &view_id in retired {
            self.frame_services.frame_resources.retire_view(view_id);
            self.graph_state.history_registry_mut().retire_view(view_id);
            let _ = self.occlusion.retire_view(view_id);
        }
    }

    /// Builds the narrow graph-execution access packet from disjoint backend owners.
    pub(crate) fn graph_access(&mut self) -> BackendGraphAccess<'_> {
        let scene_color_format = self.scene_color_format_wgpu();
        let gpu_limits = self.gpu_limits().cloned();
        let msaa_depth_resolve = self.frame_services.msaa_depth_resolve();
        let live_gtao_settings = self.live_gtao_settings();
        let live_bloom_settings = self.live_bloom_settings();
        let live_auto_exposure_settings = self.live_auto_exposure_settings();
        let wall_frame_time_ms = self.debug_frame_time_ms();
        let (transient_pool, history_registry, upload_arena) =
            self.graph_state.execution_resources_mut();
        let (frame_resources, mesh_preprocess, mesh_deform_scratch, skin_cache) =
            self.frame_services.graph_access_slices();
        BackendGraphAccess {
            occlusion: &mut self.occlusion,
            frame_resources,
            materials: &self.materials,
            asset_transfers: &mut self.asset_transfers,
            mesh_preprocess,
            mesh_deform_scratch,
            skin_cache,
            world_mesh_frame_planner: &self.world_mesh_frame_planner,
            transient_pool,
            history_registry,
            upload_arena,
            debug_hud: self.diagnostics.bundle_mut(),
            scene_color_format,
            gpu_limits,
            msaa_depth_resolve,
            live_gtao_settings,
            live_bloom_settings,
            live_auto_exposure_settings,
            wall_frame_time_ms,
        }
    }
}

#[cfg(test)]
mod post_processing_rebuild_tests {
    use std::sync::{Arc, RwLock};

    use super::*;
    use crate::config::{GtaoSettings, RendererSettings, TonemapMode, TonemapSettings};
    use crate::render_graph::{GraphCacheKey, post_process_chain::PostProcessChainSignature};
    use hashbrown::HashMap;

    fn settings_handle(post: PostProcessingSettings) -> RendererSettingsHandle {
        Arc::new(RwLock::new(RendererSettings {
            post_processing: post,
            ..Default::default()
        }))
    }

    /// Returns the current cached graph key.
    fn cached_graph_key(backend: &RenderBackend) -> GraphCacheKey {
        backend
            .graph_state
            .frame_graph_cache
            .last_key()
            .expect("graph key should exist after sync")
    }

    fn limits_with_format_usage(
        format: wgpu::TextureFormat,
        allowed_usages: wgpu::TextureUsages,
    ) -> GpuLimits {
        let mut format_features = HashMap::new();
        format_features.insert(
            format,
            wgpu::TextureFormatFeatures {
                allowed_usages,
                flags: wgpu::TextureFormatFeatureFlags::empty(),
            },
        );
        GpuLimits::synthetic_for_tests(
            wgpu::Limits {
                max_texture_dimension_2d: 4096,
                max_storage_buffer_binding_size: 256 * 1024,
                ..Default::default()
            },
            wgpu::Features::empty(),
            format_features,
        )
    }

    /// First sync builds the graph and stores the live signature.
    #[test]
    fn first_sync_builds_graph_and_records_signature() {
        let mut backend = RenderBackend::new();
        let handle = settings_handle(PostProcessingSettings {
            enabled: true,
            tonemap: TonemapSettings {
                mode: TonemapMode::AcesFitted,
            },
            ..Default::default()
        });
        backend.renderer_settings = Some(handle);
        backend.ensure_frame_graph_in_sync(false);
        assert!(
            backend.frame_graph_pass_count() > 0,
            "graph should be built"
        );
        assert_eq!(
            cached_graph_key(&backend).post_processing,
            PostProcessChainSignature {
                aces_tonemap: true,
                agx_tonemap: false,
                auto_exposure: true,
                bloom: true,
                bloom_max_mip_dimension: 512,
                gtao: true,
                gtao_denoise_passes: GtaoSettings::default().denoise_passes.min(3),
            }
        );
    }

    /// Toggling the master enable flips the signature and rebuilds the graph with an extra pass.
    #[test]
    fn signature_change_triggers_rebuild() {
        let mut backend = RenderBackend::new();
        let handle = settings_handle(PostProcessingSettings {
            enabled: false,
            ..Default::default()
        });
        backend.renderer_settings = Some(Arc::clone(&handle));
        backend.ensure_frame_graph_in_sync(false);
        let initial_passes = backend.frame_graph_pass_count();
        let initial_signature = cached_graph_key(&backend).post_processing;

        if let Ok(mut g) = handle.write() {
            g.post_processing.enabled = true;
            g.post_processing.tonemap.mode = TonemapMode::AcesFitted;
        }
        backend.ensure_frame_graph_in_sync(false);

        assert_ne!(
            cached_graph_key(&backend).post_processing,
            initial_signature,
            "signature must update after rebuild"
        );
        assert!(
            backend.frame_graph_pass_count() > initial_passes,
            "enabling ACES should add a graph pass"
        );
    }

    /// Repeat sync without HUD edits is a no-op (no rebuild, signature and pass count unchanged).
    #[test]
    fn unchanged_signature_does_not_rebuild() {
        let mut backend = RenderBackend::new();
        let handle = settings_handle(PostProcessingSettings {
            enabled: true,
            tonemap: TonemapSettings {
                mode: TonemapMode::AcesFitted,
            },
            ..Default::default()
        });
        backend.renderer_settings = Some(handle);
        backend.ensure_frame_graph_in_sync(false);
        let signature = cached_graph_key(&backend).post_processing;
        let pass_count = backend.frame_graph_pass_count();

        backend.ensure_frame_graph_in_sync(false);
        assert_eq!(cached_graph_key(&backend).post_processing, signature);
        assert_eq!(backend.frame_graph_pass_count(), pass_count);
    }

    /// Switching between mono and stereo multiview should flip the graph key in one place so the
    /// runtime does not rely on implicit backend assumptions when VR starts or stops.
    #[test]
    fn multiview_change_updates_graph_key() {
        let mut backend = RenderBackend::new();
        backend.renderer_settings = Some(settings_handle(PostProcessingSettings::default()));

        backend.ensure_frame_graph_in_sync(false);
        let mono_key = cached_graph_key(&backend);
        backend.ensure_frame_graph_in_sync(true);
        let stereo_key = cached_graph_key(&backend);

        assert!(!mono_key.multiview_stereo);
        assert!(stereo_key.multiview_stereo);
        assert_ne!(mono_key, stereo_key);
    }

    #[test]
    fn scene_color_format_falls_back_when_requested_format_is_not_renderable() {
        let limits = limits_with_format_usage(
            wgpu::TextureFormat::Rg11b10Ufloat,
            wgpu::TextureUsages::TEXTURE_BINDING,
        );

        assert_eq!(
            effective_scene_color_format(wgpu::TextureFormat::Rg11b10Ufloat, &limits, false),
            wgpu::TextureFormat::Rgba16Float
        );
    }

    #[test]
    fn scene_color_format_promotes_unsigned_when_signed_rgb_is_required() {
        let limits = limits_with_format_usage(
            wgpu::TextureFormat::Rg11b10Ufloat,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );

        assert_eq!(
            effective_scene_color_format(wgpu::TextureFormat::Rg11b10Ufloat, &limits, true),
            wgpu::TextureFormat::Rgba16Float
        );
        assert_eq!(
            effective_scene_color_format(wgpu::TextureFormat::Rg11b10Ufloat, &limits, false),
            wgpu::TextureFormat::Rg11b10Ufloat
        );
    }
}
