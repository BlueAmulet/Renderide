//! Narrow backend access packet used by render-graph execution.

use std::sync::Arc;

use crate::backend::AssetTransferQueue;
use crate::diagnostics::{DebugHudEncodeError, PerViewHudConfig, PerViewHudOutputs};
use crate::gpu::{GpuLimits, MsaaDepthResolveResources};
use crate::materials::MaterialSystem;
use crate::mesh_deform::{GpuSkinCache, MeshDeformScratch, MeshPreprocessPipelines};
use crate::render_graph::TransientPool;
use crate::render_graph::blackboard::Blackboard;
use crate::render_graph::compiled::FrameView;
use crate::render_graph::frame_params::{GraphPassFrame, PerViewFramePlan};
use crate::render_graph::frame_upload_batch::GraphUploadSink;
use crate::render_graph::upload_arena::PersistentUploadArena;
use crate::world_mesh::WorldMeshDrawPlanSlot;

use super::super::debug_hud_bundle::DebugHudBundle;
use super::super::{FrameResourceManager, HistoryRegistry};
use crate::occlusion::OcclusionSystem;

/// Disjoint backend slices assembled into [`crate::render_graph::GraphPassFrame`].
type GraphFrameParamsSplit<'a> = (
    &'a OcclusionSystem,
    &'a FrameResourceManager,
    &'a MaterialSystem,
    &'a AssetTransferQueue,
    Option<&'a MeshPreprocessPipelines>,
    Option<&'a mut MeshDeformScratch>,
    Option<&'a mut GpuSkinCache>,
    Option<Arc<GpuLimits>>,
    Option<Arc<MsaaDepthResolveResources>>,
    PerViewHudConfig,
);

/// Narrow backend packet used by the render graph executor.
///
/// This is intentionally a bundle of disjoint backend owners instead of `&mut RenderBackend`:
/// graph execution can mutate transient/history/frame/HUD state without gaining access to IPC,
/// facade-only helpers, or unrelated backend orchestration fields.
pub(crate) struct BackendGraphAccess<'a> {
    /// Hi-Z and temporal occlusion state.
    pub(crate) occlusion: &'a mut OcclusionSystem,
    /// Frame-global and per-view bind resources.
    pub(crate) frame_resources: &'a mut FrameResourceManager,
    /// Material registry, routes, and property data.
    pub(crate) materials: &'a MaterialSystem,
    /// Asset upload queues and resident GPU pools.
    pub(crate) asset_transfers: &'a mut AssetTransferQueue,
    /// Mesh preprocess pipelines for frame-global deform work.
    pub(crate) mesh_preprocess: Option<&'a MeshPreprocessPipelines>,
    /// Mesh deform scratch buffers for frame-global deform work.
    pub(crate) mesh_deform_scratch: Option<&'a mut MeshDeformScratch>,
    /// Skin cache mutably borrowed by frame-global deform and shared by per-view draws afterwards.
    pub(crate) skin_cache: Option<&'a mut GpuSkinCache>,
    /// Backend-owned world-mesh forward frame planner.
    pub(crate) world_mesh_frame_planner: &'a crate::backend::BackendWorldMeshFramePlanner,
    /// Render-graph transient pool retained across frames.
    pub(crate) transient_pool: &'a mut TransientPool,
    /// Persistent ping-pong history registry.
    pub(crate) history_registry: &'a mut HistoryRegistry,
    /// Persistent upload staging arena used by graph buffer upload drains.
    pub(crate) upload_arena: &'a mut PersistentUploadArena,
    /// Debug HUD state and encoder.
    pub(crate) debug_hud: &'a mut DebugHudBundle,
    /// Scene-color format snapshot selected before graph execution borrows backend fields.
    pub(super) scene_color_format: wgpu::TextureFormat,
    /// GPU limits snapshot selected before graph execution borrows backend fields.
    pub(super) gpu_limits: Option<Arc<GpuLimits>>,
    /// MSAA depth-resolve resources selected before graph execution borrows backend fields.
    pub(super) msaa_depth_resolve: Option<Arc<MsaaDepthResolveResources>>,
    /// GTAO settings snapshot selected before graph execution borrows backend fields.
    pub(super) live_gtao_settings: crate::config::GtaoSettings,
    /// Bloom settings snapshot selected before graph execution borrows backend fields.
    pub(super) live_bloom_settings: crate::config::BloomSettings,
    /// Auto-exposure settings snapshot selected before graph execution borrows backend fields.
    pub(super) live_auto_exposure_settings: crate::config::AutoExposureSettings,
    /// Wall-frame delta snapshot in milliseconds.
    pub(super) wall_frame_time_ms: f64,
}

impl<'a> BackendGraphAccess<'a> {
    /// Mutable transient pool for graph resource resolution.
    pub(crate) fn transient_pool_mut(&mut self) -> &mut TransientPool {
        self.transient_pool
    }

    /// Shared history registry for import resolution.
    pub(crate) fn history_registry(&self) -> &HistoryRegistry {
        self.history_registry
    }

    /// Mutable history registry for frame advance and view-scoped registrations.
    pub(crate) fn history_registry_mut(&mut self) -> &mut HistoryRegistry {
        self.history_registry
    }

    /// Persistent upload staging arena.
    pub(crate) fn upload_arena_mut(&mut self) -> &mut PersistentUploadArena {
        self.upload_arena
    }

    /// Scene-color format snapshot selected for this graph frame.
    pub(crate) fn scene_color_format_wgpu(&self) -> wgpu::TextureFormat {
        self.scene_color_format
    }

    /// GPU limits snapshot after attach.
    pub(crate) fn gpu_limits(&self) -> Option<&Arc<GpuLimits>> {
        self.gpu_limits.as_ref()
    }

    /// Optional MSAA depth resolve resources.
    pub(crate) fn msaa_depth_resolve(&self) -> Option<Arc<MsaaDepthResolveResources>> {
        self.msaa_depth_resolve.clone()
    }

    /// Live GTAO settings snapshot for this graph frame.
    pub(crate) fn live_gtao_settings(&self) -> crate::config::GtaoSettings {
        self.live_gtao_settings
    }

    /// Live bloom settings snapshot for this graph frame.
    pub(crate) fn live_bloom_settings(&self) -> crate::config::BloomSettings {
        self.live_bloom_settings
    }

    /// Live auto-exposure settings snapshot for this graph frame.
    pub(crate) fn live_auto_exposure_settings(&self) -> crate::config::AutoExposureSettings {
        self.live_auto_exposure_settings
    }

    /// Wall-frame delta snapshot for this graph frame, in seconds.
    pub(crate) fn wall_frame_delta_seconds(&self) -> f32 {
        (self.wall_frame_time_ms / 1000.0).clamp(0.0, 1.0) as f32
    }

    /// Shared frame resources.
    pub(crate) fn frame_resources(&self) -> &FrameResourceManager {
        self.frame_resources
    }

    /// Shared material system.
    pub(crate) fn materials(&self) -> &MaterialSystem {
        self.materials
    }

    /// Shared asset-transfer state.
    pub(crate) fn asset_transfers(&self) -> &AssetTransferQueue {
        self.asset_transfers
    }

    /// Shared occlusion state.
    pub(crate) fn occlusion(&self) -> &OcclusionSystem {
        self.occlusion
    }

    /// Mutable occlusion state.
    pub(crate) fn occlusion_mut(&mut self) -> &mut OcclusionSystem {
        self.occlusion
    }

    /// Optional mesh preprocess pipelines.
    pub(crate) fn mesh_preprocess(&self) -> Option<&MeshPreprocessPipelines> {
        self.mesh_preprocess
    }

    /// Optional read-only skin cache for per-view forward draws.
    pub(crate) fn skin_cache(&self) -> Option<&GpuSkinCache> {
        self.skin_cache.as_deref()
    }

    /// Warms backend-owned assets required by caller-seeded per-view blackboards.
    pub(crate) fn pre_warm_view_assets_from_blackboards(
        &mut self,
        device: &wgpu::Device,
        views: &[FrameView<'_>],
    ) {
        profiling::scope!("graph::pre_warm_view_assets");
        let mut mesh_ids_needing_uv1_stream = hashbrown::HashSet::new();
        let mut mesh_ids_needing_tangent_stream = hashbrown::HashSet::new();
        let mut mesh_ids_needing_uv2_stream = hashbrown::HashSet::new();
        let mut mesh_ids_needing_uv3_stream = hashbrown::HashSet::new();
        for view in views {
            let Some(draw_plan) = view.initial_blackboard.get::<WorldMeshDrawPlanSlot>() else {
                continue;
            };
            let Some(collection) = draw_plan.as_prefetched() else {
                continue;
            };
            for item in &collection.items {
                if item.mesh_asset_id < 0 {
                    continue;
                }
                if item.batch_key.embedded_needs_uv1 {
                    mesh_ids_needing_uv1_stream.insert(item.mesh_asset_id);
                }
                if item.batch_key.embedded_needs_tangent {
                    mesh_ids_needing_tangent_stream.insert(item.mesh_asset_id);
                }
                if item.batch_key.embedded_needs_uv2 {
                    mesh_ids_needing_uv2_stream.insert(item.mesh_asset_id);
                }
                if item.batch_key.embedded_needs_uv3 {
                    mesh_ids_needing_uv3_stream.insert(item.mesh_asset_id);
                }
            }
        }
        logger::trace!(
            "graph pre-warm view assets: views={} uv1_stream_meshes={} tangent_stream_meshes={} uv2_stream_meshes={} uv3_stream_meshes={}",
            views.len(),
            mesh_ids_needing_uv1_stream.len(),
            mesh_ids_needing_tangent_stream.len(),
            mesh_ids_needing_uv2_stream.len(),
            mesh_ids_needing_uv3_stream.len(),
        );
        let mesh_ids_needing_all_extended_streams: hashbrown::HashSet<i32> =
            mesh_ids_needing_tangent_stream
                .iter()
                .filter(|mesh_asset_id| {
                    mesh_ids_needing_uv1_stream.contains(*mesh_asset_id)
                        && mesh_ids_needing_uv2_stream.contains(*mesh_asset_id)
                        && mesh_ids_needing_uv3_stream.contains(*mesh_asset_id)
                })
                .copied()
                .collect();
        for &mesh_asset_id in &mesh_ids_needing_all_extended_streams {
            let _ = self
                .asset_transfers
                .mesh_pool_mut()
                .ensure_extended_vertex_streams(device, mesh_asset_id);
        }
        for mesh_asset_id in mesh_ids_needing_uv1_stream {
            if mesh_ids_needing_all_extended_streams.contains(&mesh_asset_id) {
                continue;
            }
            let _ = self
                .asset_transfers
                .mesh_pool_mut()
                .ensure_uv1_vertex_stream(device, mesh_asset_id);
        }
        for mesh_asset_id in mesh_ids_needing_tangent_stream {
            if mesh_ids_needing_all_extended_streams.contains(&mesh_asset_id) {
                continue;
            }
            let _ = self
                .asset_transfers
                .mesh_pool_mut()
                .ensure_tangent_vertex_stream(device, mesh_asset_id);
        }
        for mesh_asset_id in mesh_ids_needing_uv2_stream {
            if mesh_ids_needing_all_extended_streams.contains(&mesh_asset_id) {
                continue;
            }
            let _ = self
                .asset_transfers
                .mesh_pool_mut()
                .ensure_uv2_vertex_stream(device, mesh_asset_id);
        }
        for mesh_asset_id in mesh_ids_needing_uv3_stream {
            if mesh_ids_needing_all_extended_streams.contains(&mesh_asset_id) {
                continue;
            }
            let _ = self
                .asset_transfers
                .mesh_pool_mut()
                .ensure_uv3_vertex_stream(device, mesh_asset_id);
        }
    }

    /// Lets backend-specific systems consume and enrich one per-view blackboard before recording.
    pub(crate) fn prepare_view_blackboard(
        &self,
        device: &wgpu::Device,
        uploads: GraphUploadSink<'_>,
        gpu_limits: &GpuLimits,
        frame: &GraphPassFrame<'_>,
        frame_plan: &PerViewFramePlan,
        blackboard: &mut Blackboard,
    ) {
        crate::backend::prepare_world_mesh_view_blackboard(
            self.world_mesh_frame_planner,
            device,
            uploads,
            gpu_limits,
            frame,
            frame_plan,
            blackboard,
        );
    }

    /// Debug HUD flags consumed by per-view recording.
    pub(crate) fn per_view_hud_config(&self) -> PerViewHudConfig {
        PerViewHudConfig {
            main_enabled: self.debug_hud.main_enabled(),
            textures_enabled: self.debug_hud.textures_enabled(),
        }
    }

    /// Whether the HUD will draw visible content this frame.
    pub(crate) fn debug_hud_has_visible_content(&self) -> bool {
        self.debug_hud.has_visible_content()
    }

    /// Encodes the debug HUD overlay.
    pub(crate) fn encode_debug_hud_overlay(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        backbuffer: &wgpu::TextureView,
        extent: (u32, u32),
    ) -> Result<(), DebugHudEncodeError> {
        profiling::scope!("hud::encode");
        self.debug_hud
            .encode_overlay(device, queue, encoder, backbuffer, extent)
    }

    /// Clears cached input-capture state when HUD encoding is skipped.
    pub(crate) fn clear_debug_hud_input_capture(&mut self) {
        self.debug_hud.clear_input_capture();
    }

    /// Applies one deferred per-view HUD payload.
    pub(crate) fn apply_per_view_hud_outputs(&mut self, outputs: &PerViewHudOutputs) {
        self.debug_hud.apply_per_view_outputs(outputs);
    }

    /// Disjoint mutable borrows and attach-time snapshots for frame-global pass params.
    pub(crate) fn split_for_graph_frame_params(&mut self) -> GraphFrameParamsSplit<'_> {
        let per_view_hud_config = PerViewHudConfig {
            main_enabled: self.debug_hud.main_enabled(),
            textures_enabled: self.debug_hud.textures_enabled(),
        };
        (
            self.occlusion,
            self.frame_resources,
            self.materials,
            self.asset_transfers,
            self.mesh_preprocess,
            self.mesh_deform_scratch.as_deref_mut(),
            self.skin_cache.as_deref_mut(),
            self.gpu_limits.clone(),
            self.msaa_depth_resolve.clone(),
            per_view_hud_config,
        )
    }
}
