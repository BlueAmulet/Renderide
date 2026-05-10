//! Backend-owned frame extraction helpers and read-only draw-preparation views.

use hashbrown::{HashMap, HashSet};

use crate::backend::FrameLightViewDesc;
use crate::gpu_pools::MeshPool;
use crate::materials::ShaderPermutation;
use crate::materials::host_data::MaterialPropertyStore;
use crate::materials::{MaterialPipelinePropertyIds, MaterialRouter};
use crate::reflection_probes::specular::ReflectionProbeFrameSelection;
use crate::scene::{RenderSpaceId, SceneApplyReport, SceneCacheFlushReport, SceneCoordinator};
use crate::shared::RenderingContext;
use crate::world_mesh::{
    FrameMaterialBatchCache, FramePreparedRenderables, WorldMeshDrawCollectParallelism,
};

use super::draw_preparation::DrawPreparationExtractDesc;
use super::{OcclusionSystem, RenderBackend};
use crate::backend::resource_scope::ReleasedRenderSpaceResources;

/// Immutable backend-owned extraction snapshot produced by [`RenderBackend::extract_frame_shared`].
///
/// This is the runtime/backend hand-off for CPU-side world-mesh draw collection: the runtime owns
/// view planning while the backend owns material routing, resolved-material caching, prepared
/// renderables, and occlusion state.
pub(crate) struct ExtractedFrameShared<'a> {
    /// Scene after cache flush for world-matrix lookups and cull evaluation.
    pub(crate) scene: &'a SceneCoordinator,
    /// Mesh GPU asset pool queried for bounds and skinning metadata during draw collection.
    pub(crate) mesh_pool: &'a MeshPool,
    /// Property store backing [`crate::materials::host_data::MaterialDictionary::new`].
    pub(crate) property_store: &'a MaterialPropertyStore,
    /// Resolved raster pipeline selection for embedded materials.
    pub(crate) router: &'a MaterialRouter,
    /// Registry of renderer-side property ids used by the pipeline selector.
    pub(crate) pipeline_property_ids: MaterialPipelinePropertyIds,
    /// Mono/stereo/overlay render context applied this tick.
    pub(crate) render_context: RenderingContext,
    /// Persistent material batch caches keyed by [`ShaderPermutation`], refreshed once per frame
    /// for every distinct permutation appearing across this tick's prepared views. Per-view draw
    /// collection looks up the entry matching the view's permutation rather than building a
    /// per-view local cache (the previous mono-only fast path is now subsumed by this map).
    pub(crate) material_caches: &'a HashMap<ShaderPermutation, FrameMaterialBatchCache>,
    /// Dense draw-prep snapshot from the backend render-world cache.
    pub(crate) prepared_renderables: &'a FramePreparedRenderables,
    /// Shared occlusion state used for Hi-Z snapshots and temporal cull data.
    pub(crate) occlusion: &'a OcclusionSystem,
    /// CPU-side specular reflection-probe selector for per-object probe assignment.
    pub(crate) reflection_probes: &'a ReflectionProbeFrameSelection,
    /// Rayon parallelism tier for each view's inner walk.
    pub(crate) inner_parallelism: WorldMeshDrawCollectParallelism,
}

impl RenderBackend {
    /// Applies scene mutation reports to backend-owned CPU render-world caches.
    pub(crate) fn note_scene_apply_report(
        &mut self,
        report: &SceneApplyReport,
        scene: &SceneCoordinator,
    ) {
        self.draw_preparation.note_scene_apply_report(report);
        let released = self
            .resource_scopes
            .apply_scene_report(report, scene, &self.materials);
        self.purge_released_render_space_resources(released);
    }

    /// Applies world-cache flush reports to backend-owned CPU render-world caches.
    pub(crate) fn note_scene_cache_flush_report(&mut self, report: &SceneCacheFlushReport) {
        self.draw_preparation.note_scene_cache_flush_report(report);
    }

    fn purge_released_render_space_resources(&mut self, released: ReleasedRenderSpaceResources) {
        if released.is_empty() {
            return;
        }
        profiling::scope!("backend::purge_released_render_space_resources");

        self.materials.purge_texture_reference_caches();
        self.reflection_probes
            .purge_render_space_resources(&released.removed_spaces, &released.assets);
        let retired_views = self.retire_views_for_render_spaces(&released.removed_spaces);
        let skin_entries = self
            .frame_services
            .purge_skin_cache_spaces(&released.removed_spaces);
        self.materials.purge_released_material_assets(
            &released.assets.materials,
            &released.assets.property_blocks,
        );
        let asset_summary = self
            .asset_transfers
            .purge_render_space_assets(&released.assets);

        logger::info!(
            "world-close resource purge: spaces={} zero_owner_assets={} asset_purges={} views={} skin_entries={}",
            released.removed_spaces.len(),
            released.assets.total_len(),
            asset_summary.total(),
            retired_views,
            skin_entries
        );
    }

    fn retire_views_for_render_spaces(&mut self, spaces: &[RenderSpaceId]) -> usize {
        if spaces.is_empty() {
            return 0;
        }
        let removed_spaces: HashSet<RenderSpaceId> = spaces.iter().copied().collect();
        let retired = self.graph_state.retire_views_where(|view_id| {
            view_id
                .render_space_id()
                .is_some_and(|space_id| removed_spaces.contains(&space_id))
        });
        if retired.is_empty() {
            return 0;
        }
        logger::debug!(
            "retiring {} view-scoped resource sets for closed render spaces",
            retired.len()
        );
        self.world_mesh_frame_planner
            .release_view_resources(&retired);
        for &view_id in &retired {
            self.frame_services.frame_resources.retire_view(view_id);
            self.graph_state.history_registry_mut().retire_view(view_id);
            let _ = self.occlusion.retire_view(view_id);
        }
        retired.len()
    }

    /// Prepares clustered-light frame resources for the planned views in one graph submission.
    pub(crate) fn prepare_lights_for_views<I>(&mut self, scene: &SceneCoordinator, views: I)
    where
        I: IntoIterator<Item = FrameLightViewDesc>,
    {
        self.frame_services
            .frame_resources
            .prepare_lights_for_views(scene, views);
    }

    /// Drains completed Hi-Z readbacks into CPU snapshots at the top of the tick.
    pub(crate) fn hi_z_begin_frame_readback(&self, device: &wgpu::Device) {
        self.occlusion.hi_z_begin_frame_readback(device);
    }

    /// Refreshes backend-owned draw-prep state and returns the immutable frame setup used by the
    /// runtime's per-view draw collection stage.
    ///
    /// `view_shader_permutations` lists the [`ShaderPermutation`] each prepared view will use; one
    /// material batch cache is refreshed per distinct permutation so multi-view frames (e.g. VR
    /// stereo + a secondary camera) do not pay an O(materials x pipeline_property_ids) walk per
    /// view. The implicit `ShaderPermutation(0)` mono cache is always refreshed so the prepared
    /// renderables walk warms the steady-state working set.
    pub(crate) fn extract_frame_shared<'a>(
        &'a mut self,
        scene: &'a SceneCoordinator,
        render_context: RenderingContext,
        inner_parallelism: WorldMeshDrawCollectParallelism,
        view_shader_permutations: impl IntoIterator<Item = ShaderPermutation>,
    ) -> ExtractedFrameShared<'a> {
        self.draw_preparation
            .extract_frame_shared(DrawPreparationExtractDesc {
                scene,
                materials: &self.materials,
                asset_transfers: &self.asset_transfers,
                occlusion: &self.occlusion,
                reflection_probes: self.reflection_probes.selection(),
                render_context,
                inner_parallelism,
                view_shader_permutations,
            })
    }
}
