//! CPU draw-preparation ownership behind the backend facade.

use hashbrown::HashMap;

use crate::materials::host_data::{MaterialDictionary, MaterialPropertyStore};
use crate::materials::{MaterialPipelinePropertyIds, MaterialRouter, RasterPipelineKind};
use crate::reflection_probes::specular::ReflectionProbeFrameSelection;
use crate::scene::{SceneApplyReport, SceneCacheFlushReport, SceneCoordinator};
use crate::shared::RenderingContext;
use crate::world_mesh::{
    FrameMaterialBatchCache, FramePreparedRenderables, RenderWorld, WorldMeshDrawCollectParallelism,
};

use crate::assets::asset_transfer_queue::AssetTransferQueue;
use crate::materials::{MaterialSystem, ShaderPermutation};
use crate::occlusion::OcclusionSystem;

use super::frame_packet::ExtractedFrameShared;

/// Inputs for one backend draw-preparation extraction.
pub(super) struct DrawPreparationExtractDesc<'a, I>
where
    I: IntoIterator<Item = ShaderPermutation>,
{
    /// Scene after cache flush for world-matrix lookups and cull evaluation.
    pub(super) scene: &'a SceneCoordinator,
    /// Material registry, routes, and property data.
    pub(super) materials: &'a MaterialSystem,
    /// Asset upload queues and resident GPU pools.
    pub(super) asset_transfers: &'a AssetTransferQueue,
    /// Shared occlusion state used for Hi-Z snapshots and temporal cull data.
    pub(super) occlusion: &'a OcclusionSystem,
    /// CPU-side specular reflection-probe selector for per-object probe assignment.
    pub(super) reflection_probes: &'a ReflectionProbeFrameSelection,
    /// Mono/stereo/overlay render context applied this tick.
    pub(super) render_context: RenderingContext,
    /// Rayon parallelism tier for each view's inner walk.
    pub(super) inner_parallelism: WorldMeshDrawCollectParallelism,
    /// Shader permutations used by prepared views this tick.
    pub(super) view_shader_permutations: I,
}

/// Backend-owned CPU draw-preparation caches.
pub(super) struct BackendDrawPreparation {
    /// Fallback router used before any embedded-material registry is available.
    null_material_router: MaterialRouter,
    /// Persistent resolved-material caches keyed by shader permutation.
    material_batch_caches: HashMap<ShaderPermutation, FrameMaterialBatchCache>,
    /// Backend-owned CPU render-world cache used to amortize draw preparation.
    render_world: RenderWorld,
}

impl BackendDrawPreparation {
    /// Creates empty draw-preparation caches.
    pub(super) fn new() -> Self {
        Self {
            null_material_router: MaterialRouter::new(RasterPipelineKind::Null),
            material_batch_caches: HashMap::new(),
            render_world: RenderWorld::new(RenderingContext::default()),
        }
    }

    /// Applies scene mutation reports to backend-owned CPU render-world caches.
    pub(super) fn note_scene_apply_report(&mut self, report: &SceneApplyReport) {
        self.render_world.note_scene_apply_report(report);
    }

    /// Applies world-cache flush reports to backend-owned CPU render-world caches.
    pub(super) fn note_scene_cache_flush_report(&mut self, report: &SceneCacheFlushReport) {
        self.render_world.note_cache_flush_report(report);
    }

    /// Refreshes backend-owned draw-prep state and returns the immutable frame setup.
    pub(super) fn extract_frame_shared<'a, I>(
        &'a mut self,
        desc: DrawPreparationExtractDesc<'a, I>,
    ) -> ExtractedFrameShared<'a>
    where
        I: IntoIterator<Item = ShaderPermutation>,
    {
        let DrawPreparationExtractDesc {
            scene,
            materials,
            asset_transfers,
            occlusion,
            reflection_probes,
            render_context,
            inner_parallelism,
            view_shader_permutations,
        } = desc;
        let Self {
            null_material_router,
            material_batch_caches,
            render_world,
        } = self;
        let property_store = materials.material_property_store();
        let router = materials
            .material_registry()
            .map_or(&*null_material_router, |registry| &registry.router);
        let pipeline_property_ids = materials.pipeline_property_resolver().resolve();

        let prepared_renderables = {
            profiling::scope!("render::build_frame_prepared_renderables");
            render_world.prepare_for_frame(scene, asset_transfers.mesh_pool(), render_context)
        };

        refresh_material_caches(
            material_batch_caches,
            prepared_renderables,
            property_store,
            router,
            &pipeline_property_ids,
            view_shader_permutations,
        );

        ExtractedFrameShared {
            scene,
            mesh_pool: asset_transfers.mesh_pool(),
            property_store,
            router,
            pipeline_property_ids,
            render_context,
            material_caches: material_batch_caches,
            prepared_renderables,
            occlusion,
            reflection_probes,
            inner_parallelism,
        }
    }
}

fn refresh_material_caches(
    material_batch_caches: &mut HashMap<ShaderPermutation, FrameMaterialBatchCache>,
    prepared_renderables: &FramePreparedRenderables,
    property_store: &MaterialPropertyStore,
    router: &MaterialRouter,
    pipeline_property_ids: &MaterialPipelinePropertyIds,
    view_shader_permutations: impl IntoIterator<Item = ShaderPermutation>,
) {
    profiling::scope!("render::build_frame_material_cache");
    let dict = MaterialDictionary::new(property_store);
    material_batch_caches
        .entry(ShaderPermutation(0))
        .or_default()
        .refresh_for_prepared(
            prepared_renderables,
            &dict,
            router,
            pipeline_property_ids,
            ShaderPermutation(0),
        );
    for perm in view_shader_permutations {
        if perm == ShaderPermutation(0) {
            continue;
        }
        material_batch_caches
            .entry(perm)
            .or_default()
            .refresh_for_prepared(
                prepared_renderables,
                &dict,
                router,
                pipeline_property_ids,
                perm,
            );
    }
}
