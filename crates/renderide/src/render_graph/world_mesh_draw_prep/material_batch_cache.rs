//! Per-frame cache of material-derived batch key fields, keyed by `(material_asset_id, property_block_id)`.
//!
//! All values in [`ResolvedMaterialBatch`] are functions of
//! `(material_asset_id, property_block_id, shader_perm)` and are constant for the lifetime of a
//! single collection call. Caching them amortises repeated dictionary and router lookups across
//! all draws that share the same material: in a typical scene, hundreds of draws share a few dozen
//! materials.

use hashbrown::HashMap;

use crate::assets::material::{MaterialDictionary, MaterialPropertyLookupIds};
use crate::materials::{
    embedded_stem_needs_color_stream, embedded_stem_needs_extended_vertex_streams,
    embedded_stem_needs_uv0_stream, embedded_stem_requires_intersection_pass,
    embedded_stem_uses_alpha_blending, material_blend_mode_for_lookup,
    material_render_state_for_lookup, resolve_raster_pipeline, MaterialBlendMode,
    MaterialPipelinePropertyIds, MaterialRenderState, MaterialRouter, RasterPipelineKind,
};
use crate::pipelines::ShaderPermutation;

/// Batch key fields derived from one `(material_asset_id, property_block_id)` pair.
///
/// All fields mirror what `batch_key_for_slot` computes on every draw; caching here avoids
/// repeating those dictionary and router lookups for every draw that uses the same material.
#[derive(Clone)]
pub(super) struct ResolvedMaterialBatch {
    /// Host shader asset id from material `set_shader` (`-1` when unknown).
    pub shader_asset_id: i32,
    /// Resolved raster pipeline kind for this material's shader.
    pub pipeline: RasterPipelineKind,
    /// Whether the active shader permutation requires a UV0 vertex stream.
    pub embedded_needs_uv0: bool,
    /// Whether the active shader permutation requires a color vertex stream.
    pub embedded_needs_color: bool,
    /// Whether the active shader permutation requires extended vertex streams (tangent, UV1-3).
    pub embedded_needs_extended_vertex_streams: bool,
    /// Whether the material requires a second forward subpass with a depth snapshot.
    pub embedded_requires_intersection_pass: bool,
    /// Resolved material blend mode.
    pub blend_mode: MaterialBlendMode,
    /// Runtime color, stencil, and depth state for this material/property-block pair.
    pub render_state: MaterialRenderState,
    /// Whether draws using this material should be sorted back-to-front.
    pub alpha_blended: bool,
}

/// Per-frame lookup table mapping `(material_asset_id, property_block_id)` →
/// [`ResolvedMaterialBatch`].
///
/// Built lazily via [`Self::get_or_insert`] on first miss, or eagerly by calling
/// [`Self::get_or_insert`] for every material in the scene before the parallel collection phase
/// begins. Once built, the table can be shared as an immutable reference across rayon workers.
pub(super) struct FrameMaterialBatchCache {
    entries: HashMap<(i32, Option<i32>), ResolvedMaterialBatch>,
    shader_perm: ShaderPermutation,
}

impl FrameMaterialBatchCache {
    /// Creates an empty cache for the given shader permutation.
    pub fn new(shader_perm: ShaderPermutation) -> Self {
        Self {
            entries: HashMap::new(),
            shader_perm,
        }
    }

    /// Returns the cached [`ResolvedMaterialBatch`] for the given key, inserting on first miss.
    pub fn get_or_insert(
        &mut self,
        material_asset_id: i32,
        property_block_id: Option<i32>,
        dict: &MaterialDictionary<'_>,
        router: &MaterialRouter,
        pipeline_property_ids: &MaterialPipelinePropertyIds,
    ) -> &ResolvedMaterialBatch {
        let shader_perm = self.shader_perm;
        self.entries
            .entry((material_asset_id, property_block_id))
            .or_insert_with(|| {
                resolve_material_batch(
                    material_asset_id,
                    property_block_id,
                    dict,
                    router,
                    pipeline_property_ids,
                    shader_perm,
                )
            })
    }

    /// Returns a cached entry without inserting.
    ///
    /// Returns `None` when the entry was never populated via [`Self::get_or_insert`].
    pub fn get(
        &self,
        material_asset_id: i32,
        property_block_id: Option<i32>,
    ) -> Option<&ResolvedMaterialBatch> {
        self.entries.get(&(material_asset_id, property_block_id))
    }

    /// Clears all entries while retaining allocated capacity for reuse.
    ///
    /// Reserved for a future pooled-cache optimization; currently a fresh cache is built each
    /// frame via [`Self::new`].
    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

/// Computes all batch key fields for one `(material_asset_id, property_block_id)` pair.
fn resolve_material_batch(
    material_asset_id: i32,
    property_block_id: Option<i32>,
    dict: &MaterialDictionary<'_>,
    router: &MaterialRouter,
    pipeline_property_ids: &MaterialPipelinePropertyIds,
    shader_perm: ShaderPermutation,
) -> ResolvedMaterialBatch {
    let shader_asset_id = dict
        .shader_asset_for_material(material_asset_id)
        .unwrap_or(-1);
    let pipeline = resolve_raster_pipeline(shader_asset_id, router);
    let embedded_needs_uv0 = match &pipeline {
        RasterPipelineKind::EmbeddedStem(stem) => {
            embedded_stem_needs_uv0_stream(stem.as_ref(), shader_perm)
        }
        RasterPipelineKind::DebugWorldNormals => false,
    };
    let embedded_needs_color = match &pipeline {
        RasterPipelineKind::EmbeddedStem(stem) => {
            embedded_stem_needs_color_stream(stem.as_ref(), shader_perm)
        }
        RasterPipelineKind::DebugWorldNormals => false,
    };
    let embedded_needs_extended_vertex_streams = match &pipeline {
        RasterPipelineKind::EmbeddedStem(stem) => {
            embedded_stem_needs_extended_vertex_streams(stem.as_ref(), shader_perm)
        }
        RasterPipelineKind::DebugWorldNormals => false,
    };
    let embedded_requires_intersection_pass = match &pipeline {
        RasterPipelineKind::EmbeddedStem(stem) => {
            embedded_stem_requires_intersection_pass(stem.as_ref(), shader_perm)
        }
        RasterPipelineKind::DebugWorldNormals => false,
    };
    let lookup_ids = MaterialPropertyLookupIds {
        material_asset_id,
        mesh_property_block_slot0: property_block_id,
    };
    let blend_mode = material_blend_mode_for_lookup(dict, lookup_ids, pipeline_property_ids);
    let render_state = material_render_state_for_lookup(dict, lookup_ids, pipeline_property_ids);
    let alpha_blended = match &pipeline {
        RasterPipelineKind::EmbeddedStem(stem) => embedded_stem_uses_alpha_blending(stem.as_ref()),
        RasterPipelineKind::DebugWorldNormals => false,
    } || blend_mode.is_transparent();
    ResolvedMaterialBatch {
        shader_asset_id,
        pipeline,
        embedded_needs_uv0,
        embedded_needs_color,
        embedded_needs_extended_vertex_streams,
        embedded_requires_intersection_pass,
        blend_mode,
        render_state,
        alpha_blended,
    }
}

#[cfg(test)]
mod tests {
    use crate::assets::material::{MaterialDictionary, MaterialPropertyStore, PropertyIdRegistry};
    use crate::materials::{MaterialPipelinePropertyIds, MaterialRouter, RasterPipelineKind};
    use crate::pipelines::ShaderPermutation;

    use super::FrameMaterialBatchCache;

    fn make_test_deps() -> (MaterialPropertyStore, MaterialRouter, PropertyIdRegistry) {
        let store = MaterialPropertyStore::new();
        let router = MaterialRouter::new(RasterPipelineKind::DebugWorldNormals);
        let reg = PropertyIdRegistry::new();
        (store, router, reg)
    }

    #[test]
    fn get_or_insert_caches_by_material_id() {
        let (store, router, reg) = make_test_deps();
        let dict = MaterialDictionary::new(&store);
        let ids = MaterialPipelinePropertyIds::new(&reg);
        let mut cache = FrameMaterialBatchCache::new(ShaderPermutation(0));

        let a = cache
            .get_or_insert(42, None, &dict, &router, &ids)
            .shader_asset_id;
        let b = cache
            .get_or_insert(42, None, &dict, &router, &ids)
            .shader_asset_id;
        assert_eq!(a, b);
        // Unknown material → shader id -1.
        assert_eq!(a, -1);
    }

    #[test]
    fn distinct_material_ids_produce_separate_entries() {
        let (store, router, reg) = make_test_deps();
        let dict = MaterialDictionary::new(&store);
        let ids = MaterialPipelinePropertyIds::new(&reg);
        let mut cache = FrameMaterialBatchCache::new(ShaderPermutation(0));

        cache.get_or_insert(1, None, &dict, &router, &ids);
        cache.get_or_insert(2, None, &dict, &router, &ids);
        assert_eq!(cache.entries.len(), 2);
        assert!(cache.get(1, None).is_some());
        assert!(cache.get(2, None).is_some());
        assert!(cache.get(99, None).is_none());
    }

    #[test]
    fn property_block_id_produces_separate_entry() {
        let (store, router, reg) = make_test_deps();
        let dict = MaterialDictionary::new(&store);
        let ids = MaterialPipelinePropertyIds::new(&reg);
        let mut cache = FrameMaterialBatchCache::new(ShaderPermutation(0));

        cache.get_or_insert(10, None, &dict, &router, &ids);
        cache.get_or_insert(10, Some(99), &dict, &router, &ids);
        assert_eq!(cache.entries.len(), 2);
    }

    #[test]
    fn clear_removes_entries_but_retains_capacity() {
        let (store, router, reg) = make_test_deps();
        let dict = MaterialDictionary::new(&store);
        let ids = MaterialPipelinePropertyIds::new(&reg);
        let mut cache = FrameMaterialBatchCache::new(ShaderPermutation(0));

        cache.get_or_insert(5, None, &dict, &router, &ids);
        cache.clear();
        assert_eq!(cache.entries.len(), 0);
        assert!(cache.get(5, None).is_none());
    }
}
