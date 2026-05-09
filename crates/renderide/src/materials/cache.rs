//! Cache of [`wgpu::RenderPipeline`] per [`RasterPipelineKind`] + permutation + attachment formats.
//!
//! Lookup keys intentionally **do not** include a WGSL layout fingerprint: reflecting the full
//! shader on every cache probe would dominate CPU cost. Embedded targets are stable per
//! `(kind, permutation, [`MaterialPipelineDesc`])`. If hot-reload or dynamic WGSL is introduced,
//! extend the key with a content hash or version.
//!
//! The cache is LRU-bounded to avoid unbounded growth when many format/permutation combinations appear.

use std::num::{NonZeroU32, NonZeroUsize};
use std::sync::Arc;

use lru::LruCache;
use parking_lot::Mutex;

use crate::concurrency::{KeyedSingleFlight, SingleFlightPermit};
use crate::gpu_resource::AtomicCacheCounters;
use crate::materials::ShaderPermutation;
use crate::materials::embedded_raster_pipeline::{
    EmbeddedRasterPipelineSource, build_embedded_wgsl, create_embedded_render_pipelines,
};
use crate::materials::null_pipeline::{build_null_wgsl, create_null_render_pipeline};
use crate::materials::raster_pipeline::ShaderModuleBuildRefs;
use crate::materials::{
    MaterialBlendMode, MaterialRenderState, RasterFrontFace, RasterPipelineKind,
    RasterPrimitiveTopology,
};

use super::family::MaterialPipelineDesc;
use super::pipeline_build_error::PipelineBuildError;

/// Maximum raster pipelines retained (LRU eviction).
const MAX_CACHED_PIPELINES: usize = 512;

/// Non-zero raster pipeline cache capacity.
fn max_cached_pipelines() -> NonZeroUsize {
    NonZeroUsize::new(MAX_CACHED_PIPELINES).unwrap_or(NonZeroUsize::MIN)
}

/// Material-driven pipeline variant: selectors that affect [`wgpu::RenderPipeline`] state but are
/// not derived from [`MaterialPipelineDesc`] attachment formats.
///
/// Bundled together so registry / cache lookups carry a single argument instead of five
/// loose scalars, and so any future axis (e.g. additional shader permutations) lands here without
/// growing call signatures.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MaterialPipelineVariantSpec {
    /// Stereo multiview / single-view permutation for the pipeline.
    pub permutation: ShaderPermutation,
    /// Material-level blend override for stems without explicit pass directives.
    pub blend_mode: MaterialBlendMode,
    /// Material-level stencil and color write state.
    pub render_state: MaterialRenderState,
    /// Front-face winding for draw transforms in this pipeline bucket.
    pub front_face: RasterFrontFace,
    /// Primitive topology baked into [`wgpu::PrimitiveState::topology`] for this pipeline bucket.
    pub primitive_topology: RasterPrimitiveTopology,
}

/// Key for [`MaterialPipelineCache`] lookups (no WGSL parse -- see module docs).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MaterialPipelineCacheKey {
    /// Which WGSL program backs the pipeline (embedded stem or null fallback).
    pub kind: RasterPipelineKind,
    /// Stereo multiview / single-view permutation for the pipeline.
    pub permutation: ShaderPermutation,
    /// Color attachment format (swapchain or offscreen).
    pub surface_format: wgpu::TextureFormat,
    /// Depth/stencil format when depth attachment is used.
    pub depth_stencil_format: Option<wgpu::TextureFormat>,
    /// MSAA sample count for the color target.
    pub sample_count: u32,
    /// OpenXR / multiview view mask when compiling multiview pipelines.
    pub multiview_mask: Option<NonZeroU32>,
    /// Material-level blend override for stems without explicit pass directives.
    pub blend_mode: MaterialBlendMode,
    /// Material-level stencil and color write state.
    pub render_state: MaterialRenderState,
    /// Front-face winding for draw transforms in this pipeline bucket.
    pub front_face: RasterFrontFace,
    /// Primitive topology baked into [`wgpu::PrimitiveState::topology`] for this pipeline bucket.
    ///
    /// `wgpu::RenderPipeline` immutably bakes its primitive topology, so two draws of the same
    /// shader/material that differ in topology must build separate pipelines.
    pub primitive_topology: RasterPrimitiveTopology,
}

/// One or more pipelines for a material entry (one per declared `//#pass`).
///
/// Materials without pass directives have `len == 1`; OverlayFresnel and other multi-pass shaders
/// have `len >= 2`. The forward encode loop dispatches every pipeline in order for each draw.
pub type MaterialPipelineSet = Arc<[wgpu::RenderPipeline]>;

/// Lazily built pipeline sets; LRU-evicted when over [`MAX_CACHED_PIPELINES`].
#[derive(Debug)]
pub struct MaterialPipelineCache {
    device: Arc<wgpu::Device>,
    limits: Arc<crate::gpu::GpuLimits>,
    pipelines: Mutex<LruCache<MaterialPipelineCacheKey, MaterialPipelineSet>>,
    compiles: KeyedSingleFlight<MaterialPipelineCacheKey>,
    stats: AtomicCacheCounters,
}

impl MaterialPipelineCache {
    /// Creates an empty cache for `device` with the device's effective [`crate::gpu::GpuLimits`].
    pub fn new(device: Arc<wgpu::Device>, limits: Arc<crate::gpu::GpuLimits>) -> Self {
        Self {
            device,
            limits,
            pipelines: Mutex::new(LruCache::new(max_cached_pipelines())),
            compiles: KeyedSingleFlight::default(),
            stats: AtomicCacheCounters::default(),
        }
    }

    /// Returns or builds the pipeline set for `kind`, `desc`, and `permutation`.
    ///
    /// On a cache hit, does not compose WGSL or run reflection; those run only when inserting a new entry.
    pub fn get_or_create(
        &self,
        kind: &RasterPipelineKind,
        desc: &MaterialPipelineDesc,
        variant: MaterialPipelineVariantSpec,
    ) -> Result<MaterialPipelineSet, PipelineBuildError> {
        profiling::scope!("materials::get_or_create_pipeline");
        let key = Self::cache_key(kind, desc, variant);
        loop {
            if let Some(hit) = self.cached_pipeline_set(&key) {
                self.stats.note_hit();
                return Ok(hit);
            }

            let leader = match self.compiles.acquire(key.clone()) {
                SingleFlightPermit::Leader(leader) => leader,
                SingleFlightPermit::Waiter(waiter) => {
                    profiling::scope!("materials::pipeline_single_flight_wait");
                    waiter.wait();
                    continue;
                }
            };

            if let Some(hit) = self.cached_pipeline_set(&key) {
                self.stats.note_hit();
                return Ok(hit);
            }

            self.stats.note_miss();
            let set = self.build_pipeline_set(kind, desc, variant)?;
            self.insert_pipeline_set(key, set.clone());
            drop(leader);
            return Ok(set);
        }
    }

    fn cache_key(
        kind: &RasterPipelineKind,
        desc: &MaterialPipelineDesc,
        variant: MaterialPipelineVariantSpec,
    ) -> MaterialPipelineCacheKey {
        let MaterialPipelineVariantSpec {
            permutation,
            blend_mode,
            render_state,
            front_face,
            primitive_topology,
        } = variant;
        MaterialPipelineCacheKey {
            kind: kind.clone(),
            permutation,
            surface_format: desc.surface_format,
            depth_stencil_format: desc.depth_stencil_format,
            sample_count: desc.sample_count,
            multiview_mask: desc.multiview_mask,
            blend_mode,
            render_state,
            front_face,
            primitive_topology,
        }
    }

    fn cached_pipeline_set(&self, key: &MaterialPipelineCacheKey) -> Option<MaterialPipelineSet> {
        // A hit is real use; promote it so hot pipelines do not get evicted.
        self.pipelines.lock().get(key).cloned()
    }

    fn build_pipeline_set(
        &self,
        kind: &RasterPipelineKind,
        desc: &MaterialPipelineDesc,
        variant: MaterialPipelineVariantSpec,
    ) -> Result<MaterialPipelineSet, PipelineBuildError> {
        let MaterialPipelineVariantSpec {
            permutation,
            blend_mode,
            render_state,
            front_face,
            primitive_topology,
        } = variant;
        let wgsl = match kind {
            RasterPipelineKind::EmbeddedStem(stem) => build_embedded_wgsl(stem, permutation)?,
            RasterPipelineKind::Null => build_null_wgsl(permutation)?,
        };
        let device = self.device.clone();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raster_material_shader"),
            source: wgpu::ShaderSource::Wgsl(wgsl.clone().into()),
        });
        let pipelines: Vec<wgpu::RenderPipeline> = match kind {
            RasterPipelineKind::EmbeddedStem(stem) => create_embedded_render_pipelines(
                EmbeddedRasterPipelineSource {
                    stem: stem.clone(),
                    permutation,
                    blend_mode,
                    render_state,
                    front_face,
                    primitive_topology,
                },
                ShaderModuleBuildRefs {
                    device: &device,
                    limits: &self.limits,
                    module: &module,
                    desc,
                    wgsl_source: &wgsl,
                },
            )?,
            RasterPipelineKind::Null => {
                vec![create_null_render_pipeline(
                    &device,
                    &self.limits,
                    &module,
                    desc,
                    &wgsl,
                    front_face,
                    primitive_topology,
                )?]
            }
        };
        Ok(Arc::from(pipelines.into_boxed_slice()))
    }

    fn insert_pipeline_set(&self, key: MaterialPipelineCacheKey, set: MaterialPipelineSet) {
        let mut cache = self.pipelines.lock();
        self.stats.note_insertion();
        let evicted = cache.push(key, set);
        drop(cache);

        if let Some((_evicted_key, evicted)) = evicted {
            drop(evicted);
            self.stats.note_eviction();
            let stats = self.stats.snapshot();
            logger::trace!(
                "MaterialPipelineCache: evicted LRU pipeline entry hits={} misses={} insertions={} evictions={}",
                stats.hits,
                stats.misses,
                stats.insertions,
                stats.evictions
            );
        }
    }
}
