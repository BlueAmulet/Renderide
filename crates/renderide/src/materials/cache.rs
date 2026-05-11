//! Cache of [`wgpu::RenderPipeline`] per [`RasterPipelineKind`] + permutation + attachment formats.
//!
//! Lookup keys intentionally **do not** include a WGSL layout fingerprint: reflecting the full
//! shader on every cache probe would dominate CPU cost. Embedded targets are stable per
//! `(kind, permutation, [`MaterialPipelineDesc`])`. If hot-reload or dynamic WGSL is introduced,
//! extend the key with a content hash or version.
//!
//! The cache is LRU-bounded to avoid unbounded growth when many format/permutation combinations appear.

use std::num::{NonZeroU32, NonZeroUsize};
use std::sync::{Arc, OnceLock};

use hashbrown::{HashMap, HashSet};
use lru::LruCache;
use parking_lot::Mutex;

use crate::gpu_resource::AtomicCacheCounters;
use crate::materials::ShaderPermutation;
use crate::materials::embedded::stem_metadata::{
    EmbeddedRasterPipelineSource, build_embedded_wgsl, create_embedded_render_pipelines,
};
use crate::materials::null_pipeline::{build_null_wgsl, create_null_render_pipeline};
use crate::materials::raster_pipeline::ShaderModuleBuildRefs;
use crate::materials::{
    MaterialBlendMode, MaterialRenderState, RasterFrontFace, RasterPipelineKind,
    RasterPrimitiveTopology,
};

use super::pipeline_build_error::PipelineBuildError;
use super::raster_pipeline::MaterialPipelineDesc;

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

/// Nonblocking cache lookup result.
pub(super) enum MaterialPipelineLookup {
    /// The requested pipeline set is available for this frame.
    Ready(MaterialPipelineSet),
    /// A background worker is building the requested pipeline set.
    Pending,
    /// The requested pipeline failed to build; callers may use a fallback.
    Failed(String),
}

struct PipelineBuildRequest {
    key: MaterialPipelineCacheKey,
    kind: RasterPipelineKind,
    desc: MaterialPipelineDesc,
    variant: MaterialPipelineVariantSpec,
    device: Arc<wgpu::Device>,
    limits: Arc<crate::gpu::GpuLimits>,
    tx: crossbeam_channel::Sender<PipelineBuildOutcome>,
}

struct PipelineBuildOutcome {
    key: MaterialPipelineCacheKey,
    kind: RasterPipelineKind,
    result: Result<MaterialPipelineSet, String>,
}

/// Lazily built pipeline sets; LRU-evicted when over [`MAX_CACHED_PIPELINES`].
pub struct MaterialPipelineCache {
    device: Arc<wgpu::Device>,
    limits: Arc<crate::gpu::GpuLimits>,
    pipelines: Mutex<LruCache<MaterialPipelineCacheKey, MaterialPipelineSet>>,
    pipeline_build_tx: crossbeam_channel::Sender<PipelineBuildOutcome>,
    pipeline_build_rx: crossbeam_channel::Receiver<PipelineBuildOutcome>,
    pending_pipeline_builds: Mutex<HashSet<MaterialPipelineCacheKey>>,
    failed_pipeline_builds: Mutex<HashMap<MaterialPipelineCacheKey, String>>,
    stats: AtomicCacheCounters,
}

impl MaterialPipelineCache {
    /// Creates an empty cache for `device` with the device's effective [`crate::gpu::GpuLimits`].
    pub fn new(device: Arc<wgpu::Device>, limits: Arc<crate::gpu::GpuLimits>) -> Self {
        let (pipeline_build_tx, pipeline_build_rx) = crossbeam_channel::unbounded();
        Self {
            device,
            limits,
            pipelines: Mutex::new(LruCache::new(max_cached_pipelines())),
            pipeline_build_tx,
            pipeline_build_rx,
            pending_pipeline_builds: Mutex::new(HashSet::new()),
            failed_pipeline_builds: Mutex::new(HashMap::new()),
            stats: AtomicCacheCounters::default(),
        }
    }

    /// Returns the cached pipeline set or queues a background build for a miss.
    ///
    /// On a cache hit, does not compose WGSL or run reflection; those run only on the worker.
    pub(super) fn get_or_queue(
        &self,
        kind: &RasterPipelineKind,
        desc: &MaterialPipelineDesc,
        variant: MaterialPipelineVariantSpec,
    ) -> MaterialPipelineLookup {
        profiling::scope!("materials::get_or_create_pipeline");
        self.drain_completed_pipeline_builds();
        let key = Self::cache_key(kind, desc, variant);

        if let Some(hit) = self.cached_pipeline_set(&key) {
            self.stats.note_hit();
            return MaterialPipelineLookup::Ready(hit);
        }
        let failed_build = self.failed_pipeline_builds.lock().get(&key).cloned();
        if let Some(error) = failed_build {
            return MaterialPipelineLookup::Failed(error);
        }

        if self.pending_pipeline_builds.lock().contains(&key) {
            return MaterialPipelineLookup::Pending;
        }

        self.queue_pipeline_build(key, kind.clone(), *desc, variant);
        MaterialPipelineLookup::Pending
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

    fn queue_pipeline_build(
        &self,
        key: MaterialPipelineCacheKey,
        kind: RasterPipelineKind,
        desc: MaterialPipelineDesc,
        variant: MaterialPipelineVariantSpec,
    ) {
        {
            let mut pending = self.pending_pipeline_builds.lock();
            if !pending.insert(key.clone()) {
                return;
            }
        }
        self.stats.note_miss();

        let request = PipelineBuildRequest {
            key: key.clone(),
            kind: kind.clone(),
            desc,
            variant,
            device: self.device.clone(),
            limits: self.limits.clone(),
            tx: self.pipeline_build_tx.clone(),
        };
        if let Err(e) = spawn_pipeline_build(request) {
            self.pending_pipeline_builds.lock().remove(&key);
            self.failed_pipeline_builds.lock().insert(key, e.clone());
            logger::warn!("MaterialPipelineCache: could not queue {kind:?} pipeline build: {e}");
        }
    }

    fn drain_completed_pipeline_builds(&self) {
        while let Ok(outcome) = self.pipeline_build_rx.try_recv() {
            self.pending_pipeline_builds.lock().remove(&outcome.key);
            match outcome.result {
                Ok(set) => {
                    self.failed_pipeline_builds.lock().remove(&outcome.key);
                    self.insert_pipeline_set(outcome.key, set);
                }
                Err(e) => {
                    logger::warn!(
                        "MaterialPipelineCache: async pipeline build failed for {:?}: {e}",
                        outcome.kind
                    );
                    self.failed_pipeline_builds.lock().insert(outcome.key, e);
                }
            }
        }
    }

    fn build_pipeline_set_for(
        device: Arc<wgpu::Device>,
        limits: Arc<crate::gpu::GpuLimits>,
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
                    limits: &limits,
                    module: &module,
                    desc,
                    wgsl_source: &wgsl,
                },
            )?,
            RasterPipelineKind::Null => {
                vec![create_null_render_pipeline(
                    &device,
                    &limits,
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

fn spawn_pipeline_build(request: PipelineBuildRequest) -> Result<(), String> {
    let pool = material_pipeline_compile_pool()?;
    pool.spawn(move || {
        profiling::scope!("materials::async_pipeline_compile");
        let PipelineBuildRequest {
            key,
            kind,
            desc,
            variant,
            device,
            limits,
            tx,
        } = request;
        let result =
            MaterialPipelineCache::build_pipeline_set_for(device, limits, &kind, &desc, variant)
                .map_err(|e| e.to_string());
        let _ = tx.send(PipelineBuildOutcome { key, kind, result });
    });
    Ok(())
}

fn material_pipeline_compile_pool() -> Result<&'static rayon::ThreadPool, String> {
    static POOL: OnceLock<Result<rayon::ThreadPool, String>> = OnceLock::new();
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .thread_name(|_| "material-pipeline-worker".to_string())
            .build()
            .map_err(|e| format!("material pipeline worker pool creation failed: {e}"))
    })
    .as_ref()
    .map_err(Clone::clone)
}
