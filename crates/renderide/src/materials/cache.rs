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

use crate::materials::embedded_raster_pipeline::{
    build_embedded_wgsl, create_embedded_render_pipeline,
};
use crate::materials::RasterPipelineKind;
use crate::pipelines::raster::debug_world_normals::{
    build_debug_world_normals_wgsl, create_debug_world_normals_render_pipeline,
};
use crate::pipelines::ShaderPermutation;

use super::family::MaterialPipelineDesc;

/// Maximum raster pipelines retained (LRU eviction).
const MAX_CACHED_PIPELINES: usize = 512;

/// Key for [`MaterialPipelineCache`] lookups (no WGSL parse — see module docs).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MaterialPipelineCacheKey {
    /// Which WGSL program backs the pipeline (embedded stem or debug fallback).
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
}

/// Lazily built pipelines; LRU-evicted when over [`MAX_CACHED_PIPELINES`].
#[derive(Debug)]
pub struct MaterialPipelineCache {
    device: Arc<wgpu::Device>,
    pipelines: LruCache<MaterialPipelineCacheKey, wgpu::RenderPipeline>,
}

impl MaterialPipelineCache {
    /// Creates an empty cache for `device`.
    pub fn new(device: Arc<wgpu::Device>) -> Self {
        Self {
            device,
            pipelines: LruCache::new(
                NonZeroUsize::new(MAX_CACHED_PIPELINES).expect("MAX_CACHED_PIPELINES > 0"),
            ),
        }
    }

    /// Device used for `create_shader_module` / `create_render_pipeline`.
    pub fn device(&self) -> &Arc<wgpu::Device> {
        &self.device
    }

    /// Returns or builds a pipeline for `kind`, `desc`, and `permutation`.
    ///
    /// On a cache hit, does not compose WGSL or run reflection; those run only when inserting a new entry.
    pub fn get_or_create(
        &mut self,
        kind: &RasterPipelineKind,
        desc: &MaterialPipelineDesc,
        permutation: ShaderPermutation,
    ) -> &wgpu::RenderPipeline {
        let key = MaterialPipelineCacheKey {
            kind: kind.clone(),
            permutation,
            surface_format: desc.surface_format,
            depth_stencil_format: desc.depth_stencil_format,
            sample_count: desc.sample_count,
            multiview_mask: desc.multiview_mask,
        };
        let device = self.device.clone();
        let cache = &mut self.pipelines;
        if cache.peek(&key).is_some() {
            return cache.get(&key).expect("pipeline cache peek hit");
        }
        let wgsl = match kind {
            RasterPipelineKind::EmbeddedStem(stem) => build_embedded_wgsl(stem, permutation),
            RasterPipelineKind::DebugWorldNormals => build_debug_world_normals_wgsl(permutation),
        };
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raster_material_shader"),
            source: wgpu::ShaderSource::Wgsl(wgsl.clone().into()),
        });
        let pipeline = match kind {
            RasterPipelineKind::EmbeddedStem(stem) => {
                create_embedded_render_pipeline(stem, &device, &module, desc, &wgsl)
            }
            RasterPipelineKind::DebugWorldNormals => {
                create_debug_world_normals_render_pipeline(&device, &module, desc, &wgsl)
            }
        };
        if let Some(evicted) = cache.put(key.clone(), pipeline) {
            drop(evicted);
            logger::trace!("MaterialPipelineCache: evicted LRU pipeline entry");
        }
        cache.get(&key).expect("pipeline just inserted")
    }
}
