//! Render pipeline cache keyed by pass-specific descriptors.

use std::hash::Hash;
use std::sync::Arc;

use super::cache::GpuCache;

/// Typed cache for `wgpu::RenderPipeline` values.
#[derive(Debug)]
pub(crate) struct RenderPipelineMap<K> {
    /// Shared map storing pipelines behind `Arc` so record paths can clone handles cheaply.
    inner: GpuCache<K, Arc<wgpu::RenderPipeline>>,
}

impl<K> Default for RenderPipelineMap<K> {
    fn default() -> Self {
        Self {
            inner: GpuCache::new(),
        }
    }
}

impl<K> RenderPipelineMap<K>
where
    K: Clone + Eq + Hash,
{
    /// Returns a cached render pipeline or builds one outside the map lock.
    pub(crate) fn get_or_create(
        &self,
        key: K,
        build: impl FnOnce(&K) -> wgpu::RenderPipeline,
    ) -> Arc<wgpu::RenderPipeline> {
        self.inner.get_or_create(key, |key| Arc::new(build(key)))
    }
}
