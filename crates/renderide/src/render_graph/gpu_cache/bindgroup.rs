//! Bind group cache keyed by pass-specific descriptors.

use std::hash::Hash;

use super::cache::GpuCache;

/// Typed cache for `wgpu::BindGroup` values.
#[derive(Debug)]
pub(crate) struct BindGroupMap<K> {
    /// Shared map storing bind groups keyed by pass-specific resource identity.
    inner: GpuCache<K, wgpu::BindGroup>,
}

impl<K> Default for BindGroupMap<K> {
    fn default() -> Self {
        Self {
            inner: GpuCache::new(),
        }
    }
}

impl<K> BindGroupMap<K>
where
    K: Clone + Eq + Hash,
{
    /// Creates an empty bind-group map with clear-on-overflow eviction.
    pub(crate) fn with_max_entries(max_entries: usize) -> Self {
        Self {
            inner: GpuCache::with_max_entries(max_entries),
        }
    }

    /// Returns a cached bind group or builds one outside the map lock.
    pub(crate) fn get_or_create(
        &self,
        key: K,
        build: impl FnOnce(&K) -> wgpu::BindGroup,
    ) -> wgpu::BindGroup {
        self.inner.get_or_create(key, build)
    }
}
