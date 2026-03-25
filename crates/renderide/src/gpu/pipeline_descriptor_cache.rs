//! Secondary index for [`wgpu::RenderPipeline`] instances by a stable numeric descriptor key.
//!
//! Built-in variants are registered with [`PipelineDescriptorCache::builtin_key`]; host-unlit
//! programs use [`PipelineDescriptorCache::host_unlit_key`].

use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use super::PipelineVariant;
use super::pipeline::RenderPipeline;

const HOST_UNLIT_CACHE_TAG: u64 = 0x48_30_53_54_55_4e_4c_54; // "H0STUNLT"

/// Maps descriptor hashes to shared pipeline [`Arc`]s.
#[derive(Default)]
pub(crate) struct PipelineDescriptorCache {
    entries: HashMap<u64, Arc<dyn RenderPipeline>>,
}

impl PipelineDescriptorCache {
    /// Hash for a built-in [`PipelineVariant`] at a given swapchain format.
    pub(crate) fn builtin_key(variant: PipelineVariant, format: wgpu::TextureFormat) -> u64 {
        let mut h = DefaultHasher::new();
        0xB0_u8.hash(&mut h);
        variant.hash(&mut h);
        format.hash(&mut h);
        h.finish()
    }

    /// Hash for a host-unlit pipeline sharing WGSL but keyed by shader asset id.
    pub(crate) fn host_unlit_key(shader_asset_id: i32, format: wgpu::TextureFormat) -> u64 {
        let mut h = DefaultHasher::new();
        HOST_UNLIT_CACHE_TAG.hash(&mut h);
        shader_asset_id.hash(&mut h);
        format.hash(&mut h);
        h.finish()
    }

    pub(crate) fn get(&self, key: u64) -> Option<Arc<dyn RenderPipeline>> {
        self.entries.get(&key).cloned()
    }

    pub(crate) fn insert(&mut self, key: u64, pipeline: Arc<dyn RenderPipeline>) {
        self.entries.insert(key, pipeline);
    }

    /// Drops a host-unlit entry when the host unloads that shader asset.
    pub(crate) fn remove_host_unlit(&mut self, shader_asset_id: i32, format: wgpu::TextureFormat) {
        self.entries
            .remove(&Self::host_unlit_key(shader_asset_id, format));
    }
}
