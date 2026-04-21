//! Clustered forward lighting: GPU buffers for per-cluster light lists and compute-only uniforms.
//!
//! [`ClusterBufferCache`] recreates storage when the viewport or Z slice count changes. Tile size and
//! per-tile caps match the clustered light compute shader and PBS fragment sampling.

use std::mem::size_of;

use crate::gpu::GpuLimits;

/// Screen tile size in pixels (DOOM-style cluster grid XY).
pub const TILE_SIZE: u32 = 16;
/// Exponential depth slice count (view-space Z bins).
pub const CLUSTER_COUNT_Z: u32 = 32;
/// Maximum lights assigned to a single cluster (buffer index order). Keep in sync with
/// `MAX_LIGHTS_PER_TILE` in `shaders/source/modules/pbs_cluster.wgsl` and
/// `shaders/source/compute/clustered_light.wgsl`. Bumped from 32 to reduce far-cluster overflow
/// that produced dark "splotches" in scenes with many lights.
pub const MAX_LIGHTS_PER_TILE: u32 = 64;
/// Uniform buffer size for clustered light compute `ClusterParams` (WGSL layout + tail padding).
pub const CLUSTER_PARAMS_UNIFORM_SIZE: u64 = 256;

/// References to GPU buffers shared by the clustered light compute pass and raster `@group(0)`.
#[derive(Clone, Copy)]
pub struct ClusterBufferRefs<'a> {
    /// One `u32` count per cluster (compute writes; fragment reads plain `u32`; one thread per cluster).
    pub cluster_light_counts: &'a wgpu::Buffer,
    /// Flattened `cluster_id * MAX_LIGHTS_PER_TILE + slot` light indices.
    pub cluster_light_indices: &'a wgpu::Buffer,
    /// Uniform block for compute only (`ClusterParams` in WGSL).
    pub params_buffer: &'a wgpu::Buffer,
}

/// Caches cluster buffers; bumps [`Self::version`] when storage is recreated.
///
/// When `stereo` is true, the counts and indices buffers are allocated at **2x** size so the
/// compute pass can write eye-0 clusters at `[0..N)` and eye-1 at `[N..2N)`.
pub struct ClusterBufferCache {
    cluster_light_counts: Option<wgpu::Buffer>,
    cluster_light_indices: Option<wgpu::Buffer>,
    params_buffer: Option<wgpu::Buffer>,
    cached_key: ClusterCacheKey,
    /// Incremented when buffers are recreated (bind group invalidation).
    pub version: u64,
}

/// Cache invalidation key: viewport size, Z slice count, and stereo mode.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
struct ClusterCacheKey {
    viewport: (u32, u32),
    cluster_count_z: u32,
    stereo: bool,
}

impl ClusterBufferCache {
    /// Empty cache; [`Self::ensure_buffers`] allocates on first use.
    pub fn new() -> Self {
        Self {
            cluster_light_counts: None,
            cluster_light_indices: None,
            params_buffer: None,
            cached_key: ClusterCacheKey::default(),
            version: 0,
        }
    }

    /// Ensures buffers exist for `viewport`, `cluster_count_z`, and `stereo`; recreates when any changes.
    ///
    /// When `stereo` is true, count and index buffers are doubled to hold per-eye cluster data.
    ///
    /// Returns [`None`] when cluster storage would exceed [`GpuLimits`] storage/buffer caps.
    pub fn ensure_buffers(
        &mut self,
        device: &wgpu::Device,
        limits: &GpuLimits,
        viewport: (u32, u32),
        cluster_count_z: u32,
        stereo: bool,
    ) -> Option<ClusterBufferRefs<'_>> {
        let (width, height) = viewport;
        if width == 0 || height == 0 {
            return None;
        }
        let cluster_count_x = width.div_ceil(TILE_SIZE);
        let cluster_count_y = height.div_ceil(TILE_SIZE);
        let clusters_per_eye = (cluster_count_x * cluster_count_y * cluster_count_z) as usize;
        let eye_multiplier = if stereo { 2 } else { 1 };
        let total_clusters = clusters_per_eye * eye_multiplier;
        let counts_bytes = (total_clusters * size_of::<u32>()) as u64;
        let indices_bytes =
            (total_clusters * MAX_LIGHTS_PER_TILE as usize * size_of::<u32>()) as u64;
        let max_bind = limits.max_storage_buffer_binding_size();
        let max_buf = limits.max_buffer_size();
        if counts_bytes > max_bind
            || indices_bytes > max_bind
            || counts_bytes > max_buf
            || indices_bytes > max_buf
        {
            logger::warn!(
                "cluster buffers: viewport {:?} stereo={} would need counts={} indices={} bytes; exceeds max_storage_buffer_binding_size ({}) or max_buffer_size ({})",
                viewport,
                stereo,
                counts_bytes,
                indices_bytes,
                max_bind,
                max_buf
            );
            return None;
        }
        let key = ClusterCacheKey {
            viewport,
            cluster_count_z,
            stereo,
        };
        if self.cluster_light_counts.is_none() || self.cached_key != key {
            self.version = self.version.wrapping_add(1);
            self.cached_key = key;

            self.cluster_light_counts = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cluster_light_counts"),
                size: (total_clusters * size_of::<u32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.cluster_light_indices = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cluster_light_indices"),
                size: (total_clusters * MAX_LIGHTS_PER_TILE as usize * size_of::<u32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.params_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cluster_params_uniform"),
                size: CLUSTER_PARAMS_UNIFORM_SIZE * eye_multiplier as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        Some(ClusterBufferRefs {
            cluster_light_counts: self.cluster_light_counts.as_ref()?,
            cluster_light_indices: self.cluster_light_indices.as_ref()?,
            params_buffer: self.params_buffer.as_ref()?,
        })
    }

    /// Returns buffers when they match the last successful [`Self::ensure_buffers`] key.
    pub fn get_buffers(
        &self,
        viewport: (u32, u32),
        cluster_count_z: u32,
        stereo: bool,
    ) -> Option<ClusterBufferRefs<'_>> {
        let key = ClusterCacheKey {
            viewport,
            cluster_count_z,
            stereo,
        };
        if self.cached_key != key {
            return None;
        }
        Some(ClusterBufferRefs {
            cluster_light_counts: self.cluster_light_counts.as_ref()?,
            cluster_light_indices: self.cluster_light_indices.as_ref()?,
            params_buffer: self.params_buffer.as_ref()?,
        })
    }
}

impl Default for ClusterBufferCache {
    fn default() -> Self {
        Self::new()
    }
}
