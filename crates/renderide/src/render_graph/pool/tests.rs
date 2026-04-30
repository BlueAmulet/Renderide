//! Unit tests for [`super::TransientPool`] keying, free-list reuse, GC, and eviction.

use super::*;
use crate::render_graph::resources::{BufferSizePolicy, TransientExtent};

fn tex_key(usage: wgpu::TextureUsages, width: u32, height: u32) -> TextureKey {
    TextureKey {
        format: wgpu::TextureFormat::Rgba8Unorm,
        extent: TransientExtent::Custom { width, height },
        mip_levels: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        array_layers: 1,
        usage_bits: u64::from(usage.bits()),
    }
}

fn buf_key(usage: wgpu::BufferUsages) -> BufferKey {
    BufferKey {
        size_policy: BufferSizePolicy::Fixed(256),
        usage_bits: u64::from(usage.bits()),
    }
}

#[test]
fn matching_descriptor_reuses_physical_texture() {
    let mut pool = TransientPool::new();
    let key = tex_key(wgpu::TextureUsages::RENDER_ATTACHMENT, 64, 64);
    let first = pool.acquire_texture(key);
    pool.release_texture(first);
    let second = pool.acquire_texture(key);
    assert_eq!(first, second);
    assert_eq!(pool.metrics().texture_hits, 1);
}

#[test]
fn different_usages_produce_different_slots() {
    let mut pool = TransientPool::new();
    let a = pool.acquire_texture(tex_key(wgpu::TextureUsages::RENDER_ATTACHMENT, 64, 64));
    pool.release_texture(a);
    let b = pool.acquire_texture(tex_key(wgpu::TextureUsages::TEXTURE_BINDING, 64, 64));
    assert_ne!(a, b);
}

#[test]
fn gc_tick_releases_after_threshold_generations() {
    let mut pool = TransientPool::new();
    let key = tex_key(wgpu::TextureUsages::RENDER_ATTACHMENT, 64, 64);
    let id = pool.acquire_texture(key);
    pool.release_texture(id);
    pool.begin_generation();
    pool.begin_generation();
    pool.gc_tick(0);
    let next = pool.acquire_texture(key);
    assert_ne!(id, next);
}

#[test]
fn viewport_resize_invalidates_backbuffer_sized_slots() {
    let mut pool = TransientPool::new();
    let a_key = tex_key(wgpu::TextureUsages::RENDER_ATTACHMENT, 64, 64);
    let b_key = tex_key(wgpu::TextureUsages::RENDER_ATTACHMENT, 128, 64);
    let a = pool.acquire_texture(a_key);
    pool.release_texture(a);
    let b = pool.acquire_texture(b_key);
    assert_ne!(a, b);
}

#[test]
fn pool_metrics_count_hits_and_misses() {
    let mut pool = TransientPool::new();
    let t_key = tex_key(wgpu::TextureUsages::RENDER_ATTACHMENT, 64, 64);
    let b_key = buf_key(wgpu::BufferUsages::COPY_DST);
    let t = pool.acquire_texture(t_key);
    pool.release_texture(t);
    let _ = pool.acquire_texture(t_key);
    let b = pool.acquire_buffer(b_key);
    pool.release_buffer(b);
    let _ = pool.acquire_buffer(b_key);
    let metrics = pool.metrics();
    assert_eq!(metrics.texture_misses, 1);
    assert_eq!(metrics.texture_hits, 1);
    assert_eq!(metrics.buffer_misses, 1);
    assert_eq!(metrics.buffer_hits, 1);
}

#[test]
fn gc_drops_gpu_texture_slots_for_stale_msaa_keys() {
    let mut pool = TransientPool::new();
    let base = tex_key(wgpu::TextureUsages::RENDER_ATTACHMENT, 64, 64);
    let mut k1 = base;
    k1.sample_count = 1;
    let mut k4 = base;
    k4.sample_count = 4;
    let mut k8 = base;
    k8.sample_count = 8;

    let id1 = pool.acquire_texture(k1);
    let id4 = pool.acquire_texture(k4);
    let id8 = pool.acquire_texture(k8);
    pool.release_texture(id1);
    pool.release_texture(id4);
    pool.release_texture(id8);

    for _ in 0..4 {
        pool.begin_generation();
    }
    pool.gc_tick(0);

    let m = pool.metrics();
    assert_eq!(
        m.retained_textures, 0,
        "dead slots should not count as retained GPU textures"
    );

    let fresh = pool.acquire_texture(k1);
    assert_ne!(
        fresh, id1,
        "after GC, acquire should allocate a new slot when the free list was pruned"
    );
}
