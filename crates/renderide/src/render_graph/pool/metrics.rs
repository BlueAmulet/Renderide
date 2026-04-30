//! Counters reported by [`super::TransientPool`] for diagnostics and HUD readout.

/// Pool statistics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TransientPoolMetrics {
    /// Texture reuse hits.
    pub texture_hits: usize,
    /// Texture allocation misses.
    pub texture_misses: usize,
    /// Buffer reuse hits.
    pub buffer_hits: usize,
    /// Buffer allocation misses.
    pub buffer_misses: usize,
    /// Pool texture slots that currently hold GPU [`wgpu::Texture`] handles (after GC drops dead entries).
    pub retained_textures: usize,
    /// Pool buffer slots that currently hold GPU [`wgpu::Buffer`] handles.
    pub retained_buffers: usize,
}
