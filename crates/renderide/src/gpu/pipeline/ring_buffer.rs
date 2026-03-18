//! Ring buffers for batched uniform data.

use std::cell::RefCell;

use nalgebra::Matrix4;

use super::core::{
    matrix4_to_wgsl_column_major, MAX_BLENDSHAPE_WEIGHTS, NUM_FRAMES_IN_FLIGHT,
    SKINNED_SLOTS_PER_FRAME, SKINNED_SLOT_STRIDE, SLOTS_PER_FRAME, UNIFORM_ALIGNMENT,
};
use super::uniforms::{OverlayStencilUniforms, SkinnedUniforms, Uniforms};

/// Ring buffer for batched uniform data. Supports arbitrary draw counts by chunking.
/// No panic or silent drop when draws exceed a single region.
pub(crate) struct UniformRingBuffer {
    pub buffer: wgpu::Buffer,
    /// Reusable scratch buffer for uploads. Avoids per-chunk heap allocation.
    scratch: RefCell<Vec<u8>>,
}

impl UniformRingBuffer {
    pub fn new(device: &wgpu::Device, label: &str) -> Self {
        let size = (NUM_FRAMES_IN_FLIGHT * SLOTS_PER_FRAME) as u64 * UNIFORM_ALIGNMENT;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let scratch = RefCell::new(Vec::with_capacity(
            (SLOTS_PER_FRAME as u64 * UNIFORM_ALIGNMENT) as usize,
        ));
        Self { buffer, scratch }
    }

    /// Uploads uniforms to the ring buffer, chunking if needed. No limit panic.
    pub fn upload(
        &self,
        queue: &wgpu::Queue,
        mvp_models: &[(Matrix4<f32>, Matrix4<f32>)],
        frame_index: u64,
    ) {
        if mvp_models.is_empty() {
            return;
        }
        let uniform_size = std::mem::size_of::<Uniforms>();
        let region_base = (frame_index as usize % NUM_FRAMES_IN_FLIGHT) * SLOTS_PER_FRAME;
        let mut scratch = self.scratch.borrow_mut();
        for (chunk_idx, chunk) in mvp_models.chunks(SLOTS_PER_FRAME).enumerate() {
            let region = (region_base + chunk_idx) % NUM_FRAMES_IN_FLIGHT;
            let buffer_offset = (region * SLOTS_PER_FRAME) as u64 * UNIFORM_ALIGNMENT;
            let need_len = (chunk.len() as u64 * UNIFORM_ALIGNMENT) as usize;
            scratch.resize(need_len, 0);
            let aligned = &mut scratch[..need_len];
            for (i, (mvp, model)) in chunk.iter().enumerate() {
                let u = Uniforms {
                    mvp: matrix4_to_wgsl_column_major(mvp),
                    model: matrix4_to_wgsl_column_major(model),
                };
                let offset = (i as u64 * UNIFORM_ALIGNMENT) as usize;
                let bytes: &[u8] = bytemuck::bytes_of(&u);
                aligned[offset..offset + uniform_size].copy_from_slice(bytes);
            }
            queue.write_buffer(&self.buffer, buffer_offset, &aligned);
        }
    }

    /// Computes dynamic offset for draw index `i` given `frame_index`.
    pub fn dynamic_offset(&self, batch_index: u32, frame_index: u64) -> u32 {
        let i = batch_index as usize;
        let chunk_idx = i / SLOTS_PER_FRAME;
        let slot_in_chunk = i % SLOTS_PER_FRAME;
        let region_base = (frame_index as usize % NUM_FRAMES_IN_FLIGHT) * SLOTS_PER_FRAME;
        let region = (region_base + chunk_idx) % NUM_FRAMES_IN_FLIGHT;
        let slot = region * SLOTS_PER_FRAME + slot_in_chunk;
        (slot as u64 * UNIFORM_ALIGNMENT) as u32
    }
}

/// Ring buffer for batched skinned uniform data (mvp + bone_matrices[256] per slot).
pub(crate) struct SkinnedUniformRingBuffer {
    /// GPU buffer backing the ring.
    pub buffer: wgpu::Buffer,
    /// Reusable scratch buffer for uploads. Avoids per-chunk heap allocation.
    scratch: RefCell<Vec<u8>>,
}

impl SkinnedUniformRingBuffer {
    /// Creates a new ring buffer sized for `NUM_FRAMES_IN_FLIGHT * SKINNED_SLOTS_PER_FRAME` slots.
    pub fn new(device: &wgpu::Device, label: &str) -> Self {
        let size =
            (NUM_FRAMES_IN_FLIGHT * SKINNED_SLOTS_PER_FRAME) as u64 * SKINNED_SLOT_STRIDE;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let scratch = RefCell::new(Vec::with_capacity(
            (SKINNED_SLOTS_PER_FRAME as u64 * SKINNED_SLOT_STRIDE) as usize,
        ));
        Self { buffer, scratch }
    }

    /// Uploads skinned uniforms to the ring buffer, chunking if needed. No limit panic.
    pub fn upload(
        &self,
        queue: &wgpu::Queue,
        items: &[(Matrix4<f32>, &[[[f32; 4]; 4]], Option<&[f32]>, u32)],
        frame_index: u64,
    ) {
        if items.is_empty() {
            return;
        }
        let uniform_size = std::mem::size_of::<SkinnedUniforms>();
        let region_base =
            (frame_index as usize % NUM_FRAMES_IN_FLIGHT) * SKINNED_SLOTS_PER_FRAME;
        let mut scratch = self.scratch.borrow_mut();
        for (chunk_idx, chunk) in items.chunks(SKINNED_SLOTS_PER_FRAME).enumerate() {
            let region = (region_base + chunk_idx) % NUM_FRAMES_IN_FLIGHT;
            let buffer_offset =
                (region * SKINNED_SLOTS_PER_FRAME) as u64 * SKINNED_SLOT_STRIDE;
            let need_len = (chunk.len() as u64 * SKINNED_SLOT_STRIDE) as usize;
            scratch.resize(need_len, 0);
            let aligned = &mut scratch[..need_len];
            for (i, (mvp, bone_matrices, blendshape_weights, num_vertices)) in chunk.iter().enumerate()
            {
                let mut u = SkinnedUniforms {
                    mvp: matrix4_to_wgsl_column_major(mvp),
                    bone_matrices: [[[0.0; 4]; 4]; 256],
                    num_blendshapes: 0,
                    num_vertices: *num_vertices,
                    _pad: [0, 0],
                    blendshape_weights: [[0.0; 4]; 32],
                };
                let n = bone_matrices.len().min(256);
                u.bone_matrices[..n].copy_from_slice(&bone_matrices[..n]);
                if let Some(weights) = blendshape_weights {
                    let count = weights.len().min(MAX_BLENDSHAPE_WEIGHTS);
                    u.num_blendshapes = count as u32;
                    for (i, &w) in weights.iter().take(MAX_BLENDSHAPE_WEIGHTS).enumerate() {
                        u.blendshape_weights[i / 4][i % 4] = w;
                    }
                }
                let offset = (i as u64 * SKINNED_SLOT_STRIDE) as usize;
                let bytes: &[u8] = bytemuck::bytes_of(&u);
                aligned[offset..offset + uniform_size].copy_from_slice(bytes);
            }
            queue.write_buffer(&self.buffer, buffer_offset, &aligned);
        }
    }

    /// Computes dynamic offset for draw index `i` given `frame_index`.
    pub fn dynamic_offset(&self, batch_index: u32, frame_index: u64) -> u32 {
        let i = batch_index as usize;
        let chunk_idx = i / SKINNED_SLOTS_PER_FRAME;
        let slot_in_chunk = i % SKINNED_SLOTS_PER_FRAME;
        let region_base =
            (frame_index as usize % NUM_FRAMES_IN_FLIGHT) * SKINNED_SLOTS_PER_FRAME;
        let region = (region_base + chunk_idx) % NUM_FRAMES_IN_FLIGHT;
        let slot = region * SKINNED_SLOTS_PER_FRAME + slot_in_chunk;
        (slot as u64 * SKINNED_SLOT_STRIDE) as u32
    }
}

/// Ring buffer for overlay stencil uniforms (mvp + model + clip_rect per slot).
pub(crate) struct OverlayStencilUniformRingBuffer {
    pub buffer: wgpu::Buffer,
    /// Reusable scratch buffer for uploads. Avoids per-chunk heap allocation.
    scratch: RefCell<Vec<u8>>,
}

impl OverlayStencilUniformRingBuffer {
    pub fn new(device: &wgpu::Device, label: &str) -> Self {
        let size = (NUM_FRAMES_IN_FLIGHT * SLOTS_PER_FRAME) as u64 * UNIFORM_ALIGNMENT;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let scratch = RefCell::new(Vec::with_capacity(
            (SLOTS_PER_FRAME as u64 * UNIFORM_ALIGNMENT) as usize,
        ));
        Self { buffer, scratch }
    }

    pub fn upload(
        &self,
        queue: &wgpu::Queue,
        items: &[(Matrix4<f32>, Matrix4<f32>, Option<[f32; 4]>)],
        frame_index: u64,
    ) {
        if items.is_empty() {
            return;
        }
        let uniform_size = std::mem::size_of::<OverlayStencilUniforms>();
        let region_base = (frame_index as usize % NUM_FRAMES_IN_FLIGHT) * SLOTS_PER_FRAME;
        let mut scratch = self.scratch.borrow_mut();
        for (chunk_idx, chunk) in items.chunks(SLOTS_PER_FRAME).enumerate() {
            let region = (region_base + chunk_idx) % NUM_FRAMES_IN_FLIGHT;
            let buffer_offset = (region * SLOTS_PER_FRAME) as u64 * UNIFORM_ALIGNMENT;
            let need_len = (chunk.len() as u64 * UNIFORM_ALIGNMENT) as usize;
            scratch.resize(need_len, 0);
            let aligned = &mut scratch[..need_len];
            for (i, (mvp, model, clip_rect)) in chunk.iter().enumerate() {
                let clip = clip_rect.unwrap_or([0.0, 0.0, 0.0, 0.0]);
                let u = OverlayStencilUniforms {
                    mvp: matrix4_to_wgsl_column_major(mvp),
                    model: matrix4_to_wgsl_column_major(model),
                    clip_rect: clip,
                    _pad: [0.0; 16],
                };
                let offset = (i as u64 * UNIFORM_ALIGNMENT) as usize;
                let bytes: &[u8] = bytemuck::bytes_of(&u);
                aligned[offset..offset + uniform_size].copy_from_slice(bytes);
            }
            queue.write_buffer(&self.buffer, buffer_offset, &aligned);
        }
    }

    pub fn dynamic_offset(&self, batch_index: u32, frame_index: u64) -> u32 {
        let i = batch_index as usize;
        let chunk_idx = i / SLOTS_PER_FRAME;
        let slot_in_chunk = i % SLOTS_PER_FRAME;
        let region_base = (frame_index as usize % NUM_FRAMES_IN_FLIGHT) * SLOTS_PER_FRAME;
        let region = (region_base + chunk_idx) % NUM_FRAMES_IN_FLIGHT;
        let slot = region * SLOTS_PER_FRAME + slot_in_chunk;
        (slot as u64 * UNIFORM_ALIGNMENT) as u32
    }
}
