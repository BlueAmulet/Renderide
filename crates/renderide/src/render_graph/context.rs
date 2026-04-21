//! Context types passed to each pass's recording method for one encoder slice.
//!
//! Four context types correspond to the four pass kinds in [`super::pass::PassNode`]:
//! - [`RasterPassCtx`] — graph has already opened `wgpu::RenderPass`; pass records draws.
//! - [`ComputePassCtx`] — pass receives the raw `wgpu::CommandEncoder` for compute work.
//! - [`CopyPassCtx`] — same as compute, semantically restricted to copy operations.
//! - [`CallbackCtx`] — no encoder; pass runs CPU prep, Queue writes, and blackboard mutations.
//!
//! [`PostSubmitContext`] is shared across all pass kinds for post-submit hooks.
//!
//! ## Lifetime parameters
//!
//! Contexts use up to three lifetime parameters:
//! - `'a` — immutable GPU handles (device, limits, queue, graph resources, views).
//! - `'encoder` — mutable encoder borrow (only compute/copy contexts).
//! - `'frame` — mutable scene/backend frame params borrow.
//!
//! Multi-view graph execution creates a **separate** encoder (and thus a separate context) for
//! frame-global work vs each view; passes never share one encoder across those slices.

use std::collections::HashSet;
use std::sync::Arc;

use crate::gpu::GpuLimits;

use super::blackboard::Blackboard;
use super::frame_params::FrameRenderParams;
use super::resources::{
    BufferHandle, ImportedBufferHandle, ImportedTextureHandle, TextureHandle, TextureResourceHandle,
};
use super::transient_pool::TransientPool;

// ─────────────────────────────────────────────────────────────────────────────
// Resolved resource types
// ─────────────────────────────────────────────────────────────────────────────

/// Resolved transient texture for one graph execution scope.
#[derive(Clone, Debug)]
pub struct ResolvedGraphTexture {
    /// Transient pool entry id.
    pub pool_id: usize,
    /// Compiler-assigned alias slot.
    pub physical_slot: usize,
    /// Texture handle.
    pub texture: wgpu::Texture,
    /// Default texture view.
    pub view: wgpu::TextureView,
    /// Per-layer D2 views for array textures.
    pub layer_views: Vec<wgpu::TextureView>,
}

impl ResolvedGraphTexture {
    /// Builds a [`wgpu::TextureView`] with [`wgpu::TextureViewDimension::D2Array`] for WGSL
    /// `texture_2d_array` bindings that expect a D2Array view.
    ///
    /// The transient pool's default [`Self::view`] uses [`wgpu::TextureViewDimension::D2`] when the
    /// backing texture has a single array layer (mono desktop). Passing that view to bindings that
    /// expect a 2D array fails wgpu validation.
    ///
    /// `multiview_stereo` must match [`FrameRenderParams::multiview_stereo`] so the view spans
    /// one layer (mono) or two (stereo).
    pub(crate) fn view_for_sampled_2d_array(&self, multiview_stereo: bool) -> wgpu::TextureView {
        let layers_in_texture = self.texture.size().depth_or_array_layers.max(1);
        let array_layer_count = if multiview_stereo {
            2.min(layers_in_texture)
        } else {
            1
        };
        self.texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("sampled_2d_array"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            array_layer_count: Some(array_layer_count),
            ..Default::default()
        })
    }
}

/// Resolved transient buffer for one graph execution scope.
#[derive(Clone, Debug)]
pub struct ResolvedGraphBuffer {
    /// Transient pool entry id.
    pub pool_id: usize,
    /// Compiler-assigned alias slot.
    pub physical_slot: usize,
    /// Buffer handle.
    pub buffer: wgpu::Buffer,
    /// Buffer size in bytes.
    pub size: u64,
}

/// Imported texture resolved from the current frame target or backend history.
#[derive(Clone, Debug)]
pub struct ResolvedImportedTexture {
    /// Texture view available to graph-owned pass descriptors.
    pub view: wgpu::TextureView,
}

/// Imported buffer resolved from backend frame resources or external state.
#[derive(Clone, Debug)]
pub struct ResolvedImportedBuffer {
    /// Buffer handle.
    pub buffer: wgpu::Buffer,
}

// ─────────────────────────────────────────────────────────────────────────────
// Resolved resource lookup table
// ─────────────────────────────────────────────────────────────────────────────

/// Execute-time resource lookup table built by [`super::compiled::CompiledRenderGraph`].
#[derive(Debug, Default)]
pub struct GraphResolvedResources {
    transient_textures: Vec<Option<ResolvedGraphTexture>>,
    transient_buffers: Vec<Option<ResolvedGraphBuffer>>,
    imported_textures: Vec<Option<ResolvedImportedTexture>>,
    imported_buffers: Vec<Option<ResolvedImportedBuffer>>,
}

impl GraphResolvedResources {
    /// Creates a lookup table with fixed handle capacities.
    pub fn with_capacity(
        transient_texture_count: usize,
        transient_buffer_count: usize,
        imported_texture_count: usize,
        imported_buffer_count: usize,
    ) -> Self {
        Self {
            transient_textures: std::iter::repeat_with(|| None)
                .take(transient_texture_count)
                .collect(),
            transient_buffers: std::iter::repeat_with(|| None)
                .take(transient_buffer_count)
                .collect(),
            imported_textures: std::iter::repeat_with(|| None)
                .take(imported_texture_count)
                .collect(),
            imported_buffers: std::iter::repeat_with(|| None)
                .take(imported_buffer_count)
                .collect(),
        }
    }

    /// Inserts a transient texture.
    pub fn set_transient_texture(&mut self, handle: TextureHandle, texture: ResolvedGraphTexture) {
        if let Some(slot) = self.transient_textures.get_mut(handle.index()) {
            *slot = Some(texture);
        }
    }

    /// Inserts a transient buffer.
    pub fn set_transient_buffer(&mut self, handle: BufferHandle, buffer: ResolvedGraphBuffer) {
        if let Some(slot) = self.transient_buffers.get_mut(handle.index()) {
            *slot = Some(buffer);
        }
    }

    /// Inserts an imported texture.
    pub fn set_imported_texture(
        &mut self,
        handle: ImportedTextureHandle,
        texture: ResolvedImportedTexture,
    ) {
        if let Some(slot) = self.imported_textures.get_mut(handle.index()) {
            *slot = Some(texture);
        }
    }

    /// Inserts an imported buffer.
    pub fn set_imported_buffer(
        &mut self,
        handle: ImportedBufferHandle,
        buffer: ResolvedImportedBuffer,
    ) {
        if let Some(slot) = self.imported_buffers.get_mut(handle.index()) {
            *slot = Some(buffer);
        }
    }

    /// Looks up a transient texture.
    pub fn transient_texture(&self, handle: TextureHandle) -> Option<&ResolvedGraphTexture> {
        self.transient_textures.get(handle.index())?.as_ref()
    }

    /// Looks up a transient buffer.
    pub fn transient_buffer(&self, handle: BufferHandle) -> Option<&ResolvedGraphBuffer> {
        self.transient_buffers.get(handle.index())?.as_ref()
    }

    /// Looks up an imported texture.
    pub fn imported_texture(
        &self,
        handle: ImportedTextureHandle,
    ) -> Option<&ResolvedImportedTexture> {
        self.imported_textures.get(handle.index())?.as_ref()
    }

    /// Looks up an imported buffer.
    pub fn imported_buffer(&self, handle: ImportedBufferHandle) -> Option<&ResolvedImportedBuffer> {
        self.imported_buffers.get(handle.index())?.as_ref()
    }

    pub(crate) fn texture_view(&self, handle: TextureResourceHandle) -> Option<&wgpu::TextureView> {
        match handle {
            TextureResourceHandle::Transient(handle) => Some(&self.transient_texture(handle)?.view),
            TextureResourceHandle::Imported(handle) => Some(&self.imported_texture(handle)?.view),
        }
    }

    pub(crate) fn release_to_pool(&self, pool: &mut TransientPool) {
        let mut texture_ids = HashSet::new();
        for texture in self.transient_textures.iter().flatten() {
            if texture_ids.insert(texture.pool_id) {
                pool.release_texture(texture.pool_id);
            }
        }
        let mut buffer_ids = HashSet::new();
        for buffer in self.transient_buffers.iter().flatten() {
            if buffer_ids.insert(buffer.pool_id) {
                pool.release_buffer(buffer.pool_id);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// New typed context structs (Phases 1–2)
// ─────────────────────────────────────────────────────────────────────────────

/// Context for [`super::pass::RasterPass::record`].
///
/// The graph has already opened a [`wgpu::RenderPass`] from the compiled attachment template;
/// the pass records draw commands into it. No encoder is exposed since the encoder is borrowed
/// by the open render pass.
pub struct RasterPassCtx<'a, 'frame> {
    /// WGPU device.
    pub device: &'a wgpu::Device,
    /// Effective limits for this frame.
    pub gpu_limits: &'a GpuLimits,
    /// Submission queue (internally synchronized by wgpu).
    pub queue: &'a Arc<wgpu::Queue>,
    /// Swapchain view when the frame acquired the surface; [`None`] for offscreen-only graphs.
    pub backbuffer: Option<&'a wgpu::TextureView>,
    /// Depth attachment for the main forward pass.
    pub depth_view: Option<&'a wgpu::TextureView>,
    /// Scene + backend frame params for this view.
    pub frame: Option<&'frame mut FrameRenderParams<'a>>,
    /// Typed graph resources resolved for this execution scope.
    pub graph_resources: Option<&'a GraphResolvedResources>,
    /// Per-scope typed blackboard (read/write; populated by prior callback passes this scope).
    pub blackboard: &'frame mut Blackboard,
}

/// Context for [`super::pass::ComputePass::record`].
///
/// The pass receives the raw [`wgpu::CommandEncoder`] and dispatches compute workgroups or
/// issues other encoder-level commands.
pub struct ComputePassCtx<'a, 'encoder, 'frame> {
    /// WGPU device.
    pub device: &'a wgpu::Device,
    /// Effective limits for this frame.
    pub gpu_limits: &'a GpuLimits,
    /// Submission queue.
    pub queue: &'a Arc<wgpu::Queue>,
    /// Active command encoder for this recording slice.
    pub encoder: &'encoder mut wgpu::CommandEncoder,
    /// Depth attachment for the main forward pass (often needed by compute passes that
    /// read or copy the depth buffer).
    pub depth_view: Option<&'a wgpu::TextureView>,
    /// Scene + backend frame params for this view.
    pub frame: Option<&'frame mut FrameRenderParams<'a>>,
    /// Typed graph resources resolved for this execution scope.
    pub graph_resources: Option<&'a GraphResolvedResources>,
    /// Per-scope typed blackboard (read/write; populated by prior callback passes this scope).
    pub blackboard: &'frame mut Blackboard,
}

/// Context for [`super::pass::CopyPass::record`].
///
/// Structurally identical to [`ComputePassCtx`]; separated by type to distinguish copy-only
/// intent from arbitrary compute dispatch.
pub type CopyPassCtx<'a, 'encoder, 'frame> = ComputePassCtx<'a, 'encoder, 'frame>;

/// Context for [`super::pass::CallbackPass::run`].
///
/// No encoder is provided. The pass runs as a CPU callback, which may issue
/// [`wgpu::Queue::write_buffer`] calls via `queue` and mutate `blackboard`.
pub struct CallbackCtx<'a, 'frame> {
    /// WGPU device.
    pub device: &'a wgpu::Device,
    /// Effective limits for this frame.
    pub gpu_limits: &'a GpuLimits,
    /// Submission queue for `write_buffer` calls.
    pub queue: &'a Arc<wgpu::Queue>,
    /// Scene + backend frame params for this view (or first view for frame-global callbacks).
    pub frame: Option<&'frame mut FrameRenderParams<'a>>,
    /// Typed graph resources resolved for this execution scope.
    pub graph_resources: Option<&'a GraphResolvedResources>,
    /// Per-scope typed blackboard (read/write; the primary output of callback passes).
    pub blackboard: &'frame mut Blackboard,
}

// ─────────────────────────────────────────────────────────────────────────────
// Post-submit context (unchanged)
// ─────────────────────────────────────────────────────────────────────────────

/// Context passed to `post_submit` after a per-view or frame-global submit.
///
/// Runs on the CPU **after** [`wgpu::Queue::submit`] so passes can start `map_async` work on
/// buffers they wrote this frame (e.g. Hi-Z readback staging rotation).
pub struct PostSubmitContext<'a> {
    /// WGPU device for `map_async` and device polling.
    pub device: &'a wgpu::Device,
    /// Hi-Z readback and temporal bookkeeping for this view after submit.
    pub occlusion: &'a mut crate::backend::OcclusionSystem,
    /// Which occlusion view this submit covered.
    pub occlusion_view: super::OcclusionViewId,
    /// Host camera snapshot for the view.
    pub host_camera: super::HostCameraFrame,
}

// ─────────────────────────────────────────────────────────────────────────────
// Legacy context type aliases (kept for test compatibility; callers should migrate)
// ─────────────────────────────────────────────────────────────────────────────

/// Legacy encoder-driven pass context. Prefer [`ComputePassCtx`] for new code.
///
/// Kept as an alias for incremental migration of tests and helper functions that reference the
/// old type. Will be removed when all callers are updated.
pub struct RenderPassContext<'a, 'encoder, 'frame> {
    /// WGPU device.
    pub device: &'a wgpu::Device,
    /// Effective limits for this frame.
    pub gpu_limits: &'a GpuLimits,
    /// Submission queue.
    pub queue: &'a Arc<wgpu::Queue>,
    /// Active command encoder.
    pub encoder: &'encoder mut wgpu::CommandEncoder,
    /// Swapchain view when this frame acquired the surface.
    pub backbuffer: Option<&'a wgpu::TextureView>,
    /// Depth attachment for the main forward pass.
    pub depth_view: Option<&'a wgpu::TextureView>,
    /// Scene + backend frame params.
    pub frame: Option<&'frame mut FrameRenderParams<'a>>,
    /// Typed graph resources resolved for this execution scope.
    pub graph_resources: Option<&'a GraphResolvedResources>,
}

/// Legacy graph-raster pass context. Prefer [`RasterPassCtx`] for new code.
///
/// Kept for incremental migration of tests and setup/compose pass helpers.
pub struct GraphRasterPassContext<'a, 'frame> {
    /// WGPU device.
    pub device: &'a wgpu::Device,
    /// Effective limits for this frame.
    pub gpu_limits: &'a GpuLimits,
    /// Submission queue.
    pub queue: &'a Arc<wgpu::Queue>,
    /// Swapchain view when this frame acquired the surface.
    pub backbuffer: Option<&'a wgpu::TextureView>,
    /// Depth attachment for the main forward pass.
    pub depth_view: Option<&'a wgpu::TextureView>,
    /// Scene + backend frame params.
    pub frame: Option<&'frame mut FrameRenderParams<'a>>,
    /// Typed graph resources resolved for this execution scope.
    pub graph_resources: Option<&'a GraphResolvedResources>,
}
