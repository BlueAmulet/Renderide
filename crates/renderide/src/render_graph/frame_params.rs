//! Per-frame parameters shared across render graph passes (scene, backend, surface state).

use crate::backend::RenderBackend;
use crate::scene::SceneCoordinator;

/// Data passes need beyond raw GPU handles: host scene, backend pools, and main-surface formats.
pub struct FrameRenderParams<'a> {
    /// World caches and mesh renderables after [`SceneCoordinator::flush_world_caches`].
    pub scene: &'a SceneCoordinator,
    /// GPU pools, materials, and deform scratch buffers.
    pub backend: &'a mut RenderBackend,
    /// Depth attachment for the main forward pass.
    pub depth_view: &'a wgpu::TextureView,
    /// Swapchain / main color format.
    pub surface_format: wgpu::TextureFormat,
    /// Main surface extent in pixels (`width`, `height`) for projection.
    pub viewport_px: (u32, u32),
}
