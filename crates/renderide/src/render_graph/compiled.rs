//! Compiled DAG: immutable pass order and per-frame execution.

use crate::backend::BackendGraphAccess;
use crate::gpu::{GpuContext, GpuLimits};
use crate::scene::SceneCoordinator;

use super::blackboard::Blackboard;
use super::error::GraphExecuteError;
use super::frame_params::FrameViewClear;
#[cfg(test)]
use super::pass::{PassKind, PassMergeHint};
use super::pass::{PassNode, RenderPassTemplate};
use super::resources::{
    ImportedBufferDecl, ImportedTextureDecl, ResourceAccess, TransientBufferDesc,
    TransientSubresourceDesc, TransientTextureDesc,
};
use super::schedule::FrameSchedule;
use crate::camera::{
    HostCameraFrame, ViewId, camera_state_motion_blur, camera_state_post_processing,
    camera_state_screen_space_reflections,
};
use crate::shared::{CameraRenderParameters, CameraState};

/// Single-view color + depth for secondary cameras rendering to a host [`crate::gpu_pools::GpuRenderTexture`].
pub struct ExternalOffscreenTargets<'a> {
    /// Host render-texture asset id for `color_view` (used to suppress self-sampling during this pass).
    pub render_texture_asset_id: i32,
    /// Color attachment (`Rgba16Float` for Unity `ARGBHalf` parity).
    pub color_view: &'a wgpu::TextureView,
    /// Depth texture backing `depth_view`.
    pub depth_texture: &'a wgpu::Texture,
    /// Depth-stencil view for the offscreen pass.
    pub depth_view: &'a wgpu::TextureView,
    /// Color/depth attachment extent in physical pixels.
    pub extent_px: (u32, u32),
    /// Color attachment format (must match pipeline targets).
    pub color_format: wgpu::TextureFormat,
}

/// Pre-acquired 2-layer color + depth targets for OpenXR multiview (no window swapchain acquire).
pub struct ExternalFrameTargets<'a> {
    /// `D2Array` color view (`array_layer_count` = 2).
    pub color_view: &'a wgpu::TextureView,
    /// Backing `D2Array` depth texture for copy/snapshot passes.
    pub depth_texture: &'a wgpu::Texture,
    /// `D2Array` depth view (`array_layer_count` = 2).
    pub depth_view: &'a wgpu::TextureView,
    /// Pixel extent per eye (`width`, `height`).
    pub extent_px: (u32, u32),
    /// Color format (must match pipeline targets).
    pub surface_format: wgpu::TextureFormat,
}

/// Where a multi-view frame writes color/depth.
pub enum FrameViewTarget<'a> {
    /// Main window swapchain (acquire + present).
    Swapchain,
    /// OpenXR stereo multiview (pre-acquired array targets).
    ExternalMultiview(ExternalFrameTargets<'a>),
    /// Secondary camera to a host render texture.
    OffscreenRt(ExternalOffscreenTargets<'a>),
}

/// Post-processing permissions requested by a single view.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ViewPostProcessing {
    /// `true` when this view should run the post-processing stack.
    pub enabled: bool,
    /// `true` when this view allows screen-space reflections to record.
    pub screen_space_reflections: bool,
    /// `true` when this view allows motion blur to record.
    pub motion_blur: bool,
}

impl ViewPostProcessing {
    /// Builds a view post-processing policy from decoded host camera settings.
    pub const fn new(enabled: bool, screen_space_reflections: bool, motion_blur: bool) -> Self {
        Self {
            enabled,
            screen_space_reflections: enabled && screen_space_reflections,
            motion_blur: enabled && motion_blur,
        }
    }

    /// Primary/HMD view policy: allow the renderer-global post-processing stack to run.
    pub const fn primary_view() -> Self {
        Self::new(true, true, true)
    }

    /// Reflection-probe and other raw-capture policy: bypass all post-processing effects.
    pub const fn disabled() -> Self {
        Self::new(false, false, false)
    }

    /// Converts host camera readback parameters into a view post-processing policy.
    ///
    /// Camera render tasks explicitly disable motion blur to match the host camera-capture path.
    pub fn from_camera_render_parameters(parameters: &CameraRenderParameters) -> Self {
        Self::new(
            parameters.post_processing,
            parameters.screen_space_reflections,
            false,
        )
    }

    /// Converts secondary render-texture camera state flags into a view post-processing policy.
    pub fn from_camera_state(state: &CameraState) -> Self {
        Self::new(
            camera_state_post_processing(state.flags),
            camera_state_screen_space_reflections(state.flags),
            camera_state_motion_blur(state.flags),
        )
    }

    /// Returns `true` when this view should run the post-processing stack.
    pub const fn is_enabled(self) -> bool {
        self.enabled
    }
}

impl Default for ViewPostProcessing {
    fn default() -> Self {
        Self::primary_view()
    }
}

/// One view to render in a multi-view frame.
pub struct FrameView<'a> {
    /// Stable logical identity for view-scoped resources and temporal state.
    pub view_id: ViewId,
    /// Clip planes, FOV, and matrix overrides for this view.
    pub host_camera: HostCameraFrame,
    /// Color/depth destination.
    pub target: FrameViewTarget<'a>,
    /// Background clear/skybox behavior for this view.
    pub clear: FrameViewClear,
    /// Post-processing permissions for this view.
    pub post_processing: ViewPostProcessing,
    /// Resource layout hints required by backend-specific pre-record preparation.
    pub resource_hints: FrameViewResourceHints,
    /// Caller-seeded per-view graph state.
    pub initial_blackboard: Blackboard,
}

/// Resource layout hints supplied by view preparation before graph execution.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameViewResourceHints {
    /// Whether passes in this view require a scene-depth snapshot resource.
    pub needs_depth_snapshot: bool,
    /// Whether passes in this view require a scene-color snapshot resource.
    pub needs_color_snapshot: bool,
}

/// Borrows shared across frame-global and per-view [`CompiledRenderGraph::execute_multi_view`] passes.
pub(super) struct MultiViewExecutionContext<'a, 'backend> {
    /// GPU context (surface, swapchain, submits).
    pub(super) gpu: &'a mut GpuContext,
    /// Scene after cache flush.
    pub(super) scene: &'a SceneCoordinator,
    /// Narrow graph-facing backend access packet.
    pub(super) backend: &'a mut BackendGraphAccess<'backend>,
    /// Device for encoders and pipeline state.
    pub(super) device: &'a wgpu::Device,
    /// Limits for pass contexts.
    pub(super) gpu_limits: &'a GpuLimits,
    /// Swapchain color view when a view targets the main window.
    pub(super) backbuffer_view_holder: &'a Option<wgpu::TextureView>,
}

impl FrameViewTarget<'_> {
    /// `true` when this target renders to a 2-layer multiview color attachment.
    pub fn is_multiview_target(&self) -> bool {
        matches!(self, FrameViewTarget::ExternalMultiview(_))
    }

    /// Viewport extent in pixels for this target.
    pub fn extent_px(&self, gpu: &GpuContext) -> (u32, u32) {
        match self {
            FrameViewTarget::ExternalMultiview(ext) => ext.extent_px,
            FrameViewTarget::OffscreenRt(ext) => ext.extent_px,
            FrameViewTarget::Swapchain => gpu.surface_extent_px(),
        }
    }

    /// Depth attachment format for this target. Lazily allocates the swapchain depth target if
    /// needed (the `Swapchain` case requires `&mut`).
    pub fn depth_format(
        &self,
        gpu: &mut GpuContext,
    ) -> Result<wgpu::TextureFormat, GraphExecuteError> {
        match self {
            FrameViewTarget::ExternalMultiview(ext) => Ok(ext.depth_texture.format()),
            FrameViewTarget::OffscreenRt(ext) => Ok(ext.depth_texture.format()),
            FrameViewTarget::Swapchain => {
                let (depth_tex, _) = gpu
                    .ensure_depth_target()
                    .map_err(GraphExecuteError::DepthTarget)?;
                Ok(depth_tex.format())
            }
        }
    }
}

impl<'a> FrameView<'a> {
    /// Stable logical identity for this view.
    pub fn view_id(&self) -> ViewId {
        self.view_id
    }

    /// `true` when this view both targets a multiview attachment AND the host camera carries stereo
    /// matrices -- i.e. the per-view record path should emit stereo clustering / multiview draws.
    ///
    /// Single source of truth; every caller that gates on "is this the stereo multiview view?"
    /// goes through this method rather than re-deriving the AND-chain.
    pub fn is_multiview_stereo_active(&self) -> bool {
        self.target.is_multiview_target() && self.host_camera.active_stereo().is_some()
    }
}

impl CompiledRenderGraph {
    /// Stores main-frame MSAA depth scratch handles used by per-view recording helpers.
    pub(crate) fn set_main_graph_msaa_transient_handles(
        &mut self,
        handles: [crate::render_graph::resources::TextureHandle; 2],
    ) {
        self.main_graph_msaa_transient_handles = Some(handles);
    }

    /// Releases any pass-local view-scoped caches for views that are no longer active.
    pub(crate) fn release_view_resources(&mut self, retired_views: &[ViewId]) {
        if retired_views.is_empty() {
            return;
        }
        for pass in &mut self.passes {
            pass.release_view_resources(retired_views);
        }
    }
}

/// Statistics emitted when building a [`CompiledRenderGraph`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CompileStats {
    /// Number of passes in the flattened schedule.
    pub pass_count: usize,
    /// Number of Kahn sweep **waves** (parallel layers) in the build-time DAG sort.
    ///
    /// Runtime execution still walks the compiled pass list in one flat order; this
    /// count is not a separate executor schedule. It is exposed in the debug HUD (with pass count)
    /// as a diagnostic and a hint for future wave-based parallel record scheduling.
    pub topo_levels: usize,
    /// Number of passes culled because their writes could not reach an import/export.
    pub culled_count: usize,
    /// Number of declared transient texture handles.
    pub transient_texture_count: usize,
    /// Number of physical transient texture slots after lifetime aliasing.
    pub transient_texture_slots: usize,
    /// Number of declared transient buffer handles.
    pub transient_buffer_count: usize,
    /// Number of physical transient buffer slots after lifetime aliasing.
    pub transient_buffer_slots: usize,
    /// Number of imported texture declarations.
    pub imported_texture_count: usize,
    /// Number of imported buffer declarations.
    pub imported_buffer_count: usize,
}

/// Inclusive pass-index lifetime for one transient resource in the retained schedule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourceLifetime {
    /// First retained pass index that touches the resource.
    pub first_pass: usize,
    /// Last retained pass index that touches the resource.
    pub last_pass: usize,
}

impl ResourceLifetime {
    /// Returns true when two lifetimes do not overlap.
    pub fn disjoint(self, other: Self) -> bool {
        self.last_pass < other.first_pass || other.last_pass < self.first_pass
    }
}

/// Compiled metadata for a transient texture handle.
#[derive(Clone, Debug)]
pub struct CompiledTextureResource {
    /// Original descriptor.
    pub desc: TransientTextureDesc,
    /// Usage union across retained pass declarations.
    pub usage: wgpu::TextureUsages,
    /// Retained-schedule lifetime.
    pub lifetime: Option<ResourceLifetime>,
    /// Physical alias slot assigned by the compiler.
    pub physical_slot: usize,
}

/// Compiled metadata for a transient buffer handle.
#[derive(Clone, Debug)]
pub struct CompiledBufferResource {
    /// Original descriptor.
    pub desc: TransientBufferDesc,
    /// Usage union across retained pass declarations.
    pub usage: wgpu::BufferUsages,
    /// Retained-schedule lifetime.
    pub lifetime: Option<ResourceLifetime>,
    /// Physical alias slot assigned by the compiler.
    pub physical_slot: usize,
}

/// Compiled setup metadata for one retained pass.
#[derive(Clone, Debug)]
pub struct CompiledPassInfo {
    /// Pass name.
    pub name: String,
    /// Command kind.
    #[cfg(test)]
    pub kind: PassKind,
    /// Declared accesses.
    pub(crate) accesses: Vec<ResourceAccess>,
    /// Optional multiview mask for raster passes.
    #[cfg(test)]
    pub multiview_mask: Option<std::num::NonZeroU32>,
    /// Render-pass attachment template for graph-managed raster passes.
    pub raster_template: Option<RenderPassTemplate>,
    /// Backend merge hint declared at setup time. See [`PassMergeHint`].
    ///
    /// The wgpu executor currently ignores this; the field is populated for use by a future
    /// subpass-aware backend without a second migration pass across all call sites.
    #[cfg(test)]
    pub merge_hint: PassMergeHint,
}

/// Immutable execution schedule produced by [`super::GraphBuilder::build`].
///
/// ## Pass storage
///
/// Passes are stored as [`PassNode`] enum values, enabling the executor to dispatch to the
/// correct context type (raster/compute) without a runtime `graph_managed_raster()` toggle.
///
/// ## Frame-global contract
///
/// [`super::pass::PassPhase::FrameGlobal`] passes run once per tick in
/// [`CompiledRenderGraph::execute_multi_view_frame_global_passes`]. Host/scene context and
/// resource resolution for that encoder use the **first** [`FrameView`] only.
///
/// ## Submit model
///
/// The executor records frame-global work plus one command buffer per view, drains deferred
/// uploads on the main thread, and submits the assembled batch once per tick.
pub struct CompiledRenderGraph {
    /// Ordered pass nodes in execution order (culled, sorted).
    pub(super) passes: Vec<PassNode>,
    /// `true` when any pass writes an imported frame color target; frame execution
    /// acquires the swapchain once and presents after submit.
    pub needs_surface_acquire: bool,
    /// Build-time stats for tests and profiling hooks.
    pub compile_stats: CompileStats,
    /// Retained pass metadata in execution order.
    pub pass_info: Vec<CompiledPassInfo>,
    /// Compiled transient texture metadata.
    pub transient_textures: Vec<CompiledTextureResource>,
    /// Compiled transient buffer metadata.
    pub transient_buffers: Vec<CompiledBufferResource>,
    /// Declared subresource views of transient textures. Resolved lazily at execute time via
    /// [`super::context::GraphResolvedResources::subresource_view`]; see
    /// [`super::resources::SubresourceHandle`].
    pub subresources: Vec<TransientSubresourceDesc>,
    /// Imported texture declarations.
    pub imported_textures: Vec<ImportedTextureDecl>,
    /// Imported buffer declarations.
    pub imported_buffers: Vec<ImportedBufferDecl>,
    /// Single source of truth for pass ordering, phase, and wave membership.
    pub schedule: FrameSchedule,
    /// When this graph is the main frame graph from [`super::build_main_graph`], transient handles
    /// for the MSAA depth and R32-float depth-resolve scratch resources.
    pub(super) main_graph_msaa_transient_handles:
        Option<[crate::render_graph::resources::TextureHandle; 2]>,
}

pub(super) struct ResolvedView<'a> {
    pub(super) depth_texture: &'a wgpu::Texture,
    pub(super) depth_view: &'a wgpu::TextureView,
    pub(super) backbuffer: Option<&'a wgpu::TextureView>,
    pub(super) surface_format: wgpu::TextureFormat,
    pub(super) viewport_px: (u32, u32),
    pub(super) multiview_stereo: bool,
    pub(super) offscreen_write_render_texture_asset_id: Option<i32>,
    pub(super) view_id: ViewId,
    pub(super) sample_count: u32,
    pub(super) post_processing: ViewPostProcessing,
    // MSAA views are now in the per-view blackboard (MsaaViewsSlot), resolved from graph
    // transient textures by the executor. ResolvedView no longer carries them.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn view_post_processing_default_allows_primary_view_effects() {
        let policy = ViewPostProcessing::default();

        assert!(policy.is_enabled());
        assert!(policy.screen_space_reflections);
        assert!(policy.motion_blur);
    }

    #[test]
    fn view_post_processing_decodes_secondary_camera_flags() {
        let state = CameraState {
            flags: (1 << 6) | (1 << 8),
            ..Default::default()
        };
        let policy = ViewPostProcessing::from_camera_state(&state);

        assert!(policy.is_enabled());
        assert!(!policy.screen_space_reflections);
        assert!(policy.motion_blur);
    }

    #[test]
    fn view_post_processing_decodes_camera_render_parameters() {
        let parameters = CameraRenderParameters {
            post_processing: true,
            screen_space_reflections: true,
            ..Default::default()
        };
        let policy = ViewPostProcessing::from_camera_render_parameters(&parameters);

        assert!(policy.is_enabled());
        assert!(policy.screen_space_reflections);
        assert!(!policy.motion_blur);
    }

    #[test]
    fn view_post_processing_master_gate_masks_sub_effects() {
        let policy = ViewPostProcessing::new(false, true, true);

        assert!(!policy.is_enabled());
        assert!(!policy.screen_space_reflections);
        assert!(!policy.motion_blur);
    }
}

mod exec;
mod helpers;

#[cfg(test)]
mod dot;
