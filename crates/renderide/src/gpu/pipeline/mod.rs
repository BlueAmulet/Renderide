//! Pipeline abstraction: RenderPipeline trait, PipelineManager, and concrete implementations.
//!
//! Extension point for pipelines, materials, PBR.

mod core;
mod mrt;
mod normal_debug;
mod overlay_stencil;
mod overlay_stencil_skinned;
mod placeholders;
mod ring_buffer;
mod shaders;
mod skinned;
mod uniforms;
mod uv_debug;

pub use core::{matrix4_to_wgsl_column_major, RenderPipeline, UniformData, MAX_BLENDSHAPE_WEIGHTS};
pub use mrt::{NormalDebugMRTPipeline, SkinnedMRTPipeline, UvDebugMRTPipeline};
pub use normal_debug::NormalDebugPipeline;
pub use overlay_stencil::{
    OverlayStencilMaskClearPipeline, OverlayStencilMaskWritePipeline, OverlayStencilPipeline,
};
pub use overlay_stencil_skinned::{
    OverlayStencilMaskClearSkinnedPipeline, OverlayStencilMaskWriteSkinnedPipeline,
    OverlayStencilSkinnedPipeline,
};
pub use placeholders::{MaterialPipeline, PbrPipeline};
pub use skinned::SkinnedPipeline;
pub use uv_debug::UvDebugPipeline;
