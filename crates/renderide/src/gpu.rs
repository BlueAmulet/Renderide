//! GPU device, adapter, swapchain, frame uniforms, and VR mirror blit.
//!
//! Layout: [`context`] ([`GpuContext`]), [`instance_limits`] ([`instance_flags_for_gpu_init`]),
//! [`frame_globals`] ([`FrameGpuUniforms`]), [`frame_cpu_gpu_timing`] (debug HUD CPU/GPU intervals),
//! [`present`] (surface acquire / clear helpers), [`vr_mirror`] (HMD eye -> staging -> window).

mod adapter;
pub mod bind_layout;
mod context;
pub mod depth;
pub mod display_blit;
pub mod driver_thread;
pub mod frame_bindings;
pub(crate) mod frame_bracket;
pub(crate) mod frame_cpu_gpu_timing;
mod instance_limits;
pub mod limits;
pub mod msaa_depth_resolve;
pub mod output_depth_mode;
pub mod present;
mod submission_state;
mod sync;
mod vr_mirror;

pub mod frame_globals;

pub use context::{GpuContext, GpuError};
pub use depth::{
    MAIN_FORWARD_DEPTH_CLEAR, MAIN_FORWARD_DEPTH_COMPARE, main_forward_depth_stencil_format,
};
pub use display_blit::DisplayBlitResources;
pub use frame_bindings::{
    CLUSTER_LIGHT_RANGE_WORDS, CLUSTER_PARAMS_UNIFORM_SIZE, GpuLight, GpuReflectionProbeMetadata,
    MAX_LIGHTS, REFLECTION_PROBE_ATLAS_FORMAT, REFLECTION_PROBE_METADATA_BOX_PROJECTION,
    REFLECTION_PROBE_METADATA_SH2_SOURCE_LOCAL, REFLECTION_PROBE_METADATA_SH2_SOURCE_SKYBOX,
    empty_material_bind_group_layout, frame_bind_group_layout, frame_bind_group_layout_entries,
};
pub use instance_limits::{RENDERER_MAX_TEXTURE_DIMENSION_2D, instance_flags_for_gpu_init};
pub use limits::{CUBEMAP_ARRAY_LAYERS, GpuLimits};
pub use msaa_depth_resolve::{
    MsaaDepthResolveMonoTargets, MsaaDepthResolveResources, MsaaDepthResolveStereoTargets,
};
pub use output_depth_mode::OutputDepthMode;
pub use sync::queue_access_gate::GpuQueueAccessGate;
pub use vr_mirror::{VR_MIRROR_EYE_LAYER, VrMirrorBlitResources};
