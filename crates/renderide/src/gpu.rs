//! GPU device, adapter, swapchain, frame uniforms, profiling, and VR mirror blit.
//!
//! Top-level layout:
//! - [`context`] -- [`GpuContext`] (instance, surface, device, swapchain) and construction.
//! - [`adapter`] -- adapter selection, device creation, feature negotiation, MSAA probing.
//! - [`limits`] -- [`GpuLimits`] capability snapshot and bounds helpers.
//! - [`depth`] -- reverse-Z conventions, depth-stencil format choice, and [`OutputDepthMode`].
//! - [`frame_globals`] -- WGSL-matched per-frame uniform structs.
//! - [`frame_bindings`] -- shader ABI: `@group(0)` BGL, light rows, reflection-probe rows.
//! - [`profiling`] -- frame-bracket GPU timestamps and CPU/GPU wall-clock timing.
//! - [`sync`] -- Vulkan queue serialisation and mapped-buffer health.
//! - [`driver_thread`] -- dedicated submit/present worker.
//! - [`present`], [`display_blit`], [`vr_mirror`], [`msaa_depth_resolve`] -- presentation passes.
//! - [`bind_layout`] -- reusable [`wgpu::BindGroupLayoutEntry`] factories.
//! - [`instance_setup`] -- renderer-policy clamps applied at instance/device creation.
//!
//! `blit_kit` (private) holds helpers shared by [`display_blit`] and [`vr_mirror`].

mod adapter;
mod blit_kit;
mod context;
mod instance_setup;
mod submission_state;
mod sync;
mod vr_mirror;

pub mod bind_layout;
pub mod depth;
pub mod display_blit;
pub mod driver_thread;
pub mod frame_bindings;
pub mod frame_globals;
pub mod limits;
pub mod msaa_depth_resolve;
pub mod present;
pub mod profiling;

// --- External-API re-exports (cross-top-level-module contract) ---
pub use context::{GpuContext, GpuError};
pub use depth::{
    MAIN_FORWARD_DEPTH_CLEAR, MAIN_FORWARD_DEPTH_COMPARE, OutputDepthMode,
    main_forward_depth_stencil_format,
};
pub use display_blit::DisplayBlitResources;
pub use frame_bindings::{
    CLUSTER_LIGHT_RANGE_WORDS, CLUSTER_PARAMS_UNIFORM_SIZE, GpuLight, GpuReflectionProbeMetadata,
    MAX_LIGHTS, REFLECTION_PROBE_ATLAS_FORMAT, REFLECTION_PROBE_METADATA_BOX_PROJECTION,
    REFLECTION_PROBE_METADATA_SH2_SOURCE_LOCAL, REFLECTION_PROBE_METADATA_SH2_SOURCE_SKYBOX,
    empty_material_bind_group_layout, frame_bind_group_layout, frame_bind_group_layout_entries,
};
pub use instance_setup::{RENDERER_MAX_TEXTURE_DIMENSION_2D, instance_flags_for_gpu_init};
pub use limits::{CUBEMAP_ARRAY_LAYERS, GpuLimits};
pub use msaa_depth_resolve::{
    MsaaDepthResolveMonoTargets, MsaaDepthResolveResources, MsaaDepthResolveStereoTargets,
};
pub use vr_mirror::{VR_MIRROR_EYE_LAYER, VrMirrorBlitResources};

// --- Legacy submodule-path re-exports (preserve external `crate::gpu::<x>::*` paths) ---
//
// External code references `crate::gpu::frame_cpu_gpu_timing::*` and
// `crate::gpu::GpuQueueAccessGate`; both physically live under newer parent modules now.
pub(crate) use profiling::frame_bracket;
pub use profiling::frame_cpu_gpu_timing;
pub use sync::queue_access_gate::GpuQueueAccessGate;
