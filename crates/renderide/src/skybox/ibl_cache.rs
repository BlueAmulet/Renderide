//! Unified IBL bake cache for specular reflection sources.
//!
//! Owns one in-flight bake job tracker, three lazily-built mip-0 producer pipelines (analytic
//! procedural / gradient skies, host cubemaps, and Projection360 equirect Texture2Ds), one
//! source-pyramid downsample pipeline, and one GGX convolve pipeline. For each new active
//! reflection source the cache:
//!
//! 1. Allocates a source Rgba16Float cubemap and a filtered output cubemap with full mip chains.
//! 2. Records a mip-0 producer compute pass that converts the source into the source cube's mip 0.
//! 3. Copies source mip 0 into filtered output mip 0 for mirror-smooth reflections.
//! 4. Records downsample passes that build the source radiance mip pyramid.
//! 5. Records one GGX convolve compute pass per filtered mip in `1..N`, sampling the full source
//!    pyramid with solid-angle source-mip selection.
//! 6. Submits the encoder through [`crate::backend::gpu_jobs::GpuSubmitJobTracker`] and parks the
//!    cube in `pending` until the submit-completion callback promotes it to `completed`.
//!
//! The completed prefiltered cube is reused by reflection probes so every source type reaches
//! shader sampling through a single GGX-prefiltered cube.

mod bind_groups;
mod cache;
mod convolver;
mod encode;
mod errors;
mod key;
mod mip_loop;
mod pipeline;
mod pipeline_store;
mod resources;
mod sampler;

pub(crate) use cache::SkyboxIblCache;
pub(crate) use convolver::SkyboxIblConvolver;
pub(crate) use key::{SkyboxIblKey, build_key, clamp_face_size, mip_extent, mip_levels_for_edge};
