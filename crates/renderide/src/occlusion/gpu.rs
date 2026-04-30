//! GPU hierarchical depth pyramid construction and readback for Hi-Z occlusion culling.
//!
//! Used by [`crate::occlusion::OcclusionSystem`] and [`crate::passes::HiZBuildPass`].

mod hi_z_encode;
mod hi_z_gpu;
mod hi_z_pipelines;
mod readback_ring;

pub use hi_z_encode::{HiZBuildRecord, HiZHistoryTarget, encode_hi_z_build};
pub(crate) use hi_z_gpu::HIZ_MAX_MIPS;
pub use hi_z_gpu::HiZGpuState;
