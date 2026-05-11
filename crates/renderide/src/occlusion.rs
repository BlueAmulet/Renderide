//! Hi-Z occlusion subsystem: CPU helpers (mip layout, readback unpacking, screen-space tests),
//! GPU pyramid build (`gpu`), and the [`OcclusionSystem`] facade that owns per-view temporal state.

pub(crate) mod cpu;
pub mod gpu;
mod hook;
mod system;

pub use cpu::pyramid::{hi_z_pyramid_dimensions, mip_levels_for_extent};
pub use cpu::query::{hi_z_view_proj_matrices, mesh_fully_occluded_in_hiz, stereo_hiz_keeps_draw};
#[cfg(test)]
pub(crate) use cpu::snapshot::HiZCpuSnapshot;
pub use cpu::snapshot::HiZCullData;
pub use hook::OcclusionGraphHook;
pub(crate) use system::HiZBuildInput;
pub use system::OcclusionSystem;
