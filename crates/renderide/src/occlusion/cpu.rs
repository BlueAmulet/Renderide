//! CPU-side Hi-Z helpers: data types, mip-pyramid math, GPU-readback decoding, and the
//! AABB-vs-pyramid occlusion test driving CPU world-mesh culling.

pub mod pyramid;
pub mod query;
pub mod readback;
pub mod snapshot;
