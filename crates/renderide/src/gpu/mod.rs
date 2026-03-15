//! GPU state, pipelines, and mesh rendering.

pub mod mesh;
pub mod pipeline;
pub mod state;

pub use mesh::{create_mesh_buffers, compute_vertex_stride_from_mesh, GpuMeshBuffers, MeshPipeline};
pub use state::{create_depth_texture, init_gpu, GpuState};
