//! GPU state, pipelines, and mesh rendering.

pub mod mesh;
pub mod pipeline;
pub mod state;

pub use mesh::{GpuMeshBuffers, compute_vertex_stride_from_mesh, create_mesh_buffers};
pub use pipeline::{PipelineManager, RenderPipeline, UniformData};
pub use state::{GpuState, create_depth_texture, init_gpu};
