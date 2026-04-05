//! Mesh layout (host `MeshBuffer` contract) and GPU upload.

mod gpu_mesh;
mod layout;

pub use gpu_mesh::{try_upload_mesh_from_raw, GpuMesh};
pub use layout::{
    attribute_offset_and_size, compute_index_count, compute_mesh_buffer_layout,
    compute_vertex_stride, extract_bind_poses, extract_blendshape_offsets, index_bytes_per_element,
    synthetic_bone_data_for_blendshape_only, MeshBufferLayout, BLENDSHAPE_OFFSET_GPU_STRIDE,
    MAX_BUFFER_SIZE,
};
