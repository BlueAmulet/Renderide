//! Mesh layout (host `MeshBuffer` contract) and GPU upload.

mod gpu_mesh;
mod gpu_mesh_fingerprint;
mod gpu_mesh_hints;
mod gpu_mesh_validation;
mod layout;
#[cfg(test)]
mod layout_tests;
mod tangent_generation;
mod upload_impl;

pub use gpu_mesh::GpuMesh;
pub use gpu_mesh_fingerprint::mesh_upload_input_fingerprint;
pub use gpu_mesh_validation::{compute_and_validate_mesh_layout, try_upload_mesh_from_raw};
pub use layout::{
    BLENDSHAPE_PACKED_VECTOR_SPARSE_ENTRY_WORDS, BLENDSHAPE_POSITION_SPARSE_ENTRY_SIZE,
    BLENDSHAPE_POSITION_SPARSE_ENTRY_WORDS, BlendshapeFrameRange, BlendshapeFrameSpan,
    BlendshapeGpuPack, MeshBufferLayout, blendshape_deform_is_active,
    select_blendshape_frame_coefficients,
};
