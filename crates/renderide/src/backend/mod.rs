//! GPU and host-facing resource layer: pools, material tables, uploads, preprocess pipelines.
//!
//! This module owns **wgpu** [`wgpu::Device`] / [`wgpu::Queue`], mesh and texture pools, the
//! [`MaterialPropertyStore`](crate::assets::material::MaterialPropertyStore), and code paths that
//! turn shared-memory asset payloads into resident GPU resources. It does **not** own IPC queues,
//! [`SharedMemoryAccessor`](crate::ipc::SharedMemoryAccessor), or scene graph state; callers pass
//! those in where a command requires both transport and GPU work.

mod render_backend;

pub use render_backend::{
    RenderBackend, MAX_PENDING_MATERIAL_BATCHES, MAX_PENDING_MESH_UPLOADS,
    MAX_PENDING_TEXTURE_UPLOADS,
};
