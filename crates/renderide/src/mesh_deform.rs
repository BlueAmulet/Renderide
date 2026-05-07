//! Mesh skinning / blendshape scatter compute preprocess, sparse buffer checks, and per-draw
//! uniform packing for `@group(2)` in the world mesh forward pass.

pub mod range_alloc;
pub mod skin_cache;

mod blendshape_bind_chunks;
mod mesh_preprocess;
mod per_draw_uniforms;
mod scratch;
mod skinning_palette;

pub use skinning_palette::{SkinningPaletteParams, write_skinning_palette_bytes};

pub use blendshape_bind_chunks::{
    BLENDSHAPE_SPARSE_MIN_BUFFER_BYTES, blendshape_sparse_buffers_fit_device,
    plan_blendshape_scatter_chunks,
};
pub use mesh_preprocess::MeshPreprocessPipelines;
pub use per_draw_uniforms::{
    INITIAL_PER_DRAW_UNIFORM_SLOTS, PER_DRAW_UNIFORM_STRIDE, PaddedPerDrawUniforms,
    write_per_draw_uniform_slab,
};
pub use range_alloc::Range;
pub use scratch::{
    BlendshapeBindGroupKey, MeshDeformScratch, SkinningBindGroupKey, advance_slab_cursor,
    buffer_identity,
};
pub use skin_cache::{
    DeformSignature, EntryNeed, GpuSkinCache, SkinCacheEntry, SkinCacheKey, SkinCacheRendererKind,
};
