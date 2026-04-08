//! Per-draw dynamic uniform slot (`@group(2)`) shared by mesh materials.
//! Import with `#import renderide::per_draw as pd` from `source/materials/*.wgsl` and use `pd::draw`.
//! Do not redeclare `@group(2)` or `draw` in material roots.
//!
//! CPU packing must match [`crate::gpu::PaddedPerDrawUniforms`].

#define_import_path renderide::per_draw

struct PerDrawUniforms {
    view_proj_left: mat4x4<f32>,
    view_proj_right: mat4x4<f32>,
    model: mat4x4<f32>,
    _pad: array<vec4<f32>, 4>,
}

@group(2) @binding(0) var<uniform> draw: PerDrawUniforms;
