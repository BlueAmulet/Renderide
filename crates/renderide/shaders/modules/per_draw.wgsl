//! Per-draw instance data (`@group(2)`) shared by mesh materials -- storage buffer indexed by
//! `@builtin(instance_index)`.
//! Import with `#import renderide::per_draw as pd` from `shaders/materials/*.wgsl` and use
//! `pd::get_draw(instance_index)` in `vs_main`. Do not redeclare `@group(2)` in material roots.
//!
//! CPU packing must match [`crate::backend::mesh_deform::PaddedPerDrawUniforms`].

#define_import_path renderide::per_draw

struct PerDrawUniforms {
    view_proj_left: mat4x4<f32>,
    view_proj_right: mat4x4<f32>,
    model: mat4x4<f32>,
    /// Inverse transpose of the upper 3x3 of `model` (correct normals under non-uniform scale).
    normal_matrix: mat3x3<f32>,
    /// Metadata. `x` marks world-space position streams; `yzw` pack reflection-probe selection.
    _pad: vec4<f32>,
}

/// `_pad.x` marker for world-space position streams.
const POSITION_STREAM_WORLD_SPACE_FLAG: f32 = 1.0;

@group(2) @binding(0) var<storage, read> instances: array<PerDrawUniforms>;

fn get_draw(instance_idx: u32) -> PerDrawUniforms {
    return instances[instance_idx];
}

/// `true` when the bound position stream has already been transformed into world space.
fn position_stream_is_world_space(draw: PerDrawUniforms) -> bool {
    return draw._pad.x > 0.5 * POSITION_STREAM_WORLD_SPACE_FLAG;
}

fn reflection_probe_indices(draw: PerDrawUniforms) -> vec2<u32> {
    let packed = bitcast<u32>(draw._pad.y);
    return vec2<u32>(packed & 0xFFFFu, packed >> 16u);
}

fn reflection_probe_second_weight(draw: PerDrawUniforms) -> f32 {
    return clamp(draw._pad.z, 0.0, 1.0);
}

fn reflection_probe_hit_count(draw: PerDrawUniforms) -> u32 {
    return min(u32(draw._pad.w + 0.5), 2u);
}
