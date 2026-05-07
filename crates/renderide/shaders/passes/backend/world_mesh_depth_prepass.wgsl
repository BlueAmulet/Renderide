//! Generic world-mesh depth prepass.
//!
//! Uses the same per-draw slab packing and position transform convention as material forward
//! vertex stages, but emits only clip-space position and relies on fixed-function depth writes.

struct PerDrawUniforms {
    view_proj_left: mat4x4<f32>,
    view_proj_right: mat4x4<f32>,
    model: mat4x4<f32>,
    normal_matrix: mat3x3<f32>,
    _pad: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> instances: array<PerDrawUniforms>;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
}

fn select_view_proj(draw: PerDrawUniforms, view_idx: u32) -> mat4x4<f32> {
    if (view_idx == 0u) {
        return draw.view_proj_left;
    }
    return draw.view_proj_right;
}

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
) -> VertexOutput {
    let draw = instances[instance_index];
#ifdef MULTIVIEW
    let view_layer = view_idx;
#else
    let view_layer = 0u;
#endif
    let vp = select_view_proj(draw, view_layer);
    let world_p = draw.model * vec4<f32>(pos.xyz, 1.0);

    var out: VertexOutput;
    out.clip_pos = vp * world_p;
    return out;
}

@fragment
fn fs_main() {
}
