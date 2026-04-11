//! Debug raster: world-space normals (RGB).
//!
//! Build emits two targets from this file via [`MULTIVIEW`](https://docs.rs/naga_oil) shader defs:
//! - `debug_world_normals_default.wgsl` — `MULTIVIEW` off (single-view desktop)
//! - `debug_world_normals_multiview.wgsl` — `MULTIVIEW` on (stereo `@builtin(view_index)`)
//!
//! [`PerDrawUniforms`] lives in [`renderide::per_draw`].

#import renderide::globals as rg
#import renderide::per_draw as pd
#import renderide::view_proj as vp
#import renderide::globals_retention as ret

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_n: vec3<f32>,
}

@vertex
fn vs_main(
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) normal: vec4<f32>,
) -> VertexOutput {
    let world_p = pd::draw.model * vec4<f32>(pos.xyz, 1.0);
    let world_n = normalize((pd::draw.model * vec4<f32>(normal.xyz, 0.0)).xyz);
#ifdef MULTIVIEW
    let vpm = vp::view_projection_for_eye(view_idx);
#else
    let vpm = vp::view_projection_for_eye(0u);
#endif
    var out: VertexOutput;
    out.clip_pos = vpm * world_p;
    out.world_n = world_n;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let n = in.world_n * 0.5 + 0.5;
    let c = vec3<f32>(n) + rg::frame.camera_world_pos.xyz * 0.0001 + ret::fragment_watermark_rgb();
    return vec4<f32>(c, 1.0);
}
