//! Grab-pass channel-matrix filter (`Shader "Filters/ChannelMatrix"`).
//!
//! Reads scene color via the grab pass, applies a 3x3 channel matrix with per-channel offsets,
//! then clamps the result. The `RECTCLIP` variant bit (Unity `#pragma multi_compile _ RECTCLIP`)
//! discards fragments outside the object-space `_Rect` rectangle.

#import renderide::post::filter_vertex as fv
#import renderide::frame::globals as rg
#import renderide::frame::grab_pass as gp
#import renderide::material::variant_bits as vb
#import renderide::ui::rect_clip as rc

struct FiltersChannelMatrixMaterial {
    _LevelsR: vec4<f32>,
    _LevelsG: vec4<f32>,
    _LevelsB: vec4<f32>,
    _ClampMin: vec4<f32>,
    _ClampMax: vec4<f32>,
    _Rect: vec4<f32>,
    _RenderideVariantBits: u32,
    _pad0: vec3<f32>,
}

const CHANNELMATRIX_KW_RECTCLIP: u32 = 1u << 0u;

@group(1) @binding(0) var<uniform> mat: FiltersChannelMatrixMaterial;

fn kw_RECTCLIP() -> bool {
    return vb::enabled(mat._RenderideVariantBits, CHANNELMATRIX_KW_RECTCLIP);
}

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
    @location(4) t: vec4<f32>,
) -> fv::RectVertexOutput {
#ifdef MULTIVIEW
    return fv::rect_vertex_main(instance_index, view_idx, pos, n, t, uv0);
#else
    return fv::rect_vertex_main(instance_index, 0u, pos, n, t, uv0);
#endif
}

//#pass forward
@fragment
fn fs_main(
    @builtin(position) frag_pos: vec4<f32>,
    @location(3) @interpolate(flat) view_layer: u32,
    @location(5) obj_xy: vec2<f32>,
) -> @location(0) vec4<f32> {
    if (rc::should_clip_rect_b(obj_xy, mat._Rect, kw_RECTCLIP())) {
        discard;
    }
    let c = gp::sample_scene_color(gp::frag_screen_uv(frag_pos), view_layer);
    let remapped = vec3<f32>(
        dot(mat._LevelsR.xyz, c.rgb) + mat._LevelsR.w,
        dot(mat._LevelsG.xyz, c.rgb) + mat._LevelsG.w,
        dot(mat._LevelsB.xyz, c.rgb) + mat._LevelsB.w,
    );
    let filtered = clamp(remapped, mat._ClampMin.xyz, mat._ClampMax.xyz);
    return rg::retain_globals_additive(vec4<f32>(filtered, c.a));
}
