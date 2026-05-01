//! Per-object grab-pass 3D LUT filter (`Shader "Filters/LUT_PerObject"`).


#import renderide::filter_vertex as fv
#import renderide::globals as rg
#import renderide::grab_pass as gp
#import renderide::texture_sampling as ts
#import renderide::ui::rect_clip as uirc

struct FiltersLutPerObjectMaterial {
    _Rect: vec4<f32>,
    _Lerp: f32,
    _LUT_LodBias: f32,
    _SecondaryLUT_LodBias: f32,
    _RectClip: f32,
    LERP: f32,
    _pad0: f32,
    _pad1: vec2<f32>,
}

@group(1) @binding(0) var<uniform> mat: FiltersLutPerObjectMaterial;
@group(1) @binding(1) var _LUT: texture_3d<f32>;
@group(1) @binding(2) var _LUT_sampler: sampler;
@group(1) @binding(3) var _SecondaryLUT: texture_3d<f32>;
@group(1) @binding(4) var _SecondaryLUT_sampler: sampler;

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
fn fs_main(vout: fv::RectVertexOutput) -> @location(0) vec4<f32> {
    if (uirc::should_clip_rect(vout.obj_xy, mat._Rect, mat._RectClip)) {
        discard;
    }

    let c = gp::sample_scene_color(gp::frag_screen_uv(vout.clip_pos), vout.view_layer);
    let coords = clamp(c.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    var filtered = ts::sample_tex_3d(_LUT, _LUT_sampler, coords, mat._LUT_LodBias).rgb;
    if (mat.LERP > 0.5) {
        let secondary = ts::sample_tex_3d(_SecondaryLUT, _SecondaryLUT_sampler, coords, mat._SecondaryLUT_LodBias).rgb;
        filtered = mix(filtered, secondary, mat._Lerp);
    }
    return rg::retain_globals_additive(vec4<f32>(filtered, c.a));
}
