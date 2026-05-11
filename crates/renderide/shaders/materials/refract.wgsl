//! Grab-pass refraction filter (`Shader "Filters/Refract"`).
//!
//! Froox variant bits populate `_RenderideVariantBits`; this shader decodes Refract's
//! shader-specific keyword bits locally.

#import renderide::post::filter_math as fm
#import renderide::post::filter_vertex as fv
#import renderide::frame::globals as rg
#import renderide::frame::grab_pass as gp
#import renderide::core::normal_decode as nd
#import renderide::frame::scene_depth_sample as sds
#import renderide::material::variant_bits as vb
#import renderide::ui::rect_clip as uirc
#import renderide::core::uv as uvu

struct FiltersRefractMaterial {
    _NormalMap_ST: vec4<f32>,
    _Rect: vec4<f32>,
    _RefractionStrength: f32,
    _DepthBias: f32,
    _DepthDivisor: f32,
    _RenderideVariantBits: u32,
}

const REFRACT_KW_RECTCLIP: u32 = 1u << 0u;
const REFRACT_KW_NORMALMAP: u32 = 1u << 1u;

@group(1) @binding(0) var<uniform> mat: FiltersRefractMaterial;
@group(1) @binding(1) var _NormalMap: texture_2d<f32>;
@group(1) @binding(2) var _NormalMap_sampler: sampler;

fn refract_kw(mask: u32) -> bool {
    return vb::enabled(mat._RenderideVariantBits, mask);
}

fn kw_RECTCLIP() -> bool {
    return refract_kw(REFRACT_KW_RECTCLIP);
}

fn kw_NORMALMAP() -> bool {
    return refract_kw(REFRACT_KW_NORMALMAP);
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

fn refract_offset(uv0: vec2<f32>, view_n: vec3<f32>, clip_recip_w: f32) -> vec2<f32> {
    var n = normalize(view_n);
    if (kw_NORMALMAP()) {
        let ts = nd::decode_ts_normal_with_placeholder_sample(
            textureSample(_NormalMap, _NormalMap_sampler, uvu::apply_st(uv0, mat._NormalMap_ST)),
            1.0,
        );
        n = normalize(vec3<f32>(n.xy + ts.xy, n.z));
    }
    return n.xy * clip_recip_w * mat._RefractionStrength;
}

fn refracted_screen_uv(
    screen_uv: vec2<f32>,
    uv0: vec2<f32>,
    view_n: vec3<f32>,
    frag_pos: vec4<f32>,
    world_pos: vec3<f32>,
    view_layer: u32,
) -> vec2<f32> {
    let fade = sds::depth_fade(frag_pos, world_pos, view_layer, mat._DepthDivisor);
    let offset = refract_offset(uv0, view_n, frag_pos.w) * fade * fm::screen_vignette(screen_uv);
    let grab_uv = screen_uv - offset;
    let sampled_depth = sds::scene_linear_depth_at_uv(grab_uv, view_layer);
    let fragment_depth = sds::fragment_linear_depth(world_pos, view_layer);
    if (sampled_depth > fragment_depth + mat._DepthBias) {
        return screen_uv;
    }
    return grab_uv;
}

//#pass forward
@fragment
fn fs_main(in: fv::RectVertexOutput) -> @location(0) vec4<f32> {
    if (uirc::should_clip_rect_kw(in.obj_xy, mat._Rect, kw_RECTCLIP())) {
        discard;
    }
    let screen_uv = gp::frag_screen_uv(in.clip_pos);
    let color = gp::sample_scene_color(
        refracted_screen_uv(screen_uv, in.primary_uv, in.view_n, in.clip_pos, in.world_pos, in.view_layer),
        in.view_layer,
    );
    return rg::retain_globals_additive(color);
}
