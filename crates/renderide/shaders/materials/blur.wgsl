//! Grab-pass blur filter (`Shader "Filters/Blur"`).
//!
//! Reads scene color via the grab pass and accumulates either a circular or Poisson-disc tap set
//! around the fragment's screen UV, optionally offset by view-space refraction (with normal-map
//! perturbation). The `SPREAD_TEX`, `REFRACT`/`REFRACT_NORMALMAP`, `RECTCLIP`, and `POISSON_DISC`
//! variant bits mirror Unity `#pragma multi_compile _ ...` groups on the source shader.

#import renderide::post::filter_math as fm
#import renderide::post::filter_vertex as fv
#import renderide::frame::globals as rg
#import renderide::frame::grab_pass as gp
#import renderide::core::normal_decode as nd
#import renderide::frame::scene_depth_sample as sds
#import renderide::core::uv as uvu
#import renderide::pbs::normal as pnorm
#import renderide::material::variant_bits as vb
#import renderide::ui::rect_clip as uirc

struct FiltersBlurMaterial {
    _Spread: vec4<f32>,
    _SpreadTex_ST: vec4<f32>,
    _NormalMap_ST: vec4<f32>,
    _Rect: vec4<f32>,
    _Iterations: f32,
    _RefractionStrength: f32,
    _DepthDivisor: f32,
    _RenderideVariantBits: u32,
}

const BLUR_KW_POISSON_DISC: u32 = 1u << 0u;
const BLUR_KW_RECTCLIP: u32 = 1u << 1u;
const BLUR_KW_REFRACT: u32 = 1u << 2u;
const BLUR_KW_REFRACT_NORMALMAP: u32 = 1u << 3u;
const BLUR_KW_SPREAD_TEX: u32 = 1u << 4u;

@group(1) @binding(0) var<uniform> mat: FiltersBlurMaterial;
@group(1) @binding(1) var _SpreadTex: texture_2d<f32>;
@group(1) @binding(2) var _SpreadTex_sampler: sampler;
@group(1) @binding(3) var _NormalMap: texture_2d<f32>;
@group(1) @binding(4) var _NormalMap_sampler: sampler;

fn blur_kw(mask: u32) -> bool {
    return vb::enabled(mat._RenderideVariantBits, mask);
}

fn kw_POISSON_DISC() -> bool {
    return blur_kw(BLUR_KW_POISSON_DISC);
}

fn kw_RECTCLIP() -> bool {
    return blur_kw(BLUR_KW_RECTCLIP);
}

fn kw_REFRACT() -> bool {
    return blur_kw(BLUR_KW_REFRACT);
}

fn kw_REFRACT_NORMALMAP() -> bool {
    return blur_kw(BLUR_KW_REFRACT_NORMALMAP);
}

fn kw_SPREAD_TEX() -> bool {
    return blur_kw(BLUR_KW_SPREAD_TEX);
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

fn refraction_enabled() -> bool {
    return kw_REFRACT() || kw_REFRACT_NORMALMAP();
}

fn refract_offset(uv0: vec2<f32>, view_n: vec3<f32>, clip_recip_w: f32) -> vec2<f32> {
    if (!refraction_enabled()) {
        return vec2<f32>(0.0);
    }
    var n = normalize(view_n);
    if (kw_REFRACT_NORMALMAP()) {
        let ts = nd::decode_ts_normal_with_placeholder_sample(
            textureSample(_NormalMap, _NormalMap_sampler, uvu::apply_st(uv0, mat._NormalMap_ST)),
            1.0,
        );
        n = normalize(vec3<f32>(n.xy + ts.xy, n.z));
    }
    return n.xy * clip_recip_w * mat._RefractionStrength;
}

fn spread_modulation(uv0: vec2<f32>) -> vec2<f32> {
    if (!kw_SPREAD_TEX()) {
        return vec2<f32>(1.0);
    }
    return textureSample(_SpreadTex, _SpreadTex_sampler, uvu::apply_st(uv0, mat._SpreadTex_ST)).rg;
}

fn sample_blur(center_uv: vec2<f32>, spread: vec2<f32>, iterations: u32, view_layer: u32) -> vec4<f32> {
    var c = vec4<f32>(0.0);
    let use_poisson = kw_POISSON_DISC();
    for (var i = 0u; i < 128u; i = i + 1u) {
        if (i >= iterations) {
            break;
        }
        let offset = select(
            fm::circular_blur_offset(i, iterations, spread),
            fm::poisson_blur_offset(i, spread),
            use_poisson,
        );
        c = c + gp::sample_scene_color(center_uv + offset, view_layer);
    }
    return c / max(f32(iterations), 1.0);
}

//#pass forward
@fragment
fn fs_main(in: fv::RectVertexOutput) -> @location(0) vec4<f32> {
    if (uirc::should_clip_rect_kw(in.obj_xy, mat._Rect, kw_RECTCLIP())) {
        discard;
    }
    let screen_uv = gp::frag_screen_uv(in.clip_pos);
    let fade = sds::depth_fade(in.clip_pos, in.world_pos, in.view_layer, mat._DepthDivisor);
    let spread = mat._Spread.xy * spread_modulation(in.primary_uv) * fade;
    let center_uv = screen_uv - refract_offset(in.primary_uv, in.view_n, in.clip_pos.w) * fade;
    let iterations = u32(clamp(mat._Iterations, 1.0, 128.0));
    return rg::retain_globals_additive(sample_blur(center_uv, spread, iterations, in.view_layer));
}
