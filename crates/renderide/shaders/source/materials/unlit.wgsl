//! World Unlit (`Shader "Unlit"`): texture × tint, optional alpha test.
//!
//! Build emits `unlit_default` / `unlit_multiview` targets via [`MULTIVIEW`](https://docs.rs/naga_oil).
//! `@group(1)` identifiers match Unity material property names (`_Color`, `_Tex`, …) for host binding by reflection.
//!
//! Per-frame bindings (`@group(0)`) are imported from `globals.wgsl` so composed targets match the frame bind group layout used by the renderer.
//! Per-draw uniforms (`@group(2)`) use [`renderide::per_draw`].
//!
//! Previously-carried `flags: u32` bitfield was removed: FrooxEngine routed its multi-compile
//! keywords (`_OFFSET_TEXTURE`, `_MASK_TEXTURE_MUL`, `_MUL_RGB_BY_ALPHA`, …) exclusively through
//! the `ShaderKeywords.Variant` bitmask that the renderer never receives, so every inferred bit
//! except `has texture` and `alpha-test` was permanently zero. The default-white texture fallback
//! handles "no host texture bound" without needing a separate flag bit, and alpha-test is read
//! directly from `_Cutoff`.

#import renderide::globals as rg
#import renderide::per_draw as pd
#import renderide::alpha_clip_sample as acs
#import renderide::uv_utils as uvu

struct UnlitMaterial {
    _Color: vec4<f32>,
    _Tex_ST: vec4<f32>,
    _Cutoff: f32,
    _PolarPow: f32,
    _SrcBlend: f32,
    _DstBlend: f32,
    _ZWrite: f32,
    _Cull: f32,
    _ZTest: f32,
}

@group(1) @binding(0) var<uniform> mat: UnlitMaterial;
@group(1) @binding(1) var _Tex: texture_2d<f32>;
@group(1) @binding(2) var _Tex_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) _n: vec4<f32>,
    @location(2) uv: vec2<f32>,
) -> VertexOutput {
    let d = pd::get_draw(instance_index);
    let world_p = d.model * vec4<f32>(pos.xyz, 1.0);
#ifdef MULTIVIEW
    var vp: mat4x4<f32>;
    if (view_idx == 0u) {
        vp = d.view_proj_left;
    } else {
        vp = d.view_proj_right;
    }
#else
    let vp = d.view_proj_left;
#endif
    var out: VertexOutput;
    out.clip_pos = vp * world_p;
    out.uv = uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv_main = uvu::apply_st(in.uv, mat._Tex_ST);
    let t = textureSample(_Tex, _Tex_sampler, uv_main);
    let clip_a = mat._Color.a * acs::texture_alpha_base_mip(_Tex, _Tex_sampler, uv_main);
    let albedo = mat._Color * t;

    // Alpha test is active when `_Cutoff` is a meaningful value in (0, 1); otherwise every
    // alpha at exactly 0 or exactly 1 would either never discard or always discard.
    if (mat._Cutoff > 0.0 && mat._Cutoff < 1.0 && clip_a <= mat._Cutoff) {
        discard;
    }

    return rg::retain_globals_additive(albedo);
}
