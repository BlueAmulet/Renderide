//! Canvas UI Unlit (`Shader "UI/Unlit"`): sprite texture, tint, optional alpha clip.
//!
//! Build emits `ui_unlit_default` / `ui_unlit_multiview` via [`MULTIVIEW`](https://docs.rs/naga_oil).
//! `@group(1)` global names match Unity `UI_Unlit.shader` material property names for host reflection.
//!
//! **Vertex color:** Unity multiplies `vertex_color * _Tint`. The mesh pass provides a dense
//! float4 color stream at `@location(3)` with opaque-white fallback when the host mesh lacks color.
//!
//! The previous `flags: u32` bitfield + `_MaskTex` + `_Rect` / `_OverlayTint` paths were all
//! keyword-driven in Unity (`RECTCLIP`, `OVERLAY`, `_MASK_TEXTURE_*`) and FrooxEngine sets those
//! only via `ShaderKeywords.SetKeyword`, which the renderer never receives. The inferred flag
//! bits were permanently zero, so those branches were dead. Alpha-test is read directly from
//! `_Cutoff`.
//!
//! Per-draw uniforms (`@group(2)`) use [`renderide::per_draw`].

//#pass main: blend=src_alpha,one_minus_src_alpha,add, alpha=one,one_minus_src_alpha,add, zwrite=off, cull=none, write=all, material=forward_base

#import renderide::globals as rg
#import renderide::per_draw as pd
#import renderide::alpha_clip_sample as acs
#import renderide::uv_utils as uvu

struct UiUnlitMaterial {
    _MainTex_ST: vec4<f32>,
    _Tint: vec4<f32>,
    _Cutoff: f32,
    _SrcBlend: f32,
    _DstBlend: f32,
    _ZWrite: f32,
    _Cull: f32,
    _StencilComp: f32,
    _Stencil: f32,
    _StencilOp: f32,
    _StencilWriteMask: f32,
    _StencilReadMask: f32,
    _ColorMask: f32,
}

@group(1) @binding(0) var<uniform> mat: UiUnlitMaterial;
@group(1) @binding(1) var _MainTex: texture_2d<f32>;
@group(1) @binding(2) var _MainTex_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
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
    @location(3) color: vec4<f32>,
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
    out.color = color * mat._Tint;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv_s = uvu::apply_st(in.uv, mat._MainTex_ST);
    let t = textureSample(_MainTex, _MainTex_sampler, uv_s);
    let clip_a = in.color.a * acs::texture_alpha_base_mip(_MainTex, _MainTex_sampler, uv_s);
    let color = in.color * t;

    if (mat._Cutoff > 0.0 && mat._Cutoff < 1.0 && clip_a <= mat._Cutoff) {
        discard;
    }

    return rg::retain_globals_additive(color);
}
