//! Unity Projection360 (`Shader "Projection360"`): equirectangular/cubemap projection with
//! optional second texture, tint texture, offset map, and rectangular clipping.
//!
//! Froox variant bits populate `_RenderideVariantBits`; this shader decodes Projection360's
//! shader-specific keyword bits locally. Implicit defaults from each multi_compile group
//! (`_VIEW`, `EQUIRECTANGULAR`, `OUTSIDE_CLIP`, `TINT_TEX_NONE`) reserve a bit but never
//! need their helper consulted - the fallthrough branches handle them.

#import renderide::skybox::cubemap_storage as cubemap_storage
#import renderide::frame::globals as rg
#import renderide::draw::per_draw as pd
#import renderide::core::math as rmath
#import renderide::material::variant_bits as vb
#import renderide::mesh::vertex as mv
#import renderide::core::uv as uvu
#import renderide::skybox::projection360 as p360

struct Projection360Material {
    _Tint: vec4<f32>,
    _OutsideColor: vec4<f32>,
    _Tint0: vec4<f32>,
    _Tint1: vec4<f32>,
    _FOV: vec4<f32>,
    _SecondTexOffset: vec4<f32>,
    _OffsetMagnitude: vec4<f32>,
    _PerspectiveFOV: vec4<f32>,
    _MainTex_ST: vec4<f32>,
    _RightEye_ST: vec4<f32>,
    _TintTex_ST: vec4<f32>,
    _OffsetTex_ST: vec4<f32>,
    _Rect: vec4<f32>,
    _TextureLerp: f32,
    _CubeLOD: f32,
    _MainCube_StorageVInverted: f32,
    _SecondCube_StorageVInverted: f32,
    _Exposure: f32,
    _Gamma: f32,
    _MaxIntensity: f32,
    _RenderideVariantBits: u32,
}

const P360_KW_CUBEMAP: u32 = 1u << 0u;
const P360_KW_CUBEMAP_LOD: u32 = 1u << 1u;
const P360_KW_EQUIRECTANGULAR: u32 = 1u << 2u;
const P360_KW_OUTSIDE_CLAMP: u32 = 1u << 3u;
const P360_KW_OUTSIDE_CLIP: u32 = 1u << 4u;
const P360_KW_OUTSIDE_COLOR: u32 = 1u << 5u;
const P360_KW_RECTCLIP: u32 = 1u << 6u;
const P360_KW_SECOND_TEXTURE: u32 = 1u << 7u;
const P360_KW_TINT_TEX_DIRECT: u32 = 1u << 8u;
const P360_KW_TINT_TEX_LERP: u32 = 1u << 9u;
const P360_KW_TINT_TEX_NONE: u32 = 1u << 10u;
const P360_KW_CLAMP_INTENSITY: u32 = 1u << 11u;
const P360_KW_NORMAL: u32 = 1u << 12u;
const P360_KW_OFFSET: u32 = 1u << 13u;
const P360_KW_PERSPECTIVE: u32 = 1u << 14u;
const P360_KW_RIGHT_EYE_ST: u32 = 1u << 15u;
const P360_KW_VIEW: u32 = 1u << 16u;
const P360_KW_WORLD_VIEW: u32 = 1u << 17u;

@group(1) @binding(0) var<uniform> mat: Projection360Material;
@group(1) @binding(1) var _MainTex: texture_2d<f32>;
@group(1) @binding(2) var _MainTex_sampler: sampler;
@group(1) @binding(3) var _SecondTex: texture_2d<f32>;
@group(1) @binding(4) var _SecondTex_sampler: sampler;
@group(1) @binding(5) var _TintTex: texture_2d<f32>;
@group(1) @binding(6) var _TintTex_sampler: sampler;
@group(1) @binding(7) var _OffsetTex: texture_2d<f32>;
@group(1) @binding(8) var _OffsetTex_sampler: sampler;
@group(1) @binding(9) var _OffsetMask: texture_2d<f32>;
@group(1) @binding(10) var _OffsetMask_sampler: sampler;
@group(1) @binding(11) var _MainCube: texture_cube<f32>;
@group(1) @binding(12) var _MainCube_sampler: sampler;
@group(1) @binding(13) var _SecondCube: texture_cube<f32>;
@group(1) @binding(14) var _SecondCube_sampler: sampler;

fn proj360_kw(mask: u32) -> bool {
    return vb::enabled(mat._RenderideVariantBits, mask);
}

fn kw_CUBEMAP() -> bool { return proj360_kw(P360_KW_CUBEMAP); }
fn kw_CUBEMAP_LOD() -> bool { return proj360_kw(P360_KW_CUBEMAP_LOD); }
fn kw_OUTSIDE_CLAMP() -> bool { return proj360_kw(P360_KW_OUTSIDE_CLAMP); }
fn kw_OUTSIDE_COLOR() -> bool { return proj360_kw(P360_KW_OUTSIDE_COLOR); }
fn kw_RECTCLIP() -> bool { return proj360_kw(P360_KW_RECTCLIP); }
fn kw_SECOND_TEXTURE() -> bool { return proj360_kw(P360_KW_SECOND_TEXTURE); }
fn kw_TINT_TEX_DIRECT() -> bool { return proj360_kw(P360_KW_TINT_TEX_DIRECT); }
fn kw_TINT_TEX_LERP() -> bool { return proj360_kw(P360_KW_TINT_TEX_LERP); }
fn kw_CLAMP_INTENSITY() -> bool { return proj360_kw(P360_KW_CLAMP_INTENSITY); }
fn kw_NORMAL() -> bool { return proj360_kw(P360_KW_NORMAL); }
fn kw_OFFSET() -> bool { return proj360_kw(P360_KW_OFFSET); }
fn kw_PERSPECTIVE() -> bool { return proj360_kw(P360_KW_PERSPECTIVE); }
fn kw_RIGHT_EYE_ST() -> bool { return proj360_kw(P360_KW_RIGHT_EYE_ST); }
fn kw_WORLD_VIEW() -> bool { return proj360_kw(P360_KW_WORLD_VIEW); }

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) pos_os: vec3<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) normal_os: vec3<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) dist: f32,
    @location(5) local_xy: vec2<f32>,
    @location(6) @interpolate(flat) view_layer: u32,
    @location(7) object_view_dir: vec3<f32>,
}

/// Object-space view direction at the vertex, intentionally un-normalized.
///
/// The result is linear in `world_pos`, so perspective-correct interpolation across the
/// triangle yields the per-fragment direction from the surface point to the camera.
/// Normalizing per vertex would skew the interpolated direction (the angular error scales
/// with the triangle's angular extent and breaks narrow-FOV projections); the fragment
/// shader normalizes the interpolated value, matching the per-fragment recompute used by
/// the original Unity shader.
fn object_space_view_dir(model: mat4x4<f32>, world_pos: vec3<f32>, view_layer: u32) -> vec3<f32> {
    let model3 = mat3x3<f32>(model[0].xyz, model[1].xyz, model[2].xyz);
    return transpose(model3) * (rg::camera_world_pos_for_view(view_layer) - world_pos);
}

fn perspective_view_dir(uv: vec2<f32>) -> vec3<f32> {
    return p360::perspective_view_dir_from_ndc((uv - vec2<f32>(0.5)) * 2.0, mat._PerspectiveFOV);
}

fn base_view_dir(in: VertexOutput) -> vec3<f32> {
    if (kw_PERSPECTIVE()) {
        return perspective_view_dir(in.uv);
    }
    if (kw_NORMAL()) {
        return normalize(in.normal_os);
    }
    if (kw_WORLD_VIEW()) {
        return rg::view_dir_for_world_pos(in.world_pos, in.view_layer);
    }
    return normalize(in.object_view_dir);
}

fn apply_offset(view_dir: vec3<f32>) -> vec3<f32> {
    if (!kw_OFFSET()) {
        return view_dir;
    }

    let offset_uv = p360::dir_to_uv(view_dir, mat._FOV);
    let offset_sample =
        textureSampleLevel(_OffsetTex, _OffsetTex_sampler, uvu::apply_st(offset_uv, mat._OffsetTex_ST), 0.0).rg;
    let offset_mask = textureSampleLevel(_OffsetMask, _OffsetMask_sampler, offset_uv, 0.0).rg;
    let offset = (offset_sample * 2.0 - vec2<f32>(1.0)) * offset_mask * mat._OffsetMagnitude.xy;
    return p360::rotate_dir(view_dir, offset);
}

fn sample_equirect(view_dir: vec3<f32>, view_layer: u32) -> vec4<f32> {
    var uv = p360::dir_to_uv(view_dir, mat._FOV);
    if (p360::is_outside_uv(uv)) {
        if (kw_OUTSIDE_COLOR()) {
            return mat._OutsideColor;
        }
        if (!kw_OUTSIDE_CLAMP()) {
            discard;
        }
    }
    uv = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0));

    var st = mat._MainTex_ST;
    if (kw_RIGHT_EYE_ST() && view_layer != 0u) {
        st = mat._RightEye_ST;
    }
    // `uv` is procedurally derived from `view_dir` (not a Unity mesh UV), but the texture is
    // still authored in Unity convention. Storage-inverted native compressed uploads keep their
    // block bytes unchanged, so they opt out of the final shader V flip.
    let sample_uv = uvu::apply_st(uv, st);
    var c = textureSampleLevel(_MainTex, _MainTex_sampler, sample_uv, 0.0);
    if (kw_SECOND_TEXTURE()) {
        // `_SecondTexOffset` is authored in Unity texel space; preserve the relative shift after
        // the V-flip by negating the y component.
        let secondary_offset = vec2<f32>(mat._SecondTexOffset.x, -mat._SecondTexOffset.y);
        let sc = textureSampleLevel(_SecondTex, _SecondTex_sampler, sample_uv + secondary_offset, 0.0);
        c = mix(c, sc, clamp(mat._TextureLerp, 0.0, 1.0));
    }

    if (kw_TINT_TEX_DIRECT()) {
        c = c * textureSampleLevel(_TintTex, _TintTex_sampler, sample_uv, 0.0);
    } else if (kw_TINT_TEX_LERP()) {
        let tint_uv = uvu::apply_st(uv, vec4<f32>(mat._TintTex_ST.xy, mat._TintTex_ST.w, mat._TintTex_ST.z));
        let l = textureSampleLevel(_TintTex, _TintTex_sampler, tint_uv, 0.0).r;
        c = c * mix(mat._Tint0, mat._Tint1, l);
    }
    return c;
}

fn sample_cubemap(view_dir: vec3<f32>) -> vec4<f32> {
    let dir = normalize(-view_dir);
    var lod = 0.0;
    if (kw_CUBEMAP_LOD()) {
        lod = mat._CubeLOD;
    }
    let main_dir = cubemap_storage::sample_dir(dir, mat._MainCube_StorageVInverted);
    var c = textureSampleLevel(_MainCube, _MainCube_sampler, main_dir, lod);
    if (kw_SECOND_TEXTURE()) {
        let second_dir = cubemap_storage::sample_dir(dir, mat._SecondCube_StorageVInverted);
        let sc = textureSampleLevel(_SecondCube, _SecondCube_sampler, second_dir, lod);
        c = mix(c, sc, clamp(mat._TextureLerp, 0.0, 1.0));
    }
    return c;
}

fn finish_color(c_in: vec4<f32>, dist: f32) -> vec4<f32> {
    var c = c_in;
    let fade = clamp((dist - 0.05) * 10.0, 0.0, 1.0);
    var tint = mat._Tint;
    tint.a = tint.a * fade;

    c = p360::apply_tint_exposure_and_clamp(
        c,
        tint,
        mat._Gamma,
        mat._Exposure,
        kw_CLAMP_INTENSITY(),
        mat._MaxIntensity,
    );
    return rg::retain_globals_additive(c);
}

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv: vec2<f32>,
) -> VertexOutput {
    let d = pd::get_draw(instance_index);
    let world_p = mv::world_position(d, pos);
#ifdef MULTIVIEW
    let vp = mv::select_view_proj(d, view_idx);
    let layer = view_idx;
#else
    let vp = mv::select_view_proj(d, 0u);
    let layer = 0u;
#endif
    let clip = vp * world_p;

    var out: VertexOutput;
    out.clip_pos = clip;
    out.pos_os = pos.xyz;
    out.world_pos = world_p.xyz;
    out.normal_os = normalize(-n.xyz);
    out.uv = uv;
    out.dist = clip.w;
    out.local_xy = pos.xy;
    out.object_view_dir = object_space_view_dir(d.model, world_p.xyz, layer);
    out.view_layer = layer;
    return out;
}

//#pass forward
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if (kw_RECTCLIP() && rmath::outside_rect(in.local_xy, mat._Rect)) {
        discard;
    }

    let view_dir = apply_offset(base_view_dir(in));
    var c: vec4<f32>;
    if (kw_CUBEMAP() || kw_CUBEMAP_LOD()) {
        c = sample_cubemap(view_dir);
    } else {
        c = sample_equirect(view_dir, in.view_layer);
    }
    return finish_color(c, in.dist);
}
