struct PerDrawUniformsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX {
    view_proj_left: mat4x4<f32>,
    view_proj_right: mat4x4<f32>,
    model: mat4x4<f32>,
    _pad: array<vec4<f32>, 4>,
}

struct FrameGlobalsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX {
    camera_world_pos: vec4<f32>,
    view_space_z_coeffs: vec4<f32>,
    view_space_z_coeffs_right: vec4<f32>,
    cluster_count_x: u32,
    cluster_count_y: u32,
    cluster_count_z: u32,
    stereo_cluster_layers: u32,
    near_clip: f32,
    far_clip: f32,
    light_count: u32,
    viewport_width: u32,
    viewport_height: u32,
}

struct GpuLightX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX {
    position: vec3<f32>,
    align_pad_vec3_pos: f32,
    direction: vec3<f32>,
    align_pad_vec3_dir: f32,
    color: vec3<f32>,
    intensity: f32,
    range: f32,
    spot_cos_half_angle: f32,
    light_type: u32,
    align_pad_before_shadow: u32,
    shadow_strength: f32,
    shadow_near_plane: f32,
    shadow_bias: f32,
    shadow_normal_bias: f32,
    shadow_type: u32,
    align_pad_vec3_tail: vec3<u32>,
}

struct UnlitMaterial {
    _Color: vec4<f32>,
    _Tex_ST: vec4<f32>,
    _MaskTex_ST: vec4<f32>,
    _OffsetTex_ST: vec4<f32>,
    _OffsetMagnitude: vec4<f32>,
    _Cutoff: f32,
    _PolarPow: f32,
    flags: u32,
    _SrcBlend: f32,
    _DstBlend: f32,
    _ZWrite: f32,
    _Cull: f32,
    _ZTest: f32,
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

const CLIP_COVERAGE_LODX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX: f32 = 0f;

@group(2) @binding(0) 
var<uniform> drawX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX: PerDrawUniformsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX;
@group(0) @binding(0) 
var<uniform> frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX: FrameGlobalsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX;
@group(0) @binding(1) 
var<storage> lightsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX: array<GpuLightX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX>;
@group(0) @binding(2) 
var<storage> cluster_light_countsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX: array<u32>;
@group(0) @binding(3) 
var<storage> cluster_light_indicesX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX: array<u32>;
@group(1) @binding(0) 
var<uniform> mat: UnlitMaterial;
@group(1) @binding(1) 
var _Tex: texture_2d<f32>;
@group(1) @binding(2) 
var _Tex_sampler: sampler;
@group(1) @binding(3) 
var _OffsetTex: texture_2d<f32>;
@group(1) @binding(4) 
var _OffsetTex_sampler: sampler;
@group(1) @binding(5) 
var _MaskTex: texture_2d<f32>;
@group(1) @binding(6) 
var _MaskTex_sampler: sampler;

fn texture_alpha_base_mipX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX(tex: texture_2d<f32>, samp: sampler, uv_1: vec2<f32>) -> f32 {
    let _e4: vec4<f32> = textureSampleLevel(tex, samp, uv_1, CLIP_COVERAGE_LODX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX);
    return _e4.w;
}

fn mask_luminance_mul_base_mipX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX(tex_1: texture_2d<f32>, samp_1: sampler, uv_2: vec2<f32>) -> f32 {
    let mask: vec4<f32> = textureSampleLevel(tex_1, samp_1, uv_2, CLIP_COVERAGE_LODX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX);
    return ((((mask.x + mask.y) + mask.z) * 0.33333334f) * mask.w);
}

fn apply_st(uv_3: vec2<f32>, st: vec4<f32>) -> vec2<f32> {
    let uv_st: vec2<f32> = ((uv_3 * st.xy) + st.zw);
    return vec2<f32>(uv_st.x, (1f - uv_st.y));
}

@vertex 
fn vs_main(@builtin(view_index) view_idx: u32, @location(0) pos: vec4<f32>, @location(1) _n: vec4<f32>, @location(2) uv: vec2<f32>) -> VertexOutput {
    var vp: mat4x4<f32>;
    var out: VertexOutput;

    let _e3: mat4x4<f32> = drawX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX.model;
    let world_p: vec4<f32> = (_e3 * vec4<f32>(pos.xyz, 1f));
    if (view_idx == 0u) {
        let _e13: mat4x4<f32> = drawX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX.view_proj_left;
        vp = _e13;
    } else {
        let _e17: mat4x4<f32> = drawX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX.view_proj_right;
        vp = _e17;
    }
    let _e20: mat4x4<f32> = vp;
    out.clip_pos = (_e20 * world_p);
    out.uv = uv;
    let _e24: VertexOutput = out;
    return _e24;
}

@fragment 
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var albedo: vec4<f32>;
    var clip_a: f32;
    var uv_main: vec2<f32>;
    var local: bool;
    var lit: u32 = 0u;

    let _e3: vec4<f32> = mat._Color;
    albedo = _e3;
    let _e8: f32 = mat._Color.w;
    clip_a = _e8;
    let _e12: u32 = mat.flags;
    if ((_e12 & 1u) != 0u) {
        let _e21: vec4<f32> = mat._Tex_ST;
        let _e22: vec2<f32> = apply_st(in.uv, _e21);
        uv_main = _e22;
        let _e26: u32 = mat.flags;
        if ((_e26 & 4u) != 0u) {
            let _e34: vec4<f32> = mat._OffsetTex_ST;
            let _e35: vec2<f32> = apply_st(in.uv, _e34);
            let offset_s: vec4<f32> = textureSample(_OffsetTex, _OffsetTex_sampler, _e35);
            let _e39: vec2<f32> = uv_main;
            let _e43: vec4<f32> = mat._OffsetMagnitude;
            uv_main = (_e39 + (offset_s.xy * _e43.xy));
        }
        let _e49: vec2<f32> = uv_main;
        let t: vec4<f32> = textureSample(_Tex, _Tex_sampler, _e49);
        let _e54: f32 = mat._Color.w;
        let _e55: vec2<f32> = uv_main;
        let _e58: f32 = texture_alpha_base_mipX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX(_Tex, _Tex_sampler, _e55);
        clip_a = (_e54 * _e58);
        let _e60: vec4<f32> = albedo;
        albedo = (_e60 * t);
    }
    let _e64: u32 = mat.flags;
    if ((_e64 & 24u) != 0u) {
        let _e72: vec4<f32> = mat._MaskTex_ST;
        let _e73: vec2<f32> = apply_st(in.uv, _e72);
        let mask_1: vec4<f32> = textureSample(_MaskTex, _MaskTex_sampler, _e73);
        let mul: f32 = ((((mask_1.x + mask_1.y) + mask_1.z) * 0.33333334f) * mask_1.w);
        let _e88: f32 = mask_luminance_mul_base_mipX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX(_MaskTex, _MaskTex_sampler, _e73);
        let _e91: u32 = mat.flags;
        if ((_e91 & 8u) != 0u) {
            let _e98: f32 = albedo.w;
            albedo.w = (_e98 * mul);
            let _e100: f32 = clip_a;
            clip_a = (_e100 * _e88);
        }
        let _e104: u32 = mat.flags;
        if ((_e104 & 16u) != 0u) {
            let _e111: f32 = mat._Cutoff;
            if (_e88 <= _e111) {
                discard;
            }
        }
    }
    let _e115: u32 = mat.flags;
    if ((_e115 & 2u) != 0u) {
        let _e122: u32 = mat.flags;
        local = ((_e122 & 16u) == 0u);
    } else {
        local = false;
    }
    let _e130: bool = local;
    if _e130 {
        let _e131: f32 = clip_a;
        let _e134: f32 = mat._Cutoff;
        if (_e131 <= _e134) {
            discard;
        }
    }
    let _e138: u32 = mat.flags;
    if ((_e138 & 32u) != 0u) {
        let _e143: vec4<f32> = albedo;
        let _e146: f32 = albedo.w;
        let _e149: f32 = albedo.w;
        albedo = vec4<f32>((_e143.xyz * _e146), _e149);
    }
    let _e153: u32 = mat.flags;
    if ((_e153 & 64u) != 0u) {
        let _e159: f32 = albedo.x;
        let _e161: f32 = albedo.y;
        let _e164: f32 = albedo.z;
        let lum: f32 = (((_e159 + _e161) + _e164) * 0.33333334f);
        let _e170: f32 = albedo.w;
        albedo.w = (_e170 * lum);
    }
    let _e174: u32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.light_count;
    if (_e174 > 0u) {
        let _e180: u32 = lightsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX[0].light_type;
        lit = _e180;
    }
    let _e184: u32 = cluster_light_countsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX[0];
    let _e192: u32 = cluster_light_indicesX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX[0];
    let _e201: vec4<f32> = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.view_space_z_coeffs_right;
    let _e212: u32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.stereo_cluster_layers;
    let cluster_touch: f32 = (((f32((_e184 & 255u)) * 0.0000000001f) + (f32((_e192 & 255u)) * 0.0000000001f)) + ((dot(_e201, vec4<f32>(1f, 1f, 1f, 1f)) * 0.0000000001f) + (f32(_e212) * 0.0000000001f)));
    let _e218: vec4<f32> = albedo;
    let _e219: u32 = lit;
    return (_e218 + vec4<f32>(vec3(((f32(_e219) * 0.0000000001f) + cluster_touch)), 0f));
}
