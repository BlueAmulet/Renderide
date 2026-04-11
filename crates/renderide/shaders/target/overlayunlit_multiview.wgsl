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

struct OverlayUnlitMaterial {
    _BehindColor: vec4<f32>,
    _FrontColor: vec4<f32>,
    _BehindTex_ST: vec4<f32>,
    _FrontTex_ST: vec4<f32>,
    _Cutoff: f32,
    _PolarPow: f32,
    _SrcBlend: f32,
    _DstBlend: f32,
    _ZWrite: f32,
    _Cull: f32,
    _POLARUV: f32,
    _MUL_RGB_BY_ALPHA: f32,
    _MUL_ALPHA_INTENSITY: f32,
    _pad0_: f32,
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
var<uniform> mat: OverlayUnlitMaterial;
@group(1) @binding(1) 
var _BehindTex: texture_2d<f32>;
@group(1) @binding(2) 
var _BehindTex_sampler: sampler;
@group(1) @binding(3) 
var _FrontTex: texture_2d<f32>;
@group(1) @binding(4) 
var _FrontTex_sampler: sampler;

fn texture_rgba_base_mipX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX(tex: texture_2d<f32>, samp: sampler, uv_1: vec2<f32>) -> vec4<f32> {
    let _e4: vec4<f32> = textureSampleLevel(tex, samp, uv_1, CLIP_COVERAGE_LODX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX);
    return _e4;
}

fn apply_st(uv_2: vec2<f32>, st: vec4<f32>) -> vec2<f32> {
    let uv_st: vec2<f32> = ((uv_2 * st.xy) + st.zw);
    return vec2<f32>(uv_st.x, (1f - uv_st.y));
}

fn polar_uv(raw_uv: vec2<f32>, radius_pow: f32) -> vec2<f32> {
    let centered: vec2<f32> = ((raw_uv * 2f) - vec2(1f));
    let radius: f32 = pow(length(centered), radius_pow);
    let angle: f32 = (atan2(centered.x, centered.y) + (6.2831855f * 0.5f));
    return vec2<f32>((angle / 6.2831855f), radius);
}

fn sample_layer(tex_1: texture_2d<f32>, samp_1: sampler, tint: vec4<f32>, uv_3: vec2<f32>, st_1: vec4<f32>) -> vec4<f32> {
    let _e2: f32 = mat._POLARUV;
    let use_polar: bool = (_e2 > 0.99f);
    let _e7: vec2<f32> = apply_st(uv_3, st_1);
    let _e10: f32 = mat._PolarPow;
    let _e11: vec2<f32> = polar_uv(uv_3, _e10);
    let _e12: vec2<f32> = apply_st(_e11, st_1);
    let sample_uv: vec2<f32> = select(_e7, _e12, use_polar);
    let _e17: vec4<f32> = textureSample(tex_1, samp_1, sample_uv);
    return (_e17 * tint);
}

fn sample_layer_lod0_(tex_2: texture_2d<f32>, samp_2: sampler, tint_1: vec4<f32>, uv_4: vec2<f32>, st_2: vec4<f32>) -> vec4<f32> {
    let _e2: f32 = mat._POLARUV;
    let use_polar_1: bool = (_e2 > 0.99f);
    let _e7: vec2<f32> = apply_st(uv_4, st_2);
    let _e10: f32 = mat._PolarPow;
    let _e11: vec2<f32> = polar_uv(uv_4, _e10);
    let _e12: vec2<f32> = apply_st(_e11, st_2);
    let sample_uv_1: vec2<f32> = select(_e7, _e12, use_polar_1);
    let _e16: vec4<f32> = texture_rgba_base_mipX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX(tex_2, samp_2, sample_uv_1);
    return (_e16 * tint_1);
}

fn alpha_over(front: vec4<f32>, behind: vec4<f32>) -> vec4<f32> {
    let out_a: f32 = (front.w + (behind.w * (1f - front.w)));
    if (out_a <= 0.000001f) {
        return vec4(0f);
    }
    let out_rgb: vec3<f32> = (((front.xyz * front.w) + ((behind.xyz * behind.w) * (1f - front.w))) / vec3(out_a));
    return vec4<f32>(out_rgb, out_a);
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
    var color: vec4<f32>;
    var local: bool;
    var local_1: bool;
    var lit: u32 = 0u;

    let _e4: vec4<f32> = mat._BehindColor;
    let _e8: vec4<f32> = mat._BehindTex_ST;
    let _e11: vec4<f32> = sample_layer(_BehindTex, _BehindTex_sampler, _e4, in.uv, _e8);
    let _e14: vec4<f32> = mat._FrontColor;
    let _e18: vec4<f32> = mat._FrontTex_ST;
    let _e21: vec4<f32> = sample_layer(_FrontTex, _FrontTex_sampler, _e14, in.uv, _e18);
    let _e22: vec4<f32> = alpha_over(_e21, _e11);
    color = _e22;
    let _e26: vec4<f32> = mat._BehindColor;
    let _e30: vec4<f32> = mat._BehindTex_ST;
    let _e33: vec4<f32> = sample_layer_lod0_(_BehindTex, _BehindTex_sampler, _e26, in.uv, _e30);
    let _e36: vec4<f32> = mat._FrontColor;
    let _e40: vec4<f32> = mat._FrontTex_ST;
    let _e43: vec4<f32> = sample_layer_lod0_(_FrontTex, _FrontTex_sampler, _e36, in.uv, _e40);
    let _e44: vec4<f32> = alpha_over(_e43, _e33);
    let _e47: f32 = mat._Cutoff;
    if (_e47 > 0f) {
        let _e52: f32 = mat._Cutoff;
        local = (_e52 < 1f);
    } else {
        local = false;
    }
    let _e58: bool = local;
    if _e58 {
        let _e62: f32 = mat._Cutoff;
        local_1 = (_e44.w <= _e62);
    } else {
        local_1 = false;
    }
    let _e67: bool = local_1;
    if _e67 {
        discard;
    }
    let _e70: f32 = mat._MUL_RGB_BY_ALPHA;
    if (_e70 > 0.99f) {
        let _e73: vec4<f32> = color;
        let _e76: f32 = color.w;
        let _e79: f32 = color.w;
        color = vec4<f32>((_e73.xyz * _e76), _e79);
    }
    let _e83: f32 = mat._MUL_ALPHA_INTENSITY;
    if (_e83 > 0.99f) {
        let _e87: f32 = color.x;
        let _e89: f32 = color.y;
        let _e92: f32 = color.z;
        let lum: f32 = (((_e87 + _e89) + _e92) * 0.33333334f);
        let _e98: f32 = color.w;
        color.w = (_e98 * lum);
    }
    let _e102: u32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.light_count;
    if (_e102 > 0u) {
        let _e108: u32 = lightsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX[0].light_type;
        lit = _e108;
    }
    let _e112: u32 = cluster_light_countsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX[0];
    let _e120: u32 = cluster_light_indicesX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX[0];
    let _e129: vec4<f32> = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.view_space_z_coeffs_right;
    let _e140: u32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.stereo_cluster_layers;
    let cluster_touch: f32 = (((f32((_e112 & 255u)) * 0.0000000001f) + (f32((_e120 & 255u)) * 0.0000000001f)) + ((dot(_e129, vec4<f32>(1f, 1f, 1f, 1f)) * 0.0000000001f) + (f32(_e140) * 0.0000000001f)));
    let _e146: vec4<f32> = color;
    let _e147: u32 = lit;
    return (_e146 + vec4<f32>(vec3(((f32(_e147) * 0.0000000001f) + cluster_touch)), 0f));
}
