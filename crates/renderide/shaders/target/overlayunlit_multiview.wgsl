struct PerDrawUniformsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX {
    view_proj_left: mat4x4<f32>,
    view_proj_right: mat4x4<f32>,
    model: mat4x4<f32>,
    _pad: array<vec4<f32>, 4>,
}

struct FrameGlobalsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX {
    camera_world_pos: vec4<f32>,
    view_space_z_coeffs: vec4<f32>,
    cluster_count_x: u32,
    cluster_count_y: u32,
    cluster_count_z: u32,
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

fn apply_st(uv_1: vec2<f32>, st: vec4<f32>) -> vec2<f32> {
    let uv_st: vec2<f32> = ((uv_1 * st.xy) + st.zw);
    return vec2<f32>(uv_st.x, (1f - uv_st.y));
}

fn polar_uv(raw_uv: vec2<f32>, radius_pow: f32) -> vec2<f32> {
    let centered: vec2<f32> = ((raw_uv * 2f) - vec2(1f));
    let radius: f32 = pow(length(centered), radius_pow);
    let angle: f32 = (atan2(centered.x, centered.y) + (6.2831855f * 0.5f));
    return vec2<f32>((angle / 6.2831855f), radius);
}

fn sample_layer(tex: texture_2d<f32>, samp: sampler, tint: vec4<f32>, uv_2: vec2<f32>, st_1: vec4<f32>) -> vec4<f32> {
    let _e2: f32 = mat._POLARUV;
    let use_polar: bool = (_e2 > 0.99f);
    let _e7: vec2<f32> = apply_st(uv_2, st_1);
    let _e10: f32 = mat._PolarPow;
    let _e11: vec2<f32> = polar_uv(uv_2, _e10);
    let _e12: vec2<f32> = apply_st(_e11, st_1);
    let sample_uv: vec2<f32> = select(_e7, _e12, use_polar);
    let _e17: vec4<f32> = textureSample(tex, samp, sample_uv);
    return (_e17 * tint);
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
    let _e26: f32 = mat._Cutoff;
    if (_e26 > 0f) {
        let _e31: f32 = mat._Cutoff;
        local = (_e31 < 1f);
    } else {
        local = false;
    }
    let _e37: bool = local;
    if _e37 {
        let _e39: f32 = color.w;
        let _e42: f32 = mat._Cutoff;
        local_1 = (_e39 <= _e42);
    } else {
        local_1 = false;
    }
    let _e47: bool = local_1;
    if _e47 {
        discard;
    }
    let _e50: f32 = mat._MUL_RGB_BY_ALPHA;
    if (_e50 > 0.99f) {
        let _e53: vec4<f32> = color;
        let _e56: f32 = color.w;
        let _e59: f32 = color.w;
        color = vec4<f32>((_e53.xyz * _e56), _e59);
    }
    let _e63: f32 = mat._MUL_ALPHA_INTENSITY;
    if (_e63 > 0.99f) {
        let _e67: f32 = color.x;
        let _e69: f32 = color.y;
        let _e72: f32 = color.z;
        let lum: f32 = (((_e67 + _e69) + _e72) * 0.33333334f);
        let _e78: f32 = color.w;
        color.w = (_e78 * lum);
    }
    let _e82: u32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.light_count;
    if (_e82 > 0u) {
        let _e88: u32 = lightsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX[0].light_type;
        lit = _e88;
    }
    let _e92: u32 = cluster_light_countsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX[0];
    let _e100: u32 = cluster_light_indicesX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX[0];
    let cluster_touch: f32 = ((f32((_e92 & 255u)) * 0.0000000001f) + (f32((_e100 & 255u)) * 0.0000000001f));
    let _e107: vec4<f32> = color;
    let _e108: u32 = lit;
    return (_e107 + vec4<f32>(vec3(((f32(_e108) * 0.0000000001f) + cluster_touch)), 0f));
}
