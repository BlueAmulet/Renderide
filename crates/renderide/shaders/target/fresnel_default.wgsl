struct PerDrawUniformsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX {
    view_proj_left: mat4x4<f32>,
    view_proj_right: mat4x4<f32>,
    model: mat4x4<f32>,
    normal_matrix: mat3x3<f32>,
    _pad: vec4<f32>,
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

struct FrameGlobalsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX {
    camera_world_pos: vec4<f32>,
    view_space_z_coeffs: vec4<f32>,
    view_space_z_coeffs_right: vec4<f32>,
    cluster_count_x: u32,
    cluster_count_y: u32,
    cluster_count_z: u32,
    near_clip: f32,
    far_clip: f32,
    light_count: u32,
    viewport_width: u32,
    viewport_height: u32,
}

struct FresnelMaterial {
    _FarColor: vec4<f32>,
    _NearColor: vec4<f32>,
    _FarTex_ST: vec4<f32>,
    _NearTex_ST: vec4<f32>,
    _MaskTex_ST: vec4<f32>,
    _Exp: f32,
    _GammaCurve: f32,
    _NormalScale: f32,
    _Cutoff: f32,
    _SrcBlend: f32,
    _DstBlend: f32,
    _ZWrite: f32,
    _Cull: f32,
    _ZTest: f32,
    _PolarPow: f32,
    _POLARUV: f32,
    _NORMALMAP: f32,
    _MASK_TEXTURE_MUL: f32,
    _MASK_TEXTURE_CLIP: f32,
    _MUL_ALPHA_INTENSITY: f32,
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

const CLIP_COVERAGE_LODX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX: f32 = 0f;

@group(2) @binding(0) 
var<storage> instancesX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX: array<PerDrawUniformsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX>;
@group(0) @binding(0) 
var<uniform> frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX: FrameGlobalsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX;
@group(0) @binding(1) 
var<storage> lightsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX: array<GpuLightX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX>;
@group(0) @binding(2) 
var<storage> cluster_light_countsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX: array<u32>;
@group(0) @binding(3) 
var<storage> cluster_light_indicesX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX: array<u32>;
@group(1) @binding(0) 
var<uniform> mat: FresnelMaterial;
@group(1) @binding(1) 
var _FarTex: texture_2d<f32>;
@group(1) @binding(2) 
var _FarTex_sampler: sampler;
@group(1) @binding(3) 
var _NearTex: texture_2d<f32>;
@group(1) @binding(4) 
var _NearTex_sampler: sampler;
@group(1) @binding(5) 
var _NormalMap: texture_2d<f32>;
@group(1) @binding(6) 
var _NormalMap_sampler: sampler;
@group(1) @binding(7) 
var _MaskTex: texture_2d<f32>;
@group(1) @binding(8) 
var _MaskTex_sampler: sampler;

fn get_drawX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX(instance_idx: u32) -> PerDrawUniformsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX {
    let _e3: PerDrawUniformsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX = instancesX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX[instance_idx];
    return _e3;
}

fn apply_stX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ2XMX3VORUWY4YX(uv_in: vec2<f32>, st: vec4<f32>) -> vec2<f32> {
    let uv_st: vec2<f32> = ((uv_in * st.xy) + st.zw);
    return vec2<f32>(uv_st.x, (1f - uv_st.y));
}

fn polar_uvX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ2XMX3VORUWY4YX(raw_uv: vec2<f32>, radius_pow: f32) -> vec2<f32> {
    let centered: vec2<f32> = ((raw_uv * 2f) - vec2(1f));
    let radius: f32 = pow(length(centered), radius_pow);
    let angle: f32 = (atan2(centered.x, centered.y) + (6.2831855f * 0.5f));
    return vec2<f32>((angle / 6.2831855f), radius);
}

fn texture_rgba_base_mipX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX(tex: texture_2d<f32>, samp: sampler, uv_1: vec2<f32>) -> vec4<f32> {
    let _e4: vec4<f32> = textureSampleLevel(tex, samp, uv_1, CLIP_COVERAGE_LODX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX);
    return _e4;
}

fn mask_luminance_mul_base_mipX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX(tex_1: texture_2d<f32>, samp_1: sampler, uv_2: vec2<f32>) -> f32 {
    let mask: vec4<f32> = textureSampleLevel(tex_1, samp_1, uv_2, CLIP_COVERAGE_LODX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX);
    return ((((mask.x + mask.y) + mask.z) * 0.33333334f) * mask.w);
}

fn orthonormal_tbnX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(n_2: vec3<f32>) -> mat3x3<f32> {
    let sign_: f32 = select(-1f, 1f, (n_2.z >= 0f));
    let a: f32 = (-1f / (sign_ + n_2.z));
    let b: f32 = ((n_2.x * n_2.y) * a);
    let t: vec3<f32> = vec3<f32>((1f + (((sign_ * n_2.x) * n_2.x) * a)), (sign_ * b), (-(sign_) * n_2.x));
    let bitan: vec3<f32> = vec3<f32>(b, (sign_ + ((n_2.y * n_2.y) * a)), -(n_2.y));
    return mat3x3<f32>(normalize(t), normalize(bitan), n_2);
}

fn decode_ts_normal_sample_rawX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJXG64TNMFWF6ZDFMNXWIZIX(s: vec4<f32>) -> vec3<f32> {
    var local_5: bool;

    let uniform_white_rgb: bool = all((s.xyz > vec3<f32>(0.99f, 0.99f, 0.99f)));
    if uniform_white_rgb {
        return s.xyz;
    }
    let all_r_high: bool = (s.x > 0.98f);
    let gb_close: bool = (abs((s.y - s.z)) < 0.03f);
    if all_r_high {
        local_5 = gb_close;
    } else {
        local_5 = false;
    }
    let _e21: bool = local_5;
    if _e21 {
        return vec3<f32>(s.w, s.y, s.z);
    }
    return s.xyz;
}

fn decode_ts_normal_with_placeholderX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJXG64TNMFWF6ZDFMNXWIZIX(raw: vec3<f32>, scale: f32) -> vec3<f32> {
    if all((raw > vec3<f32>(0.99f, 0.99f, 0.99f))) {
        return vec3<f32>(0f, 0f, 1f);
    }
    let nm_xy: vec2<f32> = (((raw.xy * 2f) - vec2(1f)) * scale);
    let z: f32 = max(sqrt(max((1f - dot(nm_xy, nm_xy)), 0f)), 0.000001f);
    return normalize(vec3<f32>(nm_xy, z));
}

fn decode_ts_normal_with_placeholder_sampleX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJXG64TNMFWF6ZDFMNXWIZIX(s_1: vec4<f32>, scale_1: f32) -> vec3<f32> {
    let _e1: vec3<f32> = decode_ts_normal_sample_rawX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJXG64TNMFWF6ZDFMNXWIZIX(s_1);
    let _e3: vec3<f32> = decode_ts_normal_with_placeholderX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJXG64TNMFWF6ZDFMNXWIZIX(_e1, scale_1);
    return _e3;
}

fn retain_globals_additiveX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX(color_1: vec4<f32>) -> vec4<f32> {
    var lit: u32 = 0u;

    let _e3: u32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.light_count;
    if (_e3 > 0u) {
        let _e9: u32 = lightsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX[0].light_type;
        lit = _e9;
    }
    let _e13: u32 = cluster_light_countsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX[0];
    let _e21: u32 = cluster_light_indicesX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX[0];
    let cluster_touch: f32 = ((f32((_e13 & 255u)) * 0.0000000001f) + (f32((_e21 & 255u)) * 0.0000000001f));
    let _e28: u32 = lit;
    return (color_1 + vec4<f32>(vec3(((f32(_e28) * 0.0000000001f) + cluster_touch)), 0f));
}

fn sample_color(tex_2: texture_2d<f32>, samp_2: sampler, uv_3: vec2<f32>, st_1: vec4<f32>) -> vec4<f32> {
    let _e2: f32 = mat._POLARUV;
    let use_polar: bool = (_e2 > 0.99f);
    let _e7: vec2<f32> = apply_stX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ2XMX3VORUWY4YX(uv_3, st_1);
    let _e10: f32 = mat._PolarPow;
    let _e11: vec2<f32> = polar_uvX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ2XMX3VORUWY4YX(uv_3, _e10);
    let _e12: vec2<f32> = apply_stX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ2XMX3VORUWY4YX(_e11, st_1);
    let sample_uv: vec2<f32> = select(_e7, _e12, use_polar);
    let _e16: vec4<f32> = textureSample(tex_2, samp_2, sample_uv);
    return _e16;
}

fn sample_color_lod0_(tex_3: texture_2d<f32>, samp_3: sampler, uv_4: vec2<f32>, st_2: vec4<f32>) -> vec4<f32> {
    let _e2: f32 = mat._POLARUV;
    let use_polar_1: bool = (_e2 > 0.99f);
    let _e7: vec2<f32> = apply_stX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ2XMX3VORUWY4YX(uv_4, st_2);
    let _e10: f32 = mat._PolarPow;
    let _e11: vec2<f32> = polar_uvX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ2XMX3VORUWY4YX(uv_4, _e10);
    let _e12: vec2<f32> = apply_stX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ2XMX3VORUWY4YX(_e11, st_2);
    let sample_uv_1: vec2<f32> = select(_e7, _e12, use_polar_1);
    let _e16: vec4<f32> = texture_rgba_base_mipX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX(tex_3, samp_3, sample_uv_1);
    return _e16;
}

@vertex 
fn vs_main(@builtin(instance_index) instance_index: u32, @location(0) pos: vec4<f32>, @location(1) n: vec4<f32>, @location(2) uv: vec2<f32>) -> VertexOutput {
    var out: VertexOutput;

    let _e1: PerDrawUniformsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX = get_drawX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX(instance_index);
    let world_p: vec4<f32> = (_e1.model * vec4<f32>(pos.xyz, 1f));
    let wn: vec3<f32> = normalize((_e1.normal_matrix * n.xyz));
    let vp: mat4x4<f32> = _e1.view_proj_left;
    out.clip_pos = (vp * world_p);
    out.world_pos = world_p.xyz;
    out.world_n = wn;
    out.uv = uv;
    let _e22: VertexOutput = out;
    return _e22;
}

@fragment 
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var n_1: vec3<f32>;
    var fres: f32;
    var color: vec4<f32>;
    var clip_a: f32;
    var local: bool;
    var local_1: bool;
    var local_2: bool;
    var local_3: bool;
    var local_4: bool;

    n_1 = normalize(in.world_n);
    let _e6: f32 = mat._NORMALMAP;
    if (_e6 > 0.99f) {
        let uv_n: vec2<f32> = vec2<f32>(in.uv.x, (1f - in.uv.y));
        let _e16: vec3<f32> = n_1;
        let _e17: mat3x3<f32> = orthonormal_tbnX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(_e16);
        let _e20: vec4<f32> = textureSample(_NormalMap, _NormalMap_sampler, uv_n);
        let _e23: f32 = mat._NormalScale;
        let _e24: vec3<f32> = decode_ts_normal_with_placeholder_sampleX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJXG64TNMFWF6ZDFMNXWIZIX(_e20, _e23);
        n_1 = normalize((_e17 * _e24));
    }
    let _e29: vec4<f32> = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.camera_world_pos;
    let view_dir: vec3<f32> = normalize((_e29.xyz - in.world_pos));
    let _e34: vec3<f32> = n_1;
    let _e41: f32 = mat._Exp;
    fres = pow((1f - abs(dot(_e34, view_dir))), max(_e41, 0.0001f));
    let _e46: f32 = fres;
    let _e52: f32 = mat._GammaCurve;
    fres = pow(clamp(_e46, 0f, 1f), max(_e52, 0.0001f));
    let _e58: vec4<f32> = mat._FarColor;
    let _e62: vec4<f32> = mat._FarTex_ST;
    let _e65: vec4<f32> = sample_color(_FarTex, _FarTex_sampler, in.uv, _e62);
    let far_color: vec4<f32> = (_e58 * _e65);
    let _e69: vec4<f32> = mat._NearColor;
    let _e73: vec4<f32> = mat._NearTex_ST;
    let _e76: vec4<f32> = sample_color(_NearTex, _NearTex_sampler, in.uv, _e73);
    let near_color: vec4<f32> = (_e69 * _e76);
    let _e78: f32 = fres;
    color = mix(near_color, far_color, clamp(_e78, 0f, 1f));
    let _e86: vec4<f32> = mat._FarColor;
    let _e90: vec4<f32> = mat._FarTex_ST;
    let _e93: vec4<f32> = sample_color_lod0_(_FarTex, _FarTex_sampler, in.uv, _e90);
    let far_clip: vec4<f32> = (_e86 * _e93);
    let _e97: vec4<f32> = mat._NearColor;
    let _e101: vec4<f32> = mat._NearTex_ST;
    let _e104: vec4<f32> = sample_color_lod0_(_NearTex, _NearTex_sampler, in.uv, _e101);
    let near_clip: vec4<f32> = (_e97 * _e104);
    let _e108: f32 = fres;
    clip_a = mix(near_clip.w, far_clip.w, clamp(_e108, 0f, 1f));
    let _e116: f32 = mat._MASK_TEXTURE_MUL;
    if !((_e116 > 0.99f)) {
        let _e122: f32 = mat._MASK_TEXTURE_CLIP;
        local = (_e122 > 0.99f);
    } else {
        local = true;
    }
    let _e128: bool = local;
    if _e128 {
        let _e132: vec4<f32> = mat._MaskTex_ST;
        let _e133: vec2<f32> = apply_stX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJ2XMX3VORUWY4YX(in.uv, _e132);
        let mask_1: vec4<f32> = textureSample(_MaskTex, _MaskTex_sampler, _e133);
        let mul: f32 = ((((mask_1.x + mask_1.y) + mask_1.z) * 0.33333334f) * mask_1.w);
        let _e148: f32 = mask_luminance_mul_base_mipX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX(_MaskTex, _MaskTex_sampler, _e133);
        let _e151: f32 = mat._MASK_TEXTURE_MUL;
        if (_e151 > 0.99f) {
            let _e156: f32 = color.w;
            color.w = (_e156 * mul);
            let _e158: f32 = clip_a;
            clip_a = (_e158 * _e148);
        }
        let _e162: f32 = mat._MASK_TEXTURE_CLIP;
        if (_e162 > 0.99f) {
            let _e167: f32 = mat._Cutoff;
            local_1 = (_e148 <= _e167);
        } else {
            local_1 = false;
        }
        let _e172: bool = local_1;
        if _e172 {
            discard;
        }
    }
    let _e175: f32 = mat._MASK_TEXTURE_CLIP;
    if !((_e175 > 0.99f)) {
        let _e181: f32 = mat._Cutoff;
        local_2 = (_e181 > 0f);
    } else {
        local_2 = false;
    }
    let _e187: bool = local_2;
    if _e187 {
        let _e190: f32 = mat._Cutoff;
        local_3 = (_e190 < 1f);
    } else {
        local_3 = false;
    }
    let _e196: bool = local_3;
    if _e196 {
        let _e197: f32 = clip_a;
        let _e200: f32 = mat._Cutoff;
        local_4 = (_e197 <= _e200);
    } else {
        local_4 = false;
    }
    let _e205: bool = local_4;
    if _e205 {
        discard;
    }
    let _e208: f32 = mat._MUL_ALPHA_INTENSITY;
    if (_e208 > 0.99f) {
        let _e212: f32 = color.x;
        let _e214: f32 = color.y;
        let _e217: f32 = color.z;
        let lum: f32 = (((_e212 + _e214) + _e217) * 0.33333334f);
        let _e223: f32 = color.w;
        color.w = ((_e223 * lum) * lum);
    }
    let _e226: vec4<f32> = color;
    let _e227: vec4<f32> = retain_globals_additiveX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX(_e226);
    return _e227;
}
