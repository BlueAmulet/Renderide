struct PerDrawUniformsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX {
    view_proj_left: mat4x4<f32>,
    view_proj_right: mat4x4<f32>,
    model: mat4x4<f32>,
    _pad: array<vec4<f32>, 4>,
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
    stereo_cluster_layers: u32,
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

fn texture_rgba_base_mipX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX(tex: texture_2d<f32>, samp: sampler, uv_1: vec2<f32>) -> vec4<f32> {
    let _e4: vec4<f32> = textureSampleLevel(tex, samp, uv_1, CLIP_COVERAGE_LODX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX);
    return _e4;
}

fn mask_luminance_mul_base_mipX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX(tex_1: texture_2d<f32>, samp_1: sampler, uv_2: vec2<f32>) -> f32 {
    let mask: vec4<f32> = textureSampleLevel(tex_1, samp_1, uv_2, CLIP_COVERAGE_LODX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX);
    return ((((mask.x + mask.y) + mask.z) * 0.33333334f) * mask.w);
}

fn orthonormal_tbnX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(n_2: vec3<f32>) -> mat3x3<f32> {
    let up: vec3<f32> = select(vec3<f32>(0f, 1f, 0f), vec3<f32>(1f, 0f, 0f), (abs(n_2.y) > 0.99f));
    let t: vec3<f32> = normalize(cross(up, n_2));
    let b: vec3<f32> = cross(n_2, t);
    return mat3x3<f32>(t, b, n_2);
}

fn apply_st(uv_3: vec2<f32>, st: vec4<f32>) -> vec2<f32> {
    let uv_st: vec2<f32> = ((uv_3 * st.xy) + st.zw);
    return vec2<f32>(uv_st.x, (1f - uv_st.y));
}

fn polar_uv(raw_uv: vec2<f32>, radius_pow: f32) -> vec2<f32> {
    let centered: vec2<f32> = ((raw_uv * 2f) - vec2(1f));
    let radius: f32 = pow(length(centered), radius_pow);
    let angle: f32 = (atan2(centered.x, centered.y) + (6.2831855f * 0.5f));
    return vec2<f32>((angle / 6.2831855f), radius);
}

fn decode_ts_normal(raw: vec3<f32>, scale: f32) -> vec3<f32> {
    let nm_xy: vec2<f32> = (((raw.xy * 2f) - vec2(1f)) * scale);
    let z: f32 = max(sqrt(max((1f - dot(nm_xy, nm_xy)), 0f)), 0.000001f);
    return normalize(vec3<f32>(nm_xy, z));
}

fn sample_color(tex_2: texture_2d<f32>, samp_2: sampler, uv_4: vec2<f32>, st_1: vec4<f32>) -> vec4<f32> {
    let _e2: f32 = mat._POLARUV;
    let use_polar: bool = (_e2 > 0.99f);
    let _e7: vec2<f32> = apply_st(uv_4, st_1);
    let _e10: f32 = mat._PolarPow;
    let _e11: vec2<f32> = polar_uv(uv_4, _e10);
    let _e12: vec2<f32> = apply_st(_e11, st_1);
    let sample_uv: vec2<f32> = select(_e7, _e12, use_polar);
    let _e16: vec4<f32> = textureSample(tex_2, samp_2, sample_uv);
    return _e16;
}

fn sample_color_lod0_(tex_3: texture_2d<f32>, samp_3: sampler, uv_5: vec2<f32>, st_2: vec4<f32>) -> vec4<f32> {
    let _e2: f32 = mat._POLARUV;
    let use_polar_1: bool = (_e2 > 0.99f);
    let _e7: vec2<f32> = apply_st(uv_5, st_2);
    let _e10: f32 = mat._PolarPow;
    let _e11: vec2<f32> = polar_uv(uv_5, _e10);
    let _e12: vec2<f32> = apply_st(_e11, st_2);
    let sample_uv_1: vec2<f32> = select(_e7, _e12, use_polar_1);
    let _e16: vec4<f32> = texture_rgba_base_mipX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX(tex_3, samp_3, sample_uv_1);
    return _e16;
}

@vertex 
fn vs_main(@builtin(view_index) view_idx: u32, @location(0) pos: vec4<f32>, @location(1) n: vec4<f32>, @location(2) uv: vec2<f32>) -> VertexOutput {
    var vp: mat4x4<f32>;
    var out: VertexOutput;

    let _e3: mat4x4<f32> = drawX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX.model;
    let world_p: vec4<f32> = (_e3 * vec4<f32>(pos.xyz, 1f));
    let _e11: mat4x4<f32> = drawX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX.model;
    let wn: vec3<f32> = normalize((_e11 * vec4<f32>(n.xyz, 0f)).xyz);
    if (view_idx == 0u) {
        let _e23: mat4x4<f32> = drawX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX.view_proj_left;
        vp = _e23;
    } else {
        let _e27: mat4x4<f32> = drawX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGK4S7MRZGC5YX.view_proj_right;
        vp = _e27;
    }
    let _e30: mat4x4<f32> = vp;
    out.clip_pos = (_e30 * world_p);
    out.world_pos = world_p.xyz;
    out.world_n = wn;
    out.uv = uv;
    let _e37: VertexOutput = out;
    return _e37;
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
    var lit: u32 = 0u;

    n_1 = normalize(in.world_n);
    let _e7: f32 = mat._NORMALMAP;
    if (_e7 > 0.99f) {
        let uv_n: vec2<f32> = vec2<f32>(in.uv.x, (1f - in.uv.y));
        let _e17: vec3<f32> = n_1;
        let _e18: mat3x3<f32> = orthonormal_tbnX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJYGE4Z2HJRHEZDGX(_e17);
        let _e21: vec4<f32> = textureSample(_NormalMap, _NormalMap_sampler, uv_n);
        let _e25: f32 = mat._NormalScale;
        let _e26: vec3<f32> = decode_ts_normal(_e21.xyz, _e25);
        n_1 = normalize((_e18 * _e26));
    }
    let _e31: vec4<f32> = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.camera_world_pos;
    let view_dir: vec3<f32> = normalize((_e31.xyz - in.world_pos));
    let _e36: vec3<f32> = n_1;
    let _e43: f32 = mat._Exp;
    fres = pow((1f - abs(dot(_e36, view_dir))), max(_e43, 0.0001f));
    let _e48: f32 = fres;
    let _e54: f32 = mat._GammaCurve;
    fres = pow(clamp(_e48, 0f, 1f), max(_e54, 0.0001f));
    let _e60: vec4<f32> = mat._FarColor;
    let _e64: vec4<f32> = mat._FarTex_ST;
    let _e67: vec4<f32> = sample_color(_FarTex, _FarTex_sampler, in.uv, _e64);
    let far_color: vec4<f32> = (_e60 * _e67);
    let _e71: vec4<f32> = mat._NearColor;
    let _e75: vec4<f32> = mat._NearTex_ST;
    let _e78: vec4<f32> = sample_color(_NearTex, _NearTex_sampler, in.uv, _e75);
    let near_color: vec4<f32> = (_e71 * _e78);
    let _e80: f32 = fres;
    color = mix(near_color, far_color, clamp(_e80, 0f, 1f));
    let _e88: vec4<f32> = mat._FarColor;
    let _e92: vec4<f32> = mat._FarTex_ST;
    let _e95: vec4<f32> = sample_color_lod0_(_FarTex, _FarTex_sampler, in.uv, _e92);
    let far_clip: vec4<f32> = (_e88 * _e95);
    let _e99: vec4<f32> = mat._NearColor;
    let _e103: vec4<f32> = mat._NearTex_ST;
    let _e106: vec4<f32> = sample_color_lod0_(_NearTex, _NearTex_sampler, in.uv, _e103);
    let near_clip: vec4<f32> = (_e99 * _e106);
    let _e110: f32 = fres;
    clip_a = mix(near_clip.w, far_clip.w, clamp(_e110, 0f, 1f));
    let _e118: f32 = mat._MASK_TEXTURE_MUL;
    if !((_e118 > 0.99f)) {
        let _e124: f32 = mat._MASK_TEXTURE_CLIP;
        local = (_e124 > 0.99f);
    } else {
        local = true;
    }
    let _e130: bool = local;
    if _e130 {
        let _e134: vec4<f32> = mat._MaskTex_ST;
        let _e135: vec2<f32> = apply_st(in.uv, _e134);
        let mask_1: vec4<f32> = textureSample(_MaskTex, _MaskTex_sampler, _e135);
        let mul: f32 = ((((mask_1.x + mask_1.y) + mask_1.z) * 0.33333334f) * mask_1.w);
        let _e150: f32 = mask_luminance_mul_base_mipX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJQWY4DIMFPWG3DJOBPXGYLNOBWGKX(_MaskTex, _MaskTex_sampler, _e135);
        let _e153: f32 = mat._MASK_TEXTURE_MUL;
        if (_e153 > 0.99f) {
            let _e158: f32 = color.w;
            color.w = (_e158 * mul);
            let _e160: f32 = clip_a;
            clip_a = (_e160 * _e150);
        }
        let _e164: f32 = mat._MASK_TEXTURE_CLIP;
        if (_e164 > 0.99f) {
            let _e169: f32 = mat._Cutoff;
            local_1 = (_e150 <= _e169);
        } else {
            local_1 = false;
        }
        let _e174: bool = local_1;
        if _e174 {
            discard;
        }
    }
    let _e177: f32 = mat._MASK_TEXTURE_CLIP;
    if !((_e177 > 0.99f)) {
        let _e183: f32 = mat._Cutoff;
        local_2 = (_e183 > 0f);
    } else {
        local_2 = false;
    }
    let _e189: bool = local_2;
    if _e189 {
        let _e192: f32 = mat._Cutoff;
        local_3 = (_e192 < 1f);
    } else {
        local_3 = false;
    }
    let _e198: bool = local_3;
    if _e198 {
        let _e199: f32 = clip_a;
        let _e202: f32 = mat._Cutoff;
        local_4 = (_e199 <= _e202);
    } else {
        local_4 = false;
    }
    let _e207: bool = local_4;
    if _e207 {
        discard;
    }
    let _e210: f32 = mat._MUL_ALPHA_INTENSITY;
    if (_e210 > 0.99f) {
        let _e214: f32 = color.x;
        let _e216: f32 = color.y;
        let _e219: f32 = color.z;
        let lum: f32 = (((_e214 + _e216) + _e219) * 0.33333334f);
        let _e225: f32 = color.w;
        color.w = ((_e225 * lum) * lum);
    }
    let _e230: u32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.light_count;
    if (_e230 > 0u) {
        let _e236: u32 = lightsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX[0].light_type;
        lit = _e236;
    }
    let _e240: u32 = cluster_light_countsX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX[0];
    let _e248: u32 = cluster_light_indicesX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX[0];
    let _e257: vec4<f32> = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.view_space_z_coeffs_right;
    let _e268: u32 = frameX_naga_oil_mod_XOJSW4ZDFOJUWIZJ2HJTWY33CMFWHGX.stereo_cluster_layers;
    let cluster_touch: f32 = (((f32((_e240 & 255u)) * 0.0000000001f) + (f32((_e248 & 255u)) * 0.0000000001f)) + ((dot(_e257, vec4<f32>(1f, 1f, 1f, 1f)) * 0.0000000001f) + (f32(_e268) * 0.0000000001f)));
    let _e274: vec4<f32> = color;
    let _e275: u32 = lit;
    return (_e274 + vec4<f32>(vec3(((f32(_e275) * 0.0000000001f) + cluster_touch)), 0f));
}
