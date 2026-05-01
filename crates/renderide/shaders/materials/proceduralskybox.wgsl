//! Unity ProceduralSkybox asset (`Shader "ProceduralSky"`): analytic sky material with
//! per-vertex Rayleigh+Mie scattering and three sun-disk modes (NONE / SIMPLE / HIGH_QUALITY).
//!
//! The renderer pipeline operates entirely in linear color space, so this port implements
//! the linear branch of the original shader only; the gamma-space branch and
//! `SKYBOX_COLOR_IN_TARGET_COLOR_SPACE` short-circuit are intentionally omitted. Sun-disk
//! mode is selected at runtime via the `_SUNDISK_*` keyword floats, mirroring the host's
//! material keyword routing.


#import renderide::globals as rg
#import renderide::per_draw as pd
#import renderide::mesh::vertex as mv
#import renderide::uv_utils as uvu

struct ProceduralSkyboxMaterial {
    _SkyTint: vec4<f32>,
    _GroundColor: vec4<f32>,
    _SunColor: vec4<f32>,
    _SunDirection: vec4<f32>,
    _Exposure: f32,
    _SunSize: f32,
    _AtmosphereThickness: f32,
    _SUNDISK_NONE: f32,
    _SUNDISK_SIMPLE: f32,
    _SUNDISK_HIGH_QUALITY: f32,
}

@group(1) @binding(0) var<uniform> mat: ProceduralSkyboxMaterial;

const PI: f32 = 3.14159265358979323846;
const OUTER_RADIUS: f32 = 1.025;
const INNER_RADIUS: f32 = 1.0;
const OUTER_RADIUS_SQ: f32 = OUTER_RADIUS * OUTER_RADIUS;
const INNER_RADIUS_SQ: f32 = INNER_RADIUS * INNER_RADIUS;
const CAMERA_HEIGHT: f32 = 0.0001;
const KMIE: f32 = 0.0010;
const KSUN_BRIGHTNESS: f32 = 20.0;
const KMAX_SCATTER: f32 = 50.0;
const KSUN_SCALE: f32 = 400.0 * KSUN_BRIGHTNESS;
const KKM_ESUN: f32 = KMIE * KSUN_BRIGHTNESS;
const KKM_4PI: f32 = KMIE * 4.0 * PI;
const KSCALE: f32 = 1.0 / (OUTER_RADIUS - 1.0);
const KSCALE_DEPTH: f32 = 0.25;
const KSCALE_OVER_SCALE_DEPTH: f32 = (1.0 / (OUTER_RADIUS - 1.0)) / 0.25;
const KSAMPLES: f32 = 2.0;
const MIE_G: f32 = -0.990;
const MIE_G2: f32 = 0.9801;
const SKY_GROUND_THRESHOLD: f32 = 0.02;
const GAMMA: f32 = 2.2;

const DEFAULT_SCATTERING_WAVELENGTH: vec3<f32> = vec3<f32>(0.65, 0.57, 0.475);
const VARIABLE_RANGE_SCATTERING_WAVELENGTH: vec3<f32> = vec3<f32>(0.15, 0.15, 0.15);

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_ray: vec3<f32>,
    @location(1) ground_color: vec3<f32>,
    @location(2) sky_color: vec3<f32>,
    @location(3) sun_color: vec3<f32>,
}

fn scale_factor(in_cos: f32) -> f32 {
    let x = 1.0 - in_cos;
    return 0.25 * exp(-0.00287 + x * (0.459 + x * (3.83 + x * (-6.80 + x * 5.25))));
}

fn rayleigh_phase_from_cos2(eye_cos2: f32) -> f32 {
    return 0.75 + 0.75 * eye_cos2;
}

fn rayleigh_phase(light: vec3<f32>, ray: vec3<f32>) -> f32 {
    let eye_cos = dot(light, ray);
    return rayleigh_phase_from_cos2(eye_cos * eye_cos);
}

fn mie_phase(eye_cos: f32, eye_cos2: f32, sun_size: f32) -> f32 {
    var temp = 1.0 + MIE_G2 - 2.0 * MIE_G * eye_cos;
    temp = pow(temp, pow(sun_size, 0.65) * 10.0);
    temp = max(temp, 1.0e-4);
    return 1.5 * ((1.0 - MIE_G2) / (2.0 + MIE_G2)) * (1.0 + eye_cos2) / temp;
}

fn calc_sun_spot(v1: vec3<f32>, v2: vec3<f32>, sun_size: f32) -> f32 {
    let delta = v1 - v2;
    let dist = length(delta);
    let spot = 1.0 - smoothstep(0.0, sun_size, dist);
    return KSUN_SCALE * spot * spot;
}

struct ScatteringStep {
    contribution: vec3<f32>,
    attenuate: vec3<f32>,
}

struct ScatteringOutput {
    c_in: vec3<f32>,
    c_out: vec3<f32>,
}

fn scattering_inscatter_step(
    sample_point: vec3<f32>,
    eye_ray: vec3<f32>,
    sun_dir: vec3<f32>,
    inv_wavelength: vec3<f32>,
    kkr_4pi: f32,
    start_offset: f32,
    scaled_length: f32,
) -> ScatteringStep {
    let h = length(sample_point);
    let depth = exp(KSCALE_OVER_SCALE_DEPTH * (INNER_RADIUS - h));
    let light_angle = dot(sun_dir, sample_point) / h;
    let camera_angle = dot(eye_ray, sample_point) / h;
    let scatter = start_offset + depth * (scale_factor(light_angle) - scale_factor(camera_angle));
    let attenuate = exp(-clamp(scatter, 0.0, KMAX_SCATTER) * (inv_wavelength * kkr_4pi + KKM_4PI));
    return ScatteringStep(attenuate * (depth * scaled_length), attenuate);
}

fn ground_inscatter_step(
    sample_point: vec3<f32>,
    inv_wavelength: vec3<f32>,
    kkr_4pi: f32,
    temp: f32,
    camera_offset: f32,
    scaled_length: f32,
) -> ScatteringStep {
    let h = length(sample_point);
    let depth = exp(KSCALE_OVER_SCALE_DEPTH * (INNER_RADIUS - h));
    let scatter = depth * temp - camera_offset;
    let attenuate = exp(-clamp(scatter, 0.0, KMAX_SCATTER) * (inv_wavelength * kkr_4pi + KKM_4PI));
    return ScatteringStep(attenuate * (depth * scaled_length), attenuate);
}

fn evaluate_scattering(
    eye_ray: vec3<f32>,
    sun_dir: vec3<f32>,
    inv_wavelength: vec3<f32>,
    krayleigh: f32,
) -> ScatteringOutput {
    let kkr_esun = krayleigh * KSUN_BRIGHTNESS;
    let kkr_4pi = krayleigh * 4.0 * PI;
    let camera_pos = vec3<f32>(0.0, INNER_RADIUS + CAMERA_HEIGHT, 0.0);

    var c_in: vec3<f32>;
    var c_out: vec3<f32>;

    if (eye_ray.y >= 0.0) {
        let far = sqrt(OUTER_RADIUS_SQ + INNER_RADIUS_SQ * eye_ray.y * eye_ray.y - INNER_RADIUS_SQ)
            - INNER_RADIUS * eye_ray.y;
        let height = INNER_RADIUS + CAMERA_HEIGHT;
        let depth_init = exp(KSCALE_OVER_SCALE_DEPTH * (-CAMERA_HEIGHT));
        let start_angle = dot(eye_ray, camera_pos) / height;
        let start_offset = depth_init * scale_factor(start_angle);

        let sample_length = far / KSAMPLES;
        let scaled_length = sample_length * KSCALE;
        let sample_ray = eye_ray * sample_length;
        var sample_point = camera_pos + sample_ray * 0.5;

        var front_color = vec3<f32>(0.0);
        let s0 = scattering_inscatter_step(
            sample_point, eye_ray, sun_dir, inv_wavelength, kkr_4pi, start_offset, scaled_length,
        );
        front_color = front_color + s0.contribution;
        sample_point = sample_point + sample_ray;

        let s1 = scattering_inscatter_step(
            sample_point, eye_ray, sun_dir, inv_wavelength, kkr_4pi, start_offset, scaled_length,
        );
        front_color = front_color + s1.contribution;

        c_in = front_color * (inv_wavelength * kkr_esun);
        c_out = front_color * KKM_ESUN;
    } else {
        let far = (-CAMERA_HEIGHT) / min(-0.001, eye_ray.y);
        let pos = camera_pos + far * eye_ray;
        let depth = exp((-CAMERA_HEIGHT) * (1.0 / KSCALE_DEPTH));
        let camera_angle = dot(-eye_ray, pos);
        let light_angle = dot(sun_dir, pos);
        let camera_scale = scale_factor(camera_angle);
        let light_scale = scale_factor(light_angle);
        let camera_offset = depth * camera_scale;
        let temp = light_scale + camera_scale;

        let sample_length = far / KSAMPLES;
        let scaled_length = sample_length * KSCALE;
        let sample_ray = eye_ray * sample_length;
        let sample_point = camera_pos + sample_ray * 0.5;

        let g = ground_inscatter_step(
            sample_point, inv_wavelength, kkr_4pi, temp, camera_offset, scaled_length,
        );
        let front_color = g.contribution;

        c_in = front_color * (inv_wavelength * kkr_esun + KKM_ESUN);
        c_out = clamp(g.attenuate, vec3<f32>(0.0), vec3<f32>(1.0));
    }

    return ScatteringOutput(c_in, c_out);
}

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
) -> VertexOutput {
    let d = pd::get_draw(instance_index);
    let world_p = mv::world_position(d, pos);
#ifdef MULTIVIEW
    let vp = mv::select_view_proj(d, view_idx);
#else
    let vp = mv::select_view_proj(d, 0u);
#endif

    let eye_ray = normalize(mv::model_vector(d, pos.xyz));
    let sun_dir = normalize(mat._SunDirection.xyz);

    let krayleigh = mix(0.0, 0.0025, pow(max(mat._AtmosphereThickness, 0.0), 2.5));
    let sky_tint_gamma = pow(max(mat._SkyTint.rgb, vec3<f32>(0.0)), vec3<f32>(1.0 / GAMMA));
    let scattering_wavelength = mix(
        DEFAULT_SCATTERING_WAVELENGTH - VARIABLE_RANGE_SCATTERING_WAVELENGTH,
        DEFAULT_SCATTERING_WAVELENGTH + VARIABLE_RANGE_SCATTERING_WAVELENGTH,
        vec3<f32>(1.0) - sky_tint_gamma,
    );
    let inv_wavelength = 1.0 / pow(scattering_wavelength, vec3<f32>(4.0));

    let scattering = evaluate_scattering(eye_ray, sun_dir, inv_wavelength, krayleigh);

    var out: VertexOutput;
    out.clip_pos = vp * world_p;
    out.world_ray = -mv::model_vector(d, pos.xyz);
    out.ground_color = mat._Exposure * (scattering.c_in + mat._GroundColor.rgb * scattering.c_out);
    out.sky_color = mat._Exposure * (scattering.c_in * rayleigh_phase(sun_dir, -eye_ray));
    out.sun_color = mat._Exposure * (scattering.c_out * mat._SunColor.rgb);
    return out;
}

//#pass forward
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let ray = normalize(in.world_ray);
    let y = ray.y / SKY_GROUND_THRESHOLD;
    var col = mix(in.sky_color, in.ground_color, clamp(y, 0.0, 1.0));

    if (!uvu::kw_enabled(mat._SUNDISK_NONE) && y < 0.0) {
        let sun_dir = normalize(mat._SunDirection.xyz);
        let sun_size = clamp(mat._SunSize, 1.0e-4, 1.0);
        var mie: f32;
        if (uvu::kw_enabled(mat._SUNDISK_HIGH_QUALITY)) {
            let eye_cos = dot(sun_dir, ray);
            mie = mie_phase(eye_cos, eye_cos * eye_cos, sun_size);
        } else {
            mie = calc_sun_spot(sun_dir, -ray, sun_size);
        }
        col = col + mie * in.sun_color;
    }

    return rg::retain_globals_additive(vec4<f32>(max(col, vec3<f32>(0.0)), 1.0));
}
