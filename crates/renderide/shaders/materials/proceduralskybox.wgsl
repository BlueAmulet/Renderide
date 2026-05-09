//! Unity ProceduralSkybox asset (`Shader "ProceduralSky"`): analytic sky material with
//! Rayleigh+Mie scattering and three sun-disk modes (NONE / SIMPLE / HIGH_QUALITY).
//!
//! The renderer pipeline operates entirely in linear color space, so this port implements
//! the linear branch of the original shader only; the gamma-space branch and
//! `SKYBOX_COLOR_IN_TARGET_COLOR_SPACE` short-circuit are intentionally omitted. Sun-disk
//! mode is selected at runtime via the `_SUNDISK_*` keyword floats, mirroring the host's
//! material keyword routing.


#import renderide::globals as rg
#import renderide::per_draw as pd
#import renderide::procedural_sky as ps
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

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) eye_ray: vec3<f32>,
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

    var out: VertexOutput;
    out.clip_pos = vp * world_p;
    out.eye_ray = mv::model_vector(d, pos.xyz);
    return out;
}

//#pass forward
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let params = ps::ProceduralSkyParams(
        mat._SkyTint.rgb,
        mat._GroundColor.rgb,
        mat._SunColor.rgb,
        mat._SunDirection.xyz,
        mat._Exposure,
        mat._SunSize,
        mat._AtmosphereThickness,
        procedural_sun_disk_mode(),
    );
    return rg::retain_globals_additive(vec4<f32>(ps::sample(params, in.eye_ray), 1.0));
}

fn procedural_sun_disk_mode() -> f32 {
    if (uvu::kw_enabled(mat._SUNDISK_NONE)) {
        return 0.0;
    }
    if (uvu::kw_enabled(mat._SUNDISK_HIGH_QUALITY)) {
        return 2.0;
    }
    return 1.0;
}
