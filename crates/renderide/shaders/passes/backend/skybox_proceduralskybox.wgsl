//! Fullscreen ProceduralSkybox sky draw.

#import renderide::frame::globals as rg
#import renderide::skybox::procedural as ps
#import renderide::skybox::common as skybox
#import renderide::core::uv as uvu

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
@group(2) @binding(0) var<uniform> view: skybox::SkyboxView;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) ndc: vec2<f32>,
    @location(1) @interpolate(flat) view_layer: u32,
}

fn procedural_color(ray: vec3<f32>) -> vec4<f32> {
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
    return rg::retain_globals_additive(vec4<f32>(ps::sample(params, ray), 1.0));
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

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
) -> VertexOutput {
    let clip = skybox::fullscreen_clip_pos(vertex_index);
    var out: VertexOutput;
    out.clip_pos = clip;
    out.ndc = vec2<f32>(clip.x, clip.y * view.ndc_y_sign_pad.x);
#ifdef MULTIVIEW
    out.view_layer = view_idx;
#else
    out.view_layer = 0u;
#endif
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let proj_params = select(rg::frame.proj_params_left, rg::frame.proj_params_right, in.view_layer != 0u);
    let view_ray = skybox::view_ray_from_ndc(
        in.ndc,
        proj_params,
        skybox::view_is_orthographic(view, in.view_layer),
    );
    let world_ray = skybox::world_ray_from_view_ray(view_ray, view, in.view_layer);
    return procedural_color(world_ray);
}
