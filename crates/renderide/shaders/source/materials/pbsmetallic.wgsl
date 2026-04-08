//! Unity Standard metallic PBS (`Shader "PBSMetallic"`): clustered forward + Cook–Torrance BRDF.
//!
//! Build emits `pbsmetallic_default` / `pbsmetallic_multiview`. `@group(1)` names match Unity material
//! properties. ForwardAdd / lightmaps / reflection probes are not implemented yet.
//!
//! Per-draw uniforms (`@group(2)`) use [`renderide::per_draw`].

// unity-shader-name: PBSMetallic

#import renderide::globals as rg
#import renderide::per_draw as pd
#import renderide::pbs::brdf as brdf
#import renderide::pbs::cluster as pcls

struct PbsMetallicMaterial {
    _Color: vec4<f32>,
    _Cutoff: f32,
    _Glossiness: f32,
    _GlossMapScale: f32,
    _SmoothnessTextureChannel: f32,
    _Metallic: f32,
    _BumpScale: f32,
    _Parallax: f32,
    _OcclusionStrength: f32,
    _EmissionColor: vec4<f32>,
    _DetailNormalMapScale: f32,
    _UVSec: f32,
    _SpecularHighlights: f32,
    _GlossyReflections: f32,
    _pad: vec2<f32>,
}

@group(1) @binding(0) var<uniform> mat: PbsMetallicMaterial;
@group(1) @binding(1) var _MainTex: texture_2d<f32>;
@group(1) @binding(2) var _MainTex_sampler: sampler;
@group(1) @binding(3) var _MetallicGlossMap: texture_2d<f32>;
@group(1) @binding(4) var _MetallicGlossMap_sampler: sampler;
@group(1) @binding(5) var _BumpMap: texture_2d<f32>;
@group(1) @binding(6) var _BumpMap_sampler: sampler;
@group(1) @binding(7) var _OcclusionMap: texture_2d<f32>;
@group(1) @binding(8) var _OcclusionMap_sampler: sampler;
@group(1) @binding(9) var _EmissionMap: texture_2d<f32>;
@group(1) @binding(10) var _EmissionMap_sampler: sampler;
@group(1) @binding(11) var _DetailAlbedoMap: texture_2d<f32>;
@group(1) @binding(12) var _DetailAlbedoMap_sampler: sampler;
@group(1) @binding(13) var _DetailNormalMap: texture_2d<f32>;
@group(1) @binding(14) var _DetailNormalMap_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv0: vec2<f32>,
    // Secondary UV for `_UVSec` / detail; copied from uv0 until TEXCOORD1 is bound in the mesh pass.
    @location(3) uv1: vec2<f32>,
}

fn orthonormal_tbn(n: vec3<f32>) -> mat3x3<f32> {
    let up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.y) > 0.99);
    let t = normalize(cross(up, n));
    let b = cross(n, t);
    return mat3x3<f32>(t, b, n);
}

fn sample_normal_world(uv: vec2<f32>, world_n: vec3<f32>, bump_scale: f32) -> vec3<f32> {
    let tbn = orthonormal_tbn(world_n);
    let raw = textureSample(_BumpMap, _BumpMap_sampler, uv).xyz * 2.0 - 1.0;
    let nm = vec3<f32>(raw.xy * bump_scale, raw.z);
    let nt = normalize(vec3<f32>(nm.xy, max(sqrt(max(1.0 - dot(nm.xy, nm.xy), 0.0)), 1e-6)));
    return normalize(tbn * nt);
}

fn metallic_roughness(uv: vec2<f32>) -> vec2<f32> {
    let mg = textureSample(_MetallicGlossMap, _MetallicGlossMap_sampler, uv);
    var metallic = mat._Metallic;
    var smoothness = mat._Glossiness * mat._GlossMapScale;
    metallic = metallic * mg.x;
    let smooth_from_a = select(mg.w, mg.y, mat._SmoothnessTextureChannel < 0.5);
    smoothness = smoothness * smooth_from_a;
    let roughness = clamp(1.0 - smoothness, 0.045, 1.0);
    metallic = clamp(metallic, 0.0, 1.0);
    return vec2<f32>(metallic, roughness);
}

fn direct_radiance(
    light: rg::GpuLight,
    world_pos: vec3<f32>,
    n: vec3<f32>,
    v: vec3<f32>,
    roughness: f32,
    metallic: f32,
    base_color: vec3<f32>,
    f0: vec3<f32>,
) -> vec3<f32> {
    let light_pos = light.position.xyz;
    let light_dir = light.direction.xyz;
    let light_color = light.color.xyz;
    var l: vec3<f32>;
    var attenuation: f32;
    if light.light_type == 0u {
        let to_light = light_pos - world_pos;
        let dist = length(to_light);
        l = normalize(to_light);
        attenuation = select(
            0.0,
            light.intensity / max(dist * dist, 0.0001) * (1.0 - smoothstep(light.range * 0.9, light.range, dist)),
            light.range > 0.0
        );
    } else if light.light_type == 1u {
        let dir_len_sq = dot(light_dir, light_dir);
        l = select(vec3<f32>(0.0, 0.0, 1.0), normalize(-light_dir), dir_len_sq > 1e-16);
        attenuation = light.intensity;
    } else {
        let to_light = light_pos - world_pos;
        let dist = length(to_light);
        l = normalize(to_light);
        let spot_cos = dot(-l, normalize(light_dir));
        let spot_atten = smoothstep(light.spot_cos_half_angle, light.spot_cos_half_angle + 0.1, spot_cos);
        attenuation = select(
            0.0,
            light.intensity * spot_atten * (1.0 - smoothstep(light.range * 0.9, light.range, dist)) / max(dist * dist, 0.0001),
            light.range > 0.0
        );
    }
    let h = normalize(v + l);
    let n_dot_l = max(dot(n, l), 0.0);
    let n_dot_v = max(dot(n, v), 0.0001);
    let n_dot_h = max(dot(n, h), 0.0);
    let radiance = light_color * attenuation * n_dot_l;
    if n_dot_l <= 0.0 {
        return vec3<f32>(0.0);
    }
    let f = brdf::fresnel_schlick(max(dot(h, v), 0.0), f0);
    let spec = (brdf::distribution_ggx(n_dot_h, roughness) * brdf::geometry_smith(n_dot_v, n_dot_l, roughness) * f)
        / max(4.0 * n_dot_v * n_dot_l, 0.0001);
    let kd = (vec3<f32>(1.0) - f) * (1.0 - metallic);
    let diffuse = kd * base_color / 3.14159265;
    return (diffuse + spec) * radiance;
}

fn diffuse_only(
    light: rg::GpuLight,
    world_pos: vec3<f32>,
    n: vec3<f32>,
    base_color: vec3<f32>,
) -> vec3<f32> {
    let light_pos = light.position.xyz;
    let light_dir = light.direction.xyz;
    let light_color = light.color.xyz;
    var l: vec3<f32>;
    var attenuation: f32;
    if light.light_type == 0u {
        let to_light = light_pos - world_pos;
        let dist = length(to_light);
        l = normalize(to_light);
        attenuation = select(
            0.0,
            light.intensity / max(dist * dist, 0.0001) * (1.0 - smoothstep(light.range * 0.9, light.range, dist)),
            light.range > 0.0
        );
    } else if light.light_type == 1u {
        let dir_len_sq = dot(light_dir, light_dir);
        l = select(vec3<f32>(0.0, 0.0, 1.0), normalize(-light_dir), dir_len_sq > 1e-16);
        attenuation = light.intensity;
    } else {
        let to_light = light_pos - world_pos;
        let dist = length(to_light);
        l = normalize(to_light);
        let spot_cos = dot(-l, normalize(light_dir));
        let spot_atten = smoothstep(light.spot_cos_half_angle, light.spot_cos_half_angle + 0.1, spot_cos);
        attenuation = select(
            0.0,
            light.intensity * spot_atten * (1.0 - smoothstep(light.range * 0.9, light.range, dist)) / max(dist * dist, 0.0001),
            light.range > 0.0
        );
    }
    let n_dot_l = max(dot(n, l), 0.0);
    return base_color / 3.14159265 * light_color * attenuation * n_dot_l;
}

@vertex
fn vs_main(
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
) -> VertexOutput {
    let world_p = pd::draw.model * vec4<f32>(pos.xyz, 1.0);
    let wn = normalize((pd::draw.model * vec4<f32>(n.xyz, 0.0)).xyz);
#ifdef MULTIVIEW
    var vp: mat4x4<f32>;
    if (view_idx == 0u) {
        vp = pd::draw.view_proj_left;
    } else {
        vp = pd::draw.view_proj_right;
    }
#else
    let vp = pd::draw.view_proj_left;
#endif
    var out: VertexOutput;
    out.clip_pos = vp * world_p;
    out.world_pos = world_p.xyz;
    out.world_n = wn;
    out.uv0 = uv0;
    // Mesh forward only binds TEXCOORD0 (`@location(2)`). Duplicate until a second UV stream is plumbed.
    out.uv1 = uv0;
    return out;
}

@fragment
fn fs_main(
    @builtin(position) frag_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv0: vec2<f32>,
    @location(3) uv1: vec2<f32>,
) -> @location(0) vec4<f32> {
    let uv1_pick = select(uv0, uv1, mat._UVSec > 0.5);
    let albedo_s = textureSample(_MainTex, _MainTex_sampler, uv0);
    var base_color = mat._Color.xyz * albedo_s.xyz;
    let alpha = mat._Color.a * albedo_s.a;
    if alpha < mat._Cutoff {
        discard;
    }

    let mr = metallic_roughness(uv0);
    let metallic = mr.x;
    let roughness = mr.y;

    let occ_s = textureSample(_OcclusionMap, _OcclusionMap_sampler, uv0).x;
    let occlusion = mix(1.0, occ_s, mat._OcclusionStrength);

    var n = normalize(world_n);
    n = sample_normal_world(uv0, n, mat._BumpScale);

    let em = textureSample(_EmissionMap, _EmissionMap_sampler, uv0).xyz * mat._EmissionColor.xyz;

    let detail = textureSample(_DetailAlbedoMap, _DetailAlbedoMap_sampler, uv1_pick).xyz * 2.0;
    base_color = base_color * detail;

    let cam = rg::frame.camera_world_pos.xyz;
    let v = normalize(cam - world_pos);
    let f0 = mix(vec3<f32>(0.04), base_color, metallic);

    let cluster_id = pcls::cluster_id_from_frag(
        frag_pos.xy,
        world_pos,
        rg::frame.view_space_z_coeffs,
        rg::frame.viewport_width,
        rg::frame.viewport_height,
        rg::frame.cluster_count_x,
        rg::frame.cluster_count_y,
        rg::frame.cluster_count_z,
        rg::frame.near_clip,
        rg::frame.far_clip,
    );

    let count = rg::cluster_light_counts[cluster_id];
    let base_idx = cluster_id * pcls::MAX_LIGHTS_PER_TILE;
    var lo = vec3<f32>(0.0);
    let spec_on = mat._SpecularHighlights > 0.5;
    let i_max = min(count, pcls::MAX_LIGHTS_PER_TILE);
    for (var i = 0u; i < i_max; i++) {
        let li = rg::cluster_light_indices[base_idx + i];
        if li >= rg::frame.light_count {
            continue;
        }
        let light = rg::lights[li];
        if spec_on {
            lo = lo + direct_radiance(light, world_pos, n, v, roughness, metallic, base_color, f0);
        } else {
            lo = lo + diffuse_only(light, world_pos, n, base_color);
        }
    }

    let amb = select(vec3<f32>(0.03), vec3<f32>(0.0), mat._GlossyReflections < 0.5);
    let color = (amb * base_color * occlusion + lo * occlusion) + em;
    return vec4<f32>(color, alpha);
}
