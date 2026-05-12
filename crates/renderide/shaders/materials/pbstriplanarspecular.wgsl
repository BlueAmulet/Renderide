//! Unity surface shader `Shader "PBSTriplanarSpecular"`: Standard SpecularSetup with triplanar
//! projection sampled from world or object space.
//!
//! Mirrors [`pbstriplanar`](super::pbstriplanar) for the SpecularSetup workflow: per-axis sampling
//! of `_MainTex`, `_SpecularMap`, `_EmissionMap`, `_NormalMap`, `_OcclusionMap` blended by
//! `pow(abs(world_normal), _TriBlendPower)` with Reoriented Normal Mapping per plane, and the same
//! back-face normal flip when the host disables culling.
//!
//! Froox variant bits populate `_RenderideVariantBits`; this shader decodes
//! PBSTriplanarSpecular's shader-specific keyword bits locally.


#import renderide::draw::per_draw as pd
#import renderide::material::variant_bits as vb
#import renderide::mesh::vertex as mv
#import renderide::pbs::families::triplanar as ptri
#import renderide::pbs::lighting as plight
#import renderide::pbs::sampling as psamp
#import renderide::pbs::surface as psurf

/// Material uniforms for `PBSTriplanarSpecular`.
struct PbsTriplanarSpecularMaterial {
    /// Tint color (`Color`).
    _Color: vec4<f32>,
    /// Emission color (`EmissionColor`).
    _EmissionColor: vec4<f32>,
    /// Tinted specular color when `_SPECULARMAP` is disabled (RGB = f0, A = smoothness).
    _SpecularColor: vec4<f32>,
    /// Albedo `_ST` applied to all three projected planes.
    _MainTex_ST: vec4<f32>,
    /// Tangent-space normal scale.
    _NormalScale: f32,
    /// Triplanar blend exponent -- higher values produce sharper transitions between planes.
    _TriBlendPower: f32,
    /// Renderer-reserved Froox shader variant bitmask.
    _RenderideVariantBits: u32,
}

const PBSTRIPLANARSPEC_KW_ALBEDOTEX: u32 = 1u << 0u;
const PBSTRIPLANARSPEC_KW_EMISSIONTEX: u32 = 1u << 1u;
const PBSTRIPLANARSPEC_KW_NORMALMAP: u32 = 1u << 2u;
const PBSTRIPLANARSPEC_KW_OBJECTSPACE: u32 = 1u << 3u;
const PBSTRIPLANARSPEC_KW_OCCLUSION: u32 = 1u << 4u;
const PBSTRIPLANARSPEC_KW_SPECULARMAP: u32 = 1u << 5u;
const PBSTRIPLANARSPEC_KW_WORLDSPACE: u32 = 1u << 6u;

@group(1) @binding(0)  var<uniform> mat: PbsTriplanarSpecularMaterial;
@group(1) @binding(1)  var _MainTex: texture_2d<f32>;
@group(1) @binding(2)  var _MainTex_sampler: sampler;
@group(1) @binding(3)  var _NormalMap: texture_2d<f32>;
@group(1) @binding(4)  var _NormalMap_sampler: sampler;
@group(1) @binding(5)  var _SpecularMap: texture_2d<f32>;
@group(1) @binding(6)  var _SpecularMap_sampler: sampler;
@group(1) @binding(7)  var _EmissionMap: texture_2d<f32>;
@group(1) @binding(8)  var _EmissionMap_sampler: sampler;
@group(1) @binding(9)  var _OcclusionMap: texture_2d<f32>;
@group(1) @binding(10) var _OcclusionMap_sampler: sampler;

fn pbstriplanarspec_kw(mask: u32) -> bool {
    return vb::enabled(mat._RenderideVariantBits, mask);
}

fn kw_ALBEDOTEX() -> bool {
    return pbstriplanarspec_kw(PBSTRIPLANARSPEC_KW_ALBEDOTEX);
}

fn kw_EMISSIONTEX() -> bool {
    return pbstriplanarspec_kw(PBSTRIPLANARSPEC_KW_EMISSIONTEX);
}

fn kw_NORMALMAP() -> bool {
    return pbstriplanarspec_kw(PBSTRIPLANARSPEC_KW_NORMALMAP);
}

fn kw_OBJECTSPACE() -> bool {
    return pbstriplanarspec_kw(PBSTRIPLANARSPEC_KW_OBJECTSPACE);
}

fn kw_OCCLUSION() -> bool {
    return pbstriplanarspec_kw(PBSTRIPLANARSPEC_KW_OCCLUSION);
}

fn kw_SPECULARMAP() -> bool {
    return pbstriplanarspec_kw(PBSTRIPLANARSPEC_KW_SPECULARMAP);
}

/// Interpolated vertex output forwarded to both forward-base and forward-add fragments.
struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) proj_pos: vec3<f32>,
    @location(3) proj_n: vec3<f32>,
    @location(4) @interpolate(flat) normal_to_world_x: vec3<f32>,
    @location(5) @interpolate(flat) normal_to_world_y: vec3<f32>,
    @location(6) @interpolate(flat) normal_to_world_z: vec3<f32>,
    @location(7) @interpolate(flat) view_layer: u32,
}

/// Resolved per-fragment shading inputs for the SpecularSetup path.
struct SurfaceData {
    base_color: vec3<f32>,
    alpha: f32,
    f0: vec3<f32>,
    roughness: f32,
    occlusion: f32,
    normal: vec3<f32>,
    emission: vec3<f32>,
}

fn projection_normal_to_world(
    n_proj: vec3<f32>,
    normal_to_world_x: vec3<f32>,
    normal_to_world_y: vec3<f32>,
    normal_to_world_z: vec3<f32>,
) -> vec3<f32> {
    return normalize(mat3x3<f32>(normal_to_world_x, normal_to_world_y, normal_to_world_z) * n_proj);
}

/// Resolve the [`SurfaceData`] for a fragment, mirroring Unity's triplanar `surf` for `PBSTriplanarSpecular`.
fn sample_surface(
    world_n: vec3<f32>,
    proj_pos: vec3<f32>,
    proj_n: vec3<f32>,
    normal_to_world_x: vec3<f32>,
    normal_to_world_y: vec3<f32>,
    normal_to_world_z: vec3<f32>,
    front_facing: bool,
) -> SurfaceData {
    let uvs = ptri::build_planar_uvs(proj_pos, proj_n, mat._MainTex_ST);
    let weights = ptri::triplanar_weights(proj_n, mat._TriBlendPower);

    var c = mat._Color;
    if (kw_ALBEDOTEX()) {
        c = c * ptri::sample_rgba(_MainTex, _MainTex_sampler, uvs, weights);
    }

    var spec = mat._SpecularColor;
    if (kw_SPECULARMAP()) {
        spec = ptri::sample_rgba(_SpecularMap, _SpecularMap_sampler, uvs, weights);
    }
    let f0 = clamp(spec.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    let smoothness = clamp(spec.a, 0.0, 1.0);
    let roughness = psamp::roughness_from_smoothness(smoothness);

    var occlusion = 1.0;
    if (kw_OCCLUSION()) {
        let occ = ptri::sample_rgba(_OcclusionMap, _OcclusionMap_sampler, uvs, weights);
        occlusion = occ.g;
    }

    var emission = mat._EmissionColor;
    if (kw_EMISSIONTEX()) {
        emission = emission * ptri::sample_rgba(_EmissionMap, _EmissionMap_sampler, uvs, weights);
    }

    let n_proj = ptri::sample_normal_projection(
        kw_NORMALMAP(),
        _NormalMap,
        _NormalMap_sampler,
        uvs,
        mat._NormalScale,
        proj_n,
        weights,
    );
    let n_world = projection_normal_to_world(n_proj, normal_to_world_x, normal_to_world_y, normal_to_world_z);
    let n = ptri::flip_normal_for_back_face(n_world, world_n, front_facing);

    return SurfaceData(
        c.rgb,
        c.a,
        f0,
        roughness,
        occlusion,
        n,
        emission.rgb,
    );
}

/// Vertex stage: forward world position, world-space normal, and projection-space position.
@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
) -> VertexOutput {
    let d = pd::get_draw(instance_index);
    let world_p = mv::world_position(d, pos);
    let wn = mv::world_normal(d, n);
    let object_space = kw_OBJECTSPACE();
    let proj_n = select(wn, normalize(transpose(d.normal_matrix) * wn), object_space);
#ifdef MULTIVIEW
    let vp = mv::select_view_proj(d, view_idx);
#else
    let vp = mv::select_view_proj(d, 0u);
#endif

    var out: VertexOutput;
    out.clip_pos = vp * world_p;
    out.world_pos = world_p.xyz;
    out.world_n = wn;
    out.proj_pos = select(world_p.xyz, pos.xyz, object_space);
    out.proj_n = proj_n;
    out.normal_to_world_x = select(vec3<f32>(1.0, 0.0, 0.0), d.model[0].xyz, object_space);
    out.normal_to_world_y = select(vec3<f32>(0.0, 1.0, 0.0), d.model[1].xyz, object_space);
    out.normal_to_world_z = select(vec3<f32>(0.0, 0.0, 1.0), d.model[2].xyz, object_space);
#ifdef MULTIVIEW
    out.view_layer = mv::packed_view_layer(instance_index, view_idx);
#else
    out.view_layer = mv::packed_view_layer(instance_index, 0u);
#endif
    return out;
}

/// Forward-base pass: ambient + directional lighting + emission.
//#pass forward
@fragment
fn fs_forward_base(
    @builtin(position) frag_pos: vec4<f32>,
    @builtin(front_facing) front_facing: bool,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) proj_pos: vec3<f32>,
    @location(3) proj_n: vec3<f32>,
    @location(4) @interpolate(flat) normal_to_world_x: vec3<f32>,
    @location(5) @interpolate(flat) normal_to_world_y: vec3<f32>,
    @location(6) @interpolate(flat) normal_to_world_z: vec3<f32>,
    @location(7) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    let s = sample_surface(
        world_n,
        proj_pos,
        proj_n,
        normal_to_world_x,
        normal_to_world_y,
        normal_to_world_z,
        front_facing,
    );
    let surface = psurf::specular(
        s.base_color,
        s.alpha,
        s.f0,
        s.roughness,
        s.occlusion,
        s.normal,
        s.emission,
    );
    return vec4<f32>(
        plight::shade_specular_clustered(
            frag_pos.xy,
            world_pos,
            view_layer,
            surface,
            plight::default_lighting_options(),
        ),
        s.alpha,
    );
}
