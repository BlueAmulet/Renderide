//! Specular reflection-probe sampling for CPU-selected per-draw probes.

#define_import_path renderide::reflection_probes

#import renderide::globals as rg
#import renderide::pbs::brdf as brdf
#import renderide::per_draw as pd

const PROBE_FLAG_BOX_PROJECTION: f32 = 1.0;

fn selected_draw(view_layer: u32) -> pd::PerDrawUniforms {
    return pd::get_draw(rg::draw_index_from_layer(view_layer));
}

fn selected_probe_hit_count(view_layer: u32) -> u32 {
    return pd::reflection_probe_hit_count(selected_draw(view_layer));
}

fn has_indirect_specular(view_layer: u32, enabled: bool) -> bool {
    return enabled && selected_probe_hit_count(view_layer) > 0u;
}

fn dominant_reflection_dir(n: vec3<f32>, v: vec3<f32>, perceptual_roughness: f32) -> vec3<f32> {
    let r = reflect(-v, n);
    let blend = perceptual_roughness * perceptual_roughness;
    return normalize(mix(r, n, blend));
}

fn roughness_lod(perceptual_roughness: f32, max_lod: f32) -> f32 {
    let r = clamp(perceptual_roughness, 0.0, 1.0);
    return clamp(max_lod * r * (2.0 - r), 0.0, max_lod);
}

fn box_project_dir(probe: rg::GpuReflectionProbe, world_pos: vec3<f32>, dir: vec3<f32>) -> vec3<f32> {
    if (probe.params.z < PROBE_FLAG_BOX_PROJECTION) {
        return dir;
    }
    let safe_dir = select(vec3<f32>(1e-6), dir, abs(dir) > vec3<f32>(1e-6));
    let plane = select(probe.box_min.xyz, probe.box_max.xyz, safe_dir > vec3<f32>(0.0));
    let t = (plane - world_pos) / safe_dir;
    let distance = min(t.x, min(t.y, t.z));
    if (distance <= 0.0) {
        return dir;
    }
    return normalize(world_pos + safe_dir * distance - probe.position.xyz);
}

fn sample_probe_radiance(
    atlas_index: u32,
    world_pos: vec3<f32>,
    dir: vec3<f32>,
    perceptual_roughness: f32,
) -> vec3<f32> {
    if (atlas_index == 0u) {
        return vec3<f32>(0.0);
    }
    let probe = rg::reflection_probes[atlas_index];
    let intensity = max(probe.params.x, 0.0);
    if (intensity <= 0.0) {
        return vec3<f32>(0.0);
    }
    let sample_dir = box_project_dir(probe, world_pos, dir);
    let lod = roughness_lod(perceptual_roughness, max(probe.params.y, 0.0));
    return textureSampleLevel(
        rg::reflection_probe_specular,
        rg::reflection_probe_specular_sampler,
        sample_dir,
        i32(atlas_index),
        lod,
    ).rgb * intensity;
}

fn indirect_radiance(
    world_pos: vec3<f32>,
    n: vec3<f32>,
    v: vec3<f32>,
    perceptual_roughness: f32,
    view_layer: u32,
    enabled: bool,
) -> vec3<f32> {
    if (!enabled) {
        return vec3<f32>(0.0);
    }
    let draw = selected_draw(view_layer);
    let count = pd::reflection_probe_hit_count(draw);
    if (count == 0u) {
        return vec3<f32>(0.0);
    }
    let indices = pd::reflection_probe_indices(draw);
    let dir = dominant_reflection_dir(n, v, perceptual_roughness);
    let first = sample_probe_radiance(indices.x, world_pos, dir, perceptual_roughness);
    if (count < 2u || indices.y == 0u) {
        return first;
    }
    let second = sample_probe_radiance(indices.y, world_pos, dir, perceptual_roughness);
    return mix(first, second, pd::reflection_probe_second_weight(draw));
}

fn indirect_specular_with_energy(
    world_pos: vec3<f32>,
    n: vec3<f32>,
    v: vec3<f32>,
    perceptual_roughness: f32,
    specular_energy: vec3<f32>,
    occlusion: f32,
    enabled: bool,
    view_layer: u32,
) -> vec3<f32> {
    let radiance = indirect_radiance(world_pos, n, v, perceptual_roughness, view_layer, enabled);
    return radiance * specular_energy * max(occlusion, 0.0);
}

fn indirect_specular(
    world_pos: vec3<f32>,
    n: vec3<f32>,
    v: vec3<f32>,
    perceptual_roughness: f32,
    f0: vec3<f32>,
    occlusion: f32,
    enabled: bool,
    view_layer: u32,
) -> vec3<f32> {
    if (!has_indirect_specular(view_layer, enabled)) {
        return vec3<f32>(0.0);
    }
    let n_dot_v = clamp(dot(n, v), 0.0, 1.0);
    let energy = brdf::indirect_specular_energy(perceptual_roughness, n_dot_v, f0, true);
    return indirect_specular_with_energy(world_pos, n, v, perceptual_roughness, energy, occlusion, true, view_layer);
}
