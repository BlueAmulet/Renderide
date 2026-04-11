//! Clustered forward helpers: screen tile XY and exponential Z slice (matches clustered light compute).
//!
//! Import with `#import renderide::pbs::cluster`.

#define_import_path renderide::pbs::cluster

const TILE_SIZE: u32 = 16u;
const MAX_LIGHTS_PER_TILE: u32 = 32u;

fn cluster_xy_from_frag(frag_xy: vec2<f32>, viewport_w: u32, viewport_h: u32) -> vec2<u32> {
    let max_x = max(f32(viewport_w) - 0.5, 0.5);
    let max_y = max(f32(viewport_h) - 0.5, 0.5);
    let pxy = clamp(frag_xy, vec2<f32>(0.5, 0.5), vec2<f32>(max_x, max_y));
    let tile_f = (pxy - vec2<f32>(0.5, 0.5)) / vec2<f32>(f32(TILE_SIZE));
    return vec2<u32>(u32(floor(tile_f.x)), u32(floor(tile_f.y)));
}

fn cluster_z_from_view_z(view_z: f32, near_clip: f32, far_clip: f32, cluster_count_z: u32) -> u32 {
    let d = clamp(-view_z, near_clip, far_clip);
    let z = log(d / near_clip) / log(far_clip / near_clip) * f32(cluster_count_z);
    return u32(clamp(z, 0.0, f32(cluster_count_z - 1u)));
}

fn cluster_id_from_frag(
    clip_xy: vec2<f32>,
    world_pos: vec3<f32>,
    view_space_z_coeffs: vec4<f32>,
    viewport_w: u32,
    viewport_h: u32,
    cluster_count_x: u32,
    cluster_count_y: u32,
    cluster_count_z: u32,
    near_clip: f32,
    far_clip: f32,
) -> u32 {
    let view_z = dot(view_space_z_coeffs.xyz, world_pos) + view_space_z_coeffs.w;
    let cluster_z = cluster_z_from_view_z(view_z, near_clip, far_clip, cluster_count_z);
    let cluster_xy = cluster_xy_from_frag(clip_xy, viewport_w, viewport_h);
    let cx = min(cluster_xy.x, cluster_count_x - 1u);
    let cy = min(cluster_xy.y, cluster_count_y - 1u);
    return cx + cluster_count_x * (cy + cluster_count_y * cluster_z);
}
