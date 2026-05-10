//! Shared math for the PBS Slice material family.

#define_import_path renderide::pbs::families::slice

fn plane_distance(p: vec3<f32>, normal: vec3<f32>, offset: f32) -> f32 {
    return dot(p, normal) + offset;
}

fn use_world_space(world_space_enabled: bool, object_space_enabled: bool) -> bool {
    return world_space_enabled || (!object_space_enabled);
}

fn slice_position(
    world_pos: vec3<f32>,
    object_pos: vec3<f32>,
    world_space_enabled: bool,
    object_space_enabled: bool,
) -> vec3<f32> {
    return select(object_pos, world_pos, use_world_space(world_space_enabled, object_space_enabled));
}

fn blend_detail_normal(base_ts: vec3<f32>, detail_ts: vec3<f32>) -> vec3<f32> {
    return normalize(vec3<f32>(base_ts.xy + detail_ts.xy, base_ts.z * detail_ts.z));
}
