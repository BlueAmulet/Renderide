//! Shared math for the PBS DistanceLerp material family.

#define_import_path renderide::pbs::families::distance_lerp

fn safe_inverse_range(band_start: f32, band_end: f32) -> f32 {
    let denom = band_end - band_start;
    return select(1.0 / denom, 0.0, abs(denom) < 1e-6);
}

fn band_lerp(d: f32, band_start: f32, inv_range: f32) -> f32 {
    return clamp((d - band_start) * inv_range, 0.0, 1.0);
}

fn snap_reference(p: vec3<f32>, grid_size: vec3<f32>, grid_offset: vec3<f32>) -> vec3<f32> {
    let safe_size = vec3<f32>(
        select(grid_size.x, 1.0, grid_size.x == 0.0),
        select(grid_size.y, 1.0, grid_size.y == 0.0),
        select(grid_size.z, 1.0, grid_size.z == 0.0),
    );
    let snapped = round((p + grid_offset) / safe_size) * safe_size;
    return select(snapped, p, grid_size == vec3<f32>(0.0));
}

fn point_displacement(
    d: f32,
    band_start: f32,
    inv_range: f32,
    magnitude_from: f32,
    magnitude_to: f32,
) -> f32 {
    return mix(magnitude_from, magnitude_to, band_lerp(d, band_start, inv_range));
}

fn point_emission(
    d: f32,
    band_start: f32,
    inv_range: f32,
    tint: vec4<f32>,
    emission_from: vec4<f32>,
    emission_to: vec4<f32>,
) -> vec3<f32> {
    return (tint * mix(emission_from, emission_to, band_lerp(d, band_start, inv_range))).rgb;
}
