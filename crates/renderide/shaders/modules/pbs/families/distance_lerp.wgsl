//! Shared math for the PBS DistanceLerp material family.

#define_import_path renderide::pbs::families::distance_lerp

/// Accumulated per-vertex contribution from a list of reference points.
struct DisplaceResult {
    /// Sum of point-by-point displacement magnitudes.
    displace: f32,
    /// Sum of tinted emission contributions, RGB.
    emission: vec3<f32>,
}

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

/// Iterate the first `count` (<= 16) reference points and accumulate displacement and tinted
/// emission contributions for a given reference position.
fn accumulate_points(
    reference: vec3<f32>,
    point_count: f32,
    points: array<vec4<f32>, 16>,
    tints: array<vec4<f32>, 16>,
    displace_band_start: f32,
    displace_band_end: f32,
    magnitude_from: f32,
    magnitude_to: f32,
    emission_band_start: f32,
    emission_band_end: f32,
    emission_from: vec4<f32>,
    emission_to: vec4<f32>,
) -> DisplaceResult {
    let dist_inv = safe_inverse_range(displace_band_start, displace_band_end);
    let em_inv = safe_inverse_range(emission_band_start, emission_band_end);
    let count = u32(clamp(point_count, 0.0, 16.0));
    var displace = 0.0;
    var emission = vec3<f32>(0.0);
    for (var i: u32 = 0u; i < count; i = i + 1u) {
        let pt = points[i].xyz;
        let d = distance(reference, pt);
        displace = displace + point_displacement(
            d,
            displace_band_start,
            dist_inv,
            magnitude_from,
            magnitude_to,
        );
        emission = emission + point_emission(
            d,
            emission_band_start,
            em_inv,
            tints[i],
            emission_from,
            emission_to,
        );
    }
    return DisplaceResult(displace, emission);
}
