//! Polar remap of mesh UVs for radial / angular sampling (Unity `_POLARUV` style).
//!
//! Import with `#import renderide::polar_uv as polar`. Combine with [`renderide::unity_st`] when applying
//! `_ST` tiling.

#define_import_path renderide::polar_uv

/// Maps `raw_uv` in [0,1]² to (angle, radius) in [0,1]²; `radius_pow` shapes the radial falloff.
fn polar_uv(raw_uv: vec2<f32>, radius_pow: f32) -> vec2<f32> {
    let centered = raw_uv * 2.0 - 1.0;
    let angle_len = 6.28318530718;
    let radius = pow(length(centered), radius_pow);
    let angle = atan2(centered.x, centered.y) + angle_len * 0.5;
    return vec2<f32>(angle / angle_len, radius);
}
