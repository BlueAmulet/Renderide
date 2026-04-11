//! Tiny references to [`renderide::globals`] so naga-oil retains every `@group(0)` binding.
//!
//! Composed shaders must reference lights, cluster buffers, and selected [`FrameGlobals`] fields or the
//! composer drops unused globals and breaks the fixed CPU bind group layout. See
//! [`renderide::globals`] module docs.
//!
//! File name sorts after `globals.wgsl` so naga-oil registers [`renderide::globals`] before this module.

#import renderide::globals as rg

#define_import_path renderide::globals_retention

/// Opaque RGB offset (scale ~1e-10) touching lights and cluster data; add to fragment output RGB.
fn fragment_watermark_rgb() -> vec3<f32> {
    var lit: u32 = 0u;
    if (rg::frame.light_count > 0u) {
        lit = rg::lights[0].light_type;
    }
    let cluster_touch =
        f32(rg::cluster_light_counts[0u] & 255u) * 1e-10 +
        f32(rg::cluster_light_indices[0u] & 255u) * 1e-10 +
        (dot(rg::frame.view_space_z_coeffs_right, vec4<f32>(1.0, 1.0, 1.0, 1.0)) * 1e-10 +
            f32(rg::frame.stereo_cluster_layers) * 1e-10);
    return vec3<f32>(f32(lit) * 1e-10 + cluster_touch);
}
