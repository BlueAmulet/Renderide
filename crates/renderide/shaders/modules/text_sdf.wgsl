//! Shared helpers for MSDF/SDF/raster text font-atlas shading.
//!
//! Import with `#import renderide::text_sdf as tsdf`.

#define_import_path renderide::text_sdf

/// Median of three scalars (MSDF channel combine).
fn median3(r: f32, g: f32, b: f32) -> f32 {
    return max(min(r, g), min(max(r, g), b));
}

/// Decodes `_TextMode` to `0` = MSDF, `1` = RASTER, `2` = SDF (clamped).
fn text_mode_clamped(tm: f32) -> i32 {
    return clamp(i32(round(tm)), 0, 2);
}
