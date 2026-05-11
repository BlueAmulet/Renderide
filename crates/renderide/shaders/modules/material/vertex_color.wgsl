//! Vertex-color color-space conversion helpers.
//!
//! Material shaders receive vertex colors in whichever space the host marked them with via the
//! `_VERTEX_LINEAR_COLOR` / `_VERTEX_SRGB_COLOR` / `_VERTEX_HDRSRGB_COLOR` keyword groups.
//! Linear-tagged colors are passed through. SRGB-tagged colors run through the sRGB inverse-EOTF
//! to land in linear space. The LDR variant preserves out-of-range values unchanged (HDR vertex
//! colors authored against an SDR sRGB tag); the HDR variant applies the curve unconditionally
//! with sign-mirroring so negative components stay finite.

#define_import_path renderide::material::vertex_color

fn srgb_channel_to_linear_ldr(value: f32) -> f32 {
    if (value < 1.0 && value > -1.0) {
        if (value <= 0.04045) {
            return value / 12.92;
        }
        return pow((value + 0.055) / 1.055, 2.4);
    }
    return value;
}

fn srgb_channel_to_linear_hdr(value: f32) -> f32 {
    let sign_v = sign(value);
    let abs_v = abs(value);
    if (abs_v <= 0.04045) {
        return value / 12.92;
    }
    return sign_v * pow((abs_v + 0.055) / 1.055, 2.4);
}

/// Convert an sRGB-tagged LDR vertex color to linear. Alpha is passed through.
fn srgb_to_linear_ldr(color: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        srgb_channel_to_linear_ldr(color.r),
        srgb_channel_to_linear_ldr(color.g),
        srgb_channel_to_linear_ldr(color.b),
        color.a,
    );
}

/// Convert an sRGB-tagged HDR vertex color to linear, sign-mirroring the curve for negative
/// components and continuing it past 1.0 for super-white values. Alpha is passed through.
fn srgb_to_linear_hdr(color: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        srgb_channel_to_linear_hdr(color.r),
        srgb_channel_to_linear_hdr(color.g),
        srgb_channel_to_linear_hdr(color.b),
        color.a,
    );
}
