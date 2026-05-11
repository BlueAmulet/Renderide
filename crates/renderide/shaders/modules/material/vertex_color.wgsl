//! Vertex color encoding decoders selected by shader variant bits.
//!
//! Unity `_VERTEX_LINEAR_COLOR` / `_VERTEX_SRGB_COLOR` / `_VERTEX_HDRSRGB_COLOR`
//! keywords pick how the host packed a mesh's per-vertex color before it reached
//! the renderer. The shader needs to convert back to linear before any further
//! shading math. The renderer pipeline treats linear as the default (no keyword)
//! and converts only when a sRGB or HDR-sRGB keyword bit is set.

#define_import_path renderide::material::vertex_color

#import renderide::material::variant_bits as vb

/// Converts a single sRGB channel value (in `[0, 1]`) to linear space.
/// Values outside `[-1, 1]` pass through; this matches the unlit pattern that
/// preserves any signed/HDR component the host packed alongside sRGB-encoded
/// channels.
fn srgb_channel_to_linear(value: f32) -> f32 {
    if (value < 1.0 && value > -1.0) {
        if (value <= 0.04045) {
            return value / 12.92;
        }
        return pow((value + 0.055) / 1.055, 2.4);
    }
    return value;
}

/// Converts a four-component color whose RGB channels are sRGB-encoded to linear,
/// preserving the alpha channel.
fn srgb_to_linear(color: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        srgb_channel_to_linear(color.r),
        srgb_channel_to_linear(color.g),
        srgb_channel_to_linear(color.b),
        color.a,
    );
}

/// Converts an HDR-sRGB encoded RGB triple to linear space, preserving alpha.
/// HDR-sRGB extends the standard sRGB transfer into values above `1.0` by
/// applying the high-end power curve unconditionally; channels at or below the
/// linear-segment threshold use the same scale as the standard transfer.
fn hdr_srgb_to_linear(color: vec4<f32>) -> vec4<f32> {
    let rgb = color.rgb;
    let abs_rgb = abs(rgb);
    let sign_rgb = sign(rgb);
    let linear_segment = abs_rgb / 12.92;
    let curve_segment = pow((abs_rgb + 0.055) / 1.055, vec3<f32>(2.4));
    let use_curve = abs_rgb > vec3<f32>(0.04045);
    let magnitude = select(linear_segment, curve_segment, use_curve);
    return vec4<f32>(sign_rgb * magnitude, color.a);
}

/// Selects the vertex-color decode requested by the shader's variant bits.
///
/// `srgb_mask` is the bit for the shader's `_VERTEX_SRGB_COLOR` keyword.
/// `hdr_srgb_mask` is the bit for `_VERTEX_HDRSRGB_COLOR` (pass `0u` if the
/// shader does not declare it). When neither bit is set the input is returned
/// verbatim, matching `_VERTEX_LINEAR_COLOR` / no-keyword behavior.
fn decode_vertex_color(
    color: vec4<f32>,
    bits: u32,
    srgb_mask: u32,
    hdr_srgb_mask: u32,
) -> vec4<f32> {
    if (hdr_srgb_mask != 0u && vb::enabled(bits, hdr_srgb_mask)) {
        return hdr_srgb_to_linear(color);
    }
    if (srgb_mask != 0u && vb::enabled(bits, srgb_mask)) {
        return srgb_to_linear(color);
    }
    return color;
}
