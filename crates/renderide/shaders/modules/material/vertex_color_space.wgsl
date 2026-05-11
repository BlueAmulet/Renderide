//! Vertex-color color-space decoding shared by Unlit-family materials.
//!
//! Mirrors `ApplyVertexColor` / `ConvertProfileToLinear` from the Resonite Common.cginc:
//! per-channel sRGB-to-linear curve clamped to |x| < 1, with HDR profiles extending
//! out-of-range values via a 1/1.5 power and HDR-alpha variants gamma-22-decoding alpha.
//!
//! Each consumer passes the variant bitmask and the shader-local masks for whichever
//! of the four color-space keywords it actually declares; pass `0u` for any keyword the
//! shader does not opt into.

#define_import_path renderide::material::vertex_color_space

#import renderide::material::variant_bits as vb

const PROFILE_NONE: u32 = 0u;
const PROFILE_LINEAR: u32 = 1u;
const PROFILE_SRGB: u32 = 2u;
const PROFILE_HDRSRGB: u32 = 3u;
const PROFILE_HDRSRGBALPHA: u32 = 4u;

fn srgb_channel_to_linear(v: f32) -> f32 {
    if (v <= 0.04045) {
        return v / 12.92;
    }
    return pow((v + 0.055) / 1.055, 2.4);
}

fn convert_channel(v: f32, profile: u32) -> f32 {
    if (profile == PROFILE_LINEAR || profile == PROFILE_NONE) {
        return v;
    }
    let in_range = v < 1.0 && v > -1.0;
    if (profile == PROFILE_SRGB) {
        if (in_range) {
            return srgb_channel_to_linear(v);
        }
        return v;
    }
    if (in_range) {
        return srgb_channel_to_linear(v);
    }
    return pow(v, 1.0 / 1.5);
}

fn select_profile(
    bits: u32,
    linear_mask: u32,
    srgb_mask: u32,
    hdrsrgb_mask: u32,
    hdrsrgba_mask: u32,
) -> u32 {
    if (vb::enabled(bits, hdrsrgba_mask)) {
        return PROFILE_HDRSRGBALPHA;
    }
    if (vb::enabled(bits, hdrsrgb_mask)) {
        return PROFILE_HDRSRGB;
    }
    if (vb::enabled(bits, srgb_mask)) {
        return PROFILE_SRGB;
    }
    if (vb::enabled(bits, linear_mask)) {
        return PROFILE_LINEAR;
    }
    return PROFILE_NONE;
}

fn apply(
    color: vec4<f32>,
    bits: u32,
    linear_mask: u32,
    srgb_mask: u32,
    hdrsrgb_mask: u32,
    hdrsrgba_mask: u32,
) -> vec4<f32> {
    let profile = select_profile(bits, linear_mask, srgb_mask, hdrsrgb_mask, hdrsrgba_mask);
    if (profile == PROFILE_NONE) {
        return color;
    }
    let rgb = vec3<f32>(
        convert_channel(color.r, profile),
        convert_channel(color.g, profile),
        convert_channel(color.b, profile),
    );
    var a = color.a;
    if (profile == PROFILE_HDRSRGBALPHA) {
        a = pow(color.a, 1.0 / 2.2);
    }
    return vec4<f32>(rgb, a);
}
