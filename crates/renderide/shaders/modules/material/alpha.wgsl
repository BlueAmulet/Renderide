//! Shared alpha and blend helpers for material shaders.

#define_import_path renderide::material::alpha

#import renderide::core::math as rmath

fn apply_premultiply(color: vec3<f32>, alpha: f32, enabled: bool) -> vec3<f32> {
    return select(color, color * alpha, enabled);
}

fn mask_luminance(mask_sample: vec4<f32>) -> f32 {
    return mask_sample.a * rmath::luminance_rgb(mask_sample.rgb);
}

fn apply_alpha_mask(color: vec4<f32>, mask_sample: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(color.rgb, color.a * mask_luminance(mask_sample));
}

fn alpha_intensity(alpha: f32, rgb: vec3<f32>) -> f32 {
    return alpha * rmath::luminance_rgb(rgb);
}

fn alpha_intensity_squared(alpha: f32, rgb: vec3<f32>) -> f32 {
    let lum = rmath::luminance_rgb(rgb);
    return alpha * lum * lum;
}

fn should_clip_alpha(alpha: f32, cutoff: f32, enabled: bool) -> bool {
    return enabled && alpha <= cutoff;
}

fn alpha_over(front: vec4<f32>, behind: vec4<f32>) -> vec4<f32> {
    let out_a = front.a + behind.a * (1.0 - front.a);
    if (out_a <= 1e-6) {
        return vec4<f32>(0.0);
    }

    let out_rgb =
        (front.rgb * front.a + behind.rgb * behind.a * (1.0 - front.a)) / out_a;
    return vec4<f32>(out_rgb, out_a);
}
