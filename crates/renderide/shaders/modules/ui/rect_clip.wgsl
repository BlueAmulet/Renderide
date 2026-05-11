//! Shared UI rect clipping predicates.

#define_import_path renderide::ui::rect_clip

#import renderide::core::math as rmath
#import renderide::material::variant_bits as vb

fn rect_clip_enabled(rect: vec4<f32>, rect_clip: f32) -> bool {
    return rect_clip > 0.5 && rmath::rect_has_area(rect);
}

fn outside_rect_clip(p: vec2<f32>, rect: vec4<f32>) -> bool {
    return rmath::outside_rect(p, rect);
}

fn should_clip_rect(p: vec2<f32>, rect: vec4<f32>, rect_clip: f32) -> bool {
    return rect_clip_enabled(rect, rect_clip) && outside_rect_clip(p, rect);
}

/// Variant-bit gated rect clip. Returns true when `mask` is set in `bits`
/// and the point lies outside a non-degenerate rect.
fn should_clip_rect_bit(p: vec2<f32>, rect: vec4<f32>, bits: u32, mask: u32) -> bool {
    return vb::enabled(bits, mask)
        && rmath::rect_has_area(rect)
        && rmath::outside_rect(p, rect);
}
