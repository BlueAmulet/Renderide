//! Shared rect clipping predicates used by UI and screen-space filter materials.

#define_import_path renderide::ui::rect_clip

#import renderide::core::math as rmath

fn rect_clip_enabled(rect: vec4<f32>, rect_clip: f32) -> bool {
    return rect_clip > 0.5 && rmath::rect_has_area(rect);
}

fn rect_clip_enabled_b(rect: vec4<f32>, enabled: bool) -> bool {
    return enabled && rmath::rect_has_area(rect);
}

fn outside_rect_clip(p: vec2<f32>, rect: vec4<f32>) -> bool {
    return rmath::outside_rect(p, rect);
}

fn should_clip_rect(p: vec2<f32>, rect: vec4<f32>, rect_clip: f32) -> bool {
    return rect_clip_enabled(rect, rect_clip) && outside_rect_clip(p, rect);
}

fn should_clip_rect_b(p: vec2<f32>, rect: vec4<f32>, enabled: bool) -> bool {
    return rect_clip_enabled_b(rect, enabled) && outside_rect_clip(p, rect);
}
