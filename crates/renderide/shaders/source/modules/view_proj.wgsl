//! View–projection matrix for the active stereo eye from [`renderide::per_draw`].
//!
//! Use in `vs_main` after `#import renderide::view_proj as vp`. With `MULTIVIEW` off, pass `0u` for
//! `view_idx` (unused). With `MULTIVIEW` on, pass `@builtin(view_index)`.

#import renderide::per_draw as pd

#define_import_path renderide::view_proj

/// Selects left or right view–projection when multiview is enabled; otherwise always the left matrix.
fn view_projection_for_eye(view_idx: u32) -> mat4x4<f32> {
#ifdef MULTIVIEW
    if (view_idx == 0u) {
        return pd::draw.view_proj_left;
    }
    return pd::draw.view_proj_right;
#else
    return pd::draw.view_proj_left;
#endif
}
