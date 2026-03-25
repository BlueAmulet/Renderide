#define_import_path ui_common

/// Returns true when `local_xy` is inside axis-aligned rect (x, y, width, height) in local canvas space.
fn inside_rect_clip(local_xy: vec2f, rect: vec4f) -> bool {
    let min_v = rect.xy;
    let max_v = rect.xy + rect.zw;
    return local_xy.x >= min_v.x && local_xy.x <= max_v.x && local_xy.y >= min_v.y && local_xy.y <= max_v.y;
}

fn apply_main_tex_st(uv: vec2f, st: vec4f) -> vec2f {
    return uv * st.xy + st.zw;
}
