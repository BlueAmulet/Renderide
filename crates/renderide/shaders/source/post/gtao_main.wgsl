//! GTAO main pass: produces the raw AO term and the packed-edges side-channel that feed the
//! bilateral denoise pass and the apply pass.
//!
//! Reads only the depth buffer — view-space normals are reconstructed from depth derivatives, the
//! analytic horizon-cosine integral runs once per pixel (one slice, jittered spatiotemporally),
//! and the **scaled visibility factor** is written to MRT location 0 (`R16Float`, single channel),
//! while the packed edge stoppers from [`renderide::gtao_packing::pack_edges`] go to MRT location
//! 1 (`R8Unorm`). Splitting the visibility factor and modulation across two passes (this pass,
//! plus `post/gtao_apply.wgsl`) is what makes XeGTAO's denoise port possible: the denoise kernel
//! is bilateral on the AO term alone, so it must not see scene color.
//!
//! Build script composes this into `gtao_main_default` (mono; depth as `texture_depth_2d`) and
//! `gtao_main_multiview` (stereo; `@builtin(view_index)` selects the eye and depth is
//! `texture_depth_2d_array`) via naga-oil's `#ifdef MULTIVIEW` conditional compilation.
//!
//! Bind group (`@group(0)`):
//! - `@binding(0)` scene depth (`texture_depth_2d` mono, `texture_depth_2d_array` multiview).
//! - `@binding(1)` `FrameGlobals` uniform (per-eye proj coefficients + near/far + frame index).
//! - `@binding(2)` `GtaoParams` uniform (user-tunable radius/intensity/steps).

#import renderide::gtao_packing as gp

#ifdef MULTIVIEW
@group(0) @binding(0) var scene_depth: texture_depth_2d_array;
#else
@group(0) @binding(0) var scene_depth: texture_depth_2d;
#endif

/// Matches [`crate::gpu::frame_globals::FrameGpuUniforms`] exactly (128 bytes).
///
/// We duplicate the struct here rather than `#import renderide::globals` because post-processing
/// shaders run with a bespoke `@group(0)` layout (no lights / cluster storage), so the full
/// globals module would introduce unreferenced bindings that naga-oil would drop.
struct FrameGlobals {
    camera_world_pos: vec4<f32>,
    view_space_z_coeffs: vec4<f32>,
    view_space_z_coeffs_right: vec4<f32>,
    cluster_count_x: u32,
    cluster_count_y: u32,
    cluster_count_z: u32,
    near_clip: f32,
    far_clip: f32,
    light_count: u32,
    viewport_width: u32,
    viewport_height: u32,
    proj_params_left: vec4<f32>,
    proj_params_right: vec4<f32>,
    frame_tail: vec4<u32>,
}

@group(0) @binding(1) var<uniform> frame: FrameGlobals;

/// User-tunable GTAO parameters. Updated every record from the live
/// [`crate::config::GtaoSettings`] via the `GtaoSettingsSlot` blackboard slot. The `intensity`
/// and `albedo_multibounce` fields are consumed by the apply pass; this main pass uses only the
/// horizon-search knobs and the denoise blur beta (which is consumed downstream).
struct GtaoParams {
    radius_world: f32,
    max_pixel_radius: f32,
    intensity: f32,
    step_count: u32,
    falloff_range: f32,
    albedo_multibounce: f32,
    denoise_blur_beta: f32,
    align_pad_tail: f32,
}

@group(0) @binding(2) var<uniform> gtao: GtaoParams;

/// Fullscreen-triangle vertex output. UV covers `[0, 1]²`.
struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    let x = f32((vid << 1u) & 2u);
    let y = f32(vid & 2u);
    var out: VsOut;
    out.clip_pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, y);
    return out;
}

/// Raw depth at integer pixel coords for the current view layer.
fn load_depth(pix: vec2<i32>, view_layer: u32) -> f32 {
#ifdef MULTIVIEW
    return textureLoad(scene_depth, pix, i32(view_layer), 0);
#else
    return textureLoad(scene_depth, pix, 0);
#endif
}

/// Reverse-Z NDC depth → positive view-space Z (eye-forward distance magnitude).
fn linearize_depth(d: f32, near: f32, far: f32) -> f32 {
    let denom = d * (far - near) + near;
    return (near * far) / max(denom, 1e-6);
}

/// Selects per-eye `(P[0][0], P[1][1], P[0][2], P[1][2]) = (x_scale, y_scale, skew_x, skew_y)`
/// for the active view layer.
fn proj_params_for_view(view_layer: u32) -> vec4<f32> {
    if (view_layer == 0u) {
        return frame.proj_params_left;
    }
    return frame.proj_params_right;
}

/// Screen UV (`[0, 1]`) → view-space position, given the linearized positive view-space Z for
/// that pixel.
fn view_pos_from_uv(uv: vec2<f32>, view_z: f32, proj_params: vec4<f32>) -> vec3<f32> {
    let ndc_xy = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let view_x = (ndc_xy.x + proj_params.z) * view_z / proj_params.x;
    let view_y = (ndc_xy.y + proj_params.w) * view_z / proj_params.y;
    return vec3<f32>(view_x, view_y, view_z);
}

/// Reconstructs a view-space normal from screen-space derivatives of view-space position using
/// the four-neighbour min-depth-delta trick. See `gtao.wgsl` for the convention rationale; the
/// math is unchanged here.
fn reconstruct_view_normal(
    center_pix: vec2<i32>,
    center_view: vec3<f32>,
    view_layer: u32,
    proj_params: vec4<f32>,
    viewport: vec2<f32>,
    near: f32,
    far: f32,
) -> vec3<f32> {
    let inv_viewport = 1.0 / viewport;
    let uv_center = (vec2<f32>(center_pix) + vec2<f32>(0.5)) * inv_viewport;

    let px = center_pix + vec2<i32>(1, 0);
    let py = center_pix + vec2<i32>(0, 1);
    let nx = center_pix - vec2<i32>(1, 0);
    let ny = center_pix - vec2<i32>(0, 1);

    let zpx = linearize_depth(load_depth(px, view_layer), near, far);
    let znx = linearize_depth(load_depth(nx, view_layer), near, far);
    let zpy = linearize_depth(load_depth(py, view_layer), near, far);
    let zny = linearize_depth(load_depth(ny, view_layer), near, far);

    let dx_pos = abs(zpx - center_view.z) < abs(znx - center_view.z);
    let dy_pos = abs(zpy - center_view.z) < abs(zny - center_view.z);

    let uv_x = select(
        uv_center - vec2<f32>(inv_viewport.x, 0.0),
        uv_center + vec2<f32>(inv_viewport.x, 0.0),
        dx_pos,
    );
    let uv_y = select(
        uv_center - vec2<f32>(0.0, inv_viewport.y),
        uv_center + vec2<f32>(0.0, inv_viewport.y),
        dy_pos,
    );
    let z_x = select(znx, zpx, dx_pos);
    let z_y = select(zny, zpy, dy_pos);

    let p_x = view_pos_from_uv(uv_x, z_x, proj_params);
    let p_y = view_pos_from_uv(uv_y, z_y, proj_params);

    let dx = select(center_view - p_x, p_x - center_view, dx_pos);
    let dy = select(center_view - p_y, p_y - center_view, dy_pos);

    return normalize(cross(dx, dy));
}

/// Interleaved-gradient spatial noise (Jiménez) so adjacent pixels cover different slice angles.
fn spatial_phase(pix: vec2<i32>) -> f32 {
    let p = vec2<f32>(pix);
    let jitter = 52.9829189 * fract(0.06711056 * p.x + 0.00583715 * p.y);
    return fract(jitter);
}

/// Per-frame phase rotation across 6 directions — feeds the temporal-accumulation follow-up.
fn temporal_phase(frame_index: u32) -> f32 {
    let k = f32(frame_index % 6u);
    return k * (1.0 / 6.0);
}

/// Runs the full GTAO computation for a pixel and returns the **raw** scalar visibility factor in
/// `[0, 1]`. Intensity / multi-bounce reshaping is **not** applied here — those nonlinear steps
/// happen in the apply pass, after the bilateral denoise has averaged a linear AO term.
fn compute_gtao(
    pix: vec2<i32>,
    uv: vec2<f32>,
    view_layer: u32,
) -> f32 {
    let viewport = vec2<f32>(f32(frame.viewport_width), f32(frame.viewport_height));
    let proj_params = proj_params_for_view(view_layer);

    let raw_depth = load_depth(pix, view_layer);
    if (raw_depth <= 0.0) {
        return 1.0;
    }

    let near = frame.near_clip;
    let far = frame.far_clip;
    let view_z = linearize_depth(raw_depth, near, far);
    let view_pos = view_pos_from_uv(uv, view_z, proj_params);
    let view_normal = reconstruct_view_normal(
        pix,
        view_pos,
        view_layer,
        proj_params,
        viewport,
        near,
        far,
    );
    let view_dir = normalize(-view_pos);

    let pixel_radius_raw = gtao.radius_world * proj_params.x * viewport.x * 0.5 / max(view_z, 1e-3);
    let pixel_radius = min(gtao.max_pixel_radius, pixel_radius_raw);
    let step_count = max(gtao.step_count, 1u);
    let step_pixels = max(pixel_radius / f32(step_count), 1.0);

    let phase = spatial_phase(pix) + temporal_phase(frame.frame_tail.x);
    let angle = phase * 3.14159265359;
    let dir_ss = vec2<f32>(cos(angle), sin(angle));

    let direction_vec = vec3<f32>(dir_ss.x, dir_ss.y, 0.0);
    let ortho_direction_vec = direction_vec - view_dir * dot(direction_vec, view_dir);
    let axis_raw = cross(ortho_direction_vec, view_dir);
    let axis_len = length(axis_raw);
    if (axis_len < 1e-6) {
        return 1.0;
    }
    let axis_vec = axis_raw / axis_len;

    let projected_normal_vec = view_normal - axis_vec * dot(view_normal, axis_vec);
    let projected_normal_vec_length = length(projected_normal_vec);
    if (projected_normal_vec_length < 1e-6) {
        return 1.0;
    }
    let sign_n = sign(dot(ortho_direction_vec, projected_normal_vec));
    let cos_n = saturate(
        dot(projected_normal_vec, view_dir) / projected_normal_vec_length,
    );
    let n = sign_n * acos(cos_n);
    let sin_n = sin(n);

    let low_horizon_cos0 = -sin_n;
    let low_horizon_cos1 =  sin_n;
    var horizon_cos0 = low_horizon_cos0;
    var horizon_cos1 = low_horizon_cos1;

    let falloff_range_world = max(gtao.falloff_range, 1e-4) * gtao.radius_world;
    let falloff_mul = -1.0 / max(falloff_range_world, 1e-4);
    let falloff_add = gtao.radius_world / max(falloff_range_world, 1e-4);
    let inv_viewport = 1.0 / viewport;

    for (var s: u32 = 1u; s <= step_count; s = s + 1u) {
        let step_len = f32(s) * step_pixels;

        let pix_pos_offset = vec2<f32>(pix) + dir_ss * step_len;
        let pix_neg_offset = vec2<f32>(pix) - dir_ss * step_len;

        let ipp = vec2<i32>(pix_pos_offset);
        let inn = vec2<i32>(pix_neg_offset);
        let pp_in = ipp.x >= 0 && ipp.y >= 0
            && ipp.x < i32(viewport.x) && ipp.y < i32(viewport.y);
        let nn_in = inn.x >= 0 && inn.y >= 0
            && inn.x < i32(viewport.x) && inn.y < i32(viewport.y);

        if (pp_in) {
            let uv_pp = (vec2<f32>(ipp) + vec2<f32>(0.5)) * inv_viewport;
            let z_pp = linearize_depth(load_depth(ipp, view_layer), near, far);
            let sp_pos = view_pos_from_uv(uv_pp, z_pp, proj_params);
            let d_pos = sp_pos - view_pos;
            let d_pos_len = length(d_pos);
            if (d_pos_len > 1e-4) {
                let candidate = dot(d_pos / d_pos_len, view_dir);
                let weight = saturate(d_pos_len * falloff_mul + falloff_add);
                let shc = mix(low_horizon_cos1, candidate, weight);
                horizon_cos1 = max(horizon_cos1, shc);
            }
        }
        if (nn_in) {
            let uv_nn = (vec2<f32>(inn) + vec2<f32>(0.5)) * inv_viewport;
            let z_nn = linearize_depth(load_depth(inn, view_layer), near, far);
            let sp_neg = view_pos_from_uv(uv_nn, z_nn, proj_params);
            let d_neg = sp_neg - view_pos;
            let d_neg_len = length(d_neg);
            if (d_neg_len > 1e-4) {
                let candidate = dot(d_neg / d_neg_len, view_dir);
                let weight = saturate(d_neg_len * falloff_mul + falloff_add);
                let shc = mix(low_horizon_cos0, candidate, weight);
                horizon_cos0 = max(horizon_cos0, shc);
            }
        }
    }

    let h0 = -acos(clamp(horizon_cos1, -1.0, 1.0));
    let h1 =  acos(clamp(horizon_cos0, -1.0, 1.0));

    let iarc0 = (cos_n + 2.0 * h0 * sin_n - cos(2.0 * h0 - n)) * 0.25;
    let iarc1 = (cos_n + 2.0 * h1 * sin_n - cos(2.0 * h1 - n)) * 0.25;
    let local_visibility = projected_normal_vec_length * (iarc0 + iarc1);

    return saturate(local_visibility);
}

/// MRT output: AO term in `R16Float` (location 0) and packed edges in `R8Unorm` (location 1).
struct GtaoMainOut {
    @location(0) ao_term: f32,
    @location(1) edges: f32,
}

/// Computes the four cardinal neighbour view-space depths the edge calculator needs.
fn gather_neighbour_depths(pix: vec2<i32>, view_layer: u32, near: f32, far: f32) -> vec4<f32> {
    let l = linearize_depth(load_depth(pix - vec2<i32>(1, 0), view_layer), near, far);
    let r = linearize_depth(load_depth(pix + vec2<i32>(1, 0), view_layer), near, far);
    let t = linearize_depth(load_depth(pix - vec2<i32>(0, 1), view_layer), near, far);
    let b = linearize_depth(load_depth(pix + vec2<i32>(0, 1), view_layer), near, far);
    return vec4<f32>(l, r, t, b);
}

#ifdef MULTIVIEW
@fragment
fn fs_main(in: VsOut, @builtin(view_index) view: u32) -> GtaoMainOut {
    let pix = vec2<i32>(in.clip_pos.xy);
    let center_z = linearize_depth(load_depth(pix, view), frame.near_clip, frame.far_clip);
    let neighbours = gather_neighbour_depths(pix, view, frame.near_clip, frame.far_clip);
    var out: GtaoMainOut;
    out.ao_term = compute_gtao(pix, in.uv, view);
    out.edges = gp::pack_edges(gp::calculate_edges(
        center_z, neighbours.x, neighbours.y, neighbours.z, neighbours.w,
    ));
    return out;
}
#else
@fragment
fn fs_main(in: VsOut) -> GtaoMainOut {
    let pix = vec2<i32>(in.clip_pos.xy);
    let center_z = linearize_depth(load_depth(pix, 0u), frame.near_clip, frame.far_clip);
    let neighbours = gather_neighbour_depths(pix, 0u, frame.near_clip, frame.far_clip);
    var out: GtaoMainOut;
    out.ao_term = compute_gtao(pix, in.uv, 0u);
    out.edges = gp::pack_edges(gp::calculate_edges(
        center_z, neighbours.x, neighbours.y, neighbours.z, neighbours.w,
    ));
    return out;
}
#endif
