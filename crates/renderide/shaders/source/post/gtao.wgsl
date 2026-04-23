//! Fullscreen pass: Ground-Truth Ambient Occlusion (Jimenez et al. 2016, "Practical Realtime
//! Strategies for Accurate Indirect Occlusion"). Reads HDR scene color and the scene depth
//! buffer, reconstructs view-space positions and normals on the fly, searches screen-space
//! horizons analytically (Eq. 5–9), applies the multi-bounce fit (Eq. 10), and writes modulated
//! HDR to the post-processing chain output.
//!
//! Build script composes this into `gtao_default` (mono; depth as `texture_depth_2d`) and
//! `gtao_multiview` (stereo; `@builtin(view_index)` selects the eye and depth is
//! `texture_depth_2d_array`) via naga-oil's `#ifdef MULTIVIEW` conditional compilation. The Rust
//! side (`passes::post_processing::gtao::pipeline`) caches one pipeline per `(output_format,
//! multiview_stereo)` pair and builds a matching bind-group layout.
//!
//! Bind group (`@group(0)`):
//! - `@binding(0)` HDR scene color (always `texture_2d_array<f32>`; mono samples layer 0).
//! - `@binding(1)` linear-clamp sampler.
//! - `@binding(2)` scene depth (`texture_depth_2d` mono, `texture_depth_2d_array` multiview).
//! - `@binding(3)` `FrameGlobals` uniform (per-eye proj coefficients + near/far + frame index).
//! - `@binding(4)` `GtaoParams` uniform (user-tunable radius/intensity/steps/heuristics).

@group(0) @binding(0) var scene_color_hdr: texture_2d_array<f32>;
@group(0) @binding(1) var linear_clamp: sampler;

#ifdef MULTIVIEW
@group(0) @binding(2) var scene_depth: texture_depth_2d_array;
#else
@group(0) @binding(2) var scene_depth: texture_depth_2d;
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

@group(0) @binding(3) var<uniform> frame: FrameGlobals;

/// User-tunable GTAO parameters. Updated every record via `ctx.queue.write_buffer` from
/// [`crate::config::GtaoSettings`].
struct GtaoParams {
    radius_world: f32,
    max_pixel_radius: f32,
    intensity: f32,
    step_count: u32,
    thickness_heuristic: f32,
    albedo_multibounce: f32,
    align_pad_tail: vec2<f32>,
}

@group(0) @binding(4) var<uniform> gtao: GtaoParams;

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

/// Reverse-Z depth → positive view-space Z (eye-forward distance).
///
/// Reverse-Z projection encodes near=1.0 and far=0.0 in NDC depth. The resulting view-space Z
/// is positive going into the scene (away from the camera).
fn linearize_depth(d: f32, near: f32, far: f32) -> f32 {
    let denom = d * (far - near) + near;
    return (near * far) / max(denom, 1e-6);
}

/// Selects per-eye `(P[0][0], P[1][1], P[0][2], P[1][2])` from the active view layer.
fn proj_params_for_view(view_layer: u32) -> vec4<f32> {
    if (view_layer == 0u) {
        return frame.proj_params_left;
    }
    return frame.proj_params_right;
}

/// Screen UV (`[0, 1]`) → view-space position, given the linearized view Z for that pixel.
fn view_pos_from_uv(uv: vec2<f32>, view_z: f32, proj_params: vec4<f32>) -> vec3<f32> {
    let ndc_xy = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let view_x = (ndc_xy.x - proj_params.z) * view_z / proj_params.x;
    let view_y = (ndc_xy.y - proj_params.w) * view_z / proj_params.y;
    return vec3<f32>(view_x, view_y, view_z);
}

/// Reconstructs a view-space normal from screen-space derivatives of the view-space position.
///
/// Uses the 4-neighbor min-depth-delta trick: pick the closer neighbor on each axis so creases
/// at silhouettes pick the continuous surface instead of averaging across the depth gap. The
/// cross-product order produces an outward-pointing normal in a right-handed view space where
/// +Z points into the scene (consistent with [`linearize_depth`]).
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

    let n = normalize(cross(dy, dx));
    return n;
}

/// 4×4 interleaved gradient noise offsets (Jiminez-style) so adjacent pixels cover different
/// horizon angles. Paired with a per-frame phase rotation so temporal resolve (future) gets
/// full directional coverage.
fn spatial_phase(pix: vec2<i32>) -> f32 {
    let p = vec2<f32>(pix);
    let jitter = 52.9829189 * fract(0.06711056 * p.x + 0.00583715 * p.y);
    return fract(jitter);
}

/// Per-frame phase rotation stepping through 6 distinct directions for temporal resolve.
fn temporal_phase(frame_index: u32) -> f32 {
    let k = f32(frame_index % 6u);
    return k * (1.0 / 6.0);
}

/// Paper Eq. 7 inner integral: analytic area-weighted arc between horizon angles `h1..h2` under
/// cosine-weighted visibility, given the signed angle `gamma` between the surface normal and
/// the view vector projected into the slice plane.
fn inner_integral_cosine_weighted(h1: f32, h2: f32, gamma: f32) -> f32 {
    let cg = cos(gamma);
    let sg = sin(gamma);
    let a = 0.25 * (-cos(2.0 * h1 - gamma) + cg + 2.0 * h1 * sg);
    let b = 0.25 * (-cos(2.0 * h2 - gamma) + cg + 2.0 * h2 * sg);
    return a + b;
}

/// Paper Eq. 10 cubic fit that recovers near-field multi-bounce indirect illumination lost when
/// applying ambient occlusion alone. Uses a gray-albedo proxy since we don't sample per-pixel
/// albedo.
fn multi_bounce_fit(ao: f32, albedo: f32) -> f32 {
    let a = 2.0404 * albedo - 0.3324;
    let b = 4.7951 * albedo - 0.6417;
    let c = 2.7552 * albedo + 0.6903;
    return max(ao, ((a * ao - b) * ao + c) * ao);
}

/// Runs the full GTAO computation for a pixel and returns the scalar visibility factor in [0, 1].
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

    // Screen-space search radius scaled by view-space radius and projection focal length.
    let pixel_radius_raw = gtao.radius_world * proj_params.x * viewport.x * 0.5 / max(view_z, 1e-3);
    let pixel_radius = min(gtao.max_pixel_radius, pixel_radius_raw);
    let step_count = max(gtao.step_count, 1u);
    let step_pixels = max(pixel_radius / f32(step_count), 1.0);

    let phase = spatial_phase(pix) + temporal_phase(frame.frame_tail.x);
    let angle = (phase * 3.14159265359);
    let dir_ss = vec2<f32>(cos(angle), sin(angle));

    let slice_tangent = vec3<f32>(dir_ss.x, dir_ss.y, 0.0);
    let slice_bitangent = cross(slice_tangent, view_dir);
    let slice_plane_n = view_normal - slice_bitangent * dot(view_normal, slice_bitangent);
    let slice_plane_len = length(slice_plane_n);
    if (slice_plane_len < 1e-4) {
        return 1.0;
    }
    let n_projected_len = slice_plane_len;
    let n_in_slice = slice_plane_n / slice_plane_len;
    let sign_gamma = sign(dot(cross(slice_tangent, n_in_slice), slice_bitangent));
    let cos_gamma = clamp(dot(n_in_slice, view_dir), -1.0, 1.0);
    let gamma = sign_gamma * acos(cos_gamma);

    // Horizon search in both `+dir_ss` (updates `cos_h2`) and `-dir_ss` (updates `cos_h1`).
    //
    // Initialise both horizons to `cos(π) = -1` so that when *no* occluder is found the
    // post-loop clamp drops them back to the hemisphere bounds `γ ± π/2`, matching
    // XeGTAO / Activision's GTAO reference. Pre-populating with `cos(γ ± π/2)` (as the first
    // revision did) produced a negative analytic integral for any non-view-aligned surface,
    // which `saturate` then clamped to 0 — hence the fully black screen on anything but a
    // head-on flat wall.
    var cos_h1 = -1.0;
    var cos_h2 = -1.0;
    let radius2 = gtao.radius_world * gtao.radius_world;
    let inv_viewport = 1.0 / viewport;
    // `thickness_heuristic` ∈ [0, 1] controls how much of each horizon jump is committed.
    // 0 → full commit (standard GTAO); higher values soften the horizon update, reducing
    // over-occlusion caused by thin occluders such as foliage and branches.
    let commit_alpha = clamp(1.0 - gtao.thickness_heuristic, 0.0, 1.0);

    for (var s: u32 = 1u; s <= step_count; s = s + 1u) {
        let step_len = f32(s) * step_pixels;

        let pix_pos_offset = vec2<f32>(pix) + dir_ss * step_len;
        let pix_neg_offset = vec2<f32>(pix) - dir_ss * step_len;

        let ipp = vec2<i32>(pix_pos_offset);
        let inn = vec2<i32>(pix_neg_offset);
        if (ipp.x < 0 || ipp.y < 0 || ipp.x >= i32(viewport.x) || ipp.y >= i32(viewport.y)) {
            continue;
        }
        if (inn.x < 0 || inn.y < 0 || inn.x >= i32(viewport.x) || inn.y >= i32(viewport.y)) {
            continue;
        }

        let uv_pp = (vec2<f32>(ipp) + vec2<f32>(0.5)) * inv_viewport;
        let uv_nn = (vec2<f32>(inn) + vec2<f32>(0.5)) * inv_viewport;
        let z_pp = linearize_depth(load_depth(ipp, view_layer), near, far);
        let z_nn = linearize_depth(load_depth(inn, view_layer), near, far);
        let sp_pos = view_pos_from_uv(uv_pp, z_pp, proj_params);
        let sp_neg = view_pos_from_uv(uv_nn, z_nn, proj_params);

        let d_pos = sp_pos - view_pos;
        let d_neg = sp_neg - view_pos;
        let d_pos_len2 = dot(d_pos, d_pos);
        let d_neg_len2 = dot(d_neg, d_neg);

        if (d_pos_len2 < radius2 && d_pos_len2 > 1e-8) {
            let cos_theta = dot(d_pos * inverseSqrt(d_pos_len2), view_dir);
            if (cos_theta > cos_h2) {
                cos_h2 = mix(cos_h2, cos_theta, commit_alpha);
            }
        }
        if (d_neg_len2 < radius2 && d_neg_len2 > 1e-8) {
            let cos_theta = dot(d_neg * inverseSqrt(d_neg_len2), view_dir);
            if (cos_theta > cos_h1) {
                cos_h1 = mix(cos_h1, cos_theta, commit_alpha);
            }
        }
    }

    let half_pi = 1.57079632679;
    var h1 = -acos(clamp(cos_h1, -1.0, 1.0));
    var h2 = acos(clamp(cos_h2, -1.0, 1.0));
    // Clamp each horizon into its slice-plane hemisphere around the surface normal. Without
    // this clamp, the "no occluder found" initial cosine of -1 leaves `h1 = -π` / `h2 = π`,
    // which drives the analytic integral (Eq. 7) past its valid domain. XeGTAO uses the same
    // `[γ - π/2, γ]` / `[γ, γ + π/2]` clamp for the same reason.
    h1 = clamp(h1, gamma - half_pi, gamma);
    h2 = clamp(h2, gamma, gamma + half_pi);

    let integral = inner_integral_cosine_weighted(h1, h2, gamma);
    // The `0.5` matches the GTAO reference scaling: Eq. 7 returns twice the per-slice
    // visibility (the full unoccluded flat-wall case yields `integral = 2` with γ=0).
    let ao = saturate(0.5 * n_projected_len * integral);
    let boosted = pow(ao, max(gtao.intensity, 0.0));
    return multi_bounce_fit(boosted, gtao.albedo_multibounce);
}

#ifdef MULTIVIEW
@fragment
fn fs_main(in: VsOut, @builtin(view_index) view: u32) -> @location(0) vec4<f32> {
    let pix = vec2<i32>(in.clip_pos.xy);
    let ao = compute_gtao(pix, in.uv, view);
    let hdr = textureSample(scene_color_hdr, linear_clamp, in.uv, view);
    return vec4<f32>(hdr.rgb * ao, hdr.a);
}
#else
@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let pix = vec2<i32>(in.clip_pos.xy);
    let ao = compute_gtao(pix, in.uv, 0u);
    let hdr = textureSample(scene_color_hdr, linear_clamp, in.uv, 0u);
    return vec4<f32>(hdr.rgb * ao, hdr.a);
}
#endif
