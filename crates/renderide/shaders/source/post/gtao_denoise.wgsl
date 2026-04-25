//! GTAO bilateral denoise: edge-aware spatial filter that smooths the AO term written by the
//! main pass without crossing depth discontinuities.
//!
//! Direct port of XeGTAO's `XeGTAO_Denoise` kernel
//! (`references_external/XeGTAO/Source/Rendering/Shaders/XeGTAO.hlsli`): a 3×3 neighbourhood
//! whose four cardinal taps are weighted by the unpacked LRTB edge stoppers and whose four
//! diagonal taps use products of adjacent edge values. Symmetry is enforced against neighbour
//! edges (so two pixels separated by an edge agree on dropping each other) and an "edginess
//! leak" term re-opens highly fragmented neighbourhoods to suppress aliasing along thin features.
//!
//! Two `@fragment` entry points share the same body via the `is_final_pass` parameter:
//!
//! * `fs_denoise_intermediate` — used for every denoise pass except the last when
//!   `denoise_passes >= 2`. Centre-pixel weight is `denoise_blur_beta / 5`, so each iteration
//!   only nudges the AO term toward its neighbourhood mean; multiple iterations compound to a
//!   wider effective kernel without overshooting.
//! * `fs_denoise_final` — used for the last (or only) denoise pass. Centre-pixel weight is the
//!   full `denoise_blur_beta` (no extra blur compounding into a follow-up pass).
//!
//! Note: XeGTAO additionally multiplies the final-pass output by `OCCLUSION_TERM_SCALE = 1.5`
//! to repack its [0, 0.667]-range raw AO into the upper part of a U8 storage texture (the
//! consuming apply step then divides back out). This port stores the AO term in `R16Float`,
//! which has plenty of headroom, so we skip that scaling — applying it here and then `saturate`-
//! ing would clamp every brighter-than-0.667 pixel to 1.0 and effectively disable AO.
//!
//! The build script composes this into `gtao_denoise_default` and `gtao_denoise_multiview`. The
//! multiview variant uses `@builtin(view_index)` to address the AO and edges array layers per
//! eye independently.
//!
//! Bind group (`@group(0)`):
//! - `@binding(0)` AO term input (`texture_2d_array<f32>`; single channel R16Float).
//! - `@binding(1)` packed edges (`texture_2d_array<f32>`; single channel R8Unorm).
//! - `@binding(2)` `GtaoParams` uniform (reads `denoise_blur_beta`).

#import renderide::gtao_packing as gp

@group(0) @binding(0) var ao_in: texture_2d_array<f32>;
@group(0) @binding(1) var edges_in: texture_2d_array<f32>;

/// User-tunable GTAO parameters. The denoise pass only reads `denoise_blur_beta`, but the struct
/// matches the layout used by the main / apply passes so all three share one process-wide UBO.
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

/// Loads the AO term at integer pixel coordinates for the given view layer.
fn load_ao(pix: vec2<i32>, view_layer: u32) -> f32 {
    return textureLoad(ao_in, pix, i32(view_layer), 0).r;
}

/// Loads the packed edges value at integer pixel coordinates for the given view layer.
fn load_edges_packed(pix: vec2<i32>, view_layer: u32) -> f32 {
    return textureLoad(edges_in, pix, i32(view_layer), 0).r;
}

/// Inner core of the bilateral kernel. `is_final_pass` switches the centre-pixel weight (the
/// only meaningful difference between intermediate and final iterations once XeGTAO's U8-storage
/// scaling is dropped); the rest of the math (edge unpacking, symmetry enforcement, edginess
/// leak, weighted accumulation) is shared between the two entry points.
fn denoise_at(pix: vec2<i32>, view_layer: u32, is_final_pass: bool) -> f32 {
    let edges_c = gp::unpack_edges(load_edges_packed(pix, view_layer));
    let edges_l = gp::unpack_edges(load_edges_packed(pix + vec2<i32>(-1, 0), view_layer));
    let edges_r = gp::unpack_edges(load_edges_packed(pix + vec2<i32>( 1, 0), view_layer));
    let edges_t = gp::unpack_edges(load_edges_packed(pix + vec2<i32>( 0,-1), view_layer));
    let edges_b = gp::unpack_edges(load_edges_packed(pix + vec2<i32>( 0, 1), view_layer));

    // Symmetry: drop a neighbour weight when the neighbour itself drops the centre. Otherwise a
    // pair of pixels straddling an edge would disagree on whether the edge exists.
    let edges_sym = edges_c * vec4<f32>(edges_l.y, edges_r.x, edges_t.w, edges_b.z);

    // Edginess leak: when 3+ edges fire, allow some neighbour weight back so the kernel doesn't
    // collapse to a no-op on thin features (anti-aliasing). Matches XeGTAO's threshold/strength.
    let leak_threshold = 2.5;
    let leak_strength = 0.5;
    let edginess = saturate(4.0 - leak_threshold - dot(edges_sym, vec4<f32>(1.0)))
        / (4.0 - leak_threshold) * leak_strength;
    let edges_eff = saturate(edges_sym + vec4<f32>(edginess));

    // Diagonal weights: product of two orthogonal edges, scaled by the XeGTAO diagonal weight
    // (0.85 * 0.5).
    let diag_w = 0.425;
    let w_tl = diag_w * (edges_eff.x * edges_l.z + edges_eff.z * edges_t.x);
    let w_tr = diag_w * (edges_eff.z * edges_t.y + edges_eff.y * edges_r.z);
    let w_bl = diag_w * (edges_eff.w * edges_b.x + edges_eff.x * edges_l.w);
    let w_br = diag_w * (edges_eff.y * edges_r.w + edges_eff.w * edges_b.y);

    let beta = max(gtao.denoise_blur_beta, 1e-4);
    let centre_weight = select(beta * 0.2, beta, is_final_pass);

    let ao_c = load_ao(pix, view_layer);
    var sum = ao_c * centre_weight;
    var sum_w = centre_weight;

    let ao_l = load_ao(pix + vec2<i32>(-1, 0), view_layer);
    let ao_r = load_ao(pix + vec2<i32>( 1, 0), view_layer);
    let ao_t = load_ao(pix + vec2<i32>( 0,-1), view_layer);
    let ao_b = load_ao(pix + vec2<i32>( 0, 1), view_layer);
    sum = sum + ao_l * edges_eff.x;
    sum_w = sum_w + edges_eff.x;
    sum = sum + ao_r * edges_eff.y;
    sum_w = sum_w + edges_eff.y;
    sum = sum + ao_t * edges_eff.z;
    sum_w = sum_w + edges_eff.z;
    sum = sum + ao_b * edges_eff.w;
    sum_w = sum_w + edges_eff.w;

    let ao_tl = load_ao(pix + vec2<i32>(-1,-1), view_layer);
    let ao_tr = load_ao(pix + vec2<i32>( 1,-1), view_layer);
    let ao_bl = load_ao(pix + vec2<i32>(-1, 1), view_layer);
    let ao_br = load_ao(pix + vec2<i32>( 1, 1), view_layer);
    sum = sum + ao_tl * w_tl;
    sum_w = sum_w + w_tl;
    sum = sum + ao_tr * w_tr;
    sum_w = sum_w + w_tr;
    sum = sum + ao_bl * w_bl;
    sum_w = sum_w + w_bl;
    sum = sum + ao_br * w_br;
    sum_w = sum_w + w_br;

    return sum / max(sum_w, 1e-6);
}

#ifdef MULTIVIEW
@fragment
fn fs_denoise_intermediate(in: VsOut, @builtin(view_index) view: u32) -> @location(0) f32 {
    let pix = vec2<i32>(in.clip_pos.xy);
    return denoise_at(pix, view, false);
}

@fragment
fn fs_denoise_final(in: VsOut, @builtin(view_index) view: u32) -> @location(0) f32 {
    let pix = vec2<i32>(in.clip_pos.xy);
    return denoise_at(pix, view, true);
}
#else
@fragment
fn fs_denoise_intermediate(in: VsOut) -> @location(0) f32 {
    let pix = vec2<i32>(in.clip_pos.xy);
    return denoise_at(pix, 0u, false);
}

@fragment
fn fs_denoise_final(in: VsOut) -> @location(0) f32 {
    let pix = vec2<i32>(in.clip_pos.xy);
    return denoise_at(pix, 0u, true);
}
#endif
