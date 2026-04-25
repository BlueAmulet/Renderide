//! Edge encoding for the GTAO denoise pass.
//!
//! Mirrors the upstream XeGTAO `XeGTAO_PackEdges` / `XeGTAO_UnpackEdges` helpers (2 bits per
//! left/right/top/bottom direction, packed into the `R8Unorm` `edges` attachment that the GTAO
//! main pass writes alongside the AO term). Quantising edges to four gradient steps `{0, 1/3,
//! 2/3, 1}` keeps the storage cost at one byte per pixel while giving the bilateral kernel enough
//! resolution to stop the filter at depth discontinuities without introducing visible banding.
//!
//! Import with `#import renderide::gtao_packing as gp`. Used by `post/gtao_main.wgsl` (writer)
//! and `post/gtao_denoise.wgsl` (reader).

#define_import_path renderide::gtao_packing

/// Quantises four edge-stopper values in `[0, 1]` into a single `R8Unorm` channel using two bits
/// per LRTB direction. Bit layout (MSB→LSB): `LL RR TT BB`. The factor `2.9` (rather than `3.0`)
/// guards against `round` lifting `0.499...` to `1` after the quantise step, matching XeGTAO.
fn pack_edges(edges_lrtb: vec4<f32>) -> f32 {
    let q = round(saturate(edges_lrtb) * 2.9);
    return dot(q, vec4<f32>(64.0 / 255.0, 16.0 / 255.0, 4.0 / 255.0, 1.0 / 255.0));
}

/// Inverse of [`pack_edges`]: reads the `R8Unorm` channel back into four edge-stopper values.
fn unpack_edges(packed_val: f32) -> vec4<f32> {
    let p = u32(packed_val * 255.5);
    return vec4<f32>(
        f32((p >> 6u) & 0x03u) / 3.0,
        f32((p >> 4u) & 0x03u) / 3.0,
        f32((p >> 2u) & 0x03u) / 3.0,
        f32((p) & 0x03u) / 3.0,
    );
}

/// Computes the four LRTB edge-stoppers for the center pixel from view-space depths of the
/// center and its four cardinal neighbours. Output is in `[0, 1]` where `1` means "no edge"
/// (smooth surface, pass full neighbour weight) and `0` means "hard edge" (discontinuity, drop
/// neighbour weight to zero).
///
/// Algorithm matches XeGTAO's `XeGTAO_CalculateEdges`: linear edge gradient with slope adjustment
/// (so a planar tilted surface reads as "no edge"), then a soft cutoff at ~1.1% of the center
/// depth.
fn calculate_edges(center_z: f32, left_z: f32, right_z: f32, top_z: f32, bottom_z: f32) -> vec4<f32> {
    let edges_lrtb = vec4<f32>(left_z, right_z, top_z, bottom_z) - vec4<f32>(center_z);
    let slope_lr = (edges_lrtb.y - edges_lrtb.x) * 0.5;
    let slope_tb = (edges_lrtb.w - edges_lrtb.z) * 0.5;
    let slope_adjusted = edges_lrtb + vec4<f32>(slope_lr, -slope_lr, slope_tb, -slope_tb);
    let raw = min(abs(edges_lrtb), abs(slope_adjusted));
    return saturate(vec4<f32>(1.25) - raw / max(center_z * 0.011, 1e-6));
}
