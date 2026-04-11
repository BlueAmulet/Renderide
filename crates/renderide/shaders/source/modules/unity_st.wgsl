//! Unity `_ST` tiling/offset with WebGPU V-flip for sampled UVs.
//!
//! Unity mesh UVs use bottom-left origin; WebGPU samples with v=0 at the top row. Import with
//! `#import renderide::unity_st as st` and use [`apply_st`] for texture coordinates that follow Unity
//! `_MainTex_ST`-style `(scale.xy, offset.zw)` packing.

#define_import_path renderide::unity_st

/// Applies `_ST` scale/translate, then flips V so sampling matches Unity for a top-left WebGPU space.
fn apply_st(uv: vec2<f32>, st: vec4<f32>) -> vec2<f32> {
    let uv_st = uv * st.xy + st.zw;
    return vec2<f32>(uv_st.x, 1.0 - uv_st.y);
}
