//! GTAO apply pass: reshapes the (optionally denoised) AO term and modulates HDR scene color
//! with the resulting visibility factor.
//!
//! Reads the chain's HDR scene color (binding 0) and the AO term written by the main / denoise
//! pipeline (binding 2). The visibility factor is sharpened by `intensity` (an exponent applied
//! to the linear AO) and passed through Jiménez et al.'s multi-bounce fit (paper Eq. 10) before
//! being multiplied into RGB. Splitting this from the main pass — the historical Renderide GTAO
//! shader did both — is what lets the bilateral denoise filter a *linear* AO term (so simple
//! weighted averaging is meaningful); applying the nonlinear shaping after the denoise restores
//! the original look without re-introducing the noise.
//!
//! Build script composes this into `gtao_apply_default` and `gtao_apply_multiview`. The multiview
//! variant routes `@builtin(view_index)` through to per-eye sampling of the scene color array
//! and the AO term array.
//!
//! Bind group (`@group(0)`):
//! - `@binding(0)` HDR scene color (`texture_2d_array<f32>`; mono samples layer 0).
//! - `@binding(1)` linear-clamp sampler.
//! - `@binding(2)` AO term (`texture_2d_array<f32>`; single channel `R16Float`).
//! - `@binding(3)` `GtaoParams` uniform (intensity, multibounce albedo).

@group(0) @binding(0) var scene_color_hdr: texture_2d_array<f32>;
@group(0) @binding(1) var linear_clamp: sampler;
@group(0) @binding(2) var ao_term: texture_2d_array<f32>;

/// User-tunable GTAO parameters. Only `intensity` and `albedo_multibounce` are read here; the
/// other fields belong to the main / denoise stages but ride along on the same shared UBO.
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

@group(0) @binding(3) var<uniform> gtao: GtaoParams;

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

/// Paper Eq. 10 cubic fit: recovers near-field multi-bounce indirect illumination lost when
/// applying ambient occlusion alone. Uses a gray-albedo proxy since this pass doesn't sample
/// per-pixel albedo.
fn multi_bounce_fit(ao: f32, albedo: f32) -> f32 {
    let a = 2.0404 * albedo - 0.3324;
    let b = 4.7951 * albedo - 0.6417;
    let c = 2.7552 * albedo + 0.6903;
    return max(ao, ((a * ao - b) * ao + c) * ao);
}

/// Reshapes a raw / denoised visibility factor in `[0, 1]` into the final modulation factor:
/// applies the `intensity` exponent and the multi-bounce fit. Kept separate from the per-eye
/// fragment entry points so both `MULTIVIEW` variants share the math.
fn shape_visibility(visibility: f32) -> f32 {
    let boosted = saturate(pow(visibility, max(gtao.intensity, 0.0)));
    return multi_bounce_fit(boosted, gtao.albedo_multibounce);
}

#ifdef MULTIVIEW
@fragment
fn fs_main(in: VsOut, @builtin(view_index) view: u32) -> @location(0) vec4<f32> {
    let visibility = textureSample(ao_term, linear_clamp, in.uv, view).r;
    let factor = shape_visibility(saturate(visibility));
    let hdr = textureSample(scene_color_hdr, linear_clamp, in.uv, view);
    return vec4<f32>(hdr.rgb * factor, hdr.a);
}
#else
@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let visibility = textureSample(ao_term, linear_clamp, in.uv, 0u).r;
    let factor = shape_visibility(saturate(visibility));
    let hdr = textureSample(scene_color_hdr, linear_clamp, in.uv, 0u);
    return vec4<f32>(hdr.rgb * factor, hdr.a);
}
#endif
