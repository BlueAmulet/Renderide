#import uniform_ring
#import ui_common

/// Uniform block for group 1 binding 1 (inverse projection for `OVERLAY` depth compare).
struct NativeUiOverlayUnproject {
    inv_scene_proj: mat4x4f,
    inv_ui_proj: mat4x4f,
}

fn overlay_ndc_xy(frag_xy: vec2f, dims: vec2u) -> vec2f {
    let w = max(f32(dims.x), 1.0);
    let h = max(f32(dims.y), 1.0);
    let x = (frag_xy.x + 0.5) / w * 2.0 - 1.0;
    let y = 1.0 - (frag_xy.y + 0.5) / h * 2.0;
    return vec2f(x, y);
}

fn overlay_view_z_from_depth(d: f32, ndc_xy: vec2f, inv_proj: mat4x4f) -> f32 {
    let ndc_z = d * 2.0 - 1.0;
    let h = inv_proj * vec4f(ndc_xy.x, ndc_xy.y, ndc_z, 1.0);
    let v = h.xyz / max(h.w, 1e-8);
    return v.z;
}

fn overlay_linear_eye_dist(view_z: f32) -> f32 {
    return -view_z;
}

struct UiUnlitMaterialUniform {
    tint: vec4f,
    overlay_tint: vec4f,
    main_tex_st: vec4f,
    mask_tex_st: vec4f,
    rect: vec4f,
    cutoff: f32,
    flags: u32,
    pad_tail: vec2u,
}

@group(0) @binding(0) var<uniform> uniforms: array<uniform_ring::UniformsSlot, 64>;
@group(1) @binding(0) var scene_depth: texture_depth_2d;
@group(1) @binding(1) var<uniform> overlay_unproject: NativeUiOverlayUnproject;
@group(2) @binding(0) var<uniform> mat: UiUnlitMaterialUniform;
@group(2) @binding(1) var main_tex: texture_2d<f32>;
@group(2) @binding(2) var main_samp: sampler;
@group(2) @binding(3) var mask_tex: texture_2d<f32>;
@group(2) @binding(4) var mask_samp: sampler;

struct VertexInput {
    @location(0) position: vec3f,
    @location(1) uv: vec2f,
    @location(2) color: vec4f,
    @location(3) aux: vec4f,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) uv: vec2f,
    @location(1) color: vec4f,
    @location(2) lerp_color: vec4f,
    @location(3) local_xy: vec2f,
}

const FLAG_ALPHACLIP: u32 = 1u;
const FLAG_RECTCLIP: u32 = 2u;
const FLAG_OVERLAY: u32 = 4u;
const FLAG_TEXTURE_NORMALMAP: u32 = 8u;
const FLAG_TEXTURE_LERPCOLOR: u32 = 16u;
const FLAG_MASK_MUL: u32 = 32u;
const FLAG_MASK_CLIP: u32 = 64u;

@vertex
fn vs_main(in: VertexInput, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    let u = uniforms[instance_index];
    var out: VertexOutput;
    out.clip_position = u.mvp * vec4f(in.position, 1.0);
    out.uv = ui_common::apply_main_tex_st(in.uv, mat.main_tex_st);
    out.color = in.color * mat.tint;
    out.lerp_color = in.aux * mat.tint;
    out.local_xy = in.position.xy;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    if ((mat.flags & FLAG_RECTCLIP) != 0u) {
        if (!ui_common::inside_rect_clip(in.local_xy, mat.rect)) {
            discard;
        }
    }

    var tex_color = textureSample(main_tex, main_samp, in.uv);

    if ((mat.flags & FLAG_TEXTURE_NORMALMAP) != 0u) {
        let n = tex_color.xyz * 2.0 - 1.0;
        tex_color = vec4f(n * 0.5 + 0.5, 1.0);
    }

    var color: vec4f;
    if ((mat.flags & FLAG_TEXTURE_LERPCOLOR) != 0u) {
        let l = (tex_color.r + tex_color.g + tex_color.b) * 0.33333333;
        let lerp_c = mix(in.color, in.lerp_color, l);
        color = vec4f(lerp_c.rgb, lerp_c.a * tex_color.a);
    } else {
        color = tex_color * in.color;
    }

    let mask_uv = ui_common::apply_main_tex_st(in.uv, mat.mask_tex_st);
    if ((mat.flags & FLAG_MASK_MUL) != 0u || (mat.flags & FLAG_MASK_CLIP) != 0u) {
        let mask = textureSample(mask_tex, mask_samp, mask_uv);
        let mul = (mask.r + mask.g + mask.b) * 0.3333333 * mask.a;
        if ((mat.flags & FLAG_MASK_MUL) != 0u) {
            color.a *= mul;
        }
        if ((mat.flags & FLAG_MASK_CLIP) != 0u) {
            if (mul - mat.cutoff <= 0.0) {
                discard;
            }
        }
    }

    if ((mat.flags & FLAG_ALPHACLIP) != 0u && (mat.flags & FLAG_MASK_CLIP) == 0u) {
        if (color.a - mat.cutoff <= 0.0) {
            discard;
        }
    }

    if ((mat.flags & FLAG_OVERLAY) != 0u) {
        let dims = textureDimensions(scene_depth);
        let px = vec2i(i32(in.clip_position.x), i32(in.clip_position.y));
        let sx = clamp(px.x, 0, i32(dims.x) - 1);
        let sy = clamp(px.y, 0, i32(dims.y) - 1);
        let scene_d = textureLoad(scene_depth, vec2i(sx, sy), 0);
        let part_d = in.clip_position.z;
        let ndc_xy = overlay_ndc_xy(in.clip_position.xy, dims);
        let scene_vz = overlay_view_z_from_depth(scene_d, ndc_xy, overlay_unproject.inv_scene_proj);
        let part_vz = overlay_view_z_from_depth(part_d, ndc_xy, overlay_unproject.inv_ui_proj);
        let scene_eye = overlay_linear_eye_dist(scene_vz);
        let part_eye = overlay_linear_eye_dist(part_vz);
        if (part_eye > scene_eye) {
            color *= mat.overlay_tint;
        }
    }

    return color;
}
