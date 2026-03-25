#import uniform_ring
#import ui_common

struct UiTextUnlitMaterialUniform {
    tint_color: vec4f,
    overlay_tint: vec4f,
    outline_color: vec4f,
    background_color: vec4f,
    range_xy: vec4f,
    face_dilate: f32,
    face_softness: f32,
    outline_size: f32,
    pad_scalar: f32,
    rect: vec4f,
    flags: u32,
    pad_flags: u32,
    pad_tail: vec2u,
}

@group(0) @binding(0) var<uniform> uniforms: array<uniform_ring::UniformsSlot, 64>;
@group(1) @binding(0) var scene_depth: texture_depth_2d;
@group(2) @binding(0) var<uniform> mat: UiTextUnlitMaterialUniform;
@group(2) @binding(1) var font_atlas: texture_2d<f32>;
@group(2) @binding(2) var font_samp: sampler;

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
    @location(2) local_xy: vec2f,
}

const MODE_RASTER: u32 = 0u;
const MODE_SDF: u32 = 1u;
const MODE_MSDF: u32 = 2u;
const FLAG_OUTLINE: u32 = 256u;
const FLAG_RECTCLIP: u32 = 512u;
const FLAG_OVERLAY: u32 = 1024u;

@vertex
fn vs_main(in: VertexInput, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    let u = uniforms[instance_index];
    var out: VertexOutput;
    out.clip_position = u.mvp * vec4f(in.position, 1.0);
    out.uv = in.uv;
    out.color = in.color * mat.tint_color;
    out.local_xy = in.position.xy;
    _ = in.aux;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let mode = mat.flags & 3u;

    if ((mat.flags & FLAG_RECTCLIP) != 0u) {
        if (!ui_common::inside_rect_clip(in.local_xy, mat.rect)) {
            discard;
        }
    }

    var out_color = in.color;

    if (mode == MODE_RASTER) {
        let a = textureSample(font_atlas, font_samp, in.uv).a;
        out_color.a *= a;
    } else if (mode == MODE_SDF) {
        let s = textureSample(font_atlas, font_samp, in.uv);
        let d = s.r;
        let r = mat.range_xy.xy;
        let feather = max(mat.face_softness * r.y, 1e-6);
        let face = smoothstep(r.x - feather, r.x + feather, d - mat.face_dilate);
        var alpha = face * in.color.a;
        if ((mat.flags & FLAG_OUTLINE) != 0u && mat.outline_size > 0.0) {
            let outline_d = d - mat.outline_size;
            let outline_a = (1.0 - smoothstep(r.x - feather, r.x + feather, outline_d)) * mat.outline_color.a;
            let base_rgb = mix(mat.outline_color.rgb, in.color.rgb, face);
            out_color = vec4f(base_rgb, max(alpha, outline_a * (1.0 - face)));
        } else {
            out_color = vec4f(in.color.rgb, alpha);
        }
        out_color = mix(mat.background_color, out_color, out_color.a);
    } else {
        let s = textureSample(font_atlas, font_samp, in.uv);
        let d = min(min(s.r, s.g), s.b);
        let r = mat.range_xy.xy;
        let feather = max(mat.face_softness * r.y, 1e-6);
        let face = smoothstep(r.x - feather, r.x + feather, d - mat.face_dilate);
        out_color = vec4f(in.color.rgb, face * in.color.a);
    }

    if ((mat.flags & FLAG_OVERLAY) != 0u) {
        let dims = textureDimensions(scene_depth);
        let px = vec2i(i32(in.clip_position.x), i32(in.clip_position.y));
        let sx = clamp(px.x, 0, i32(dims.x) - 1);
        let sy = clamp(px.y, 0, i32(dims.y) - 1);
        let scene_z = textureLoad(scene_depth, vec2i(sx, sy), 0);
        if (in.clip_position.z > scene_z) {
            out_color *= mat.overlay_tint;
        }
    }

    return out_color;
}
