//! Canvas UI text unlit (Unity shader asset `UI_TextUnlit`, normalized key `ui_textunlit`): MSDF/SDF/Raster font atlas, tint, outline, rect clip.
//!
//! Build emits `ui_textunlit_default` / `ui_textunlit_multiview` via [`MULTIVIEW`](https://docs.rs/naga_oil).
//! `@group(1)` global names match Unity `UI_TextUnlit.shader` material property names for host reflection.
//!
//! **Vertex color:** Unity multiplies `_TintColor * vertexColor`. This manifest path has no vertex color stream;
//! vertex color is treated as white (`vec4(1.0)`).
//!
//! **Keywords (Unity):** RASTER, SDF, MSDF, OUTLINE, RECTCLIP, OVERLAY are shader variants. This single WGSL
//! branches at runtime: MSDF/SDF when `_Range` indicates signed-distance sampling (non-negligible `.xy`), else RASTER.
//! OUTLINE when `_OutlineSize` is significant. RECTCLIP when `_Rect` has non-zero area. **OVERLAY** depth compositing
//! is not implemented; when `_OverlayTint.a` is high, a simple tint approximation may be applied (no scene depth).
//!
//! Per-draw uniforms (`@group(2)`) use [`renderide::per_draw`].

#import renderide::globals as rg
#import renderide::per_draw as pd

struct UiTextUnlitMaterial {
    _TintColor: vec4<f32>,
    _OverlayTint: vec4<f32>,
    _OutlineColor: vec4<f32>,
    _BackgroundColor: vec4<f32>,
    _Range: vec4<f32>,
    _Rect: vec4<f32>,
    _FaceDilate: f32,
    _FaceSoftness: f32,
    _OutlineSize: f32,
    _SrcBlend: f32,
    _DstBlend: f32,
    _ZWrite: f32,
    _Cull: f32,
    _ZTest: f32,
    _StencilComp: f32,
    _Stencil: f32,
    _StencilOp: f32,
    _StencilWriteMask: f32,
    _StencilReadMask: f32,
    _ColorMask: f32,
    _pad: vec3<f32>,
}

@group(1) @binding(0) var<uniform> mat: UiTextUnlitMaterial;
@group(1) @binding(1) var _FontAtlas: texture_2d<f32>;
@group(1) @binding(2) var _FontAtlas_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) extra_data: vec4<f32>,
    @location(2) obj_xy: vec2<f32>,
}

fn median3(r: f32, g: f32, b: f32) -> f32 {
    return max(min(r, g), min(max(r, g), b));
}

@vertex
fn vs_main(
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) extra_n: vec4<f32>,
    @location(2) uv: vec2<f32>,
) -> VertexOutput {
    let world_p = pd::draw.model * vec4<f32>(pos.xyz, 1.0);
#ifdef MULTIVIEW
    var vp: mat4x4<f32>;
    if (view_idx == 0u) {
        vp = pd::draw.view_proj_left;
    } else {
        vp = pd::draw.view_proj_right;
    }
#else
    let vp = pd::draw.view_proj_left;
#endif
    var out: VertexOutput;
    out.clip_pos = vp * world_p;
    out.uv = uv;
    out.extra_data = extra_n;
    out.obj_xy = pos.xy;
    return out;
}

/// Returns true when `p` is outside the axis-aligned rect (min = xy, size = zw) in object XY.
fn outside_rect_clip(p: vec2<f32>, r: vec4<f32>) -> bool {
    let min_v = r.xy;
    let max_v = r.xy + r.zw;
    return p.x < min_v.x || p.x > max_v.x || p.y < min_v.y || p.y > max_v.y;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let vtx_color = vec4<f32>(1.0, 1.0, 1.0, 1.0);

    let rect = mat._Rect;
    let use_rect_clip = rect.z * rect.w > 1e-6;
    if (use_rect_clip && outside_rect_clip(in.obj_xy, rect)) {
        discard;
    }

    let atlas_color = textureSample(_FontAtlas, _FontAtlas_sampler, in.uv);

    let range_xy = mat._Range.xy;
    let use_sdf_path = abs(range_xy.x) + abs(range_xy.y) > 1e-7;

    var c: vec4<f32>;

    if (use_sdf_path) {
        var sig_dist: f32;
        let m = median3(atlas_color.r, atlas_color.g, atlas_color.b);
        sig_dist = m - 0.5;
        sig_dist = sig_dist + mat._FaceDilate + in.extra_data.x;

        let fw = vec2<f32>(fwidth(in.uv.x), fwidth(in.uv.y));
        let anti_aliasing = dot(range_xy, vec2<f32>(0.5) / max(fw, vec2<f32>(1e-6)));
        let aa = max(anti_aliasing, 1.0);

        var glyph_lerp = mix(sig_dist * aa, sig_dist, mat._FaceSoftness);
        glyph_lerp = clamp(glyph_lerp + 0.5, 0.0, 1.0);

        if (max(glyph_lerp, mat._BackgroundColor.a) < 0.001) {
            discard;
        }

        var fill_color = mat._TintColor * vtx_color;

        let outline_w = mat._OutlineSize + in.extra_data.y;
        if (outline_w > 1e-6 || mat._OutlineSize > 1e-6) {
            let outline_dist = sig_dist - outline_w;
            var outline_lerp = mix(outline_dist * aa, outline_dist, mat._FaceSoftness);
            outline_lerp = clamp(outline_lerp + 0.5, 0.0, 1.0);
            fill_color = mix(mat._OutlineColor * vec4<f32>(1.0, 1.0, 1.0, vtx_color.a), fill_color, outline_lerp);
        }

        c = mix(mat._BackgroundColor * vtx_color, fill_color, glyph_lerp);
    } else {
        c = atlas_color * vtx_color;
        if (c.a < 0.001) {
            discard;
        }
    }

    let o = mat._OverlayTint;
    if (o.a > 0.01) {
        c = vec4<f32>(c.rgb * mix(vec3<f32>(1.0), o.rgb, o.a), c.a);
    }

    var lit: u32 = 0u;
    if (rg::frame.light_count > 0u) {
        lit = rg::lights[0].light_type;
    }
    return c + vec4<f32>(vec3<f32>(f32(lit) * 1e-10), 0.0);
}
