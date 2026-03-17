//! WGSL shader source strings for pipeline modules.

pub(crate) const NORMAL_SHADER_SRC: &str = r#"
struct VertexInput {
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
}
struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) world_normal: vec3f,
}
struct Uniforms {
    mvp: mat4x4f,
    model: mat4x4f,
}
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.mvp * vec4f(in.position, 1.0);
    out.world_normal = (uniforms.model * vec4f(in.normal, 0.0)).xyz;
    return out;
}
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let n = normalize(in.world_normal);
    return vec4f(n * 0.5 + 0.5, 1.0);
}
"#;

pub(crate) const UV_DEBUG_SHADER_SRC: &str = r#"
struct VertexInput {
    @location(0) position: vec3f,
    @location(1) uv: vec2f,
}
struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) uv: vec2f,
}
struct Uniforms {
    mvp: mat4x4f,
    model: mat4x4f,
}
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.mvp * vec4f(in.position, 1.0);
    out.uv = in.uv;
    return out;
}
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> vec3f {
    let c = v * s;
    let h6 = h * 6.0;
    let h2 = h6 - 2.0 * floor(h6 / 2.0);
    let x = c * (1.0 - abs(h2 - 1.0));
    let m = v - c;
    var r = 0.0;
    var g = 0.0;
    var b = 0.0;
    if h6 < 1.0 {
        r = c; g = x; b = 0.0;
    } else if h6 < 2.0 {
        r = x; g = c; b = 0.0;
    } else if h6 < 3.0 {
        r = 0.0; g = c; b = x;
    } else if h6 < 4.0 {
        r = 0.0; g = x; b = c;
    } else if h6 < 5.0 {
        r = x; g = 0.0; b = c;
    } else {
        r = c; g = 0.0; b = x;
    }
    return vec3f(r + m, g + m, b + m);
}
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let u = clamp(in.uv.x, 0.0, 1.0);
    let v = clamp(in.uv.y, 0.0, 1.0);
    let hue = u * (300.0 / 360.0);
    let sat = 1.0 - v;
    let rgb = hsv_to_rgb(hue, sat, 1.0);
    return vec4f(rgb, 1.0);
}
"#;

/// Overlay stencil shader with optional rect clip (IUIX_Material.RectClip).
pub(crate) const OVERLAY_STENCIL_SHADER_SRC: &str = r#"
struct VertexInput {
    @location(0) position: vec3f,
    @location(1) uv: vec2f,
}
struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) uv: vec2f,
}
struct Uniforms {
    mvp: mat4x4f,
    model: mat4x4f,
    clip_rect: vec4f,
}
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.mvp * vec4f(in.position, 1.0);
    out.uv = in.uv;
    return out;
}
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> vec3f {
    let c = v * s;
    let h6 = h * 6.0;
    let h2 = h6 - 2.0 * floor(h6 / 2.0);
    let x = c * (1.0 - abs(h2 - 1.0));
    let m = v - c;
    var r = 0.0;
    var g = 0.0;
    var b = 0.0;
    if h6 < 1.0 {
        r = c; g = x; b = 0.0;
    } else if h6 < 2.0 {
        r = x; g = c; b = 0.0;
    } else if h6 < 3.0 {
        r = 0.0; g = c; b = x;
    } else if h6 < 4.0 {
        r = 0.0; g = x; b = c;
    } else if h6 < 5.0 {
        r = x; g = 0.0; b = c;
    } else {
        r = c; g = 0.0; b = x;
    }
    return vec3f(r + m, g + m, b + m);
}
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let rect = uniforms.clip_rect;
    if rect.z > 0.0 {
        let ndc = in.clip_position.xy / in.clip_position.w;
        let nx = (ndc.x + 1.0) * 0.5;
        let ny = 1.0 - (ndc.y + 1.0) * 0.5;
        if nx < rect.x || nx > rect.x + rect.z || ny < rect.y || ny > rect.y + rect.w {
            discard;
        }
    }
    let u = clamp(in.uv.x, 0.0, 1.0);
    let v = clamp(in.uv.y, 0.0, 1.0);
    let hue = u * (300.0 / 360.0);
    let sat = 1.0 - v;
    let rgb = hsv_to_rgb(hue, sat, 1.0);
    return vec4f(rgb, 1.0);
}
"#;

pub(crate) const SKINNED_SHADER_SRC: &str = r#"
struct VertexInput {
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
    @location(2) tangent: vec3f,
    @location(3) bone_indices: vec4i,
    @location(4) bone_weights: vec4f,
}
struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) world_normal: vec3f,
}
struct SkinnedUniforms {
    mvp: mat4x4f,
    bone_matrices: array<mat4x4f, 256>,
    num_blendshapes: u32,
    num_vertices: u32,
    blendshape_weights: array<vec4f, 32>,
}
struct BlendshapeOffset {
    position_offset: vec3f,
    normal_offset: vec3f,
    tangent_offset: vec3f,
}
@group(0) @binding(0) var<uniform> uniforms: SkinnedUniforms;
@group(0) @binding(1) var<storage, read> blendshape_offsets: array<BlendshapeOffset>;
@vertex
fn vs_main(
    in: VertexInput,
    @builtin(vertex_index) vertex_index: u32,
) -> VertexOutput {
    var out: VertexOutput;
    var pos = in.position;
    var norm = in.normal;
    var tang = in.tangent;
    for (var i = 0u; i < uniforms.num_blendshapes; i++) {
        let q = i / 4u;
        let r = i % 4u;
        let v = uniforms.blendshape_weights[q];
        let weight = select(select(select(v.x, v.y, r == 1u), select(v.z, v.w, r == 3u), r >= 2u), v.x, r == 0u);
        if weight > 0.0 {
            let offset_idx = i * uniforms.num_vertices + vertex_index;
            let offset = blendshape_offsets[offset_idx];
            pos += offset.position_offset * weight;
            norm += offset.normal_offset * weight;
            tang += offset.tangent_offset * weight;
        }
    }
    var world_pos = vec4f(0.0, 0.0, 0.0, 0.0);
    var world_normal = vec4f(0.0, 0.0, 0.0, 0.0);
    var world_tangent = vec4f(0.0, 0.0, 0.0, 0.0);
    let total_weight = in.bone_weights[0] + in.bone_weights[1] + in.bone_weights[2] + in.bone_weights[3];
    let inv_total = select(1.0, 1.0 / total_weight, total_weight > 1e-6);
    for (var i = 0; i < 4; i++) {
        let idx = clamp(in.bone_indices[i], 0, 255);
        let w = in.bone_weights[i] * inv_total;
        if w > 0.0 {
            let bone = uniforms.bone_matrices[idx];
            world_pos += w * bone * vec4f(pos, 1.0);
            world_normal += w * bone * vec4f(norm, 0.0);
            world_tangent += w * bone * vec4f(tang, 0.0);
        }
    }
    _ = world_tangent;
    out.clip_position = uniforms.mvp * world_pos;
    let n = world_normal.xyz;
    let len = length(n);
    out.world_normal = select(vec3f(0.0, 1.0, 0.0), n / len, len > 1e-6);
    return out;
}
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let n = normalize(in.world_normal);
    return vec4f(n * 0.5 + 0.5, 1.0);
}
"#;

/// Normal debug MRT shader: outputs color, world position, world normal for RTAO.
pub(crate) const NORMAL_DEBUG_MRT_SHADER_SRC: &str = r#"
struct VertexInput {
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
}
struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) world_normal: vec3f,
    @location(1) world_position: vec3f,
}
struct Uniforms {
    mvp: mat4x4f,
    model: mat4x4f,
}
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world_pos = uniforms.model * vec4f(in.position, 1.0);
    out.clip_position = uniforms.mvp * vec4f(in.position, 1.0);
    out.world_normal = (uniforms.model * vec4f(in.normal, 0.0)).xyz;
    out.world_position = world_pos.xyz;
    return out;
}
struct FragmentOutput {
    @location(0) color: vec4f,
    @location(1) position: vec4f,
    @location(2) normal: vec4f,
}
@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    let n = normalize(in.world_normal);
    return FragmentOutput(
        vec4f(n * 0.5 + 0.5, 1.0),
        vec4f(in.world_position, 1.0),
        vec4f(n, 0.0),
    );
}
"#;

/// UV debug MRT shader: outputs color, world position, world normal for RTAO.
pub(crate) const UV_DEBUG_MRT_SHADER_SRC: &str = r#"
struct VertexInput {
    @location(0) position: vec3f,
    @location(1) uv: vec2f,
}
struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) uv: vec2f,
    @location(1) world_position: vec3f,
    @location(2) world_normal: vec3f,
}
struct Uniforms {
    mvp: mat4x4f,
    model: mat4x4f,
}
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world_pos = uniforms.model * vec4f(in.position, 1.0);
    out.clip_position = uniforms.mvp * vec4f(in.position, 1.0);
    out.uv = in.uv;
    out.world_position = world_pos.xyz;
    out.world_normal = (uniforms.model * vec4f(0.0, 1.0, 0.0, 0.0)).xyz;
    return out;
}
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> vec3f {
    let c = v * s;
    let h6 = h * 6.0;
    let h2 = h6 - 2.0 * floor(h6 / 2.0);
    let x = c * (1.0 - abs(h2 - 1.0));
    let m = v - c;
    var r = 0.0;
    var g = 0.0;
    var b = 0.0;
    if h6 < 1.0 {
        r = c; g = x; b = 0.0;
    } else if h6 < 2.0 {
        r = x; g = c; b = 0.0;
    } else if h6 < 3.0 {
        r = 0.0; g = c; b = x;
    } else if h6 < 4.0 {
        r = 0.0; g = x; b = c;
    } else if h6 < 5.0 {
        r = x; g = 0.0; b = c;
    } else {
        r = c; g = 0.0; b = x;
    }
    return vec3f(r + m, g + m, b + m);
}
struct UvFragmentOutput {
    @location(0) color: vec4f,
    @location(1) position: vec4f,
    @location(2) normal: vec4f,
}
@fragment
fn fs_main(in: VertexOutput) -> UvFragmentOutput {
    let u = clamp(in.uv.x, 0.0, 1.0);
    let v = clamp(in.uv.y, 0.0, 1.0);
    let hue = u * (300.0 / 360.0);
    let sat = 1.0 - v;
    let rgb = hsv_to_rgb(hue, sat, 1.0);
    let n = normalize(in.world_normal);
    return UvFragmentOutput(
        vec4f(rgb, 1.0),
        vec4f(in.world_position, 1.0),
        vec4f(n, 0.0),
    );
}
"#;

/// Skinned MRT shader: outputs color, world position, world normal for RTAO.
pub(crate) const SKINNED_MRT_SHADER_SRC: &str = r#"
struct VertexInput {
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
    @location(2) tangent: vec3f,
    @location(3) bone_indices: vec4i,
    @location(4) bone_weights: vec4f,
}
struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) world_normal: vec3f,
    @location(1) world_position: vec3f,
}
struct SkinnedUniforms {
    mvp: mat4x4f,
    bone_matrices: array<mat4x4f, 256>,
    num_blendshapes: u32,
    num_vertices: u32,
    blendshape_weights: array<vec4f, 32>,
}
struct BlendshapeOffset {
    position_offset: vec3f,
    normal_offset: vec3f,
    tangent_offset: vec3f,
}
@group(0) @binding(0) var<uniform> uniforms: SkinnedUniforms;
@group(0) @binding(1) var<storage, read> blendshape_offsets: array<BlendshapeOffset>;
@vertex
fn vs_main(
    in: VertexInput,
    @builtin(vertex_index) vertex_index: u32,
) -> VertexOutput {
    var out: VertexOutput;
    var pos = in.position;
    var norm = in.normal;
    var tang = in.tangent;
    for (var i = 0u; i < uniforms.num_blendshapes; i++) {
        let q = i / 4u;
        let r = i % 4u;
        let v = uniforms.blendshape_weights[q];
        let weight = select(select(select(v.x, v.y, r == 1u), select(v.z, v.w, r == 3u), r >= 2u), v.x, r == 0u);
        if weight > 0.0 {
            let offset_idx = i * uniforms.num_vertices + vertex_index;
            let offset = blendshape_offsets[offset_idx];
            pos += offset.position_offset * weight;
            norm += offset.normal_offset * weight;
            tang += offset.tangent_offset * weight;
        }
    }
    var world_pos = vec4f(0.0, 0.0, 0.0, 0.0);
    var world_normal = vec4f(0.0, 0.0, 0.0, 0.0);
    var world_tangent = vec4f(0.0, 0.0, 0.0, 0.0);
    let total_weight = in.bone_weights[0] + in.bone_weights[1] + in.bone_weights[2] + in.bone_weights[3];
    let inv_total = select(1.0, 1.0 / total_weight, total_weight > 1e-6);
    for (var i = 0; i < 4; i++) {
        let idx = clamp(in.bone_indices[i], 0, 255);
        let w = in.bone_weights[i] * inv_total;
        if w > 0.0 {
            let bone = uniforms.bone_matrices[idx];
            world_pos += w * bone * vec4f(pos, 1.0);
            world_normal += w * bone * vec4f(norm, 0.0);
            world_tangent += w * bone * vec4f(tang, 0.0);
        }
    }
    _ = world_tangent;
    out.clip_position = uniforms.mvp * world_pos;
    let n = world_normal.xyz;
    let len = length(n);
    out.world_normal = select(vec3f(0.0, 1.0, 0.0), n / len, len > 1e-6);
    out.world_position = world_pos.xyz;
    return out;
}
struct SkinnedFragmentOutput {
    @location(0) color: vec4f,
    @location(1) position: vec4f,
    @location(2) normal: vec4f,
}
@fragment
fn fs_main(in: VertexOutput) -> SkinnedFragmentOutput {
    let n = normalize(in.world_normal);
    return SkinnedFragmentOutput(
        vec4f(n * 0.5 + 0.5, 1.0),
        vec4f(in.world_position, 1.0),
        vec4f(n, 0.0),
    );
}
"#;
