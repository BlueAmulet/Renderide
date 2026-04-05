// Debug raster: visualize world-space normals (RGB = n * 0.5 + 0.5).

struct GlobalUniforms {
    view_proj: mat4x4<f32>,
}

struct DrawUniforms {
    model: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> globals: GlobalUniforms;
@group(0) @binding(1) var<uniform> draw: DrawUniforms;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_n: vec3<f32>,
}

@vertex
fn vs_main(
    @location(0) pos: vec4<f32>,
    @location(1) normal: vec4<f32>,
) -> VertexOutput {
    let world_p = draw.model * vec4<f32>(pos.xyz, 1.0);
    let world_n = normalize((draw.model * vec4<f32>(normal.xyz, 0.0)).xyz);
    var out: VertexOutput;
    out.clip_pos = globals.view_proj * world_p;
    out.world_n = world_n;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.world_n * 0.5 + 0.5, 1.0);
}
