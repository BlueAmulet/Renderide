#import uniform_ring

struct HostUnlitParams {
    base_color: vec4f,
}

@group(1) @binding(0) var<uniform> host_unlit: HostUnlitParams;

struct VertexInput {
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
}
struct VertexOutput {
    @builtin(position) clip_position: vec4f,
}
@group(0) @binding(0) var<uniform> uniforms: array<uniform_ring::UniformsSlot, 64>;
@vertex
fn vs_main(in: VertexInput, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    let u = uniforms[instance_index];
    var out: VertexOutput;
    out.clip_position = u.mvp * vec4f(in.position, 1.0);
    return out;
}
@fragment
fn fs_main(_in: VertexOutput) -> @location(0) vec4f {
    return host_unlit.base_color;
}
