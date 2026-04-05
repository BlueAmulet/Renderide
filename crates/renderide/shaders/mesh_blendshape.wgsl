// Weighted blendshape deltas (compute). `deltas` uses `BLENDSHAPE_OFFSET_GPU_STRIDE` (48) bytes per
// (shape_index, vertex): three vec4 chunks (position, normal, tangent deltas). This entry point
// consumes only `.pos.xyz` per shape weight.

struct Params {
    vertex_count: u32,
    shape_count: u32,
    vertices_per_shape: u32,
    _pad: u32,
}

struct DeltaPacked {
    pos: vec4<f32>,
    _norm: vec4<f32>,
    _tang: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> base_pos: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> deltas: array<DeltaPacked>;
@group(0) @binding(3) var<storage, read> weights: array<f32>;
@group(0) @binding(4) var<storage, read_write> out_pos: array<vec4<f32>>;

@compute @workgroup_size(64)
fn blendshape_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.vertex_count) {
        return;
    }
    var acc = base_pos[i].xyz;
    for (var s = 0u; s < params.shape_count; s = s + 1u) {
        let wi = weights[s];
        if (wi != 0.0) {
            let di = deltas[s * params.vertices_per_shape + i];
            acc += wi * di.pos.xyz;
        }
    }
    out_pos[i] = vec4<f32>(acc, base_pos[i].w);
}
