//! Per-face downsample pass for the IBL source radiance pyramid.
//!
//! Writes mip *i* from mip *i - 1* so GGX filtered-importance sampling can choose a source LOD
//! whose texel footprint roughly matches each importance sample.

struct DownsampleParams {
    /// Destination mip face edge in texels.
    dst_size: u32,
    /// Source mip face edge in texels.
    src_size: u32,
    /// Reserved padding.
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> p: DownsampleParams;
@group(0) @binding(1) var src_mip: texture_2d_array<f32>;
@group(0) @binding(2) var dst_mip: texture_storage_2d_array<rgba16float, write>;

fn load_src(x: u32, y: u32, face: u32) -> vec3<f32> {
    let sx = min(x, max(p.src_size, 1u) - 1u);
    let sy = min(y, max(p.src_size, 1u) - 1u);
    return textureLoad(src_mip, vec2i(i32(sx), i32(sy)), i32(face), 0).rgb;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_size = max(p.dst_size, 1u);
    if (gid.x >= dst_size || gid.y >= dst_size || gid.z >= 6u) {
        return;
    }

    let sx = gid.x * 2u;
    let sy = gid.y * 2u;
    let color =
        load_src(sx, sy, gid.z) +
        load_src(sx + 1u, sy, gid.z) +
        load_src(sx, sy + 1u, gid.z) +
        load_src(sx + 1u, sy + 1u, gid.z);

    textureStore(
        dst_mip,
        vec2i(i32(gid.x), i32(gid.y)),
        i32(gid.z),
        vec4<f32>(color * 0.25, 1.0),
    );
}
