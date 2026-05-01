//! Mip-0 producer for cubemap-source IBL bakes.
//!
//! Resamples a host-uploaded cubemap into the destination Rgba16Float cube at mip 0. Output is
//! always written in canonical orientation; the optional `storage_v_inverted` flip is applied to
//! the *input* sampling direction so native-compressed faces with flipped Y still resolve correctly.

#import renderide::ggx_prefilter as ggx

struct Mip0CubeParams {
    /// Destination cube face edge in texels.
    dst_size: u32,
    /// Source cube face edge in texels (mip 0).
    src_face_size: u32,
    /// Storage-orientation V-flip flag. Non-zero means the source face data is V-inverted and the
    /// sample direction must compensate.
    storage_v_inverted: u32,
    /// Reserved padding to keep the struct 16-byte aligned.
    _pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Mip0CubeParams;
@group(0) @binding(1) var src_cube: texture_cube<f32>;
@group(0) @binding(2) var src_sampler: sampler;
@group(0) @binding(3) var dst_mip: texture_storage_2d_array<rgba16float, write>;

/// Cubemap direction fixup matching the skybox specular storage-orientation compensation.
fn storage_dir(dir: vec3<f32>) -> vec3<f32> {
    if (params.storage_v_inverted == 0u) {
        return dir;
    }
    let a = abs(dir);
    if (a.y >= a.x && a.y >= a.z) {
        return vec3<f32>(dir.x, dir.y, -dir.z);
    }
    return vec3<f32>(dir.x, -dir.y, dir.z);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_size = max(params.dst_size, 1u);
    if (gid.x >= dst_size || gid.y >= dst_size || gid.z >= 6u) {
        return;
    }
    let dir = storage_dir(ggx::cube_dir(gid.z, gid.x, gid.y, dst_size));
    let rgb = textureSampleLevel(src_cube, src_sampler, dir, 0.0).rgb;
    textureStore(
        dst_mip,
        vec2i(i32(gid.x), i32(gid.y)),
        i32(gid.z),
        vec4<f32>(rgb, 1.0),
    );
}
