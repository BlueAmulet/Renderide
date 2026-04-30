#import renderide::skybox_evaluator as sky

struct PrefilterParams {
    dst_size: u32,
    mip_index: u32,
    mip_count: u32,
    sample_count: u32,
}

@group(0) @binding(0) var<uniform> sky_params: sky::SkyboxEvaluatorParams;
@group(0) @binding(1) var<uniform> prefilter: PrefilterParams;
@group(0) @binding(2) var output_cube: texture_storage_2d_array<rgba16float, write>;

const PI: f32 = 3.14159265358979323846;

fn radical_inverse_vdc(bits_in: u32) -> f32 {
    var bits = bits_in;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return f32(bits) * 2.3283064365386963e-10;
}

fn hammersley(i: u32, n: u32) -> vec2<f32> {
    return vec2<f32>(f32(i) / f32(max(n, 1u)), radical_inverse_vdc(i));
}

fn tangent_to_world(local_dir: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let up = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(n.z) < 0.999);
    let tangent = normalize(cross(up, n));
    let bitangent = cross(n, tangent);
    return normalize(tangent * local_dir.x + bitangent * local_dir.y + n * local_dir.z);
}

fn importance_sample_ggx(xi: vec2<f32>, roughness: f32, n: vec3<f32>) -> vec3<f32> {
    let alpha = max(roughness * roughness, 0.0001);
    let alpha_sq = alpha * alpha;
    let phi = 2.0 * PI * xi.x;
    let cos_theta = sqrt((1.0 - xi.y) / max(1.0 + (alpha_sq - 1.0) * xi.y, 0.000001));
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    let h = vec3<f32>(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
    return tangent_to_world(h, n);
}

fn prefilter_sky(ray: vec3<f32>) -> vec3<f32> {
    let max_mip = f32(max(prefilter.mip_count, 1u) - 1u);
    let roughness = clamp(f32(prefilter.mip_index) / max(max_mip, 1.0), 0.0, 1.0);
    let sample_count = max(prefilter.sample_count, 1u);
    var color = vec3<f32>(0.0);
    var weight_sum = 0.0;
    for (var i = 0u; i < sample_count; i = i + 1u) {
        let h = importance_sample_ggx(hammersley(i, sample_count), roughness, ray);
        let l = normalize(2.0 * dot(ray, h) * h - ray);
        let n_dot_l = max(dot(ray, l), 0.0);
        if (n_dot_l > 0.0) {
            color = color + sky::sample_sky(sky_params, l) * n_dot_l;
            weight_sum = weight_sum + n_dot_l;
        }
    }
    return max(color / max(weight_sum, 0.000001), vec3<f32>(0.0));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_size = max(prefilter.dst_size, 1u);
    if (gid.x >= dst_size || gid.y >= dst_size || gid.z >= 6u) {
        return;
    }
    let ray = sky::cube_dir(gid.z, gid.x, gid.y, dst_size);
    textureStore(
        output_cube,
        vec2i(i32(gid.x), i32(gid.y)),
        i32(gid.z),
        vec4<f32>(prefilter_sky(ray), 1.0)
    );
}
