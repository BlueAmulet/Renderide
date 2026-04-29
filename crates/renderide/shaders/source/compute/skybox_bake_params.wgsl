struct Params {
    sample_size: u32,
    mode: u32,
    gradient_count: u32,
    _pad0: u32,
    color0: vec4<f32>,
    color1: vec4<f32>,
    direction: vec4<f32>,
    scalars: vec4<f32>,
    dirs_spread: array<vec4<f32>, 16>,
    gradient_color0: array<vec4<f32>, 16>,
    gradient_color1: array<vec4<f32>, 16>,
    gradient_params: array<vec4<f32>, 16>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var output_cube: texture_storage_2d_array<rgba16float, write>;

fn cube_dir(face: u32, x: u32, y: u32, n: u32) -> vec3<f32> {
    let u = (f32(x) + 0.5) / f32(n);
    let v = (f32(y) + 0.5) / f32(n);
    if (face == 0u) { return normalize(vec3<f32>(1.0, v * -2.0 + 1.0, u * -2.0 + 1.0)); }
    if (face == 1u) { return normalize(vec3<f32>(-1.0, v * -2.0 + 1.0, u * 2.0 - 1.0)); }
    if (face == 2u) { return normalize(vec3<f32>(u * 2.0 - 1.0, 1.0, v * 2.0 - 1.0)); }
    if (face == 3u) { return normalize(vec3<f32>(u * 2.0 - 1.0, -1.0, v * -2.0 + 1.0)); }
    if (face == 4u) { return normalize(vec3<f32>(u * 2.0 - 1.0, v * -2.0 + 1.0, 1.0)); }
    return normalize(vec3<f32>(u * -2.0 + 1.0, v * -2.0 + 1.0, -1.0));
}

fn sample_procedural(ray: vec3<f32>) -> vec3<f32> {
    let y = ray.y;
    let horizon = pow(1.0 - clamp(abs(y), 0.0, 1.0), 2.0);
    let sky_amount = smoothstep(-0.02, 0.08, y);
    let atmosphere = max(params.scalars.z, 0.0);
    let scatter = vec3<f32>(0.20, 0.36, 0.75) * (0.25 + atmosphere * 0.25) * max(y, 0.0);
    let sky = params.color0.rgb * (0.35 + 0.65 * max(y, 0.0)) + scatter;
    let ground = params.color1.rgb * (0.55 + 0.45 * horizon);
    var col = mix(ground, sky, sky_amount);
    col = col + params.color0.rgb * horizon * 0.18;

    if (params.scalars.w > 0.5) {
        let sun_dir = normalize(params.direction.xyz + vec3<f32>(0.0, 0.00001, 0.0));
        let sun_dot = max(dot(ray, sun_dir), 0.0);
        let size = clamp(params.scalars.y, 0.0001, 1.0);
        let exponent = mix(4096.0, 48.0, size);
        var sun = pow(sun_dot, exponent);
        if (params.scalars.w > 1.5) {
            sun = sun + pow(sun_dot, max(exponent * 0.18, 4.0)) * 0.18;
        }
        col = col + params.gradient_color0[0].rgb * sun;
    }

    return max(col * max(params.scalars.x, 0.0), vec3<f32>(0.0));
}

fn sample_gradient(ray: vec3<f32>) -> vec3<f32> {
    var color = params.color0.rgb;
    let count = min(params.gradient_count, 16u);
    for (var i = 0u; i < count; i = i + 1u) {
        let dirs_spread = params.dirs_spread[i];
        let gradient_params = params.gradient_params[i];
        let spread = max(abs(dirs_spread.w), 0.000001);
        let expv = max(gradient_params.y, 0.000001);
        let fromv = gradient_params.z;
        let tov = gradient_params.w;
        let denom = max(abs(tov - fromv), 0.000001);
        var r = (0.5 - dot(ray, normalize(dirs_spread.xyz)) * 0.5) / spread;
        if (r <= 1.0) {
            r = pow(max(r, 0.0), expv);
            r = clamp((r - fromv) / denom, 0.0, 1.0);
            let c = mix(params.gradient_color0[i], params.gradient_color1[i], r);
            if (gradient_params.x == 0.0) {
                color = color * (1.0 - c.a) + c.rgb * c.a;
            } else {
                color = color + c.rgb * c.a;
            }
        }
    }
    return max(color, vec3<f32>(0.0));
}

fn sample_sky(ray: vec3<f32>) -> vec3<f32> {
    if (params.mode == 2u) {
        return sample_gradient(ray);
    }
    return sample_procedural(ray);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let face_size = max(params.sample_size, 1u);
    if (gid.x >= face_size || gid.y >= face_size || gid.z >= 6u) {
        return;
    }
    let ray = cube_dir(gid.z, gid.x, gid.y, face_size);
    textureStore(
        output_cube,
        vec2i(i32(gid.x), i32(gid.y)),
        i32(gid.z),
        vec4<f32>(sample_sky(ray), 1.0)
    );
}
