use glam::Vec3;

use crate::shared::RenderSH2;

#[cfg(test)]
use super::Sh2ProjectParams;
#[cfg(test)]
use glam::Vec4;

/// Bit pattern for a packed float4.
pub(super) fn f32x4_bits(v: [f32; 4]) -> [u32; 4] {
    [
        v[0].to_bits(),
        v[1].to_bits(),
        v[2].to_bits(),
        v[3].to_bits(),
    ]
}

/// Analytic SH2 coefficients for a constant Lambertian diffuse color.
pub(super) fn constant_color_sh2(color: Vec3) -> RenderSH2 {
    let c = color * (4.0 * std::f32::consts::PI * SH_C0);
    RenderSH2 {
        sh0: c,
        ..RenderSH2::default()
    }
}

/// Zeroth-order SH basis constant.
pub const SH_C0: f32 = 0.282_094_8;

/// First-order SH basis constant.
#[cfg(test)]
pub const SH_C1: f32 = 0.488_602_52;

/// Second-order `xy`, `yz`, and `xz` SH basis constant.
#[cfg(test)]
pub const SH_C2: f32 = 1.092_548_5;

/// Second-order `3z^2-1` SH basis constant.
#[cfg(test)]
pub const SH_C3: f32 = 0.315_391_57;

/// Second-order `x^2-y^2` SH basis constant.
#[cfg(test)]
pub const SH_C4: f32 = 0.546_274_24;

/// Lambertian convolution factor for the zeroth SH band after diffuse BRDF division by pi.
#[cfg(test)]
pub const LAMBERT_BAND0: f32 = 1.0;

/// Lambertian convolution factor for the first SH band after diffuse BRDF division by pi.
#[cfg(test)]
pub const LAMBERT_BAND1: f32 = 2.0 / 3.0;

/// Lambertian convolution factor for the second SH band after diffuse BRDF division by pi.
#[cfg(test)]
pub const LAMBERT_BAND2: f32 = 0.25;

/// Evaluates stored RenderSH2 coefficients for a world-space normal.
#[cfg(test)]
pub(super) fn evaluate_sh2(sh: &RenderSH2, n: Vec3) -> Vec3 {
    sh.sh0 * SH_C0
        + sh.sh1 * (SH_C1 * n.y)
        + sh.sh2 * (SH_C1 * n.z)
        + sh.sh3 * (SH_C1 * n.x)
        + sh.sh4 * (SH_C2 * n.x * n.y)
        + sh.sh5 * (SH_C2 * n.y * n.z)
        + sh.sh6 * (SH_C3 * (3.0 * n.z * n.z - 1.0))
        + sh.sh7 * (SH_C2 * n.x * n.z)
        + sh.sh8 * (SH_C4 * (n.x * n.x - n.y * n.y))
}

/// Applies WGSL-style positive modulo for Projection360 angle wrapping.
#[cfg(test)]
fn positive_fmod_scalar(v: f32, wrap: f32) -> f32 {
    let mut r = v - (v / wrap).trunc() * wrap;
    r += wrap;
    r - (r / wrap).trunc() * wrap
}

/// Converts a raw texture-space direction to the pre-ST equirectangular UV convention.
#[cfg(test)]
pub(super) fn raw_equirect_uv_for_dir(dir: Vec3) -> [f32; 2] {
    [
        dir.x.atan2(dir.z) / std::f32::consts::TAU + 0.5,
        dir.y.clamp(-1.0, 1.0).acos() / std::f32::consts::PI,
    ]
}

/// Converts a Projection360 view direction to pre-ST UVs using the visible shader formula.
#[cfg(test)]
fn projection360_dir_to_uv_for_test(view_dir: Vec3, params: &Sh2ProjectParams) -> [f32; 2] {
    let angle_x = view_dir.x.atan2(view_dir.z) + params.color0[0] * 0.5 + params.color0[2];
    let angle_y = view_dir.y.clamp(-1.0, 1.0).acos() - std::f32::consts::FRAC_PI_2
        + params.color0[1] * 0.5
        + params.color0[3];
    [
        positive_fmod_scalar(angle_x, std::f32::consts::TAU)
            / params.color0[0].abs().max(0.000_001),
        positive_fmod_scalar(angle_y, std::f32::consts::PI) / params.color0[1].abs().max(0.000_001),
    ]
}

/// Applies the visible shader's `_MainTex_ST` and storage-orientation handling.
#[cfg(test)]
fn projection360_main_tex_uv_for_test(uv: [f32; 2], params: &Sh2ProjectParams) -> [f32; 2] {
    let u = uv[0].clamp(0.0, 1.0) * params.color1[0] + params.color1[2];
    let v = uv[1].clamp(0.0, 1.0) * params.color1[1] + params.color1[3];
    if params.scalars[0] > 0.5 {
        [u, v]
    } else {
        [u, 1.0 - v]
    }
}

/// Returns the texture UV that visible Projection360 equirectangular skybox sampling uses.
#[cfg(test)]
pub(super) fn projection360_equirect_uv_for_world_dir(
    world_dir: Vec3,
    params: &Sh2ProjectParams,
) -> [f32; 2] {
    projection360_main_tex_uv_for_test(
        projection360_dir_to_uv_for_test(-world_dir.normalize(), params),
        params,
    )
}

/// Returns the cubemap direction used by the visible Projection360 cubemap path.
#[cfg(test)]
pub(super) fn projection360_cubemap_sample_dir_for_world_dir(world_dir: Vec3) -> Vec3 {
    let view_dir = -world_dir.normalize();
    (-view_dir).normalize()
}

/// Evaluates the GradientSkybox color using the visible shader formula.
#[cfg(test)]
pub(super) fn gradient_sky_visible_color_for_dir(dir: Vec3, params: &Sh2ProjectParams) -> Vec3 {
    let mut color = Vec3::from_array([params.color0[0], params.color0[1], params.color0[2]]);
    let count = params.gradient_count.min(16) as usize;
    for i in 0..count {
        let dirs_spread = params.dirs_spread[i];
        let gradient_params = params.gradient_params[i];
        let axis = Vec3::new(dirs_spread[0], dirs_spread[1], dirs_spread[2]);
        let mut r = 0.5 - dir.dot(axis) * 0.5;
        r /= dirs_spread[3];
        if r <= 1.0 {
            r = r.powf(gradient_params[1]);
            r = ((r - gradient_params[2]) / (gradient_params[3] - gradient_params[2]))
                .clamp(0.0, 1.0);
            let c0 = Vec4::from_array(params.gradient_color0[i]);
            let c1 = Vec4::from_array(params.gradient_color1[i]);
            let c = c0.lerp(c1, r);
            if gradient_params[0].abs() <= f32::EPSILON {
                color = color * (1.0 - c.w) + c.truncate() * c.w;
            } else {
                color += c.truncate() * c.w;
            }
        }
    }
    color
}

/// Evaluates the ProceduralSkybox color using the visible shader formula.
#[cfg(test)]
pub(super) fn procedural_sky_visible_color_for_dir(dir: Vec3, params: &Sh2ProjectParams) -> Vec3 {
    let horizon = (1.0 - dir.y.abs().clamp(0.0, 1.0)).powi(2);
    let sky_amount = smoothstep_for_test(-0.02, 0.08, dir.y);
    let atmosphere = params.scalars[2].max(0.0);
    let scatter = Vec3::new(0.20, 0.36, 0.75) * (0.25 + atmosphere * 0.25) * dir.y.max(0.0);
    let sky_tint = Vec3::from_array([params.color0[0], params.color0[1], params.color0[2]]);
    let ground_color = Vec3::from_array([params.color1[0], params.color1[1], params.color1[2]]);
    let sky = sky_tint * (0.35 + 0.65 * dir.y.max(0.0)) + scatter;
    let ground = ground_color * (0.55 + 0.45 * horizon);
    let mut color = ground.lerp(sky, sky_amount) + sky_tint * horizon * 0.18;

    if params.scalars[3] > 0.5 {
        let sun_dir = Vec3::new(
            params.direction[0],
            params.direction[1] + 0.000_01,
            params.direction[2],
        )
        .normalize();
        let sun_dot = dir.dot(sun_dir).max(0.0);
        let size = params.scalars[1].clamp(0.0001, 1.0);
        let exponent = 4096.0 + (48.0 - 4096.0) * size;
        let mut sun = sun_dot.powf(exponent);
        if params.scalars[3] > 1.5 {
            sun += sun_dot.powf((exponent * 0.18).max(4.0)) * 0.18;
        }
        color += Vec3::from_array([
            params.gradient_color0[0][0],
            params.gradient_color0[0][1],
            params.gradient_color0[0][2],
        ]) * sun;
    }

    (color * params.scalars[0].max(0.0)).max(Vec3::ZERO)
}

/// Applies the WGSL `smoothstep` helper for CPU parity tests.
#[cfg(test)]
fn smoothstep_for_test(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Computes the cubemap texel solid-angle helper used by the GPU SH kernels.
#[cfg(test)]
fn sh2_area_element(x: f32, y: f32) -> f32 {
    (x * y).atan2((x * x + y * y + 1.0).sqrt())
}

/// Computes a cube-face texel solid angle for CPU SH regression tests.
#[cfg(test)]
fn sh2_texel_solid_angle(x: u32, y: u32, n: u32) -> f32 {
    let inv = 1.0 / n as f32;
    let x0 = (x as f32 * inv) * 2.0 - 1.0;
    let y0 = (y as f32 * inv) * 2.0 - 1.0;
    let x1 = ((x + 1) as f32 * inv) * 2.0 - 1.0;
    let y1 = ((y + 1) as f32 * inv) * 2.0 - 1.0;
    (sh2_area_element(x0, y0) - sh2_area_element(x0, y1) - sh2_area_element(x1, y0)
        + sh2_area_element(x1, y1))
    .abs()
}

/// Returns the Unity cube-face direction for one sample location.
#[cfg(test)]
fn sh2_cube_dir(face: u32, x: u32, y: u32, n: u32) -> Vec3 {
    let u = (x as f32 + 0.5) / n as f32;
    let v = (y as f32 + 0.5) / n as f32;
    match face {
        0 => Vec3::new(1.0, v * -2.0 + 1.0, u * -2.0 + 1.0).normalize(),
        1 => Vec3::new(-1.0, v * -2.0 + 1.0, u * 2.0 - 1.0).normalize(),
        2 => Vec3::new(u * 2.0 - 1.0, 1.0, v * 2.0 - 1.0).normalize(),
        3 => Vec3::new(u * 2.0 - 1.0, -1.0, v * -2.0 + 1.0).normalize(),
        4 => Vec3::new(u * 2.0 - 1.0, v * -2.0 + 1.0, 1.0).normalize(),
        _ => Vec3::new(u * -2.0 + 1.0, v * -2.0 + 1.0, -1.0).normalize(),
    }
}

/// Accumulates one weighted radiance sample into Lambertian-convolved RenderSH2 coefficients.
#[cfg(test)]
fn add_weighted_sh2_sample(sh: &mut RenderSH2, c: Vec3, dir: Vec3, weight: f32) {
    sh.sh0 += c * (SH_C0 * LAMBERT_BAND0 * weight);
    sh.sh1 += c * (SH_C1 * LAMBERT_BAND1 * dir.y * weight);
    sh.sh2 += c * (SH_C1 * LAMBERT_BAND1 * dir.z * weight);
    sh.sh3 += c * (SH_C1 * LAMBERT_BAND1 * dir.x * weight);
    sh.sh4 += c * (SH_C2 * LAMBERT_BAND2 * dir.x * dir.y * weight);
    sh.sh5 += c * (SH_C2 * LAMBERT_BAND2 * dir.y * dir.z * weight);
    sh.sh6 += c * (SH_C3 * LAMBERT_BAND2 * (3.0 * dir.z * dir.z - 1.0) * weight);
    sh.sh7 += c * (SH_C2 * LAMBERT_BAND2 * dir.x * dir.z * weight);
    sh.sh8 += c * (SH_C4 * LAMBERT_BAND2 * (dir.x * dir.x - dir.y * dir.y) * weight);
}

/// Projects a directional equirectangular lobe through the Projection360 `_VIEW` convention.
#[cfg(test)]
pub(super) fn project_projection360_equirect_lobe(
    sample_size: u32,
    bright_texture_dir: Vec3,
) -> RenderSH2 {
    let n = sample_size.max(1);
    let bright_texture_dir = bright_texture_dir.normalize();
    let mut sh = RenderSH2::default();
    for face in 0..6 {
        for y in 0..n {
            for x in 0..n {
                let world_dir = sh2_cube_dir(face, x, y, n);
                let texture_dir = -world_dir;
                let intensity = texture_dir.dot(bright_texture_dir).max(0.0).powf(16.0);
                if intensity > 0.0 {
                    add_weighted_sh2_sample(
                        &mut sh,
                        Vec3::splat(intensity),
                        world_dir,
                        sh2_texel_solid_angle(x, y, n),
                    );
                }
            }
        }
    }
    sh
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_lambertian_sh2_evaluates_to_source_color() {
        let color = Vec3::new(0.25, 0.5, 1.0);
        let sh = constant_color_sh2(color);

        for n in [Vec3::X, Vec3::Y, Vec3::Z, -Vec3::X, -Vec3::Y, -Vec3::Z] {
            let evaluated = evaluate_sh2(&sh, n);
            assert!((evaluated - color).length() < 1e-5);
        }
    }

    #[test]
    fn lambertian_projection_applies_band_factors() {
        let mut sh = RenderSH2::default();
        add_weighted_sh2_sample(&mut sh, Vec3::ONE, Vec3::X, 1.0);

        assert!((sh.sh0.x - SH_C0 * LAMBERT_BAND0).abs() < 1e-6);
        assert!((sh.sh3.x - SH_C1 * LAMBERT_BAND1).abs() < 1e-6);
        assert!((sh.sh8.x - SH_C4 * LAMBERT_BAND2).abs() < 1e-6);
    }
}
