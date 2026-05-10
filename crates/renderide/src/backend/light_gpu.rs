//! GPU packing for scene lights (`storage` buffer layout / WGSL `struct` alignment).
//!
//! [`GpuLight`] uses 16-byte alignment for `vec3` slots to match typical WGSL storage rules.
//! [`LightType`](crate::shared::LightType) and [`ShadowType`](crate::shared::ShadowType) are stored as `u32`
//! with the same numeric values as `repr(u8)` on the wire.

pub use crate::gpu::{GpuLight, MAX_LIGHTS};
use crate::scene::ResolvedLight;
use crate::shared::{LightType, ShadowType};

/// Packs a [`ResolvedLight`] for GPU consumption.
pub fn gpu_light_from_resolved(light: &ResolvedLight) -> GpuLight {
    let spot_cos_half_angle = if light.spot_angle > 0.0 && light.spot_angle < 180.0 {
        (light.spot_angle.to_radians() / 2.0).cos()
    } else {
        1.0
    };
    GpuLight {
        position: [
            light.world_position.x,
            light.world_position.y,
            light.world_position.z,
        ],
        _pad0: 0.0,
        direction: [
            light.world_direction.x,
            light.world_direction.y,
            light.world_direction.z,
        ],
        _pad1: 0.0,
        color: [light.color.x, light.color.y, light.color.z],
        intensity: light.intensity,
        range: light.range.max(0.001),
        spot_cos_half_angle,
        light_type: light_type_u32(light.light_type),
        _pad_before_shadow_params: 0,
        shadow_strength: light.shadow_strength,
        shadow_near_plane: light.shadow_near_plane,
        shadow_bias: light.shadow_bias,
        shadow_normal_bias: light.shadow_normal_bias,
        shadow_type: shadow_type_u32(light.shadow_type),
        _pad_align_vec3_trailing: [0; 4],
        _pad_trailing: [0; 3],
        _pad_struct_end: [0; 12],
    }
}

impl From<&ResolvedLight> for GpuLight {
    fn from(light: &ResolvedLight) -> Self {
        gpu_light_from_resolved(light)
    }
}

fn light_type_u32(ty: LightType) -> u32 {
    match ty {
        LightType::Point => 0,
        LightType::Directional => 1,
        LightType::Spot => 2,
    }
}

fn shadow_type_u32(ty: ShadowType) -> u32 {
    match ty {
        ShadowType::None => 0,
        ShadowType::Hard => 1,
        ShadowType::Soft => 2,
    }
}

/// Directional lights first (clustered forward compatibility); then point/spot; stable within bucket.
///
/// Sorts before applying the global [`MAX_LIGHTS`] cap so directional lights are not accidentally
/// dropped just because they arrived after many local lights in host order.
pub fn order_lights_for_clustered_shading_in_place(lights: &mut Vec<ResolvedLight>) {
    profiling::scope!("render::order_lights_for_clustered_shading");
    lights.sort_by_key(|l| match l.light_type {
        LightType::Directional => 0u8,
        LightType::Point | LightType::Spot => 1,
    });
    if lights.len() > MAX_LIGHTS {
        lights.truncate(MAX_LIGHTS);
    }
}

#[cfg(test)]
mod layout_tests {
    use std::mem::size_of;

    use glam::Vec3;

    use crate::scene::ResolvedLight;
    use crate::shared::{LightType, ShadowType};

    use super::{GpuLight, MAX_LIGHTS, order_lights_for_clustered_shading_in_place};

    #[test]
    fn gpu_light_stride_matches_wgsl() {
        assert_eq!(
            size_of::<GpuLight>(),
            112,
            "must match WGSL storage layout for `array<GpuLight>` (naga stride)"
        );
    }

    fn resolved_light(light_type: LightType) -> ResolvedLight {
        ResolvedLight {
            world_position: Vec3::ZERO,
            world_direction: Vec3::Z,
            color: Vec3::ONE,
            intensity: 1.0,
            range: 10.0,
            spot_angle: 45.0,
            light_type,
            shadow_type: ShadowType::None,
            shadow_strength: 0.0,
            shadow_near_plane: 0.0,
            shadow_bias: 0.0,
            shadow_normal_bias: 0.0,
        }
    }

    #[test]
    fn ordering_prioritizes_directionals_before_global_truncate() {
        let mut lights = vec![resolved_light(LightType::Point); MAX_LIGHTS];
        lights.push(resolved_light(LightType::Directional));

        order_lights_for_clustered_shading_in_place(&mut lights);

        assert_eq!(lights.len(), MAX_LIGHTS);
        assert_eq!(lights[0].light_type, LightType::Directional);
    }
}
