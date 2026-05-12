//! Source-key resolution for reflection-probe SH2 projection tasks.

use glam::Vec4;

use super::task_rows::TaskHeader;
use super::{
    CubemapResidency, CubemapSourceMaterialIdentity, DEFAULT_SAMPLE_SIZE, GpuSh2Source,
    Sh2SourceKey, constant_color_sh2,
};
use crate::reflection_probes::specular::{
    RuntimeReflectionProbeCaptureKey, RuntimeReflectionProbeCaptureStore,
};
use crate::scene::{RenderSpaceId, SceneCoordinator};
use crate::shared::{ReflectionProbeClear, ReflectionProbeType, RenderSH2};

/// Either a synchronous CPU result or a GPU source to project.
pub(super) enum Sh2ResolvedSource {
    /// CPU-computed SH2.
    Cpu(Box<RenderSH2>),
    /// GPU-computed SH2 source.
    Gpu(GpuSh2Source),
    /// Source is expected to become available later.
    Postpone,
}

/// Resolves a host task into a cache key and source payload.
pub(super) fn resolve_task_source(
    scene: &SceneCoordinator,
    assets: &crate::backend::AssetTransferQueue,
    captures: &RuntimeReflectionProbeCaptureStore,
    render_space_id: i32,
    task: TaskHeader,
) -> Option<(Sh2SourceKey, Sh2ResolvedSource)> {
    if task.renderable_index < 0 || task.reflection_probe_renderable_index < 0 {
        return None;
    }
    let space = scene.space(RenderSpaceId(render_space_id))?;
    let probe = space
        .reflection_probes()
        .get(task.reflection_probe_renderable_index as usize)?;
    let state = probe.state;
    if state.clear_flags == ReflectionProbeClear::Color {
        let color = state.background_color * state.intensity.max(0.0);
        let key = Sh2SourceKey::ConstantColor {
            render_space_id,
            color_bits: vec4_bits(color),
        };
        return Some((
            key,
            Sh2ResolvedSource::Cpu(Box::new(constant_color_sh2(color.truncate()))),
        ));
    }

    if state.r#type == ReflectionProbeType::Baked {
        if state.cubemap_asset_id < 0 {
            return None;
        }
        let asset_id = state.cubemap_asset_id;
        let identity = CubemapSourceMaterialIdentity::DIRECT_PROBE;
        let Some(cubemap) = assets.cubemap_pool().get(asset_id) else {
            return Some((
                Sh2SourceKey::cubemap(
                    render_space_id,
                    identity,
                    asset_id,
                    CubemapResidency::default(),
                ),
                Sh2ResolvedSource::Postpone,
            ));
        };
        let key = Sh2SourceKey::cubemap(
            render_space_id,
            identity,
            asset_id,
            cubemap_residency_from_pool(cubemap),
        );
        if cubemap.mip_levels_resident == 0 {
            return Some((key, Sh2ResolvedSource::Postpone));
        }
        return Some((
            key,
            Sh2ResolvedSource::Gpu(GpuSh2Source::Cubemap {
                asset_id,
                storage_v_inverted: cubemap.storage_v_inverted,
            }),
        ));
    }

    if state.r#type == ReflectionProbeType::OnChanges {
        return resolve_runtime_capture_source(render_space_id, probe.renderable_index, captures);
    }
    None
}

fn resolve_runtime_capture_source(
    render_space_id: i32,
    renderable_index: i32,
    captures: &RuntimeReflectionProbeCaptureStore,
) -> Option<(Sh2SourceKey, Sh2ResolvedSource)> {
    let key = RuntimeReflectionProbeCaptureKey {
        space_id: RenderSpaceId(render_space_id),
        renderable_index,
    };
    let Some(capture) = captures.get(key) else {
        return Some((
            Sh2SourceKey::RuntimeCubemap {
                render_space_id,
                renderable_index,
                generation: 0,
                size: 0,
                sample_size: DEFAULT_SAMPLE_SIZE,
            },
            Sh2ResolvedSource::Postpone,
        ));
    };
    let key = Sh2SourceKey::RuntimeCubemap {
        render_space_id,
        renderable_index,
        generation: capture.generation,
        size: capture.face_size,
        sample_size: DEFAULT_SAMPLE_SIZE,
    };
    Some((
        key,
        Sh2ResolvedSource::Gpu(GpuSh2Source::RuntimeCubemap {
            texture: capture.texture.clone(),
            view: capture.view.clone(),
        }),
    ))
}

fn cubemap_residency_from_pool(
    cubemap: &crate::gpu_pools::pools::cubemap::GpuCubemap,
) -> CubemapResidency {
    CubemapResidency {
        allocation_generation: cubemap.allocation_generation,
        size: cubemap.size,
        resident_mips: cubemap.mip_levels_resident,
        content_generation: cubemap.content_generation,
        storage_v_inverted: cubemap.storage_v_inverted,
    }
}

/// Bit pattern for a `Vec4`.
fn vec4_bits(v: Vec4) -> [u32; 4] {
    [v.x.to_bits(), v.y.to_bits(), v.z.to_bits(), v.w.to_bits()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec4_bits_preserves_exact_float_bit_patterns() {
        let bits = vec4_bits(Vec4::new(0.0, -0.0, f32::INFINITY, f32::NAN));

        assert_eq!(bits[0], 0.0f32.to_bits());
        assert_eq!(bits[1], (-0.0f32).to_bits());
        assert_eq!(bits[2], f32::INFINITY.to_bits());
        assert_eq!(bits[3], f32::NAN.to_bits());
    }

    #[test]
    fn missing_runtime_capture_postpones_onchanges_probe() {
        let captures = RuntimeReflectionProbeCaptureStore::default();

        let (key, source) = resolve_runtime_capture_source(7, 3, &captures)
            .expect("missing captures should return a stable postponed key");

        assert_eq!(
            key,
            Sh2SourceKey::RuntimeCubemap {
                render_space_id: 7,
                renderable_index: 3,
                generation: 0,
                size: 0,
                sample_size: DEFAULT_SAMPLE_SIZE,
            }
        );
        assert!(matches!(source, Sh2ResolvedSource::Postpone));
    }
}
