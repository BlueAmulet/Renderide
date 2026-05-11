//! Source-key resolution for reflection-probe SH2 projection tasks.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use glam::Vec4;

use super::task_rows::TaskHeader;
use super::{
    CubemapResidency, CubemapSourceMaterialIdentity, DEFAULT_SAMPLE_SIZE, GpuSh2Source,
    Projection360EquirectKey, Sh2ProjectParams, Sh2SourceKey, constant_color_sh2,
};
use crate::assets::texture::HostTextureAssetKind;
use crate::backend::material_property_reader::texture_property;
use crate::materials::host_data::{
    MaterialPropertyLookupIds, MaterialPropertyStore, PropertyIdRegistry,
};
use crate::reflection_probes::specular::{
    RuntimeReflectionProbeCaptureKey, RuntimeReflectionProbeCaptureStore,
};
use crate::scene::{RenderSpaceId, SceneCoordinator, reflection_probe_skybox_only};
use crate::shared::{ReflectionProbeClear, ReflectionProbeType, RenderSH2};
use crate::skybox::params::{
    gradient_sky_params, procedural_sky_params, projection360_equirect_params,
};

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
    materials: &crate::materials::MaterialSystem,
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

    if !reflection_probe_skybox_only(state.flags) {
        if state.r#type == ReflectionProbeType::OnChanges {
            return resolve_runtime_capture_source(
                render_space_id,
                probe.renderable_index,
                captures,
            );
        }
        return None;
    }
    resolve_skybox_source(
        render_space_id,
        space.skybox_material_asset_id(),
        materials,
        assets,
    )
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

/// Resolves an active skybox material into a source payload.
fn resolve_skybox_source(
    render_space_id: i32,
    material_asset_id: i32,
    materials: &crate::materials::MaterialSystem,
    assets: &crate::backend::AssetTransferQueue,
) -> Option<(Sh2SourceKey, Sh2ResolvedSource)> {
    if material_asset_id < 0 {
        return None;
    }
    let store = materials.material_property_store();
    let generation = store.material_generation(material_asset_id);
    let shader_asset_id = store.shader_asset_for_material(material_asset_id)?;
    let route_name = shader_route_name(materials, shader_asset_id);
    let route_hash = hash_route_name(route_name.as_deref().unwrap_or(""));
    let material_identity = Sh2SkyboxMaterialIdentity {
        material_asset_id,
        generation,
        route_hash,
    };
    let lookup = MaterialPropertyLookupIds {
        material_asset_id,
        mesh_property_block_slot0: None,
    };
    let registry = materials.property_id_registry();

    if route_name
        .as_deref()
        .is_some_and(|name| name.to_ascii_lowercase().contains("projection360"))
    {
        return resolve_projection360_source(
            render_space_id,
            store,
            registry,
            assets,
            lookup,
            material_identity,
        );
    }
    if route_name
        .as_deref()
        .is_some_and(|name| name.to_ascii_lowercase().contains("gradient"))
    {
        let params = gradient_sky_params(store, registry, lookup);
        let key = Sh2SourceKey::SkyParams {
            render_space_id,
            material_asset_id,
            material_generation: generation,
            sample_size: DEFAULT_SAMPLE_SIZE,
            route_hash,
        };
        return Some((
            key,
            Sh2ResolvedSource::Gpu(GpuSh2Source::SkyParams {
                params: Box::new(params),
            }),
        ));
    }
    if route_name
        .as_deref()
        .is_some_and(|name| name.to_ascii_lowercase().contains("procedural"))
    {
        let params = procedural_sky_params(store, registry, lookup);
        let key = Sh2SourceKey::SkyParams {
            render_space_id,
            material_asset_id,
            material_generation: generation,
            sample_size: DEFAULT_SAMPLE_SIZE,
            route_hash,
        };
        return Some((
            key,
            Sh2ResolvedSource::Gpu(GpuSh2Source::SkyParams {
                params: Box::new(params),
            }),
        ));
    }
    None
}

/// Returns a shader route name or stem for a shader asset id.
fn shader_route_name(
    materials: &crate::materials::MaterialSystem,
    shader_asset_id: i32,
) -> Option<String> {
    let registry = materials.material_registry()?;
    registry
        .stem_for_shader_asset(shader_asset_id)
        .map(str::to_string)
}

/// Resolves a `Projection360` material to a texture-backed source.
fn resolve_projection360_source(
    render_space_id: i32,
    store: &MaterialPropertyStore,
    registry: &PropertyIdRegistry,
    assets: &crate::backend::AssetTransferQueue,
    lookup: MaterialPropertyLookupIds,
    material_identity: Sh2SkyboxMaterialIdentity,
) -> Option<(Sh2SourceKey, Sh2ResolvedSource)> {
    let main_cube = texture_property(store, registry, lookup, "_MainCube")
        .or_else(|| texture_property(store, registry, lookup, "_Cube"));
    if let Some((asset_id, HostTextureAssetKind::Cubemap)) = main_cube {
        return Some(resolve_projection360_cubemap_source(
            render_space_id,
            assets,
            asset_id,
            material_identity,
        ));
    }

    let main_tex = texture_property(store, registry, lookup, "_MainTex")
        .or_else(|| texture_property(store, registry, lookup, "_Tex"));
    match main_tex {
        Some((asset_id, HostTextureAssetKind::Texture2D)) => {
            Some(resolve_projection360_texture2d_source(
                render_space_id,
                store,
                registry,
                assets,
                lookup,
                asset_id,
                material_identity,
            ))
        }
        Some((asset_id, HostTextureAssetKind::Cubemap)) => {
            Some(resolve_projection360_cubemap_source(
                render_space_id,
                assets,
                asset_id,
                material_identity,
            ))
        }
        _ => None,
    }
}

/// Resolves a `Projection360` cubemap binding into an SH2 source.
fn resolve_projection360_cubemap_source(
    render_space_id: i32,
    assets: &crate::backend::AssetTransferQueue,
    asset_id: i32,
    material_identity: Sh2SkyboxMaterialIdentity,
) -> (Sh2SourceKey, Sh2ResolvedSource) {
    let identity = material_identity.into_cubemap_source();
    let Some(cubemap) = assets.cubemap_pool().get(asset_id) else {
        return (
            Sh2SourceKey::cubemap(
                render_space_id,
                identity,
                asset_id,
                CubemapResidency::default(),
            ),
            Sh2ResolvedSource::Postpone,
        );
    };
    let key = Sh2SourceKey::cubemap(
        render_space_id,
        identity,
        asset_id,
        cubemap_residency_from_pool(cubemap),
    );
    if cubemap.mip_levels_resident == 0 {
        return (key, Sh2ResolvedSource::Postpone);
    }
    (
        key,
        Sh2ResolvedSource::Gpu(GpuSh2Source::Cubemap {
            asset_id,
            storage_v_inverted: cubemap.storage_v_inverted,
        }),
    )
}

/// Resolves a `Projection360` equirectangular 2D binding into an SH2 source.
fn resolve_projection360_texture2d_source(
    render_space_id: i32,
    store: &MaterialPropertyStore,
    registry: &PropertyIdRegistry,
    assets: &crate::backend::AssetTransferQueue,
    lookup: MaterialPropertyLookupIds,
    asset_id: i32,
    material_identity: Sh2SkyboxMaterialIdentity,
) -> (Sh2SourceKey, Sh2ResolvedSource) {
    let mut params = projection360_equirect_params(store, registry, lookup, false);
    let Some(tex) = assets.texture_pool().get(asset_id) else {
        return (
            projection360_equirect_source_key(
                render_space_id,
                asset_id,
                Projection360TextureSourceState::default(),
                material_identity,
                &params,
            ),
            Sh2ResolvedSource::Postpone,
        );
    };
    params = projection360_equirect_params(store, registry, lookup, tex.storage_v_inverted);
    let key = projection360_equirect_source_key(
        render_space_id,
        asset_id,
        Projection360TextureSourceState {
            width: tex.width,
            height: tex.height,
            allocation_generation: tex.view_generation,
            resident_mips: tex.mip_levels_resident,
            content_generation: tex.content_generation,
        },
        material_identity,
        &params,
    );
    if tex.mip_levels_resident == 0 {
        return (key, Sh2ResolvedSource::Postpone);
    }
    (
        key,
        Sh2ResolvedSource::Gpu(GpuSh2Source::EquirectTexture2D {
            asset_id,
            params: Box::new(params),
        }),
    )
}

#[derive(Clone, Copy, Debug, Default)]
struct Projection360TextureSourceState {
    width: u32,
    height: u32,
    allocation_generation: u64,
    resident_mips: u32,
    content_generation: u64,
}

#[derive(Clone, Copy, Debug)]
struct Sh2SkyboxMaterialIdentity {
    material_asset_id: i32,
    generation: u64,
    route_hash: u64,
}

impl Sh2SkyboxMaterialIdentity {
    fn into_cubemap_source(self) -> CubemapSourceMaterialIdentity {
        CubemapSourceMaterialIdentity {
            material_asset_id: self.material_asset_id,
            material_generation: self.generation,
            route_hash: self.route_hash,
        }
    }
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

/// Builds an equirectangular source key from texture residency and Projection360 parameters.
fn projection360_equirect_source_key(
    render_space_id: i32,
    asset_id: i32,
    texture: Projection360TextureSourceState,
    material_identity: Sh2SkyboxMaterialIdentity,
    params: &Sh2ProjectParams,
) -> Sh2SourceKey {
    Sh2SourceKey::EquirectTexture2D {
        render_space_id,
        material_asset_id: material_identity.material_asset_id,
        material_generation: material_identity.generation,
        route_hash: material_identity.route_hash,
        asset_id,
        allocation_generation: texture.allocation_generation,
        width: texture.width,
        height: texture.height,
        resident_mips: texture.resident_mips,
        content_generation: texture.content_generation,
        sample_size: DEFAULT_SAMPLE_SIZE,
        projection: Projection360EquirectKey::from_params(params),
    }
}

/// Bit pattern for a `Vec4`.
fn vec4_bits(v: Vec4) -> [u32; 4] {
    [v.x.to_bits(), v.y.to_bits(), v.z.to_bits(), v.w.to_bits()]
}

/// Hashes a route name into a stable source-key discriminator.
fn hash_route_name(route: &str) -> u64 {
    let mut h = DefaultHasher::new();
    route.hash(&mut h);
    h.finish()
}

#[cfg(test)]
mod tests {
    use crate::skybox::params::SkyboxParamMode;

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
    fn route_hash_is_stable_and_route_sensitive() {
        assert_eq!(
            hash_route_name("Projection360"),
            hash_route_name("Projection360")
        );
        assert_ne!(
            hash_route_name("Projection360"),
            hash_route_name("GradientSky")
        );
    }

    #[test]
    fn projection360_equirect_source_key_includes_texture_and_projection_state() {
        let mut params = Sh2ProjectParams::empty(SkyboxParamMode::Procedural);
        params.color0 = [1.0, 2.0, 3.0, 4.0];
        params.color1 = [0.5, 0.5, 0.25, 0.75];
        params.scalars = [1.0, 0.0, 0.0, 0.0];

        let key = projection360_equirect_source_key(
            7,
            11,
            Projection360TextureSourceState {
                width: 512,
                height: 256,
                allocation_generation: 77,
                resident_mips: 3,
                content_generation: 42,
            },
            Sh2SkyboxMaterialIdentity {
                material_asset_id: 55,
                generation: 99,
                route_hash: 1234,
            },
            &params,
        );

        assert_eq!(
            key,
            Sh2SourceKey::EquirectTexture2D {
                render_space_id: 7,
                material_asset_id: 55,
                material_generation: 99,
                route_hash: 1234,
                asset_id: 11,
                allocation_generation: 77,
                width: 512,
                height: 256,
                resident_mips: 3,
                content_generation: 42,
                sample_size: DEFAULT_SAMPLE_SIZE,
                projection: Projection360EquirectKey::from_params(&params),
            }
        );
    }
}
