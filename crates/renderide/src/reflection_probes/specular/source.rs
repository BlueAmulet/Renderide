use glam::{Vec3, Vec3A, Vec4};

use crate::backend::AssetTransferQueue;
use crate::backend::frame_gpu::{
    GpuReflectionProbeMetadata, REFLECTION_PROBE_METADATA_BOX_PROJECTION,
    REFLECTION_PROBE_METADATA_SH2_SOURCE_LOCAL, REFLECTION_PROBE_METADATA_SH2_SOURCE_SKYBOX,
};
use crate::materials::MaterialSystem;
use crate::scene::{
    ReflectionProbeEntry, RenderSpaceId, SceneCoordinator, reflection_probe_skybox_only,
    reflection_probe_use_box_projection,
};
use crate::shared::{ReflectionProbeClear, ReflectionProbeState, ReflectionProbeType, RenderSH2};
use crate::skybox::specular::{
    CubemapIblSource, SkyboxIblSource, resolve_skybox_material_ibl_source, solid_color_ibl_source,
};
use crate::world_mesh::culling::world_aabb_from_local_bounds;

use super::selection::{SpatialProbe, aabb_valid, aabb_volume};

pub(super) fn resolve_probe_source(
    skybox_material_asset_id: i32,
    probe: &ReflectionProbeEntry,
    materials: &MaterialSystem,
    assets: &AssetTransferQueue,
) -> Option<SkyboxIblSource> {
    let state = probe.state;
    if state.intensity <= 0.0 {
        return None;
    }
    if state.clear_flags == ReflectionProbeClear::Color {
        let color = state.background_color;
        return Some(solid_color_ibl_source(
            color_probe_identity(probe.renderable_index, color),
            color.to_array(),
        ));
    }
    if state.r#type == ReflectionProbeType::Baked {
        return resolve_baked_probe_source(state, assets);
    }
    if reflection_probe_skybox_only(state.flags) && skybox_material_asset_id >= 0 {
        return resolve_skybox_material_ibl_source(skybox_material_asset_id, materials, assets);
    }
    None
}

pub(super) fn resolve_space_skybox_fallback_source(
    skybox_material_asset_id: i32,
    materials: &MaterialSystem,
    assets: &AssetTransferQueue,
) -> Option<SkyboxIblSource> {
    if skybox_material_asset_id < 0 {
        return None;
    }
    resolve_skybox_material_ibl_source(skybox_material_asset_id, materials, assets)
}

pub(super) fn resolve_baked_probe_source(
    state: ReflectionProbeState,
    assets: &AssetTransferQueue,
) -> Option<SkyboxIblSource> {
    if state.cubemap_asset_id < 0 {
        return None;
    }
    let cubemap = assets.cubemap_pool().get(state.cubemap_asset_id)?;
    if cubemap.mip_levels_resident == 0 {
        return None;
    }
    Some(SkyboxIblSource::Cubemap(CubemapIblSource {
        material_asset_id: -1,
        material_generation: 0,
        route_hash: 0,
        asset_id: state.cubemap_asset_id,
        allocation_generation: cubemap.allocation_generation,
        face_size: cubemap.size,
        mip_levels_resident: cubemap.mip_levels_resident,
        content_generation: cubemap.content_generation,
        storage_v_inverted: cubemap.storage_v_inverted,
        view: cubemap.view.clone(),
    }))
}

fn color_probe_identity(renderable_index: i32, color: Vec4) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for bits in [
        renderable_index as u32,
        color.x.to_bits(),
        color.y.to_bits(),
        color.z.to_bits(),
        color.w.to_bits(),
    ] {
        hash ^= u64::from(bits);
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}

pub(super) fn spatial_probe_for_state(
    scene: &SceneCoordinator,
    space_id: RenderSpaceId,
    probe: &ReflectionProbeEntry,
    render_context: crate::shared::RenderingContext,
    atlas_index: u16,
) -> Option<SpatialProbe> {
    if probe.transform_id < 0 {
        return None;
    }
    let box_size = probe.state.box_size.abs();
    if box_size.cmplt(Vec3::splat(1e-6)).any() {
        return None;
    }
    let world =
        scene.world_matrix_for_context(space_id, probe.transform_id as usize, render_context)?;
    let bounds = crate::shared::RenderBoundingBox {
        center: Vec3::ZERO,
        extents: box_size * 0.5,
    };
    let (min, max) = world_aabb_from_local_bounds(&bounds, world)?;
    if !aabb_valid(min, max) {
        return None;
    }
    Some(SpatialProbe {
        renderable_index: probe.renderable_index,
        atlas_index,
        importance: probe.state.importance,
        aabb_min: Vec3A::from(min),
        aabb_max: Vec3A::from(max),
        center: Vec3A::from(world.transform_point3(Vec3::ZERO)),
        volume: aabb_volume(min, max),
    })
}

pub(super) fn metadata_for_spatial(
    spatial: &SpatialProbe,
    state: ReflectionProbeState,
    sh2: &RenderSH2,
) -> GpuReflectionProbeMetadata {
    let flags = if reflection_probe_use_box_projection(state.flags) {
        REFLECTION_PROBE_METADATA_BOX_PROJECTION
    } else {
        0
    };
    GpuReflectionProbeMetadata {
        box_min: [
            spatial.aabb_min.x,
            spatial.aabb_min.y,
            spatial.aabb_min.z,
            0.0,
        ],
        box_max: [
            spatial.aabb_max.x,
            spatial.aabb_max.y,
            spatial.aabb_max.z,
            0.0,
        ],
        position: [spatial.center.x, spatial.center.y, spatial.center.z, 0.0],
        params: [
            state.intensity.max(0.0),
            0.0,
            flags as f32,
            REFLECTION_PROBE_METADATA_SH2_SOURCE_LOCAL,
        ],
        sh2: pack_render_sh2_raw(sh2),
    }
}

pub(super) fn skybox_fallback_metadata(
    mip_levels: u32,
    sh2: Option<&RenderSH2>,
) -> GpuReflectionProbeMetadata {
    let mut metadata = GpuReflectionProbeMetadata {
        params: [1.0, mip_levels.saturating_sub(1) as f32, 0.0, 0.0],
        ..GpuReflectionProbeMetadata::default()
    };
    if let Some(sh2) = sh2 {
        metadata.params[3] = REFLECTION_PROBE_METADATA_SH2_SOURCE_SKYBOX;
        metadata.sh2 = pack_render_sh2_raw(sh2);
    }
    metadata
}

fn pack_render_sh2_raw(sh: &RenderSH2) -> [[f32; 4]; 9] {
    [
        [sh.sh0.x, sh.sh0.y, sh.sh0.z, 0.0],
        [sh.sh1.x, sh.sh1.y, sh.sh1.z, 0.0],
        [sh.sh2.x, sh.sh2.y, sh.sh2.z, 0.0],
        [sh.sh3.x, sh.sh3.y, sh.sh3.z, 0.0],
        [sh.sh4.x, sh.sh4.y, sh.sh4.z, 0.0],
        [sh.sh5.x, sh.sh5.y, sh.sh5.z, 0.0],
        [sh.sh6.x, sh.sh6.y, sh.sh6.z, 0.0],
        [sh.sh7.x, sh.sh7.y, sh.sh7.z, 0.0],
        [sh.sh8.x, sh.sh8.y, sh.sh8.z, 0.0],
    ]
}
