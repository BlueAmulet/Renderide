//! Projection360 cubemap/equirect source resolution from skybox materials.

use crate::assets::texture::HostTextureAssetKind;
use crate::backend::AssetTransferQueue;
use crate::backend::material_property_reader::{float4_property, texture_property};
use crate::materials::host_data::{
    MaterialPropertyLookupIds, MaterialPropertyStore, PropertyIdRegistry,
};
use crate::skybox::params::{DEFAULT_MAIN_TEX_ST, PROJECTION360_DEFAULT_FOV};

use super::source::{CubemapIblSource, EquirectIblSource, SkyboxIblSource};

/// Material identity carried into Projection360 resolution.
#[derive(Clone, Copy)]
pub(super) struct Projection360MaterialIdentity {
    pub material_asset_id: i32,
    pub material_generation: u64,
    pub route_hash: u64,
}

/// Resolves the primary Projection360 source from a skybox material.
pub(super) fn resolve_projection360_source(
    store: &MaterialPropertyStore,
    registry: &PropertyIdRegistry,
    assets: &AssetTransferQueue,
    lookup: MaterialPropertyLookupIds,
    identity: Projection360MaterialIdentity,
) -> Option<SkyboxIblSource> {
    let main_cube = texture_property(store, registry, lookup, "_MainCube")
        .or_else(|| texture_property(store, registry, lookup, "_Cube"));
    if let Some((asset_id, HostTextureAssetKind::Cubemap)) = main_cube {
        return resolve_projection360_cubemap_source(assets, asset_id, identity);
    }
    if let Some((asset_id, kind)) = main_cube {
        logger::trace!(
            "skybox specular: Projection360 _MainCube asset {asset_id} has unsupported kind {kind:?}"
        );
    }

    match texture_property(store, registry, lookup, "_MainTex")
        .or_else(|| texture_property(store, registry, lookup, "_Tex"))
    {
        Some((asset_id, HostTextureAssetKind::Texture2D)) => resolve_projection360_equirect_source(
            store, registry, assets, lookup, asset_id, identity,
        ),
        Some((asset_id, HostTextureAssetKind::Cubemap)) => {
            resolve_projection360_cubemap_source(assets, asset_id, identity)
        }
        Some((asset_id, kind)) => {
            logger::trace!(
                "skybox specular: Projection360 _MainTex asset {asset_id} has unsupported kind {kind:?}"
            );
            None
        }
        None => {
            logger::trace!("skybox specular: Projection360 skybox has no _MainCube or _MainTex");
            None
        }
    }
}

/// Resolves a resident Projection360 cubemap source.
fn resolve_projection360_cubemap_source(
    assets: &AssetTransferQueue,
    asset_id: i32,
    identity: Projection360MaterialIdentity,
) -> Option<SkyboxIblSource> {
    let Some(cubemap) = assets.cubemap_pool().get(asset_id) else {
        logger::trace!("skybox specular: cubemap asset {asset_id} is not allocated yet");
        return None;
    };
    if cubemap.mip_levels_resident == 0 {
        logger::trace!("skybox specular: cubemap asset {asset_id} has no resident mips");
        return None;
    }
    Some(SkyboxIblSource::Cubemap(CubemapIblSource {
        material_asset_id: identity.material_asset_id,
        material_generation: identity.material_generation,
        route_hash: identity.route_hash,
        asset_id,
        allocation_generation: cubemap.allocation_generation,
        face_size: cubemap.size,
        mip_levels_resident: cubemap.mip_levels_resident,
        content_generation: cubemap.content_generation,
        storage_v_inverted: cubemap.storage_v_inverted,
        view: cubemap.view.clone(),
    }))
}

/// Resolves a resident Projection360 equirectangular Texture2D source.
fn resolve_projection360_equirect_source(
    store: &MaterialPropertyStore,
    registry: &PropertyIdRegistry,
    assets: &AssetTransferQueue,
    lookup: MaterialPropertyLookupIds,
    asset_id: i32,
    identity: Projection360MaterialIdentity,
) -> Option<SkyboxIblSource> {
    let Some(texture) = assets.texture_pool().get(asset_id) else {
        logger::trace!("skybox specular: equirect Texture2D asset {asset_id} is not allocated yet");
        return None;
    };
    if texture.mip_levels_resident == 0 {
        logger::trace!("skybox specular: equirect Texture2D asset {asset_id} has no resident mips");
        return None;
    }
    Some(SkyboxIblSource::Equirect(EquirectIblSource {
        material_asset_id: identity.material_asset_id,
        material_generation: identity.material_generation,
        route_hash: identity.route_hash,
        asset_id,
        allocation_generation: texture.view_generation,
        width: texture.width,
        height: texture.height,
        mip_levels_resident: texture.mip_levels_resident,
        content_generation: texture.content_generation,
        storage_v_inverted: texture.storage_v_inverted,
        view: texture.view.clone(),
        equirect_fov: float4_property(store, registry, lookup, "_FOV", PROJECTION360_DEFAULT_FOV),
        equirect_st: float4_property(store, registry, lookup, "_MainTex_ST", DEFAULT_MAIN_TEX_ST),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::materials::host_data::MaterialPropertyValue;

    /// Packs a host texture id using the same high-bit asset-kind tag as the shared host packer.
    fn pack_host_texture(asset_id: i32, kind: HostTextureAssetKind) -> i32 {
        let type_bits = 3u32;
        let pack_type_shift = 32u32 - type_bits;
        ((asset_id as u32) | ((kind as u32) << pack_type_shift)) as i32
    }

    /// Creates a lookup and empty material property store for resolver tests.
    fn store_and_lookup(
        material_asset_id: i32,
    ) -> (MaterialPropertyStore, MaterialPropertyLookupIds) {
        (
            MaterialPropertyStore::new(),
            MaterialPropertyLookupIds {
                material_asset_id,
                mesh_property_block_slot0: None,
            },
        )
    }

    #[test]
    fn projection360_prefers_main_cube_over_main_tex() {
        let registry = PropertyIdRegistry::new();
        let (mut store, lookup) = store_and_lookup(7);
        store.set_material(
            lookup.material_asset_id,
            registry.intern("_MainTex"),
            MaterialPropertyValue::Texture(pack_host_texture(11, HostTextureAssetKind::Cubemap)),
        );
        store.set_material(
            lookup.material_asset_id,
            registry.intern("_MainCube"),
            MaterialPropertyValue::Texture(pack_host_texture(42, HostTextureAssetKind::Cubemap)),
        );

        assert_eq!(
            resolve_projection360_source_kind(&store, &registry, lookup),
            Some((42, HostTextureAssetKind::Cubemap))
        );
    }

    #[test]
    fn projection360_accepts_cubemap_main_tex_fallback() {
        let registry = PropertyIdRegistry::new();
        let (mut store, lookup) = store_and_lookup(8);
        store.set_material(
            lookup.material_asset_id,
            registry.intern("_MainTex"),
            MaterialPropertyValue::Texture(pack_host_texture(13, HostTextureAssetKind::Cubemap)),
        );

        assert_eq!(
            resolve_projection360_source_kind(&store, &registry, lookup),
            Some((13, HostTextureAssetKind::Cubemap))
        );
    }

    #[test]
    fn projection360_accepts_texture2d_main_tex() {
        let registry = PropertyIdRegistry::new();
        let (mut store, lookup) = store_and_lookup(9);
        store.set_material(
            lookup.material_asset_id,
            registry.intern("_MainTex"),
            MaterialPropertyValue::Texture(pack_host_texture(15, HostTextureAssetKind::Texture2D)),
        );

        assert_eq!(
            resolve_projection360_source_kind(&store, &registry, lookup),
            Some((15, HostTextureAssetKind::Texture2D))
        );
    }

    #[test]
    fn projection360_rejects_unsupported_main_cube_without_main_tex() {
        let registry = PropertyIdRegistry::new();
        let (mut store, lookup) = store_and_lookup(9);
        store.set_material(
            lookup.material_asset_id,
            registry.intern("_MainCube"),
            MaterialPropertyValue::Texture(pack_host_texture(15, HostTextureAssetKind::Texture2D)),
        );

        assert_eq!(
            resolve_projection360_source_kind(&store, &registry, lookup),
            None
        );
    }

    /// Resolves only the property-level source kind for unit tests that do not allocate GPU assets.
    fn resolve_projection360_source_kind(
        store: &MaterialPropertyStore,
        registry: &PropertyIdRegistry,
        lookup: MaterialPropertyLookupIds,
    ) -> Option<(i32, HostTextureAssetKind)> {
        let main_cube = texture_property(store, registry, lookup, "_MainCube")
            .or_else(|| texture_property(store, registry, lookup, "_Cube"));
        if let Some((asset_id, HostTextureAssetKind::Cubemap)) = main_cube {
            return Some((asset_id, HostTextureAssetKind::Cubemap));
        }
        let main_tex = texture_property(store, registry, lookup, "_MainTex")
            .or_else(|| texture_property(store, registry, lookup, "_Tex"));
        match main_tex {
            Some((
                asset_id,
                kind @ (HostTextureAssetKind::Texture2D | HostTextureAssetKind::Cubemap),
            )) => Some((asset_id, kind)),
            _ => None,
        }
    }
}
