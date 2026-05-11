//! Top-level dispatch from a skybox material to its [`SkyboxIblSource`].

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::backend::AssetTransferQueue;
use crate::materials::MaterialSystem;
use crate::materials::host_data::MaterialPropertyLookupIds;
use crate::skybox::params::{gradient_sky_params, procedural_sky_params};

use super::projection360::{Projection360MaterialIdentity, resolve_projection360_source};
use super::source::{AnalyticIblSource, SkyboxIblSource};

/// Resolves one skybox material into an IBL bake source.
pub(crate) fn resolve_skybox_material_ibl_source(
    material_asset_id: i32,
    materials: &MaterialSystem,
    assets: &AssetTransferQueue,
) -> Option<SkyboxIblSource> {
    let store = materials.material_property_store();
    let shader_asset_id = store.shader_asset_for_material(material_asset_id)?;
    let route_name = shader_route_name(materials, shader_asset_id)?;
    let material_generation = store.material_generation(material_asset_id);
    let route_hash = hash_route_name(&route_name);
    let route_lower = route_name.to_ascii_lowercase();
    let lookup = MaterialPropertyLookupIds {
        material_asset_id,
        mesh_property_block_slot0: None,
        mesh_renderer_property_block_id: None,
    };

    if route_lower.contains("projection360") {
        return resolve_projection360_source(
            store,
            materials.property_id_registry(),
            assets,
            lookup,
            Projection360MaterialIdentity {
                material_asset_id,
                material_generation,
                route_hash,
            },
        );
    }
    if route_lower.contains("gradient") {
        let params = gradient_sky_params(store, materials.property_id_registry(), lookup);
        return Some(SkyboxIblSource::Analytic(Box::new(AnalyticIblSource {
            material_asset_id,
            material_generation,
            route_hash,
            params,
        })));
    }
    if route_lower.contains("procedural") {
        let params = procedural_sky_params(store, materials.property_id_registry(), lookup);
        return Some(SkyboxIblSource::Analytic(Box::new(AnalyticIblSource {
            material_asset_id,
            material_generation,
            route_hash,
            params,
        })));
    }
    logger::trace!(
        "skybox specular: unsupported active skybox route '{route_name}' for material {material_asset_id}"
    );
    None
}

/// Returns a shader route name or stem for a shader asset id.
fn shader_route_name(materials: &MaterialSystem, shader_asset_id: i32) -> Option<String> {
    let registry = materials.material_registry()?;
    registry
        .stem_for_shader_asset(shader_asset_id)
        .map(str::to_string)
}

/// Hashes a shader route name into a stable cache discriminator.
fn hash_route_name(route: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    route.hash(&mut hasher);
    hasher.finish()
}
