//! Texture asset id resolution and bind signature hashing for embedded material bind groups.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::assets::material::{
    MaterialPropertyLookupIds, MaterialPropertyStore, MaterialPropertyValue,
};
use crate::assets::texture::{
    texture2d_asset_id_from_packed, unpack_host_texture_packed, HostTextureAssetKind,
};
use crate::materials::ReflectedRasterLayout;
use crate::resources::{RenderTexturePool, Texture2dSamplerState, TexturePool};

use super::embedded_material_layout::StemEmbeddedPropertyIds;

/// Resolved GPU texture binding for a material property (packed host id or primary fallback).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ResolvedTextureBinding {
    /// No texture or unsupported packed type.
    None,
    /// [`crate::resources::TexturePool`] entry (unpacked 2D asset id).
    Texture2D { asset_id: i32 },
    /// [`crate::resources::RenderTexturePool`] entry (unpacked render-texture asset id).
    RenderTexture { asset_id: i32 },
}

impl ResolvedTextureBinding {
    fn hash_for_signature(self, hasher: &mut DefaultHasher) {
        match self {
            ResolvedTextureBinding::None => {
                0u8.hash(hasher);
            }
            ResolvedTextureBinding::Texture2D { asset_id } => {
                1u8.hash(hasher);
                asset_id.hash(hasher);
            }
            ResolvedTextureBinding::RenderTexture { asset_id } => {
                2u8.hash(hasher);
                asset_id.hash(hasher);
            }
        }
    }
}

/// Resolves primary 2D texture asset id from reflected material entries.
pub(crate) fn primary_texture_2d_asset_id(
    reflected: &ReflectedRasterLayout,
    ids: &StemEmbeddedPropertyIds,
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
) -> i32 {
    for entry in &reflected.material_entries {
        if matches!(entry.ty, wgpu::BindingType::Texture { .. }) {
            let Some(pid) = ids.texture_binding_to_property_id.get(&entry.binding) else {
                continue;
            };
            if let Some(MaterialPropertyValue::Texture(packed)) = store.get_merged(lookup, *pid) {
                return texture2d_asset_id_from_packed(*packed).unwrap_or(-1);
            }
        }
    }
    -1
}

/// `true` when the first reflected `@group(1)` texture slot has a valid packed host texture of any
/// supported kind (2D, render texture, etc.).
///
/// Used for uniform `flags` bit 0 (`_TEXTURE` / main texture sampling) — Unity enables that path
/// when any texture is bound, not only [`HostTextureAssetKind::Texture2D`].
pub(crate) fn primary_texture_any_kind_present(
    reflected: &ReflectedRasterLayout,
    ids: &StemEmbeddedPropertyIds,
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
) -> bool {
    for entry in &reflected.material_entries {
        if matches!(entry.ty, wgpu::BindingType::Texture { .. }) {
            let Some(pid) = ids.texture_binding_to_property_id.get(&entry.binding) else {
                continue;
            };
            if let Some(MaterialPropertyValue::Texture(packed)) = store.get_merged(lookup, *pid) {
                return unpack_host_texture_packed(*packed).is_some();
            }
        }
    }
    false
}

pub(crate) fn should_fallback_to_primary_texture(host_name: &str) -> bool {
    matches!(host_name, "_MainTex" | "_Tex" | "_TEXTURE")
}

fn texture_property_binding(
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    property_id: i32,
) -> ResolvedTextureBinding {
    match store.get_merged(lookup, property_id) {
        Some(MaterialPropertyValue::Texture(packed)) => match unpack_host_texture_packed(*packed) {
            Some((id, HostTextureAssetKind::Texture2D)) => {
                ResolvedTextureBinding::Texture2D { asset_id: id }
            }
            Some((id, HostTextureAssetKind::RenderTexture)) => {
                ResolvedTextureBinding::RenderTexture { asset_id: id }
            }
            _ => ResolvedTextureBinding::None,
        },
        _ => ResolvedTextureBinding::None,
    }
}

/// Resolves resident texture binding for a host property name, with primary-texture fallback for 2D-only slots.
pub(crate) fn resolved_texture_binding_for_host(
    host_name: &str,
    texture_property_id: i32,
    primary_texture_2d: i32,
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
) -> ResolvedTextureBinding {
    let b = texture_property_binding(store, lookup, texture_property_id);
    if !matches!(b, ResolvedTextureBinding::None) {
        return b;
    }
    if should_fallback_to_primary_texture(host_name) && primary_texture_2d >= 0 {
        return ResolvedTextureBinding::Texture2D {
            asset_id: primary_texture_2d,
        };
    }
    ResolvedTextureBinding::None
}

/// Fingerprint for bind cache invalidation when texture views or residency change.
///
/// When `offscreen_write_render_texture_asset_id` is [`Some`], that render-texture id is treated as
/// non-resident (offscreen color target; self-sampling is masked).
#[allow(clippy::too_many_arguments)]
pub(crate) fn texture_bind_signature(
    reflected: &ReflectedRasterLayout,
    ids: &StemEmbeddedPropertyIds,
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    texture_pool: &TexturePool,
    render_texture_pool: &RenderTexturePool,
    primary_texture_2d: i32,
    offscreen_write_render_texture_asset_id: Option<i32>,
) -> u64 {
    let mut h = DefaultHasher::new();
    offscreen_write_render_texture_asset_id.hash(&mut h);
    for entry in &reflected.material_entries {
        if !matches!(entry.ty, wgpu::BindingType::Texture { .. }) {
            continue;
        }
        let Some(name) = reflected.material_group1_names.get(&entry.binding) else {
            continue;
        };
        let Some(&texture_pid) = ids.texture_binding_to_property_id.get(&entry.binding) else {
            continue;
        };
        let binding = resolved_texture_binding_for_host(
            name.as_str(),
            texture_pid,
            primary_texture_2d,
            store,
            lookup,
        );
        entry.binding.hash(&mut h);
        name.hash(&mut h);
        binding.hash_for_signature(&mut h);
        let resident = match binding {
            ResolvedTextureBinding::None => false,
            ResolvedTextureBinding::Texture2D { asset_id } => texture_pool
                .get_texture(asset_id)
                .is_some_and(|t| t.mip_levels_resident > 0),
            ResolvedTextureBinding::RenderTexture { asset_id } => {
                if offscreen_write_render_texture_asset_id == Some(asset_id) {
                    false
                } else {
                    render_texture_pool
                        .get(asset_id)
                        .is_some_and(|t| t.is_sampleable())
                }
            }
        };
        resident.hash(&mut h);
    }
    h.finish()
}

pub(crate) fn sampler_from_state(
    device: &wgpu::Device,
    state: &Texture2dSamplerState,
) -> wgpu::Sampler {
    let address_mode_u = match state.wrap_u {
        crate::shared::TextureWrapMode::Repeat => wgpu::AddressMode::Repeat,
        crate::shared::TextureWrapMode::Clamp => wgpu::AddressMode::ClampToEdge,
        crate::shared::TextureWrapMode::Mirror => wgpu::AddressMode::MirrorRepeat,
        crate::shared::TextureWrapMode::MirrorOnce => wgpu::AddressMode::ClampToEdge,
    };
    let address_mode_v = match state.wrap_v {
        crate::shared::TextureWrapMode::Repeat => wgpu::AddressMode::Repeat,
        crate::shared::TextureWrapMode::Clamp => wgpu::AddressMode::ClampToEdge,
        crate::shared::TextureWrapMode::Mirror => wgpu::AddressMode::MirrorRepeat,
        crate::shared::TextureWrapMode::MirrorOnce => wgpu::AddressMode::ClampToEdge,
    };
    let (mag, min, mipmap) = match state.filter_mode {
        crate::shared::TextureFilterMode::Point => (
            wgpu::FilterMode::Nearest,
            wgpu::FilterMode::Nearest,
            wgpu::MipmapFilterMode::Nearest,
        ),
        crate::shared::TextureFilterMode::Bilinear => (
            wgpu::FilterMode::Linear,
            wgpu::FilterMode::Linear,
            wgpu::MipmapFilterMode::Linear,
        ),
        crate::shared::TextureFilterMode::Trilinear => (
            wgpu::FilterMode::Linear,
            wgpu::FilterMode::Linear,
            wgpu::MipmapFilterMode::Linear,
        ),
        crate::shared::TextureFilterMode::Anisotropic => (
            wgpu::FilterMode::Linear,
            wgpu::FilterMode::Linear,
            wgpu::MipmapFilterMode::Linear,
        ),
    };
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("embedded_texture_sampler"),
        address_mode_u,
        address_mode_v,
        address_mode_w: address_mode_u,
        mag_filter: mag,
        min_filter: min,
        mipmap_filter: mipmap,
        ..Default::default()
    })
}
