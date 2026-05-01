//! Frame-global skybox specular source selection and fallback bindings.
//!
//! Every active skybox is converted to a GGX-prefiltered Rgba16Float cubemap before binding, so
//! this module deals only in cubemaps. Equirect Texture2D inputs are baked to a cube by the IBL
//! cache before reaching this layer.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::gpu::frame_globals::{SkyboxSpecularSourceKind, SkyboxSpecularUniformParams};
use crate::gpu_pools::SamplerState;

/// Resident skybox source bound as frame-global indirect specular.
pub enum SkyboxSpecularEnvironmentSource {
    /// A renderer-owned, GGX-prefiltered cubemap sampled through `@group(0) @binding(9)`.
    Cubemap(SkyboxSpecularCubemapSource),
}

impl SkyboxSpecularEnvironmentSource {
    /// Builds uniform parameters for this source.
    pub(super) fn uniform_params(&self) -> SkyboxSpecularUniformParams {
        match self {
            Self::Cubemap(source) => {
                SkyboxSpecularUniformParams::from_cubemap_resident_mips(source.mip_levels_resident)
            }
        }
    }
}

/// GGX-prefiltered cubemap source bound as frame-global indirect specular.
pub struct SkyboxSpecularCubemapSource {
    /// Stable renderer-side identity hash (covers source kind, asset id, generation, face size).
    pub key_hash: u64,
    /// Resident full cube texture view.
    pub view: Arc<wgpu::TextureView>,
    /// Sampler settings used for the prefiltered mip chain.
    pub sampler: SamplerState,
    /// Resident mip count available for roughness-driven LOD sampling.
    pub mip_levels_resident: u32,
}

/// Identity key for invalidating frame-global skybox specular bind groups.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SkyboxSpecularEnvironmentKey {
    /// Active source kind, or disabled for the fallback.
    source_kind: SkyboxSpecularSourceKind,
    /// Raw texture-view pointer identity so same-id replacement still invalidates.
    view_identity: usize,
    /// Resident mip count included in the sampler LOD clamp and shader roughness range.
    mip_levels_resident: u32,
    /// Hash of host sampler fields used to rebuild the wgpu sampler.
    sampler_signature: u64,
    /// Renderer-side prefiltered-source identity hash.
    source_hash: u64,
}

/// Texture/sampler resources bound to the frame-global skybox specular slots.
#[derive(Clone, Copy)]
pub(super) struct SkyboxSpecularBindGroupResources<'a> {
    /// Cubemap source bound at `@group(0) @binding(9)`.
    pub cubemap_view: &'a wgpu::TextureView,
    /// Cubemap sampler bound at `@group(0) @binding(10)`.
    pub cubemap_sampler: &'a wgpu::Sampler,
}

impl Default for SkyboxSpecularEnvironmentKey {
    fn default() -> Self {
        Self {
            source_kind: SkyboxSpecularSourceKind::Disabled,
            view_identity: 0,
            mip_levels_resident: 0,
            sampler_signature: 0,
            source_hash: 0,
        }
    }
}

impl SkyboxSpecularEnvironmentKey {
    /// Builds a key for a prefiltered skybox cubemap source.
    pub(super) fn from_source(source: &SkyboxSpecularEnvironmentSource) -> Self {
        match source {
            SkyboxSpecularEnvironmentSource::Cubemap(source) => Self {
                source_kind: SkyboxSpecularSourceKind::Cubemap,
                view_identity: Arc::as_ptr(&source.view) as usize,
                mip_levels_resident: source.mip_levels_resident,
                sampler_signature: sampler_signature(&source.sampler),
                source_hash: source.key_hash,
            },
        }
    }
}

/// Hashes sampler fields that affect the wgpu sampler descriptor.
fn sampler_signature(state: &SamplerState) -> u64 {
    let mut hasher = DefaultHasher::new();
    (state.filter_mode as i32).hash(&mut hasher);
    state.aniso_level.hash(&mut hasher);
    state.mipmap_bias.to_bits().hash(&mut hasher);
    (state.wrap_u as i32).hash(&mut hasher);
    (state.wrap_v as i32).hash(&mut hasher);
    hasher.finish()
}

/// Allocates and initializes the black cubemap used when no skybox specular environment exists.
pub(super) fn create_black_skybox_specular_fallback(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> (
    Arc<wgpu::Texture>,
    Arc<wgpu::TextureView>,
    Arc<wgpu::Sampler>,
) {
    let texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
        label: Some("frame_skybox_specular_black_cube"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 6,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    }));
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: texture.as_ref(),
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &[0u8; 24],
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 6,
        },
    );
    let view = Arc::new(texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("frame_skybox_specular_black_cube_view"),
        dimension: Some(wgpu::TextureViewDimension::Cube),
        ..Default::default()
    }));
    let sampler = Arc::new(device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("frame_skybox_specular_black_cube_sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        lod_min_clamp: 0.0,
        lod_max_clamp: 0.0,
        ..Default::default()
    }));
    (texture, view, sampler)
}
