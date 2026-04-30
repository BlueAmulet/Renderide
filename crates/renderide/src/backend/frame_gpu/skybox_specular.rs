//! Frame-global skybox specular source selection and fallback bindings.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::gpu::frame_globals::{SkyboxSpecularSourceKind, SkyboxSpecularUniformParams};
use crate::gpu_pools::SamplerState;

/// Resident skybox source that can be bound as frame-global indirect specular.
pub enum SkyboxSpecularEnvironmentSource {
    /// A resident cubemap source sampled through `@group(0) @binding(9)`.
    Cubemap(SkyboxSpecularCubemapSource),
    /// A renderer-generated cubemap sampled through `@group(0) @binding(9)`.
    GeneratedCubemap(SkyboxSpecularGeneratedCubemapSource),
    /// A resident Projection360 equirect source sampled through `@group(0) @binding(11)`.
    Projection360Equirect(SkyboxSpecularEquirectSource),
}

impl SkyboxSpecularEnvironmentSource {
    /// Builds uniform parameters for this source.
    pub(super) fn uniform_params(&self) -> SkyboxSpecularUniformParams {
        match self {
            Self::Cubemap(source) => SkyboxSpecularUniformParams::from_cubemap_resident_mips(
                source.mip_levels_resident,
                source.storage_v_inverted,
            ),
            Self::GeneratedCubemap(source) => {
                SkyboxSpecularUniformParams::from_cubemap_resident_mips(
                    source.mip_levels_resident,
                    source.storage_v_inverted,
                )
            }
            Self::Projection360Equirect(source) => {
                SkyboxSpecularUniformParams::from_equirect_resident_mips(
                    source.mip_levels_resident,
                    source.storage_v_inverted,
                    source.equirect_fov,
                    source.equirect_st,
                )
            }
        }
    }
}

/// Resident cubemap source that can be bound as frame-global indirect specular.
pub struct SkyboxSpecularCubemapSource {
    /// Host cubemap asset id.
    pub asset_id: i32,
    /// Resident full cube texture view.
    pub view: Arc<wgpu::TextureView>,
    /// Host sampler settings copied from the cubemap pool.
    pub sampler: SamplerState,
    /// Resident mip count available for roughness-driven LOD sampling.
    pub mip_levels_resident: u32,
    /// Whether shader sampling needs V-axis storage compensation.
    pub storage_v_inverted: bool,
}

/// Renderer-generated cubemap source that can be bound as frame-global indirect specular.
pub struct SkyboxSpecularGeneratedCubemapSource {
    /// Stable renderer-side source hash.
    pub key_hash: u64,
    /// Resident full cube texture view.
    pub view: Arc<wgpu::TextureView>,
    /// Sampler settings used for the generated mip chain.
    pub sampler: SamplerState,
    /// Resident mip count available for roughness-driven LOD sampling.
    pub mip_levels_resident: u32,
    /// Whether shader sampling needs V-axis storage compensation.
    pub storage_v_inverted: bool,
}

/// Resident Projection360 equirectangular source that can be bound as frame-global indirect specular.
pub struct SkyboxSpecularEquirectSource {
    /// Host Texture2D asset id.
    pub asset_id: i32,
    /// Resident full 2D texture view.
    pub view: Arc<wgpu::TextureView>,
    /// Host sampler settings copied from the Texture2D pool.
    pub sampler: SamplerState,
    /// Resident mip count available for roughness-driven LOD sampling.
    pub mip_levels_resident: u32,
    /// Whether shader sampling needs V-axis storage compensation.
    pub storage_v_inverted: bool,
    /// Projection360 `_FOV` material parameters.
    pub equirect_fov: [f32; 4],
    /// Projection360 `_MainTex_ST` material parameters.
    pub equirect_st: [f32; 4],
}

/// Identity key for invalidating frame-global skybox specular bind groups.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SkyboxSpecularEnvironmentKey {
    /// Active source kind, or disabled for the fallback.
    source_kind: SkyboxSpecularSourceKind,
    /// Host texture asset id, or `-1` for the fallback.
    asset_id: i32,
    /// Raw texture-view pointer identity so same-id replacement still invalidates.
    view_identity: usize,
    /// Resident mip count included in the sampler LOD clamp and shader roughness range.
    mip_levels_resident: u32,
    /// Storage orientation flag consumed by WGSL.
    storage_v_inverted: bool,
    /// Hash of host sampler fields used to rebuild the wgpu sampler.
    sampler_signature: u64,
    /// Renderer-side generated-source hash, or zero for host texture sources.
    generated_hash: u64,
}

/// Texture/sampler resources bound to the frame-global skybox specular slots.
#[derive(Clone, Copy)]
pub(super) struct SkyboxSpecularBindGroupResources<'a> {
    /// Cubemap source bound at `@group(0) @binding(9)`.
    pub cubemap_view: &'a wgpu::TextureView,
    /// Cubemap sampler bound at `@group(0) @binding(10)`.
    pub cubemap_sampler: &'a wgpu::Sampler,
    /// Projection360 equirect source bound at `@group(0) @binding(11)`.
    pub equirect_view: &'a wgpu::TextureView,
    /// Projection360 equirect sampler bound at `@group(0) @binding(12)`.
    pub equirect_sampler: &'a wgpu::Sampler,
}

impl Default for SkyboxSpecularEnvironmentKey {
    fn default() -> Self {
        Self {
            source_kind: SkyboxSpecularSourceKind::Disabled,
            asset_id: -1,
            view_identity: 0,
            mip_levels_resident: 0,
            storage_v_inverted: false,
            sampler_signature: 0,
            generated_hash: 0,
        }
    }
}

impl SkyboxSpecularEnvironmentKey {
    /// Builds a key for a resident skybox cubemap source.
    fn from_cubemap_source(source: &SkyboxSpecularCubemapSource) -> Self {
        Self {
            source_kind: SkyboxSpecularSourceKind::Cubemap,
            asset_id: source.asset_id,
            view_identity: Arc::as_ptr(&source.view) as usize,
            mip_levels_resident: source.mip_levels_resident,
            storage_v_inverted: source.storage_v_inverted,
            sampler_signature: sampler_signature(&source.sampler),
            generated_hash: 0,
        }
    }

    /// Builds a key for a renderer-generated skybox cubemap source.
    fn from_generated_cubemap_source(source: &SkyboxSpecularGeneratedCubemapSource) -> Self {
        Self {
            source_kind: SkyboxSpecularSourceKind::Cubemap,
            asset_id: -1,
            view_identity: Arc::as_ptr(&source.view) as usize,
            mip_levels_resident: source.mip_levels_resident,
            storage_v_inverted: source.storage_v_inverted,
            sampler_signature: sampler_signature(&source.sampler),
            generated_hash: source.key_hash,
        }
    }

    /// Builds a key for a resident Projection360 equirect source.
    fn from_equirect_source(source: &SkyboxSpecularEquirectSource) -> Self {
        Self {
            source_kind: SkyboxSpecularSourceKind::Projection360Equirect,
            asset_id: source.asset_id,
            view_identity: Arc::as_ptr(&source.view) as usize,
            mip_levels_resident: source.mip_levels_resident,
            storage_v_inverted: source.storage_v_inverted,
            sampler_signature: sampler_signature(&source.sampler),
            generated_hash: 0,
        }
    }

    /// Builds a key for any resident skybox source.
    pub(super) fn from_source(source: &SkyboxSpecularEnvironmentSource) -> Self {
        match source {
            SkyboxSpecularEnvironmentSource::Cubemap(source) => Self::from_cubemap_source(source),
            SkyboxSpecularEnvironmentSource::GeneratedCubemap(source) => {
                Self::from_generated_cubemap_source(source)
            }
            SkyboxSpecularEnvironmentSource::Projection360Equirect(source) => {
                Self::from_equirect_source(source)
            }
        }
    }
}

/// Hashes sampler fields that affect the wgpu sampler descriptor for any skybox source kind.
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

/// Allocates and initializes the black equirect Texture2D used when no skybox specular environment exists.
pub(super) fn create_black_skybox_specular_equirect_fallback(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> (
    Arc<wgpu::Texture>,
    Arc<wgpu::TextureView>,
    Arc<wgpu::Sampler>,
) {
    let texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
        label: Some("frame_skybox_specular_black_equirect"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
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
        &[0u8; 4],
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    let view = Arc::new(texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("frame_skybox_specular_black_equirect_view"),
        dimension: Some(wgpu::TextureViewDimension::D2),
        ..Default::default()
    }));
    let sampler = Arc::new(device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("frame_skybox_specular_black_equirect_sampler"),
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
