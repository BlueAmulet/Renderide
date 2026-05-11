//! IBL source identity and per-variant payload definitions.

use std::sync::Arc;

use crate::skybox::params::SkyboxEvaluatorParams;

/// Active skybox source to be baked into a GGX-prefiltered cubemap.
///
/// `Analytic` is boxed because [`AnalyticIblSource`] embeds the full
/// [`SkyboxEvaluatorParams`] (~=1.1 KiB of gradient arrays), an order of magnitude larger than the
/// other variants.
pub(crate) enum SkyboxIblSource {
    /// Analytic procedural / gradient skybox evaluator.
    Analytic(Box<AnalyticIblSource>),
    /// Resident host-uploaded cubemap (`Projection360 _MainCube`).
    Cubemap(CubemapIblSource),
    /// Resident host-uploaded equirect Texture2D (`Projection360 _MainTex`).
    Equirect(EquirectIblSource),
    /// Analytic constant-color source.
    SolidColor(SolidColorIblSource),
    /// Renderer-captured cubemap source for an OnChanges reflection probe.
    RuntimeCubemap(RuntimeCubemapIblSource),
}

/// Analytic skybox material identity and evaluator parameters.
pub(crate) struct AnalyticIblSource {
    /// Active skybox material asset id.
    pub material_asset_id: i32,
    /// Material property generation; invalidates the bake when material props change.
    pub material_generation: u64,
    /// Stable hash of the shader route stem ("gradient" / "procedural" variants).
    pub route_hash: u64,
    /// Packed evaluator parameters for the analytic mip-0 producer.
    pub params: SkyboxEvaluatorParams,
}

/// Resident cubemap source identity and GPU handle.
pub(crate) struct CubemapIblSource {
    /// Skybox material asset id when this source came from a material, or `-1` for direct probe sources.
    pub material_asset_id: i32,
    /// Material property generation when this source came from a material.
    pub material_generation: u64,
    /// Stable hash of the shader route stem when this source came from a material.
    pub route_hash: u64,
    /// Source cubemap asset id.
    pub asset_id: i32,
    /// Source GPU allocation generation; invalidates when an asset id is reallocated.
    pub allocation_generation: u64,
    /// Resident cubemap face edge in texels (mip 0).
    pub face_size: u32,
    /// Resident mip count of the source cubemap.
    pub mip_levels_resident: u32,
    /// Source cubemap content generation; invalidates bakes when texels are re-uploaded.
    pub content_generation: u64,
    /// Whether sampling needs V-axis storage compensation.
    pub storage_v_inverted: bool,
    /// Cube-dimension texture view used as the bake input.
    pub view: Arc<wgpu::TextureView>,
}

/// Resident equirect Texture2D source identity and GPU handle.
pub(crate) struct EquirectIblSource {
    /// Skybox material asset id when this source came from a material.
    pub material_asset_id: i32,
    /// Material property generation when this source came from a material.
    pub material_generation: u64,
    /// Stable hash of the shader route stem when this source came from a material.
    pub route_hash: u64,
    /// Source Texture2D asset id.
    pub asset_id: i32,
    /// Source GPU allocation generation; invalidates when an asset id is reallocated.
    pub allocation_generation: u64,
    /// Mip0 width in texels.
    pub width: u32,
    /// Mip0 height in texels.
    pub height: u32,
    /// Resident mip count of the source texture.
    pub mip_levels_resident: u32,
    /// Source texture content generation; invalidates bakes when texels are re-uploaded.
    pub content_generation: u64,
    /// Whether sampling needs V-axis storage compensation.
    pub storage_v_inverted: bool,
    /// 2D texture view used as the bake input.
    pub view: Arc<wgpu::TextureView>,
    /// Projection360 `_FOV` parameters.
    pub equirect_fov: [f32; 4],
    /// Projection360 `_MainTex_ST` parameters.
    pub equirect_st: [f32; 4],
}

/// Constant-color source identity and color.
pub(crate) struct SolidColorIblSource {
    /// Renderer-side identity for this color source.
    pub identity: u64,
    /// Linear RGB color with alpha padding.
    pub color: [f32; 4],
}

/// Renderer-owned cubemap source identity and GPU handle.
pub(crate) struct RuntimeCubemapIblSource {
    /// Render space that owns the captured probe.
    pub render_space_id: i32,
    /// Dense reflection-probe renderable index.
    pub renderable_index: i32,
    /// Monotonic renderer-side capture generation.
    pub generation: u64,
    /// Source cubemap face edge in texels.
    pub face_size: u32,
    /// Mip count allocated on the captured cubemap.
    pub mip_levels: u32,
    /// Captured texture retained with the source view.
    pub texture: Arc<wgpu::Texture>,
    /// Cube-dimension texture view used as the bake input.
    pub view: Arc<wgpu::TextureView>,
}
