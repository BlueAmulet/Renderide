//! Frame-global skybox specular sampling parameters.
//!
//! Pure data: enumerates the indirect specular source kinds and packs sampling state
//! ([`SkyboxSpecularUniformParams`]) into the trailing `vec4<f32>` slots of
//! [`crate::gpu::frame_globals::FrameGpuUniforms`].

/// Default `Projection360` field of view used by Unity material defaults.
pub(super) const PROJECTION360_DEFAULT_FOV: [f32; 4] =
    [std::f32::consts::TAU, std::f32::consts::PI, 0.0, 0.0];
/// Default texture scale/offset used by Unity `_MainTex_ST` properties.
pub(super) const DEFAULT_MAIN_TEX_ST: [f32; 4] = [1.0, 1.0, 0.0, 0.0];

/// Frame-global skybox specular source encoded in
/// [`crate::gpu::frame_globals::FrameGpuUniforms::skybox_specular`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SkyboxSpecularSourceKind {
    /// No resident indirect-specular source is bound.
    Disabled,
    /// `@group(0) @binding(9)` is a cubemap source.
    Cubemap,
    /// `@group(0) @binding(11)` is a Projection360 equirectangular `Texture2D` source.
    Projection360Equirect,
}

impl SkyboxSpecularSourceKind {
    /// Numeric tag consumed by WGSL.
    pub const fn to_f32(self) -> f32 {
        match self {
            Self::Disabled => 0.0,
            Self::Cubemap => 1.0,
            Self::Projection360Equirect => 2.0,
        }
    }
}

/// CPU-side parameters packed into
/// [`crate::gpu::frame_globals::FrameGpuUniforms::skybox_specular`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SkyboxSpecularUniformParams {
    /// Highest resident source mip available for roughness-driven sampling.
    pub max_lod: f32,
    /// Whether the frame has a resident skybox source bound for indirect specular.
    pub enabled: bool,
    /// Whether the source storage orientation needs V-axis compensation in shader sampling.
    pub storage_v_inverted: bool,
    /// Source texture kind selected by the active skybox material.
    pub source_kind: SkyboxSpecularSourceKind,
    /// Projection360 equirectangular `_FOV` parameters.
    pub equirect_fov: [f32; 4],
    /// Projection360 equirectangular `_MainTex_ST` parameters.
    pub equirect_st: [f32; 4],
}

impl SkyboxSpecularUniformParams {
    /// Disabled skybox specular environment.
    pub const fn disabled() -> Self {
        Self {
            max_lod: 0.0,
            enabled: false,
            storage_v_inverted: false,
            source_kind: SkyboxSpecularSourceKind::Disabled,
            equirect_fov: PROJECTION360_DEFAULT_FOV,
            equirect_st: DEFAULT_MAIN_TEX_ST,
        }
    }

    /// Builds enabled parameters from a resident cubemap mip count and storage orientation flag.
    pub fn from_resident_mips(mip_levels_resident: u32, storage_v_inverted: bool) -> Self {
        Self::from_cubemap_resident_mips(mip_levels_resident, storage_v_inverted)
    }

    /// Builds enabled parameters from a resident cubemap mip count and storage orientation flag.
    pub fn from_cubemap_resident_mips(mip_levels_resident: u32, storage_v_inverted: bool) -> Self {
        Self {
            max_lod: mip_levels_resident.saturating_sub(1) as f32,
            enabled: mip_levels_resident > 0,
            storage_v_inverted,
            source_kind: if mip_levels_resident > 0 {
                SkyboxSpecularSourceKind::Cubemap
            } else {
                SkyboxSpecularSourceKind::Disabled
            },
            equirect_fov: PROJECTION360_DEFAULT_FOV,
            equirect_st: DEFAULT_MAIN_TEX_ST,
        }
    }

    /// Builds enabled parameters from a resident Projection360 equirect texture.
    pub fn from_equirect_resident_mips(
        mip_levels_resident: u32,
        storage_v_inverted: bool,
        equirect_fov: [f32; 4],
        equirect_st: [f32; 4],
    ) -> Self {
        Self {
            max_lod: mip_levels_resident.saturating_sub(1) as f32,
            enabled: mip_levels_resident > 0,
            storage_v_inverted,
            source_kind: if mip_levels_resident > 0 {
                SkyboxSpecularSourceKind::Projection360Equirect
            } else {
                SkyboxSpecularSourceKind::Disabled
            },
            equirect_fov,
            equirect_st,
        }
    }

    /// Packs parameters into the `vec4<f32>` layout consumed by WGSL.
    pub fn to_vec4(self) -> [f32; 4] {
        [
            self.max_lod,
            if self.enabled { 1.0 } else { 0.0 },
            if self.storage_v_inverted { 1.0 } else { 0.0 },
            self.source_kind.to_f32(),
        ]
    }
}
