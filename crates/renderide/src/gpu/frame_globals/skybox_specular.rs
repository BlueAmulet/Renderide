//! Frame-global skybox specular sampling parameters.
//!
//! Pure data: packs sampling state ([`SkyboxSpecularUniformParams`]) into the trailing
//! `vec4<f32>` slot of [`crate::gpu::frame_globals::FrameGpuUniforms`].
//!
//! The renderer always converts the active skybox into a single GGX-prefiltered cubemap before
//! binding it as the indirect specular source, so equirect-specific sampling state is no longer
//! plumbed through the frame globals. The runtime sample path is a single
//! `textureSampleLevel(skybox_specular, dir, lod)` against that cube.

/// Frame-global indicator that the indirect-specular source is active.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SkyboxSpecularSourceKind {
    /// No resident indirect-specular cube is bound.
    Disabled,
    /// `@group(0) @binding(9)` is a GGX-prefiltered cubemap.
    Cubemap,
}

impl SkyboxSpecularSourceKind {
    /// Numeric tag consumed by WGSL.
    pub const fn to_f32(self) -> f32 {
        match self {
            Self::Disabled => 0.0,
            Self::Cubemap => 1.0,
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
    /// Active source kind (cube or disabled).
    pub source_kind: SkyboxSpecularSourceKind,
}

impl SkyboxSpecularUniformParams {
    /// Disabled skybox specular environment.
    pub const fn disabled() -> Self {
        Self {
            max_lod: 0.0,
            enabled: false,
            source_kind: SkyboxSpecularSourceKind::Disabled,
        }
    }

    /// Builds enabled parameters from a resident cubemap mip count.
    pub fn from_cubemap_resident_mips(mip_levels_resident: u32) -> Self {
        Self {
            max_lod: mip_levels_resident.saturating_sub(1) as f32,
            enabled: mip_levels_resident > 0,
            source_kind: if mip_levels_resident > 0 {
                SkyboxSpecularSourceKind::Cubemap
            } else {
                SkyboxSpecularSourceKind::Disabled
            },
        }
    }

    /// Packs parameters into the `vec4<f32>` layout consumed by WGSL.
    ///
    /// Layout: `.x` max LOD, `.y` enabled flag, `.z` source kind tag, `.w` reserved.
    pub fn to_vec4(self) -> [f32; 4] {
        [
            self.max_lod,
            if self.enabled { 1.0 } else { 0.0 },
            self.source_kind.to_f32(),
            0.0,
        ]
    }
}
