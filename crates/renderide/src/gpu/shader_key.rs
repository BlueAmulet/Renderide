//! [`ShaderKey`] describes how a drawable???s shader was resolved: optional host shader asset id
//! plus the builtin [`PipelineVariant`](super::PipelineVariant) used when no native Renderide
//! shader route applies.

use crate::assets::NativeMaterialPipelineFamily;

use super::PipelineVariant;

/// Host shader identity and fallback variant from the pre-host-resolution path.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ShaderKey {
    /// Shader asset id from a `MaterialPropertyUpdateType::set_shader` batch for this drawable???s
    /// material property block, when present.
    pub host_shader_asset_id: Option<i32>,
    /// Variant that would apply without host shader selection (debug UV, PBR, skinned, stencil, ???).
    pub fallback_variant: PipelineVariant,
}

impl ShaderKey {
    /// Builds a key with no host shader override.
    pub const fn builtin_only(fallback_variant: PipelineVariant) -> Self {
        Self {
            host_shader_asset_id: None,
            fallback_variant,
        }
    }

    /// Effective pipeline variant for batching and GPU pipeline lookup.
    ///
    /// When [`Self::host_shader_asset_id`] is set and the shader resolves to world-unlit,
    /// non-MRT non-skinned non-overlay draws use [`PipelineVariant::Material`] keyed by
    /// `material_block_id`. Native PBS / UI routes stay on their own variants.
    #[allow(clippy::too_many_arguments)]
    pub fn effective_variant(
        self,
        shader_debug_override_force_legacy: bool,
        material_block_id: i32,
        use_mrt: bool,
        is_skinned: bool,
        is_overlay: bool,
        native_material_family: NativeMaterialPipelineFamily,
    ) -> PipelineVariant {
        if shader_debug_override_force_legacy {
            return self.fallback_variant;
        }
        if self.host_shader_asset_id.is_none()
            || material_block_id < 0
            || use_mrt
            || is_skinned
            || is_overlay
        {
            return self.fallback_variant;
        }
        match native_material_family {
            NativeMaterialPipelineFamily::WorldUnlit => PipelineVariant::Material {
                material_id: material_block_id,
            },
            NativeMaterialPipelineFamily::PbsMetallic
            | NativeMaterialPipelineFamily::UiUnlit
            | NativeMaterialPipelineFamily::LegacyFallback => self.fallback_variant,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ShaderKey;
    use crate::assets::NativeMaterialPipelineFamily;
    use crate::gpu::PipelineVariant;

    #[test]
    fn effective_variant_uses_material_when_world_unlit() {
        let k = ShaderKey {
            host_shader_asset_id: Some(42),
            fallback_variant: PipelineVariant::Pbr,
        };
        let v = k.effective_variant(`r`n            false,
            7,
            false,
            false,
            false,
            NativeMaterialPipelineFamily::WorldUnlit,
        );
        assert_eq!(v, PipelineVariant::Material { material_id: 7 });
    }

    #[test]
    fn effective_variant_unsupported_stays_on_fallback() {
        let k = ShaderKey {
            host_shader_asset_id: Some(42),
            fallback_variant: PipelineVariant::Pbr,
        };
        let v = k.effective_variant(`r`n            false,
            7,
            false,
            false,
            false,
            NativeMaterialPipelineFamily::LegacyFallback,
        );
        assert_eq!(v, PipelineVariant::Pbr);
    }

    #[test]
    fn effective_variant_pbs_metallic_stays_on_fallback_pbr() {
        let k = ShaderKey {
            host_shader_asset_id: Some(42),
            fallback_variant: PipelineVariant::Pbr,
        };
        let v = k.effective_variant(`r`n            false,
            7,
            false,
            false,
            false,
            NativeMaterialPipelineFamily::PbsMetallic,
        );
        assert_eq!(v, PipelineVariant::Pbr);
    }

    #[test]
    fn effective_variant_respects_legacy_override() {
        let k = ShaderKey {
            host_shader_asset_id: Some(42),
            fallback_variant: PipelineVariant::NormalDebug,
        };
        let v = k.effective_variant(`r`n            true,
            7,
            false,
            false,
            false,
            NativeMaterialPipelineFamily::LegacyFallback,
        );
        assert_eq!(v, PipelineVariant::NormalDebug);
    }

    #[test]
    fn effective_variant_skips_overlay_and_skinned() {
        let k = ShaderKey {
            host_shader_asset_id: Some(1),
            fallback_variant: PipelineVariant::Skinned,
        };
        assert_eq!(
            k.effective_variant(`r`n                false,
                3,
                false,
                true,
                false,
                NativeMaterialPipelineFamily::LegacyFallback,
            ),
            PipelineVariant::Skinned
        );
        let k2 = ShaderKey {
            host_shader_asset_id: Some(1),
            fallback_variant: PipelineVariant::NormalDebug,
        };
        assert_eq!(
            k2.effective_variant(`r`n                false,
                3,
                false,
                false,
                true,
                NativeMaterialPipelineFamily::LegacyFallback,
            ),
            PipelineVariant::NormalDebug
        );
    }
}

