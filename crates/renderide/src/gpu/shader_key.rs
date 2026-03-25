//! [`ShaderKey`] describes how a drawable’s shader was resolved: optional host shader asset id
//! plus the legacy [`PipelineVariant`](super::PipelineVariant) used when the host path does not apply.
//!
//! This runs in parallel with the existing enum-based pipeline selection until per-draw resolution
//! fully replaces global `use_pbr` / `use_debug_uv` toggles.

use super::PipelineVariant;

/// Host shader identity and fallback variant from the pre-host-resolution path.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ShaderKey {
    /// Shader asset id from a `MaterialPropertyUpdateType::set_shader` batch for this drawable’s
    /// material property block, when present.
    pub host_shader_asset_id: Option<i32>,
    /// Variant that would apply without host shader selection (debug UV, PBR, skinned, stencil, …).
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
    /// When the host-unlit pilot is active and [`Self::host_shader_asset_id`] is set, non-MRT
    /// non-skinned mesh draws use [`PipelineVariant::Material`] keyed by `material_block_id`.
    pub fn effective_variant(
        self,
        use_host_unlit_pilot: bool,
        shader_debug_override_force_legacy: bool,
        material_block_id: i32,
        use_mrt: bool,
        is_skinned: bool,
        is_overlay: bool,
    ) -> PipelineVariant {
        if shader_debug_override_force_legacy {
            return self.fallback_variant;
        }
        if !use_host_unlit_pilot
            || self.host_shader_asset_id.is_none()
            || material_block_id < 0
            || use_mrt
            || is_skinned
            || is_overlay
        {
            return self.fallback_variant;
        }
        PipelineVariant::Material {
            material_id: material_block_id,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ShaderKey;
    use crate::gpu::PipelineVariant;

    #[test]
    fn effective_variant_uses_material_when_pilot_and_host_shader() {
        let k = ShaderKey {
            host_shader_asset_id: Some(42),
            fallback_variant: PipelineVariant::Pbr,
        };
        let v = k.effective_variant(true, false, 7, false, false, false);
        assert_eq!(v, PipelineVariant::Material { material_id: 7 });
    }

    #[test]
    fn effective_variant_respects_legacy_override() {
        let k = ShaderKey {
            host_shader_asset_id: Some(42),
            fallback_variant: PipelineVariant::NormalDebug,
        };
        let v = k.effective_variant(true, true, 7, false, false, false);
        assert_eq!(v, PipelineVariant::NormalDebug);
    }

    #[test]
    fn effective_variant_skips_overlay_and_skinned() {
        let k = ShaderKey {
            host_shader_asset_id: Some(1),
            fallback_variant: PipelineVariant::Skinned,
        };
        assert_eq!(
            k.effective_variant(true, false, 3, false, true, false),
            PipelineVariant::Skinned
        );
        let k2 = ShaderKey {
            host_shader_asset_id: Some(1),
            fallback_variant: PipelineVariant::NormalDebug,
        };
        assert_eq!(
            k2.effective_variant(true, false, 3, false, false, true),
            PipelineVariant::NormalDebug
        );
    }
}
