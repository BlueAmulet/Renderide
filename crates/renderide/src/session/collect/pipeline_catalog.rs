//! Strangler Fig **pipeline catalog**: classifies host shader / material draws into stable families and
//! exposes a catalog-driven [`resolve_pipeline_variant_for_material_draw`] that delegates to the same
//! helper chain as [`super::pipeline::resolve_pipeline_for_material_draw_internal`] so behavior stays
//! identical while the module boundary is ready for future refactors.
//!
//! **Classification** ([`ShaderPipelineFamily`]) is for diagnostics and shadow checks; it does not
//! replace [`crate::gpu::PipelineVariant`] selection by itself.

use crate::assets::{
    AssetRegistry, NativeUiShaderFamily, WorldUnlitShaderFamily, resolve_native_ui_shader_family,
    resolve_pbs_metallic_shader_family, resolve_world_unlit_shader_family,
};
use crate::config::RenderConfig;
use crate::gpu::{PipelineVariant, ShaderKey};
use crate::scene::{Drawable, Scene};

use super::pipeline::resolve_pipeline_for_material_draw_internal;

/// High-level shader routing bucket for a material draw (parallel to Unity / Renderite concepts).
///
/// This is a **classification** label: actual drawing still uses [`PipelineVariant`] from the resolver.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShaderPipelineFamily {
    /// Global debug / PBR / skinned builtins without host `Material` pipeline override.
    BuiltinGlobal,
    /// Native world-unlit material path.
    HostUnlitMaterial,
    /// Resonite `Shader "PBSMetallic"` native forward PBR ([`crate::gpu::PipelineVariant::Pbr`] family).
    PbsMetallicNative,
    /// Resonite world `Shader "Unlit"` native WGSL ([`crate::gpu::pipeline::WorldUnlitPipeline`]).
    WorldUnlitNative,
    /// Native overlay / world-space UI WGSL (`UI_Unlit` / `UI_TextUnlit` families).
    NativeUiWgsl,
    /// Could not classify (e.g. missing shader asset); rely on [`PipelineVariant`] only.
    Unknown,
}

/// Classifies `host_shader_asset_id` using the same allowlists as routing (native UI, world unlit).
pub fn classify_shader_pipeline_family(
    host_shader_asset_id: Option<i32>,
    material_block_id: i32,
    render_config: &RenderConfig,
    asset_registry: &AssetRegistry,
) -> ShaderPipelineFamily {
    let Some(sid) = host_shader_asset_id else {
        return ShaderPipelineFamily::BuiltinGlobal;
    };
    if material_block_id < 0 {
        return ShaderPipelineFamily::BuiltinGlobal;
    }
    if resolve_native_ui_shader_family(
        sid,
        render_config.native_ui_unlit_shader_id,
        render_config.native_ui_text_unlit_shader_id,
        asset_registry,
    )
    .is_some()
    {
        return ShaderPipelineFamily::NativeUiWgsl;
    }
    if resolve_world_unlit_shader_family(
        sid,
        render_config.native_world_unlit_shader_id,
        asset_registry,
    )
    .is_some()
    {
        return ShaderPipelineFamily::WorldUnlitNative;
    }
    if resolve_pbs_metallic_shader_family(sid, asset_registry) {
        return ShaderPipelineFamily::PbsMetallicNative;
    }
    ShaderPipelineFamily::Unknown
}

/// Optional detail when [`ShaderPipelineFamily::NativeUiWgsl`] or world unlit applies.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShaderFamilyDetail {
    None,
    NativeUi(NativeUiShaderFamily),
    WorldUnlit(WorldUnlitShaderFamily),
}

/// Resolves native UI or world-unlit enum detail when applicable.
pub fn shader_family_detail(
    host_shader_asset_id: Option<i32>,
    render_config: &RenderConfig,
    asset_registry: &AssetRegistry,
) -> ShaderFamilyDetail {
    let Some(sid) = host_shader_asset_id else {
        return ShaderFamilyDetail::None;
    };
    if let Some(f) = resolve_native_ui_shader_family(
        sid,
        render_config.native_ui_unlit_shader_id,
        render_config.native_ui_text_unlit_shader_id,
        asset_registry,
    ) {
        return ShaderFamilyDetail::NativeUi(f);
    }
    if let Some(w) = resolve_world_unlit_shader_family(
        sid,
        render_config.native_world_unlit_shader_id,
        asset_registry,
    ) {
        return ShaderFamilyDetail::WorldUnlit(w);
    }
    ShaderFamilyDetail::None
}

/// Catalog-driven resolver: **must** match [`super::pipeline::resolve_pipeline_for_material_draw_internal`].
///
/// Implemented by delegating to the shared internal function so the catalog path cannot drift.
#[allow(clippy::too_many_arguments)]
pub fn resolve_pipeline_variant_for_material_draw(
    scene: &Scene,
    render_config: &RenderConfig,
    drawable: &Drawable,
    use_pbr: bool,
    is_skinned: bool,
    asset_registry: &AssetRegistry,
    material_block_id: i32,
    fallback_variant: PipelineVariant,
) -> (PipelineVariant, ShaderKey) {
    resolve_pipeline_for_material_draw_internal(
        scene,
        render_config,
        drawable,
        use_pbr,
        is_skinned,
        asset_registry,
        material_block_id,
        fallback_variant,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ShaderDebugOverride;

    #[test]
    fn classify_unknown_without_pilot() {
        let reg = AssetRegistry::new();
        let rc = RenderConfig {
            use_host_unlit_pilot: false,
            ..Default::default()
        };
        assert_eq!(
            classify_shader_pipeline_family(Some(999), 1, &rc, &reg),
            ShaderPipelineFamily::Unknown
        );
    }

    #[test]
    fn classify_native_ui_when_ini_matches() {
        let reg = AssetRegistry::new();
        let rc = RenderConfig {
            native_ui_unlit_shader_id: 42,
            ..Default::default()
        };
        assert_eq!(
            classify_shader_pipeline_family(Some(42), 1, &rc, &reg),
            ShaderPipelineFamily::NativeUiWgsl
        );
    }

    #[test]
    fn catalog_resolver_matches_internal_fn() {
        let scene = Scene {
            id: 0,
            is_overlay: false,
            ..Default::default()
        };
        let reg = AssetRegistry::new();
        let drawable = Drawable::default();
        let rc = RenderConfig::default();
        let fb = PipelineVariant::NormalDebug;
        let a = resolve_pipeline_variant_for_material_draw(
            &scene, &rc, &drawable, true, false, &reg, -1, fb,
        );
        let b = resolve_pipeline_for_material_draw_internal(
            &scene, &rc, &drawable, true, false, &reg, -1, fb,
        );
        assert_eq!(a.0, b.0);
        assert_eq!(a.1, b.1);
    }

    #[test]
    fn classify_unknown_when_no_matching_family() {
        let reg = AssetRegistry::new();
        let rc = RenderConfig::default();
        assert_eq!(
            classify_shader_pipeline_family(Some(777), 3, &rc, &reg),
            ShaderPipelineFamily::Unknown
        );
    }

    #[test]
    fn classify_respects_force_legacy_for_catalog_label() {
        let reg = AssetRegistry::new();
        let rc = RenderConfig {
            use_host_unlit_pilot: true,
            native_ui_unlit_shader_id: 42,
            shader_debug_override: ShaderDebugOverride::ForceLegacyGlobalShading,
            ..Default::default()
        };
        // Shader id matches UI allowlist but global override is legacy — family still NativeUiWgsl
        // (classification is shader-identity based; actual variant uses legacy in resolver).
        assert_eq!(
            classify_shader_pipeline_family(Some(42), 1, &rc, &reg),
            ShaderPipelineFamily::NativeUiWgsl
        );
    }

    #[test]
    fn shader_family_detail_native_ui_unlit_from_ini() {
        use crate::assets::NativeUiShaderFamily;
        let reg = AssetRegistry::new();
        let rc = RenderConfig {
            native_ui_unlit_shader_id: 42,
            native_ui_text_unlit_shader_id: -1,
            ..Default::default()
        };
        assert_eq!(
            shader_family_detail(Some(42), &rc, &reg),
            ShaderFamilyDetail::NativeUi(NativeUiShaderFamily::UiUnlit)
        );
    }

    #[test]
    fn classify_pbs_metallic_native_from_shader_name() {
        let mut reg = AssetRegistry::new();
        reg.handle_shader_upload(crate::shared::ShaderUpload {
            asset_id: 100,
            file: Some("PBSMetallic".to_string()),
        });
        let rc = RenderConfig::default();
        assert_eq!(
            classify_shader_pipeline_family(Some(100), 1, &rc, &reg),
            ShaderPipelineFamily::PbsMetallicNative
        );
    }
}
