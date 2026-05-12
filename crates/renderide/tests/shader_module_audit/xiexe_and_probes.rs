//! Shader source audits for this behavior family.

use super::*;

#[test]
fn xiexe_matcap_uses_stereo_center_view_dir() -> io::Result<()> {
    let globals_src = source_file(manifest_dir().join("shaders/modules/frame/globals.wgsl"))?;
    assert!(
        globals_src.contains("fn stereo_center_view_dir_for_world_pos("),
        "globals.wgsl must expose a stereo-center view direction helper for eye-stable effects"
    );
    assert!(
        globals_src
            .contains("(frame.camera_world_pos.xyz + frame.camera_world_pos_right.xyz) * 0.5"),
        "stereo-center view direction must average the left and right camera positions in multiview"
    );

    let lighting_src =
        source_file(manifest_dir().join("shaders/modules/xiexe/toon2/lighting.wgsl"))?;
    assert!(
        lighting_src.contains(
            "let stereo_view_dir = rg::stereo_center_view_dir_for_world_pos(world_pos, view_layer);"
        ),
        "Xiexe matcap sampling must derive its view direction from the stereo-center camera"
    );
    assert!(
        lighting_src.contains("let uv = matcap_uv(stereo_view_dir, normal);"),
        "Xiexe matcap sampling must use the stereo-center view direction for matcap UVs"
    );
    assert!(
        !lighting_src.contains("let uv = matcap_uv(view_dir, normal);"),
        "Xiexe matcap UVs must not use the per-eye lighting view direction"
    );
    assert!(
        lighting_src.contains("spec = spec * (ambient + dominant_light_col_atten * 0.5);"),
        "Xiexe matcaps must receive the Unity 2.0 light-scaling term (`XSLightingFunctions.cginc:231`)"
    );
    for forbidden in [
        "reflection_is_multiplicative",
        "baked_cubemap_enabled",
        "reflection_disabled",
        "_ReflectionBlendMode",
        "_BakedCubemap",
        "apply_reflection_blend",
        "reflection_blend_weight",
    ] {
        assert!(
            !lighting_src.contains(forbidden),
            "Xiexe lighting must not retain XSToon3 extension `{forbidden}`"
        );
    }
    Ok(())
}

#[test]
fn xiexe_primary_direct_specular_uses_filament_pbr_core() -> io::Result<()> {
    let lighting_src =
        source_file(manifest_dir().join("shaders/modules/xiexe/toon2/lighting.wgsl"))?;

    for required in [
        "fn xiexe_specular_reflectance(s: xb::SurfaceData) -> vec3<f32> {",
        "fn primary_direct_specular_terms(s: xb::SurfaceData, view_dir: vec3<f32>) -> DirectSpecularTerms {",
        "let dfg = brdf::sample_ibl_dfg_lut(roughness, n_dot_v);",
        "let energy_compensation = brdf::energy_compensation_from_dfg(dfg, specular_reflectance);",
        "fn direct_specular_filament(",
        "let alpha = max(perceptual_roughness * perceptual_roughness, brdf::MIN_ALPHA);",
        "let f_term = brdf::f_schlick(specular_reflectance, brdf::f90_from_f0(specular_reflectance), ldh);",
        "var specular = max(vec3<f32>(0.0), d_term * v_term * f_term * energy_compensation);",
        "let radiance = light.color * light.attenuation * ndl;",
        "max(0.0, xb::mat._SpecularIntensity)",
        "xb::mat._SpecularAlbedoTint",
        "clamp(albedo_tint, 0.0, 1.0)",
    ] {
        assert!(
            lighting_src.contains(required),
            "Xiexe primary direct specular must use Filament/PBS term `{required}`"
        );
    }

    for forbidden in [
        "fn direct_specular_xstoon2(",
        "let roughness = 1.0 - smoothness;",
        "exp2((-5.55473 * ldh) - (6.98316 * ldh))",
        "let reflection = v_term * d_term * 3.14159265;",
        "smooth_specular",
        "xb::mat._SpecularIntensity * 0.001",
        "s.specular_mask",
        "clearcoat_direct_specular",
        "clearcoat_roughness",
    ] {
        assert!(
            !lighting_src.contains(forbidden),
            "Xiexe primary direct specular must not contain `{forbidden}`"
        );
    }

    Ok(())
}

#[test]
fn xiexe_pbr_reflections_use_pbs_probe_energy_terms() -> io::Result<()> {
    let lighting_src =
        source_file(manifest_dir().join("shaders/modules/xiexe/toon2/lighting.wgsl"))?;

    for required in [
        "return rprobe::indirect_diffuse(s.normal, view_layer, true);",
        "let indirect_enabled = rprobe::has_indirect_specular(view_layer, xvb::reflection_uses_pbr());",
        "let dfg = brdf::sample_ibl_dfg_lut(roughness, n_dot_v);",
        "let specular_energy = brdf::indirect_specular_energy_from_dfg(dfg, specular_reflectance, indirect_enabled);",
        "let specular_occlusion = brdf::specular_ao_lagarde(n_dot_v, occlusion_scalar(s), roughness);",
        "let spec = rprobe::indirect_specular_with_energy(",
        "spec = mix(spec, spec * dominant_ramp, roughness);",
        "col + reflection * clamp(s.reflectivity_mask, 0.0, 1.0)",
    ] {
        assert!(
            lighting_src.contains(required),
            "Xiexe PBR reflections must contain `{required}`"
        );
    }

    let pbr_branch_pos = lighting_src
        .find("let indirect_enabled = rprobe::has_indirect_specular(view_layer, xvb::reflection_uses_pbr());")
        .expect("Xiexe PBR reflection branch must query probe availability");
    let pbr_return_pos = lighting_src[pbr_branch_pos..]
        .find("return spec;")
        .map(|offset| pbr_branch_pos + offset)
        .expect("Xiexe PBR reflection branch must return its specular result");
    let pbr_branch = &lighting_src[pbr_branch_pos..pbr_return_pos];
    assert!(
        !pbr_branch.contains("raw_indirect_specular"),
        "Xiexe PBR reflection branch must not multiply raw probe radiance by hand-rolled Fresnel"
    );

    Ok(())
}

#[test]
fn reflection_probe_specular_samples_unity_oriented_atlas() -> io::Result<()> {
    let probe_src = module_source("lighting/reflection_probes.wgsl")?;

    for required in [
        "#import renderide::skybox::cubemap_storage as cubemap_storage",
        "const REFLECTION_PROBE_ATLAS_STORAGE_V_INVERTED: f32 = 1.0;",
        "let atlas_sample_dir = cubemap_storage::sample_dir(",
        "REFLECTION_PROBE_ATLAS_STORAGE_V_INVERTED,",
    ] {
        assert!(
            probe_src.contains(required),
            "lighting/reflection_probes.wgsl must contain `{required}`"
        );
    }
    assert!(
        probe_src.contains(
            "rg::reflection_probe_specular_sampler,\n        atlas_sample_dir,\n        i32(atlas_index),"
        ),
        "reflection probe specular sampling must use the Unity-oriented atlas sample direction"
    );
    assert!(
        !probe_src.contains(
            "rg::reflection_probe_specular_sampler,\n        sample_dir,\n        i32(atlas_index),"
        ),
        "reflection probe specular sampling must not use the uncorrected box-projected direction"
    );

    Ok(())
}

#[test]
fn xiexe_generic_stems_resolve_alpha_mode_from_variant_bits() -> io::Result<()> {
    let base_src = source_file(manifest_dir().join("shaders/modules/xiexe/toon2/base.wgsl"))?;
    assert!(
        declares_u32_field(&base_src, "_RenderideVariantBits"),
        "xiexe_toon2_base.wgsl must expose `_RenderideVariantBits` as a u32 field"
    );
    for forbidden in ["Cutout", "AlphaBlend", "Transparent"] {
        assert!(
            !declares_f32_field(&base_src, forbidden),
            "xiexe_toon2_base.wgsl must not retain the legacy `{forbidden}` f32 keyword field"
        );
    }
    assert!(
        !base_src.contains("fn resolved_alpha_mode("),
        "xiexe_toon2_base.wgsl must not retain the legacy `resolved_alpha_mode` helper"
    );

    let variant_bits_src =
        source_file(manifest_dir().join("shaders/modules/xiexe/toon2/variant_bits.wgsl"))?;
    for required in [
        "fn resolved_alpha_mode_from_bits(static_alpha_mode: u32) -> u32",
        "kw_Cutout()",
        "return xb::ALPHA_CUTOUT;",
        "kw_Transparent()",
        "return xb::ALPHA_TRANSPARENT;",
        "kw_AlphaBlend()",
        "return xb::ALPHA_FADE;",
    ] {
        assert!(
            variant_bits_src.contains(required),
            "xiexe_toon2_variant_bits.wgsl must contain `{required}`"
        );
    }

    for file_name in [
        "xstoon2.0.wgsl",
        "xstoon2.0-outlined.wgsl",
        "xstoon2.0_outlined.wgsl",
    ] {
        let src = material_source(file_name)?;
        assert!(
            src.contains("xvb::resolved_alpha_mode_from_bits(XIEE_ALPHA_MODE)"),
            "{file_name} must route the generic Xiexe alpha mode through the variant bitmask"
        );
        assert!(
            !src.contains("xb::resolved_alpha_mode("),
            "{file_name} must not retain the legacy keyword-driven alpha-mode resolver"
        );
    }
    Ok(())
}
