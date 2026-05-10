//! Shader source audits for this behavior family.

use super::*;

#[test]
fn xiexe_matcap_uses_stereo_center_view_dir() -> io::Result<()> {
    let globals_src = fs::read_to_string(manifest_dir().join("shaders/modules/globals.wgsl"))?;
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
        fs::read_to_string(manifest_dir().join("shaders/modules/xiexe_toon2_lighting.wgsl"))?;
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
    let matcap_sample_pos = lighting_src
        .find("let stereo_view_dir = rg::stereo_center_view_dir_for_world_pos(world_pos, view_layer);")
        .expect("Xiexe lighting must contain a matcap sampling branch");
    let non_matcap_branch_pos = lighting_src
        .find("if (xb::baked_cubemap_enabled()) {")
        .expect("Xiexe lighting must contain non-matcap reflection handling");
    assert!(
        matcap_sample_pos < non_matcap_branch_pos,
        "Xiexe matcaps must be sampled before non-matcap reflection branches"
    );
    let matcap_branch = &lighting_src[matcap_sample_pos..non_matcap_branch_pos];
    assert!(
        matcap_branch.contains("spec = spec * (ambient + dominant_light_col_atten * 0.5);"),
        "Xiexe matcaps must always receive the Unity light-scaling term"
    );
    assert!(
        !matcap_branch.contains("reflection_is_multiplicative()"),
        "Xiexe matcap sampling must not branch on `_ReflectionBlendMode`"
    );
    assert!(
        lighting_src.contains(
            "if (xb::matcap_enabled()) {\n        return 1.0;\n    }\n    return clamp(s.reflectivity * s.reflectivity_mask, 0.0, 1.0);"
        ),
        "Xiexe matcaps must not be blended by reflectivity or reflectivity-mask weight"
    );
    assert!(
        lighting_src.contains(
            "if (xb::matcap_enabled()) {\n        return surface + reflection;\n    }\n\n    if (reflection_is_multiplicative()) {"
        ),
        "Xiexe matcap reflections must use additive composition before `_ReflectionBlendMode` branches"
    );
    assert!(
        !lighting_src.contains("return spec * max("),
        "Xiexe matcaps must not be scaled by reflectivity or clearcoat branch strength"
    );
    Ok(())
}

#[test]
fn xiexe_primary_direct_specular_uses_filament_pbr_core() -> io::Result<()> {
    let lighting_src =
        fs::read_to_string(manifest_dir().join("shaders/modules/xiexe_toon2_lighting.wgsl"))?;

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
        "xb::mat._SpecularIntensity * s.specular_mask.r",
        "xb::mat._SpecularArea * s.specular_mask.b",
        "xb::mat._SpecularAlbedoTint * s.specular_mask.g",
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
        fs::read_to_string(manifest_dir().join("shaders/modules/xiexe_toon2_lighting.wgsl"))?;

    for required in [
        "return rprobe::indirect_diffuse(s.normal, view_layer, true);",
        "let indirect_enabled = rprobe::has_indirect_specular(view_layer, xb::reflection_uses_pbr());",
        "let dfg = brdf::sample_ibl_dfg_lut(roughness, n_dot_v);",
        "let specular_energy = brdf::indirect_specular_energy_from_dfg(dfg, specular_reflectance, indirect_enabled);",
        "let specular_occlusion = brdf::specular_ao_lagarde(n_dot_v, occlusion_scalar(s), roughness);",
        "let spec = rprobe::indirect_specular_with_energy(",
        "return clamp(s.reflectivity * s.reflectivity_mask, 0.0, 1.0);",
    ] {
        assert!(
            lighting_src.contains(required),
            "Xiexe PBR reflections must contain `{required}`"
        );
    }

    let pbr_branch_pos = lighting_src
        .find("let indirect_enabled = rprobe::has_indirect_specular(view_layer, xb::reflection_uses_pbr());")
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
    let probe_src = module_source("reflection_probes.wgsl")?;

    for required in [
        "#import renderide::cubemap_storage as cubemap_storage",
        "const REFLECTION_PROBE_ATLAS_STORAGE_V_INVERTED: f32 = 1.0;",
        "let atlas_sample_dir = cubemap_storage::sample_dir(",
        "REFLECTION_PROBE_ATLAS_STORAGE_V_INVERTED,",
    ] {
        assert!(
            probe_src.contains(required),
            "reflection_probes.wgsl must contain `{required}`"
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
fn xiexe_generic_stems_resolve_alpha_mode_from_keywords() -> io::Result<()> {
    let base_src =
        fs::read_to_string(manifest_dir().join("shaders/modules/xiexe_toon2_base.wgsl"))?;
    for field_name in ["Cutout", "AlphaBlend", "Transparent"] {
        assert!(
            declares_f32_field(&base_src, field_name),
            "xiexe_toon2_base.wgsl must expose `{field_name}` as an f32 keyword field"
        );
    }
    for required in [
        "fn resolved_alpha_mode(static_alpha_mode: u32) -> u32",
        "kw(mat.Cutout)",
        "return ALPHA_CUTOUT;",
        "kw(mat.Transparent)",
        "return ALPHA_TRANSPARENT;",
        "kw(mat.AlphaBlend)",
        "return ALPHA_FADE;",
    ] {
        assert!(
            base_src.contains(required),
            "xiexe_toon2_base.wgsl must contain `{required}`"
        );
    }

    for file_name in [
        "xstoon2.0.wgsl",
        "xstoon2.0-outlined.wgsl",
        "xstoon2.0_outlined.wgsl",
    ] {
        let src = material_source(file_name)?;
        assert!(
            src.contains("xb::resolved_alpha_mode(XIEE_ALPHA_MODE)"),
            "{file_name} must route the generic Xiexe alpha mode through material keywords"
        );
    }
    Ok(())
}
