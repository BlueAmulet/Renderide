//! Source audits for WGSL module factoring invariants.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// Returns the renderide crate directory.
fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Recursively returns all WGSL files below `relative_dir`.
fn wgsl_files_recursive(relative_dir: &str) -> io::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    collect_wgsl_files(&manifest_dir().join(relative_dir), &mut out)?;
    out.sort();
    Ok(out)
}

fn collect_wgsl_files(dir: &Path, out: &mut Vec<PathBuf>) -> io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            collect_wgsl_files(&path, out)?;
        } else if path.extension().is_some_and(|ext| ext == "wgsl") {
            out.push(path);
        }
    }
    Ok(())
}

fn file_label(path: &Path) -> String {
    normalize_file_label(
        path.strip_prefix(manifest_dir())
            .unwrap_or(path)
            .display()
            .to_string(),
    )
}

fn normalize_file_label(label: impl AsRef<str>) -> String {
    label.as_ref().replace('\\', "/")
}

fn define_import_path(src: &str) -> Option<&str> {
    src.lines().find_map(|line| {
        line.trim_start()
            .strip_prefix("#define_import_path")
            .map(str::trim)
            .filter(|path| !path.is_empty())
    })
}

fn material_source(file_name: &str) -> io::Result<String> {
    fs::read_to_string(manifest_dir().join("shaders/materials").join(file_name))
}

fn module_source(file_name: &str) -> io::Result<String> {
    fs::read_to_string(manifest_dir().join("shaders/modules").join(file_name))
}

#[test]
fn direct_light_boost_reaches_directional_and_punctual_paths() -> io::Result<()> {
    let birp = module_source("birp_light.wgsl")?;
    assert!(
        birp.contains(
            "fn direct_light_intensity(intensity: f32) -> f32 {\n    return intensity * INTENSITY_BOOST;\n}"
        ),
        "BiRP light module must expose the shared direct-light boost helper"
    );
    assert!(
        birp.contains("return lut * range_fade(t) * INTENSITY_BOOST;"),
        "punctual distance attenuation must keep the existing intensity boost"
    );

    let pbs_brdf = module_source("pbs_brdf.wgsl")?;
    assert!(
        pbs_brdf.contains("out.attenuation = bl::direct_light_intensity(light.intensity);"),
        "PBS directional lights must use the shared intensity boost"
    );
    assert!(
        pbs_brdf.contains("out.attenuation = light.intensity * distance_attenuation(dist, light.range);")
            && pbs_brdf.contains(
                "out.attenuation = light.intensity * spot_atten * distance_attenuation(dist, light.range);"
            ),
        "PBS point and spot lights must continue using boosted distance attenuation"
    );

    let xiexe = module_source("xiexe_toon2_lighting.wgsl")?;
    assert!(
        xiexe.contains("bl::direct_light_intensity(light.intensity),"),
        "Xiexe directional lights must use the shared intensity boost"
    );
    assert!(
        xiexe.contains(
            "var attenuation = bl::punctual_attenuation(light.intensity, dist, light.range);"
        ),
        "Xiexe point and spot lights must continue using boosted punctual attenuation"
    );

    for material in ["toonstandard.wgsl", "toonwater.wgsl"] {
        let src = material_source(material)?;
        assert!(
            src.contains("attenuation = bl::direct_light_intensity(light.intensity);"),
            "{material} directional lights must use the shared intensity boost"
        );
        assert!(
            src.contains(
                "attenuation = light.intensity * brdf::distance_attenuation(dist, light.range);"
            ),
            "{material} point and spot lights must continue using boosted distance attenuation"
        );
    }

    Ok(())
}

#[test]
fn standard_pbs_roots_use_unity_standard_packed_channels() -> io::Result<()> {
    let metallic = material_source("pbsmetallic.wgsl")?;
    for required in [
        "_GlossMapScale: f32",
        "_OcclusionStrength: f32",
        "smoothness = mg.a * mat._GlossMapScale;",
        "smoothness = albedo_sample.a * mat._GlossMapScale;",
        "ts::sample_tex_2d(_OcclusionMap, _OcclusionMap_sampler, uv_main, mat._OcclusionMap_LodBias).g",
        "mix(1.0, occlusion_sample, clamp(mat._OcclusionStrength, 0.0, 1.0))",
    ] {
        assert!(
            metallic.contains(required),
            "pbsmetallic.wgsl must contain `{required}`"
        );
    }

    let specular = material_source("pbsspecular.wgsl")?;
    assert!(
        specular.contains(
            "ts::sample_tex_2d(_OcclusionMap, _OcclusionMap_sampler, uv_main, mat._OcclusionMap_LodBias).g"
        ),
        "pbsspecular.wgsl must sample Unity Standard occlusion from the green channel"
    );

    Ok(())
}

#[test]
fn pbs_roughness_keeps_indirect_mirror_path_unclamped() -> io::Result<()> {
    let sampling_src =
        fs::read_to_string(manifest_dir().join("shaders/modules/pbs/sampling.wgsl"))?;
    assert!(
        sampling_src.contains("return clamp(1.0 - smoothness, 0.0, 1.0);"),
        "PBS smoothness conversion must keep perceptual roughness at 0 for mirror-smooth indirect reflections"
    );
    assert!(
        !sampling_src.contains("return clamp(1.0 - smoothness, 0.045, 1.0);"),
        "PBS smoothness conversion must not apply the direct-light roughness floor globally"
    );

    let brdf_src = fs::read_to_string(manifest_dir().join("shaders/modules/pbs_brdf.wgsl"))?;
    for required in [
        "const MIN_ALPHA: f32 = 0.002;",
        "fn direct_alpha_from_perceptual_roughness(",
        "return max(clamped * clamped, MIN_ALPHA);",
        "fn direct_perceptual_roughness(",
    ] {
        assert!(
            brdf_src.contains(required),
            "pbs_brdf.wgsl must contain `{required}`"
        );
    }

    let lighting_src =
        fs::read_to_string(manifest_dir().join("shaders/modules/pbs/lighting.wgsl"))?;
    for required in [
        "let direct_roughness = brdf::direct_perceptual_roughness(s.roughness);",
        "let direct_dfg = brdf::sample_ibl_dfg_lut(direct_roughness, n_dot_v);",
        "let indirect_dfg = brdf::sample_ibl_dfg_lut(s.roughness, n_dot_v);",
    ] {
        assert!(
            lighting_src.contains(required),
            "pbs/lighting.wgsl must contain `{required}`"
        );
    }

    for path in wgsl_files_recursive("shaders/materials")? {
        let src = fs::read_to_string(&path)?;
        for forbidden in [
            "clamp(1.0 - smoothness, 0.045, 1.0)",
            "clamp(1.0 - clamp(smoothness, 0.0, 1.0), 0.045, 1.0)",
        ] {
            assert!(
                !src.contains(forbidden),
                "{} must not contain the global PBS roughness floor `{forbidden}`",
                file_label(&path)
            );
        }
    }

    Ok(())
}

#[test]
fn pbs_lerp_preserves_variant_channels_and_raw_lerp() -> io::Result<()> {
    let metallic = material_source("pbslerp.wgsl")?;
    for required in [
        "return l;",
        "occlusion0 = textureSample(_Occlusion, _Occlusion_sampler, uv_main0).r;",
        "occlusion1 = textureSample(_Occlusion1, _Occlusion1_sampler, uv_main1).r;",
        "metallic0 = m0.r;",
        "metallic1 = m1.r;",
        "smoothness0 = m0.a;",
        "smoothness1 = m1.a;",
    ] {
        assert!(
            metallic.contains(required),
            "pbslerp.wgsl must contain `{required}`"
        );
    }
    assert!(
        !metallic.contains("return clamp(l, 0.0, 1.0);"),
        "pbslerp.wgsl must use Unity's raw lerp factor"
    );

    let specular = material_source("pbslerpspecular.wgsl")?;
    for required in [
        "return l;",
        "occlusion0 = textureSample(_Occlusion, _Occlusion_sampler, uv_main0).r;",
        "occlusion1 = textureSample(_Occlusion1, _Occlusion1_sampler, uv_main1).r;",
        "spec0 = textureSample(_SpecularMap, _SpecularMap_sampler, uv_main0);",
        "spec1 = textureSample(_SpecularMap1, _SpecularMap1_sampler, uv_main1);",
    ] {
        assert!(
            specular.contains(required),
            "pbslerpspecular.wgsl must contain `{required}`"
        );
    }
    assert!(
        !specular.contains("return clamp(l, 0.0, 1.0);"),
        "pbslerpspecular.wgsl must use Unity's raw lerp factor"
    );

    Ok(())
}

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

fn declares_f32_field(src: &str, field_name: &str) -> bool {
    src.lines().any(|line| {
        let trimmed = line.trim();
        let Some((name, ty)) = trimmed.split_once(':') else {
            return false;
        };
        name.trim() == field_name && ty.trim_start().starts_with("f32")
    })
}

fn all_texture_samples_guarded_by_keyword(src: &str, texture_name: &str, keyword: &str) -> bool {
    let sample = format!("textureSample({texture_name},");
    let guard = format!("uvu::kw_enabled(mat.{keyword})");
    let mut saw_sample = false;

    for (sample_pos, _) in src.match_indices(&sample) {
        saw_sample = true;
        let before_sample = &src[..sample_pos];
        let Some(guard_pos) = before_sample.rfind(&guard) else {
            return false;
        };
        if before_sample[guard_pos..].contains('}') {
            return false;
        }
    }

    saw_sample
}

fn normal_sampling_guarded_by_keyword(src: &str) -> bool {
    let Some(call_pos) = src.find("sample_optional_world_normal(") else {
        return false;
    };
    let call = &src[call_pos..];
    let Some(call_end) = call.find(");") else {
        return false;
    };
    call[..call_end].contains("uvu::kw_enabled(mat._NORMALMAP)")
}

fn count_font_atlas_lod_bias_samples(src: &str) -> usize {
    src.match_indices("ts::sample_tex_2d(")
        .filter(|(sample_pos, _)| {
            let call = &src[*sample_pos..];
            let call_end = call.find(");").unwrap_or(call.len());
            call[..call_end].contains("_FontAtlas")
        })
        .count()
}

#[test]
fn text_shaders_use_one_font_atlas_sample_for_coverage() -> io::Result<()> {
    for file_name in ["ui_textunlit.wgsl", "textunlit.wgsl", "textunit.wgsl"] {
        let src = material_source(file_name)?;
        assert!(
            src.contains("#import renderide::texture_sampling as ts"),
            "{file_name} must import biased texture sampling for _FontAtlas"
        );
        assert!(
            declares_f32_field(&src, "_FontAtlas_LodBias"),
            "{file_name} must expose _FontAtlas_LodBias in the material uniform"
        );
        assert_eq!(
            count_font_atlas_lod_bias_samples(&src),
            1,
            "{file_name} must sample _FontAtlas exactly once through the LOD-bias helper"
        );
        assert!(
            !src.contains("textureSample(_FontAtlas")
                && !src.contains("textureSampleLevel(_FontAtlas"),
            "{file_name} must not directly sample _FontAtlas outside the shared helper"
        );
        assert!(
            !src.contains("texture_rgba_base_mip(_FontAtlas"),
            "{file_name} must not force base-mip atlas sampling for text coverage"
        );
        assert!(
            !src.contains("atlas_clip"),
            "{file_name} must route text coverage through the same atlas sample as color"
        );
    }

    let module_src = fs::read_to_string(manifest_dir().join("shaders/modules/text_sdf.wgsl"))?;
    assert!(
        !module_src.contains("atlas_clip"),
        "text_sdf.wgsl must not expose a second atlas sample for coverage"
    );
    Ok(())
}

#[test]
fn text_shaders_route_font_extra_data_through_normal_stream() -> io::Result<()> {
    for file_name in ["ui_textunlit.wgsl", "textunlit.wgsl", "textunit.wgsl"] {
        let src = material_source(file_name)?;
        assert!(
            src.contains("@location(1) extra_n: vec4<f32>"),
            "{file_name} must read glyph extra data from the normal stream"
        );
        assert!(
            src.contains("@location(2) uv: vec2<f32>"),
            "{file_name} must keep atlas UVs on vertex location 2"
        );
        assert!(
            src.contains("@location(3) color: vec4<f32>"),
            "{file_name} must keep vertex tint on vertex location 3"
        );
        assert!(
            src.contains("out.extra_data = extra_n;"),
            "{file_name} must pass glyph extra data through to the fragment shader"
        );
    }
    Ok(())
}

#[test]
fn file_labels_use_forward_slashes_for_cross_platform_audits() {
    assert_eq!(
        normalize_file_label(r"shaders\modules\per_draw.wgsl"),
        "shaders/modules/per_draw.wgsl"
    );
}

/// Nested WGSL modules must remain discoverable and uniquely addressable by naga-oil.
#[test]
fn shader_modules_have_unique_import_paths() -> io::Result<()> {
    let mut seen: Vec<(String, String)> = Vec::new();
    let mut offenders = Vec::new();

    for path in wgsl_files_recursive("shaders/modules")? {
        let src = fs::read_to_string(&path)?;
        let Some(import_path) = define_import_path(&src) else {
            offenders.push(format!("{} has no #define_import_path", file_label(&path)));
            continue;
        };
        if let Some((_, first_path)) = seen
            .iter()
            .find(|(seen_import_path, _)| seen_import_path == import_path)
        {
            offenders.push(format!(
                "{} duplicates import path {import_path} from {first_path}",
                file_label(&path)
            ));
        }
        seen.push((import_path.to_string(), file_label(&path)));
    }

    assert!(
        offenders.is_empty(),
        "shader module import paths must be present and unique:\n  {}",
        offenders.join("\n  ")
    );
    Ok(())
}

/// Material roots using the shared PBS lighting module should not also carry their own clustered loop.
#[test]
fn shared_pbs_lighting_roots_do_not_duplicate_clustered_lighting() -> io::Result<()> {
    let mut offenders = Vec::new();
    for path in wgsl_files_recursive("shaders/materials")? {
        let src = fs::read_to_string(&path)?;
        if !src.contains("renderide::pbs::lighting") {
            continue;
        }

        for forbidden in [
            "#import renderide::sh2_ambient",
            "#import renderide::pbs::brdf",
            "#import renderide::pbs::cluster",
            "fn clustered_direct_lighting",
            "pcls::cluster_id_from_frag",
        ] {
            if src.contains(forbidden) {
                offenders.push(format!(
                    "{} still contains `{forbidden}`",
                    file_label(&path)
                ));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "materials importing renderide::pbs::lighting must delegate clustered PBS lighting:\n  {}",
        offenders.join("\n  ")
    );
    Ok(())
}

/// PBS DualSided materials should shade backfaces as a second opaque surface, not as light leaking
/// through the front side. The visible-side frame must be shared so the metallic/specular variants
/// cannot drift back to tangent-Z or world-Z flips.
#[test]
fn pbs_dualsided_shaders_use_visible_side_tbn_for_backfaces() -> io::Result<()> {
    let normal_src = fs::read_to_string(manifest_dir().join("shaders/modules/pbs_normal.wgsl"))?;
    for required in [
        "fn visible_side_tbn(",
        "select(-1.0, 1.0, front_facing)",
        "tbn[0] * side",
        "tbn[1] * side",
        "tbn[2] * side",
    ] {
        assert!(
            normal_src.contains(required),
            "pbs_normal.wgsl must define visible-side TBN helper containing `{required}`"
        );
    }

    let mut offenders = Vec::new();
    for file_name in [
        "pbsdualsided.wgsl",
        "pbsdualsidedspecular.wgsl",
        "pbsdualsidedtransparent.wgsl",
        "pbsdualsidedtransparentspecular.wgsl",
    ] {
        let src = material_source(file_name)?;
        for required in ["@builtin(front_facing)", "pnorm::visible_side_tbn("] {
            if !src.contains(required) {
                offenders.push(format!("{file_name} must contain `{required}`"));
            }
        }
        for forbidden in [
            "ts_n.z = -ts_n.z",
            "ts.z = -ts.z",
            "sample_optional_world_normal(",
        ] {
            if src.contains(forbidden) {
                offenders.push(format!("{file_name} must not contain `{forbidden}`"));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "PBS DualSided shaders must orient normals through the shared visible-side TBN:\n  {}",
        offenders.join("\n  ")
    );
    Ok(())
}

/// Standard PBS parallax must project the view vector into the material's tangent frame before
/// offsetting UVs so height maps behave consistently as lighting and camera state become active.
#[test]
fn pbs_standard_parallax_uses_tangent_space_view_dir() -> io::Result<()> {
    let module_src = fs::read_to_string(manifest_dir().join("shaders/modules/pbs/parallax.wgsl"))?;
    for required in [
        "#define_import_path renderide::pbs::parallax",
        "rg::view_dir_for_world_pos(world_pos, view_layer)",
        "pnorm::orthonormal_tbn(world_n, world_t)",
        "dot(world_view, tbn[0])",
        "dot(world_view, tbn[1])",
        "dot(world_view, tbn[2])",
        "UNITY_PARALLAX_VIEW_Z_BIAS: f32 = 0.42",
        "height_sample * height_scale - height_scale * 0.5",
    ] {
        assert!(
            module_src.contains(required),
            "parallax module should contain `{required}`"
        );
    }

    for file_name in ["pbsmetallic.wgsl", "pbsspecular.wgsl"] {
        let src = material_source(file_name)?;
        assert!(
            src.contains("#import renderide::pbs::parallax as ppar"),
            "{file_name} should use the shared parallax helper"
        );
        assert!(
            src.contains(
                "ts::sample_tex_2d(_ParallaxMap, _ParallaxMap_sampler, uv, mat._ParallaxMap_LodBias).g"
            ),
            "{file_name} should sample Unity Standard parallax height from the green channel"
        );
        assert!(
            src.contains(
                "ppar::unity_parallax_offset(h, mat._Parallax, world_pos, world_n, world_t, view_layer)"
            ),
            "{file_name} should offset parallax UVs from tangent-space view direction"
        );
        assert!(
            src.contains("uv_with_parallax(uv_base, world_pos, world_n, world_t, view_layer)"),
            "{file_name} should pass the surface frame into parallax sampling"
        );

        for forbidden in [
            "view_dir.xy / max(abs(view_dir.z), 0.25)",
            "rg::view_dir_for_world_pos(world_pos, view_layer)",
        ] {
            assert!(
                !src.contains(forbidden),
                "{file_name} should not contain the old world-space parallax expression `{forbidden}`"
            );
        }
    }

    Ok(())
}

/// PBS rim variants mirror Unity's optional-texture keywords: maps only override fallback material
/// values when their corresponding multi-compile keyword is active.
#[test]
fn pbs_rim_shaders_preserve_unity_texture_keywords() -> io::Result<()> {
    struct Case<'a> {
        file_name: &'a str,
        workflow_keyword: &'a str,
        workflow_texture: &'a str,
        forbidden_unity_fallback_mul: &'a [&'a str],
    }

    let cases = [
        Case {
            file_name: "pbsrim.wgsl",
            workflow_keyword: "_METALLICMAP",
            workflow_texture: "_MetallicMap",
            forbidden_unity_fallback_mul: &["mat._Metallic * mg", "mat._Glossiness * mg"],
        },
        Case {
            file_name: "pbsrimspecular.wgsl",
            workflow_keyword: "_SPECULARMAP",
            workflow_texture: "_SpecularMap",
            forbidden_unity_fallback_mul: &["mat._SpecularColor * spec_s"],
        },
    ];

    for case in cases {
        let src = material_source(case.file_name)?;
        for keyword in [
            "_ALBEDOTEX",
            "_EMISSIONTEX",
            "_NORMALMAP",
            case.workflow_keyword,
            "_OCCLUSION",
        ] {
            assert!(
                declares_f32_field(&src, keyword),
                "{} must declare Unity keyword uniform {keyword}",
                case.file_name
            );
        }

        for (texture, keyword) in [
            ("_MainTex", "_ALBEDOTEX"),
            ("_EmissionMap", "_EMISSIONTEX"),
            (case.workflow_texture, case.workflow_keyword),
            ("_OcclusionMap", "_OCCLUSION"),
        ] {
            assert!(
                all_texture_samples_guarded_by_keyword(&src, texture, keyword),
                "{} must sample {texture} only when {keyword} is enabled",
                case.file_name
            );
        }

        assert!(
            normal_sampling_guarded_by_keyword(&src),
            "{} must sample _NormalMap through the _NORMALMAP keyword",
            case.file_name
        );

        for forbidden in case.forbidden_unity_fallback_mul {
            assert!(
                !src.contains(forbidden),
                "{} must not multiply workflow maps by fallback material values (`{forbidden}`)",
                case.file_name
            );
        }
    }

    Ok(())
}

/// Material roots should route per-draw view-projection selection through `renderide::mesh::vertex`.
#[test]
fn material_roots_do_not_duplicate_view_projection_selection() -> io::Result<()> {
    let mut offenders = Vec::new();
    for path in wgsl_files_recursive("shaders/materials")? {
        let src = fs::read_to_string(&path)?;
        for forbidden in ["view_proj_left", "view_proj_right"] {
            if src.contains(forbidden) {
                offenders.push(format!(
                    "{} still contains `{forbidden}`",
                    file_label(&path)
                ));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "materials importing renderide::mesh::vertex must delegate view-projection selection:\n  {}",
        offenders.join("\n  ")
    );
    Ok(())
}

/// Shared modules should centralize raw per-draw matrix field access in the mesh vertex module.
#[test]
fn shader_modules_centralize_view_projection_selection() -> io::Result<()> {
    let mut offenders = Vec::new();
    for path in wgsl_files_recursive("shaders/modules")? {
        let label = file_label(&path);
        if matches!(
            label.as_str(),
            "shaders/modules/per_draw.wgsl" | "shaders/modules/mesh/vertex.wgsl"
        ) {
            continue;
        }

        let src = fs::read_to_string(&path)?;
        for forbidden in ["view_proj_left", "view_proj_right"] {
            if src.contains(forbidden) {
                offenders.push(format!("{label} still contains `{forbidden}`"));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "only per_draw and mesh::vertex should touch raw view-projection fields:\n  {}",
        offenders.join("\n  ")
    );
    Ok(())
}

/// Standard PBS-like roots should delegate clustered Standard lighting to the shared PBS module.
#[test]
fn standard_material_roots_do_not_duplicate_clustered_pbs_lighting() -> io::Result<()> {
    let mut offenders = Vec::new();
    for path in wgsl_files_recursive("shaders/materials")? {
        let label = file_label(&path);
        let src = fs::read_to_string(&path)?;
        if label == "shaders/materials/toonstandard.wgsl"
            || label == "shaders/materials/toonwater.wgsl"
        {
            continue;
        }

        for forbidden in [
            "#import renderide::sh2_ambient",
            "#import renderide::pbs::cluster",
            "cluster_id_from_frag",
            "direct_radiance_metallic",
            "direct_radiance_specular",
            "indirect_diffuse_metallic",
            "indirect_diffuse_specular",
            "indirect_specular",
        ] {
            if src.contains(forbidden) {
                offenders.push(format!("{label} still contains `{forbidden}`"));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "standard PBS-like material roots must delegate clustered PBS lighting:\n  {}",
        offenders.join("\n  ")
    );
    Ok(())
}

/// Material roots should consume shared helper modules instead of reintroducing helper copies.
#[test]
fn material_roots_do_not_redeclare_shared_helpers() -> io::Result<()> {
    let mut offenders = Vec::new();
    for path in wgsl_files_recursive("shaders/materials")? {
        let label = file_label(&path);
        let src = fs::read_to_string(&path)?;

        for forbidden in [
            "fn alpha_over",
            "fn inside_rect",
            "fn outside_rect",
            "fn roughness_from_smoothness",
            "fn safe_normalize",
            "fn shade_distance_field",
            "fn view_angle_fresnel",
        ] {
            if src.contains(forbidden) {
                offenders.push(format!("{label} still contains `{forbidden}`"));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "material roots should import shared helper modules instead of redeclaring them:\n  {}",
        offenders.join("\n  ")
    );
    Ok(())
}

#[test]
fn normal_decode_scales_xy_before_reconstructing_z() -> io::Result<()> {
    let src = fs::read_to_string(manifest_dir().join("shaders/modules/normal_decode.wgsl"))?;
    let xy_scale = src
        .find("let xy = (raw.xy * 2.0 - 1.0) * scale;")
        .expect("normal decode must scale tangent XY before Z reconstruction");
    let z_reconstruct = src
        .find("let z = sqrt(max(1.0 - dot(xy, xy), 0.0));")
        .expect("normal decode must reconstruct Z from the scaled XY vector");

    assert!(
        xy_scale < z_reconstruct,
        "normal decode must apply `_NormalScale` / `_BumpScale` before reconstructing Z"
    );
    assert!(
        !src.contains("scale-after-Z"),
        "normal decode comments must not describe the old scale-after-Z behavior"
    );
    Ok(())
}

/// Pass shaders using the fullscreen module should not duplicate fullscreen-triangle bit math.
#[test]
fn shared_fullscreen_roots_do_not_duplicate_fullscreen_triangle_setup() -> io::Result<()> {
    let mut offenders = Vec::new();
    for path in wgsl_files_recursive("shaders/passes")? {
        let src = fs::read_to_string(&path)?;
        if !src.contains("renderide::fullscreen") {
            continue;
        }

        for forbidden in ["<< 1u", "vec2(-1.0, -1.0)", "vec2(3.0, -1.0)"] {
            if src.contains(forbidden) {
                offenders.push(format!(
                    "{} still contains `{forbidden}`",
                    file_label(&path)
                ));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "passes importing renderide::fullscreen must delegate fullscreen-triangle setup:\n  {}",
        offenders.join("\n  ")
    );
    Ok(())
}
