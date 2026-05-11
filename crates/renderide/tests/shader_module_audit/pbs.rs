//! Shader source audits for this behavior family.

use super::*;

#[test]
fn direct_light_boost_reaches_directional_and_punctual_paths() -> io::Result<()> {
    let birp = module_source("lighting/birp.wgsl")?;
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
    assert!(
        birp.contains("fn spot_angle_attenuation(light: ft::GpuLight, l: vec3<f32>) -> f32"),
        "BiRP light module must expose the shared spot angle attenuation helper"
    );
    assert!(
        birp.contains("let tan2_theta = max(1.0 - rho2, 0.0) / rho2;")
            && birp.contains("let r2 = clamp(tan2_theta * light.spot_angle_scale, 0.0, 1.0);")
            && birp.contains("return quartic_edge_fade_from_t4(r2 * r2);"),
        "spot angle attenuation must use projected radial quartic falloff"
    );

    let pbs_brdf = module_source("pbs/brdf.wgsl")?;
    assert!(
        pbs_brdf.contains("out.attenuation = bl::direct_light_intensity(light.intensity);"),
        "PBS directional lights must use the shared intensity boost"
    );
    assert!(
        pbs_brdf.contains("out.attenuation = light.intensity * distance_attenuation(dist, light.range);")
            && pbs_brdf.contains("let spot_atten = bl::spot_angle_attenuation(light, out.l);")
            && pbs_brdf.contains(
                "out.attenuation = light.intensity * spot_atten * distance_attenuation(dist, light.range);"
            ),
        "PBS point and spot lights must continue using boosted distance attenuation"
    );

    let xiexe = module_source("xiexe/toon2/lighting.wgsl")?;
    assert!(
        xiexe.contains("bl::direct_light_intensity(light.intensity),"),
        "Xiexe directional lights must use the shared intensity boost"
    );
    assert!(
        xiexe.contains(
            "var attenuation = bl::punctual_attenuation(light.intensity, dist, light.range);"
        ) && xiexe.contains("attenuation = attenuation * bl::spot_angle_attenuation(light, l);"),
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
            ) && src.contains("attenuation = attenuation * bl::spot_angle_attenuation(light, l);"),
            "{material} point and spot lights must continue using boosted distance attenuation"
        );
    }

    Ok(())
}

#[test]
fn spot_lights_do_not_use_arbitrary_smoothstep_cone_fade() -> io::Result<()> {
    for (name, src) in [
        ("pbs/brdf.wgsl", module_source("pbs/brdf.wgsl")?),
        (
            "xiexe/toon2/lighting.wgsl",
            module_source("xiexe/toon2/lighting.wgsl")?,
        ),
        ("toonstandard.wgsl", material_source("toonstandard.wgsl")?),
        ("toonwater.wgsl", material_source("toonwater.wgsl")?),
    ] {
        assert!(
            !src.contains("spot_cos_half_angle + 0.1")
                && !src.contains("smoothstep(light.spot_cos_half_angle"),
            "{name} must route spot cone fade through the shared BiRP helper"
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
fn pbs_indirect_specular_energy_respects_zero_f0() -> io::Result<()> {
    let brdf = module_source("pbs/brdf.wgsl")?;

    for required in [
        "let f90 = vec3<f32>(f90_from_f0(clamped_f0));",
        "return clamped_f0 * (dfg.y - dfg.x) + f90 * dfg.x;",
    ] {
        assert!(
            brdf.contains(required),
            "pbs/brdf.wgsl must contain `{required}`"
        );
    }

    assert!(
        !brdf.contains(
            "return mix(vec3<f32>(dfg.x), vec3<f32>(dfg.y), clamp(f0, vec3<f32>(0.0), vec3<f32>(1.0)));"
        ),
        "pbs/brdf.wgsl must not use the old implicit f90=1 DFG formula"
    );
    Ok(())
}

#[test]
fn pbs_roughness_keeps_indirect_mirror_path_unclamped() -> io::Result<()> {
    let sampling_src = source_file(manifest_dir().join("shaders/modules/pbs/sampling.wgsl"))?;
    assert!(
        sampling_src.contains("return clamp(1.0 - smoothness, 0.0, 1.0);"),
        "PBS smoothness conversion must keep perceptual roughness at 0 for mirror-smooth indirect reflections"
    );
    assert!(
        !sampling_src.contains("return clamp(1.0 - smoothness, 0.045, 1.0);"),
        "PBS smoothness conversion must not apply the direct-light roughness floor globally"
    );

    let brdf_src = source_file(manifest_dir().join("shaders/modules/pbs/brdf.wgsl"))?;
    for required in [
        "const MIN_ALPHA: f32 = 0.002;",
        "fn direct_alpha_from_perceptual_roughness(",
        "return max(clamped * clamped, MIN_ALPHA);",
        "fn direct_perceptual_roughness(",
    ] {
        assert!(
            brdf_src.contains(required),
            "pbs/brdf.wgsl must contain `{required}`"
        );
    }

    let lighting_src = source_file(manifest_dir().join("shaders/modules/pbs/lighting.wgsl"))?;
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
        let src = source_file(&path)?;
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

/// Material roots using the shared PBS lighting module should not also carry their own clustered loop.
#[test]
fn shared_pbs_lighting_roots_do_not_duplicate_clustered_lighting() -> io::Result<()> {
    let mut offenders = Vec::new();
    for path in wgsl_files_recursive("shaders/materials")? {
        let src = source_file(&path)?;
        if !src.contains("renderide::pbs::lighting") {
            continue;
        }

        for forbidden in [
            "#import renderide::ibl::sh2_ambient",
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
    let normal_src = source_file(manifest_dir().join("shaders/modules/pbs/normal.wgsl"))?;
    for required in [
        "fn visible_side_tbn(",
        "select(-1.0, 1.0, front_facing)",
        "tbn[0] * side",
        "tbn[1] * side",
        "tbn[2] * side",
    ] {
        assert!(
            normal_src.contains(required),
            "pbs/normal.wgsl must define visible-side TBN helper containing `{required}`"
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
    let module_src = source_file(manifest_dir().join("shaders/modules/pbs/parallax.wgsl"))?;
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

    // `pbsrimspecular.wgsl` decodes Unity keywords from `_RenderideVariantBits` instead of
    // f32 keyword fields; its parity assertions live in `pbs_variant_bits.rs`.
    let cases = [Case {
        file_name: "pbsrim.wgsl",
        workflow_keyword: "_METALLICMAP",
        workflow_texture: "_MetallicMap",
        forbidden_unity_fallback_mul: &["mat._Metallic * mg", "mat._Glossiness * mg"],
    }];

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

/// Standard PBS-like roots should delegate clustered Standard lighting to the shared PBS module.
#[test]
fn standard_material_roots_do_not_duplicate_clustered_pbs_lighting() -> io::Result<()> {
    let mut offenders = Vec::new();
    for path in wgsl_files_recursive("shaders/materials")? {
        let label = file_label(&path);
        let src = source_file(&path)?;
        if label == "shaders/materials/toonstandard.wgsl"
            || label == "shaders/materials/toonwater.wgsl"
        {
            continue;
        }

        for forbidden in [
            "#import renderide::ibl::sh2_ambient",
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
