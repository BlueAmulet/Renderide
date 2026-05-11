//! Per-shader audits locking in the `_RenderideVariantBits` migration for the PBS material family.

use super::*;

/// Asserts that `material` declares `_RenderideVariantBits: u32`, omits the listed legacy
/// f32 keyword fields, defines each bit constant at the expected index, and contains the
/// `vb::enabled(mat._RenderideVariantBits` decode call.
fn assert_variant_bits_migration(
    file_name: &str,
    legacy_kw_fields: &[&str],
    bits: &[(&str, u32)],
) -> io::Result<()> {
    let src = material_source(file_name)?;
    assert!(
        src.contains("_RenderideVariantBits: u32"),
        "{file_name} must declare _RenderideVariantBits: u32"
    );
    assert!(
        src.contains("#import renderide::material::variant_bits as vb"),
        "{file_name} must import renderide::material::variant_bits"
    );
    assert!(
        src.contains("vb::enabled(mat._RenderideVariantBits"),
        "{file_name} must decode keywords through vb::enabled(mat._RenderideVariantBits, ...)"
    );
    for kw in legacy_kw_fields {
        assert!(
            !declares_f32_field(&src, kw),
            "{file_name} must not declare legacy f32 keyword field {kw}; \
             decode it from _RenderideVariantBits instead"
        );
    }
    for (constant_name, bit_index) in bits {
        let needle = format!("const {constant_name}: u32 = 1u << {bit_index}u;");
        assert!(
            src.contains(&needle),
            "{file_name} must define `{needle}` (Froox sorted UniqueKeywords bit order)"
        );
    }
    Ok(())
}

#[test]
fn pbsdualsided_decodes_keywords_from_variant_bits() -> io::Result<()> {
    assert_variant_bits_migration(
        "pbsdualsided.wgsl",
        &[
            "_ALBEDOTEX",
            "_ALPHACLIP",
            "_EMISSIONTEX",
            "_METALLICMAP",
            "_NORMALMAP",
            "_OCCLUSION",
            "VCOLOR_ALBEDO",
            "VCOLOR_EMIT",
            "VCOLOR_METALLIC",
        ],
        &[
            ("PBSDUALSIDED_KW_ALBEDOTEX", 0),
            ("PBSDUALSIDED_KW_ALPHACLIP", 1),
            ("PBSDUALSIDED_KW_EMISSIONTEX", 2),
            ("PBSDUALSIDED_KW_METALLICMAP", 3),
            ("PBSDUALSIDED_KW_NORMALMAP", 4),
            ("PBSDUALSIDED_KW_OCCLUSION", 5),
            ("PBSDUALSIDED_KW_VCOLOR_ALBEDO", 6),
            ("PBSDUALSIDED_KW_VCOLOR_EMIT", 7),
            ("PBSDUALSIDED_KW_VCOLOR_METALLIC", 8),
        ],
    )
}

#[test]
fn pbsintersectspecular_decodes_keywords_from_variant_bits() -> io::Result<()> {
    assert_variant_bits_migration(
        "pbsintersectspecular.wgsl",
        &[
            "_ALBEDOTEX",
            "_EMISSIONTEX",
            "_NORMALMAP",
            "_OCCLUSION",
            "_SPECULARMAP",
        ],
        &[
            ("PBSINTERSECTSPECULAR_KW_ALBEDOTEX", 0),
            ("PBSINTERSECTSPECULAR_KW_EMISSIONTEX", 1),
            ("PBSINTERSECTSPECULAR_KW_NORMALMAP", 2),
            ("PBSINTERSECTSPECULAR_KW_OCCLUSION", 3),
            ("PBSINTERSECTSPECULAR_KW_SPECULARMAP", 4),
        ],
    )
}

#[test]
fn pbsintersect_decodes_keywords_from_variant_bits() -> io::Result<()> {
    assert_variant_bits_migration(
        "pbsintersect.wgsl",
        &[
            "_ALBEDOTEX",
            "_EMISSIONTEX",
            "_METALLICMAP",
            "_NORMALMAP",
            "_OCCLUSION",
        ],
        &[
            ("PBSINTERSECT_KW_ALBEDOTEX", 0),
            ("PBSINTERSECT_KW_EMISSIONTEX", 1),
            ("PBSINTERSECT_KW_METALLICMAP", 2),
            ("PBSINTERSECT_KW_NORMALMAP", 3),
            ("PBSINTERSECT_KW_OCCLUSION", 4),
        ],
    )
}

#[test]
fn pbslerpspecular_decodes_keywords_from_variant_bits() -> io::Result<()> {
    assert_variant_bits_migration(
        "pbslerpspecular.wgsl",
        &[
            "_ALBEDOTEX",
            "_ALPHACLIP",
            "_DUALSIDED",
            "_EMISSIONTEX",
            "_LERPTEX",
            "_MULTI_VALUES",
            "_NORMALMAP",
            "_OCCLUSION",
            "_SPECULARMAP",
        ],
        &[
            ("PBSLERPSPECULAR_KW_ALBEDOTEX", 0),
            ("PBSLERPSPECULAR_KW_ALPHACLIP", 1),
            ("PBSLERPSPECULAR_KW_DUALSIDED", 2),
            ("PBSLERPSPECULAR_KW_EMISSIONTEX", 3),
            ("PBSLERPSPECULAR_KW_LERPTEX", 4),
            ("PBSLERPSPECULAR_KW_MULTI_VALUES", 5),
            ("PBSLERPSPECULAR_KW_NORMALMAP", 6),
            ("PBSLERPSPECULAR_KW_OCCLUSION", 7),
            ("PBSLERPSPECULAR_KW_SPECULARMAP", 8),
        ],
    )
}

#[test]
fn pbslerp_decodes_keywords_from_variant_bits() -> io::Result<()> {
    assert_variant_bits_migration(
        "pbslerp.wgsl",
        &[
            "_ALBEDOTEX",
            "_ALPHACLIP",
            "_DUALSIDED",
            "_EMISSIONTEX",
            "_LERPTEX",
            "_METALLICMAP",
            "_MULTI_VALUES",
            "_NORMALMAP",
            "_OCCLUSION",
        ],
        &[
            ("PBSLERP_KW_ALBEDOTEX", 0),
            ("PBSLERP_KW_ALPHACLIP", 1),
            ("PBSLERP_KW_DUALSIDED", 2),
            ("PBSLERP_KW_EMISSIONTEX", 3),
            ("PBSLERP_KW_LERPTEX", 4),
            ("PBSLERP_KW_METALLICMAP", 5),
            ("PBSLERP_KW_MULTI_VALUES", 6),
            ("PBSLERP_KW_NORMALMAP", 7),
            ("PBSLERP_KW_OCCLUSION", 8),
        ],
    )
}

#[test]
fn pbsmetallic_decodes_keywords_from_variant_bits() -> io::Result<()> {
    assert_variant_bits_migration(
        "pbsmetallic.wgsl",
        &[
            "_ALPHABLEND_ON",
            "_ALPHAPREMULTIPLY_ON",
            "_ALPHATEST_ON",
            "_DETAIL_MULX2",
            "_EMISSION",
            "_GLOSSYREFLECTIONS_OFF",
            "_METALLICGLOSSMAP",
            "_MUL_RGB_BY_ALPHA",
            "_NORMALMAP",
            "_PARALLAXMAP",
            "_SMOOTHNESS_TEXTURE_ALBEDO_CHANNEL_A",
            "_SPECULARHIGHLIGHTS_OFF",
        ],
        &[
            ("PBSMETALLIC_KW_ALPHABLEND_ON", 0),
            ("PBSMETALLIC_KW_ALPHAPREMULTIPLY_ON", 1),
            ("PBSMETALLIC_KW_ALPHATEST_ON", 2),
            ("PBSMETALLIC_KW_DETAIL_MULX2", 3),
            ("PBSMETALLIC_KW_EMISSION", 4),
            ("PBSMETALLIC_KW_GLOSSYREFLECTIONS_OFF", 5),
            ("PBSMETALLIC_KW_METALLICGLOSSMAP", 6),
            ("PBSMETALLIC_KW_NORMALMAP", 7),
            ("PBSMETALLIC_KW_PARALLAXMAP", 8),
            ("PBSMETALLIC_KW_SMOOTHNESS_TEXTURE_ALBEDO_CHANNEL_A", 9),
            ("PBSMETALLIC_KW_SPECULARHIGHLIGHTS_OFF", 10),
            ("PBSMETALLIC_KW_EDITOR_VISUALIZATION", 11),
        ],
    )
}

#[test]
fn pbsmetallic_emission_gated_by_variant_bit_not_runtime_check() -> io::Result<()> {
    let src = material_source("pbsmetallic.wgsl")?;
    assert!(
        !src.contains("dot(emission_color, emission_color)"),
        "pbsmetallic.wgsl must not use the runtime `dot(emission_color, emission_color) > 1e-8` \
         guard; the _EMISSION variant bit controls the optional emission sample"
    );
    assert!(
        src.contains("pbs_kw(PBSMETALLIC_KW_EMISSION)"),
        "pbsmetallic.wgsl must gate emission sampling on PBSMETALLIC_KW_EMISSION"
    );
    Ok(())
}

#[test]
fn pbsmultiuvspecular_decodes_keywords_from_variant_bits() -> io::Result<()> {
    assert_variant_bits_migration(
        "pbsmultiuvspecular.wgsl",
        &[
            "_ALPHACLIP",
            "_DUAL_ALBEDO",
            "_DUAL_EMISSIONTEX",
            "_EMISSIONTEX",
            "_NORMALMAP",
            "_OCCLUSION",
            "_SPECULARMAP",
        ],
        &[
            ("PBSMULTIUVSPECULAR_KW_ALPHACLIP", 0),
            ("PBSMULTIUVSPECULAR_KW_DUAL_ALBEDO", 1),
            ("PBSMULTIUVSPECULAR_KW_DUAL_EMISSIONTEX", 2),
            ("PBSMULTIUVSPECULAR_KW_EMISSIONTEX", 3),
            ("PBSMULTIUVSPECULAR_KW_NORMALMAP", 4),
            ("PBSMULTIUVSPECULAR_KW_OCCLUSION", 5),
            ("PBSMULTIUVSPECULAR_KW_SPECULARMAP", 6),
        ],
    )
}
