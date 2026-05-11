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
