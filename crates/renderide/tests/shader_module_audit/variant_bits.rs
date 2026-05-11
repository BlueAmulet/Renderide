//! Audit that every variant-bit-migrated shader still decodes its `#pragma multi_compile`
//! keywords from `_RenderideVariantBits` rather than from legacy f32 keyword uniforms.

use super::*;

fn assert_variant_bits_shader(
    file_name: &str,
    forbidden_f32_fields: &[&str],
    keyword_constants: &[(&str, u32)],
) -> io::Result<()> {
    let src = material_source(file_name)?;
    assert!(
        src.contains("_RenderideVariantBits: u32"),
        "{file_name}: must declare _RenderideVariantBits: u32"
    );
    for field_name in forbidden_f32_fields {
        assert!(
            !declares_f32_field(&src, field_name),
            "{file_name}: {field_name} must be decoded from _RenderideVariantBits, not packed as f32"
        );
    }
    for (constant_name, bit_index) in keyword_constants {
        assert!(
            src.contains(&format!("const {constant_name}: u32 = 1u << {bit_index}u;")),
            "{file_name}: {constant_name} must match the Froox sorted UniqueKeywords bit order"
        );
    }
    Ok(())
}

#[test]
fn billboardunlit_uses_reserved_variant_bits() -> io::Result<()> {
    assert_variant_bits_shader(
        "billboardunlit.wgsl",
        &[
            "_ALPHATEST",
            "_COLOR",
            "_MUL_ALPHA_INTENSITY",
            "_MUL_RGB_BY_ALPHA",
            "_OFFSET_TEXTURE",
            "_POINT_ROTATION",
            "_POINT_SIZE",
            "_POINT_UV",
            "_POLARUV",
            "_RIGHT_EYE_ST",
            "_TEXTURE",
            "_VERTEX_HDRSRGBALPHA_COLOR",
            "_VERTEX_HDRSRGB_COLOR",
            "_VERTEX_LINEAR_COLOR",
            "_VERTEX_SRGB_COLOR",
            "_VERTEXCOLORS",
        ],
        &[
            ("BILLBOARDUNLIT_KW_ALPHATEST", 0),
            ("BILLBOARDUNLIT_KW_COLOR", 1),
            ("BILLBOARDUNLIT_KW_MUL_ALPHA_INTENSITY", 2),
            ("BILLBOARDUNLIT_KW_MUL_RGB_BY_ALPHA", 3),
            ("BILLBOARDUNLIT_KW_OFFSET_TEXTURE", 4),
            ("BILLBOARDUNLIT_KW_POINT_ROTATION", 5),
            ("BILLBOARDUNLIT_KW_POINT_SIZE", 6),
            ("BILLBOARDUNLIT_KW_POINT_UV", 7),
            ("BILLBOARDUNLIT_KW_POLARUV", 8),
            ("BILLBOARDUNLIT_KW_RIGHT_EYE_ST", 9),
            ("BILLBOARDUNLIT_KW_TEXTURE", 10),
            ("BILLBOARDUNLIT_KW_VERTEX_HDRSRGBALPHA_COLOR", 11),
            ("BILLBOARDUNLIT_KW_VERTEX_HDRSRGB_COLOR", 12),
            ("BILLBOARDUNLIT_KW_VERTEX_LINEAR_COLOR", 13),
            ("BILLBOARDUNLIT_KW_VERTEX_SRGB_COLOR", 14),
            ("BILLBOARDUNLIT_KW_VERTEXCOLORS", 15),
        ],
    )
}

#[test]
fn blur_uses_reserved_variant_bits() -> io::Result<()> {
    assert_variant_bits_shader(
        "blur.wgsl",
        &[
            "POISSON_DISC",
            "RECTCLIP",
            "REFRACT",
            "REFRACT_NORMALMAP",
            "SPREAD_TEX",
        ],
        &[
            ("BLUR_KW_POISSON_DISC", 0),
            ("BLUR_KW_RECTCLIP", 1),
            ("BLUR_KW_REFRACT", 2),
            ("BLUR_KW_REFRACT_NORMALMAP", 3),
            ("BLUR_KW_SPREAD_TEX", 4),
        ],
    )
}

#[test]
fn channelmatrix_uses_reserved_variant_bits() -> io::Result<()> {
    assert_variant_bits_shader(
        "channelmatrix.wgsl",
        &["RECTCLIP"],
        &[("CHANNELMATRIX_KW_RECTCLIP", 0)],
    )
}

#[test]
fn cubemapprojection_uses_reserved_variant_bits() -> io::Result<()> {
    assert_variant_bits_shader(
        "cubemapprojection.wgsl",
        &["FLIP"],
        &[("CUBEMAPPROJECTION_KW_FLIP", 0)],
    )
}

#[test]
fn depthprojection_uses_reserved_variant_bits() -> io::Result<()> {
    assert_variant_bits_shader(
        "depthprojection.wgsl",
        &["DEPTH_HUE", "DEPTH_GRAYSCALE"],
        &[
            ("DEPTHPROJECTION_KW_DEPTH_GRAYSCALE", 0),
            ("DEPTHPROJECTION_KW_DEPTH_HUE", 1),
        ],
    )
}

#[test]
fn pixelate_uses_reserved_variant_bits() -> io::Result<()> {
    assert_variant_bits_shader(
        "pixelate.wgsl",
        &["RECTCLIP", "RESOLUTION_TEX", "_RectClip"],
        &[
            ("PIXELATE_KW_RECTCLIP", 0),
            ("PIXELATE_KW_RESOLUTION_TEX", 1),
        ],
    )
}

#[test]
fn pixelate_resolution_tex_sample_is_keyword_gated() -> io::Result<()> {
    let src = material_source("pixelate.wgsl")?;
    let kw_position = src.find("kw_RESOLUTION_TEX()").expect(
        "pixelate.wgsl must reference kw_RESOLUTION_TEX() to gate the _ResolutionTex sample",
    );
    let sample_position = src
        .find("textureSample(_ResolutionTex,")
        .expect("pixelate.wgsl must sample _ResolutionTex");
    assert!(
        kw_position < sample_position,
        "pixelate.wgsl must guard the _ResolutionTex sample on kw_RESOLUTION_TEX()"
    );
    Ok(())
}

fn is_meaningful_wrapper_line(line: &str) -> bool {
    let trimmed = line.trim();
    !trimmed.is_empty() && !trimmed.starts_with("//")
}

fn assert_source_alias_wrapper(file_name: &str, parent_stem: &str) -> io::Result<()> {
    let src = material_source(file_name)?;
    let directive = format!("//#source_alias {parent_stem}");
    assert!(
        src.lines().any(|line| line.trim() == directive),
        "{file_name}: must contain `{directive}` directive"
    );
    let meaningful_lines: Vec<&str> = src
        .lines()
        .filter(|l| is_meaningful_wrapper_line(l))
        .collect();
    assert!(
        meaningful_lines.is_empty(),
        "{file_name}: source-alias wrapper must contain no code, found: {meaningful_lines:?}"
    );
    Ok(())
}

#[test]
fn blur_perobject_is_source_alias_wrapper() -> io::Result<()> {
    assert_source_alias_wrapper("blur_perobject.wgsl", "blur")
}

#[test]
fn channelmatrix_perobject_is_source_alias_wrapper() -> io::Result<()> {
    assert_source_alias_wrapper("channelmatrix_perobject.wgsl", "channelmatrix")
}
