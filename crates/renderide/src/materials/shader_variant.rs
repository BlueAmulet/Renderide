//! Shader-specific Froox variant bitmask metadata.
//!
//! The bitmask is decoded against the matching Unity shader asset's sorted `UniqueKeywords`
//! table. It is not a global enum: bit meanings are shader-specific.

/// One keyword entry in a shader-specific variant bitmask.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ShaderVariantKeyword {
    /// Unity keyword name at this bit index.
    pub name: &'static str,
    /// True when this keyword affects Rust-side pipeline or render-state construction.
    pub affects_pipeline_state: bool,
}

/// Variant decoding metadata for one embedded shader stem.
#[derive(Debug)]
pub(crate) struct ShaderVariantMetadata {
    /// Froox/Elements `UniqueKeywords` sorted order for this shader.
    pub keywords: &'static [ShaderVariantKeyword],
}

const UNLIT_VARIANT_KEYWORDS: &[ShaderVariantKeyword] = &[
    uniform_keyword("_ALPHATEST"),
    uniform_keyword("_COLOR"),
    uniform_keyword("_MASK_TEXTURE_CLIP"),
    uniform_keyword("_MASK_TEXTURE_MUL"),
    uniform_keyword("_MUL_ALPHA_INTENSITY"),
    uniform_keyword("_MUL_RGB_BY_ALPHA"),
    uniform_keyword("_OFFSET_TEXTURE"),
    uniform_keyword("_POLARUV"),
    uniform_keyword("_RIGHT_EYE_ST"),
    uniform_keyword("_TEXTURE"),
    uniform_keyword("_TEXTURE_NORMALMAP"),
    uniform_keyword("_VERTEX_LINEAR_COLOR"),
    uniform_keyword("_VERTEX_SRGB_COLOR"),
    uniform_keyword("_VERTEXCOLORS"),
];

const UNLIT_METADATA: ShaderVariantMetadata = ShaderVariantMetadata {
    keywords: UNLIT_VARIANT_KEYWORDS,
};

const fn uniform_keyword(name: &'static str) -> ShaderVariantKeyword {
    ShaderVariantKeyword {
        name,
        affects_pipeline_state: false,
    }
}

/// Returns shader-specific variant metadata for an embedded composed or source stem.
pub(crate) fn shader_variant_metadata_for_embedded_stem(
    stem: &str,
) -> Option<&'static ShaderVariantMetadata> {
    match embedded_source_stem(stem) {
        "unlit" => Some(&UNLIT_METADATA),
        _ => None,
    }
}

/// Decodes a material-uniform keyword field from a shader variant bitmask.
///
/// Returns `None` when no authoritative bitmask is present, when the shader has no metadata, or
/// when the field is not a keyword in the shader's metadata. Callers may then use compatibility
/// fallbacks for shaders that have not been moved to variant metadata yet.
pub(crate) fn shader_variant_uniform_keyword_float(
    stem: &str,
    shader_variant_bits: Option<u32>,
    field_name: &str,
) -> Option<f32> {
    let bits = shader_variant_bits?;
    let metadata = shader_variant_metadata_for_embedded_stem(stem)?;
    let bit_index = metadata
        .keywords
        .iter()
        .position(|keyword| keyword.name == field_name && !keyword.affects_pipeline_state)?;
    Some(if variant_bit_enabled(bits, bit_index) {
        1.0
    } else {
        0.0
    })
}

/// Returns true when a shader variant has any decoded pipeline-state keyword.
pub(crate) fn shader_variant_has_pipeline_state_keywords(
    stem: &str,
    shader_variant_bits: Option<u32>,
) -> bool {
    let Some(bits) = shader_variant_bits else {
        return false;
    };
    let Some(metadata) = shader_variant_metadata_for_embedded_stem(stem) else {
        return false;
    };
    metadata
        .keywords
        .iter()
        .enumerate()
        .any(|(bit_index, keyword)| {
            keyword.affects_pipeline_state && variant_bit_enabled(bits, bit_index)
        })
}

fn embedded_source_stem(stem: &str) -> &str {
    stem.strip_suffix("_default")
        .or_else(|| stem.strip_suffix("_multiview"))
        .unwrap_or(stem)
}

fn variant_bit_enabled(bits: u32, bit_index: usize) -> bool {
    u32::try_from(bit_index)
        .ok()
        .and_then(|shift| 1u32.checked_shl(shift))
        .is_some_and(|mask| bits & mask != 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unlit_metadata_uses_froox_sorted_unique_keyword_order() {
        let metadata = shader_variant_metadata_for_embedded_stem("unlit_default").unwrap();
        let keywords: Vec<&str> = metadata
            .keywords
            .iter()
            .map(|keyword| keyword.name)
            .collect();

        assert_eq!(
            keywords,
            vec![
                "_ALPHATEST",
                "_COLOR",
                "_MASK_TEXTURE_CLIP",
                "_MASK_TEXTURE_MUL",
                "_MUL_ALPHA_INTENSITY",
                "_MUL_RGB_BY_ALPHA",
                "_OFFSET_TEXTURE",
                "_POLARUV",
                "_RIGHT_EYE_ST",
                "_TEXTURE",
                "_TEXTURE_NORMALMAP",
                "_VERTEX_LINEAR_COLOR",
                "_VERTEX_SRGB_COLOR",
                "_VERTEXCOLORS",
            ]
        );
    }

    #[test]
    fn unlit_variant_bits_decode_uniform_keyword_fields() {
        let bits = (1u32 << 1) | (1u32 << 9) | (1u32 << 13);

        assert_eq!(
            shader_variant_uniform_keyword_float("unlit_default", Some(bits), "_COLOR"),
            Some(1.0)
        );
        assert_eq!(
            shader_variant_uniform_keyword_float("unlit_default", Some(bits), "_TEXTURE"),
            Some(1.0)
        );
        assert_eq!(
            shader_variant_uniform_keyword_float("unlit_default", Some(bits), "_VERTEXCOLORS"),
            Some(1.0)
        );
        assert_eq!(
            shader_variant_uniform_keyword_float("unlit_default", Some(bits), "_ALPHATEST"),
            Some(0.0)
        );
        assert_eq!(
            shader_variant_uniform_keyword_float("unlit_default", Some(bits), "_ALPHATEST_ON"),
            None
        );
    }

    #[test]
    fn no_bitmask_keeps_compatibility_fallback_available() {
        assert_eq!(
            shader_variant_uniform_keyword_float("unlit_default", None, "_TEXTURE"),
            None
        );
    }

    #[test]
    fn unlit_variant_has_no_pipeline_state_keywords() {
        assert!(!shader_variant_has_pipeline_state_keywords(
            "unlit_default",
            Some(u32::MAX)
        ));
    }
}
