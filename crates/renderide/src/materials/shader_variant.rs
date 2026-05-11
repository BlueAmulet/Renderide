//! Shader-specific Froox variant bitmask metadata.
//!
//! The bitmask is decoded against the matching Unity shader asset's sorted `UniqueKeywords`
//! table. It is not a global enum: bit meanings are shader-specific.

/// Renderer-reserved material uniform field that carries the shader-specific Froox variant bitmask.
pub(crate) const RENDERIDE_VARIANT_BITS_FIELD: &str = "_RenderideVariantBits";

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

/// Returns the bit mask for a keyword in a shader-specific variant bitmask.
pub(crate) fn shader_variant_keyword_mask(stem: &str, keyword_name: &str) -> Option<u32> {
    let metadata = shader_variant_metadata_for_embedded_stem(stem)?;
    let bit_index = metadata
        .keywords
        .iter()
        .position(|keyword| keyword.name == keyword_name)?;
    variant_bit_mask(bit_index)
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

fn variant_bit_mask(bit_index: usize) -> Option<u32> {
    u32::try_from(bit_index)
        .ok()
        .and_then(|shift| 1u32.checked_shl(shift))
}

fn variant_bit_enabled(bits: u32, bit_index: usize) -> bool {
    variant_bit_mask(bit_index).is_some_and(|mask| bits & mask != 0)
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
    fn unlit_variant_masks_match_keyword_order() {
        assert_eq!(
            shader_variant_keyword_mask("unlit_default", "_ALPHATEST"),
            Some(1u32 << 0)
        );
        assert_eq!(
            shader_variant_keyword_mask("unlit_default", "_COLOR"),
            Some(1u32 << 1)
        );
        assert_eq!(
            shader_variant_keyword_mask("unlit_default", "_TEXTURE"),
            Some(1u32 << 9)
        );
        assert_eq!(
            shader_variant_keyword_mask("unlit_default", "_VERTEXCOLORS"),
            Some(1u32 << 13)
        );
        assert_eq!(
            shader_variant_keyword_mask("unlit_default", "_ALPHABLEND_ON"),
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
