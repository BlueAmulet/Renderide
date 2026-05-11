//! Unity-convention default tables for embedded material uniform packing.

pub(super) use crate::materials::wgsl_reflect::identifier_names::strip_writer_digit_escape as shader_writer_unescaped_field_name;

pub(super) fn default_f32_for_field(field_name: &str) -> Option<f32> {
    let field_name = shader_writer_unescaped_field_name(field_name);
    match field_name {
        "_Cutoff" => Some(0.5),
        "_Glossiness" => Some(0.5),
        "_GlossMapScale" => Some(1.0),
        "_BumpScale" | "_NormalScale" | "_DetailNormalMapScale" => Some(1.0),
        "_OcclusionStrength" => Some(1.0),
        "_Parallax" => Some(0.02),
        "_Metallic" | "_UVSec" => Some(0.0),
        _ => None,
    }
}

pub(super) fn default_vec4_for_field(field_name: &str) -> [f32; 4] {
    let field_name = shader_writer_unescaped_field_name(field_name);
    if field_name.ends_with("_ST") {
        return [1.0, 1.0, 0.0, 0.0];
    }
    match field_name {
        "_Point" => [0.0, 0.0, 0.0, 0.0],
        "_Rect" => [0.0, 0.0, 1.0, 1.0],
        "_FOV" => [std::f32::consts::TAU, std::f32::consts::PI, 0.0, 0.0],
        "_SecondTexOffset" => [0.0, 0.0, 0.0, 0.0],
        "_OffsetMagnitude" => [0.1, 0.1, 0.0, 0.0],
        "_PointSize" => [0.1, 0.1, 0.0, 0.0],
        "_PerspectiveFOV" => [
            std::f32::consts::FRAC_PI_4,
            std::f32::consts::FRAC_PI_4,
            0.0,
            0.0,
        ],
        "_Tint0" => [1.0, 0.0, 0.0, 1.0],
        "_Tint1" => [0.0, 1.0, 0.0, 1.0],
        "_OverlayTint" => [1.0, 1.0, 1.0, 0.5],
        "_BackgroundColor" => [0.0, 0.0, 0.0, 0.0],
        "_Range" => [0.001, 0.001, 0.0, 0.0],
        "_EmissionColor"
        | "_EmissionColor1"
        | "_IntersectEmissionColor"
        | "_OutsideColor"
        | "_OcclusionColor"
        | "_SSColor" => [0.0, 0.0, 0.0, 0.0],
        "_OutlineColor" => [0.0, 0.0, 0.0, 1.0],
        "_RimColor" | "_ShadowRim" | "_MatcapTint" => [1.0, 1.0, 1.0, 1.0],
        "_BehindFarColor" | "_FrontFarColor" | "_FarColor" | "_FarColor0" => [0.0, 0.0, 0.0, 1.0],
        "_FarColor1" => [0.2, 0.2, 0.2, 1.0],
        "_NearColor1" => [0.8, 0.8, 0.8, 0.8],
        "_SpecularColor" | "_SpecularColor1" => [1.0, 1.0, 1.0, 0.5],
        _ => [1.0, 1.0, 1.0, 1.0],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_vec4_known_names() {
        assert_eq!(default_vec4_for_field("_Rect"), [0.0, 0.0, 1.0, 1.0]);
        assert_eq!(default_vec4_for_field("_MainTex_ST"), [1.0, 1.0, 0.0, 0.0]);
        assert_eq!(
            default_vec4_for_field("_EmissionColor"),
            [0.0, 0.0, 0.0, 0.0]
        );
        assert_eq!(
            default_vec4_for_field("_BackgroundColor"),
            [0.0, 0.0, 0.0, 0.0]
        );
        assert_eq!(default_vec4_for_field("_Range"), [0.001, 0.001, 0.0, 0.0]);
        assert_eq!(default_vec4_for_field("_Tint0"), [1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn default_vec4_unknown_falls_back_to_white() {
        assert_eq!(default_vec4_for_field("_Unknown"), [1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn default_f32_known_standard_names() {
        assert_eq!(default_f32_for_field("_GlossMapScale"), Some(1.0));
        assert_eq!(default_f32_for_field("_OcclusionStrength"), Some(1.0));
        assert_eq!(default_f32_for_field("_UVSec"), Some(0.0));
        assert_eq!(default_f32_for_field("_UnknownScalar"), None);
    }

    #[test]
    fn default_vec4_unescaped_digit_field_resolves_to_known_default() {
        // `_Tex0_` unescapes to `_Tex0`; not a known special case, fallback is white.
        assert_eq!(default_vec4_for_field("_Tex0_"), [1.0, 1.0, 1.0, 1.0]);
    }
}
