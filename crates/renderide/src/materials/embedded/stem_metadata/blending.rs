//! Alpha-blending and scene-snapshot flag queries on composed embedded WGSL stems.

use crate::materials::ShaderPermutation;

use super::EmbeddedStemQuery;

/// `true` when the embedded material stem declares alpha blending in any `//#pass` directive.
pub fn embedded_stem_uses_alpha_blending(base_stem: &str) -> bool {
    EmbeddedStemQuery::for_stem(base_stem, ShaderPermutation(0)).uses_alpha_blending()
}

/// `true` when the composed embedded target declares a scene-depth snapshot binding.
pub fn embedded_stem_uses_scene_depth_snapshot(
    base_stem: &str,
    permutation: ShaderPermutation,
) -> bool {
    EmbeddedStemQuery::for_stem(base_stem, permutation)
        .snapshot_requirements()
        .uses_scene_depth
}

/// `true` when the composed embedded target declares a scene-color snapshot binding.
pub fn embedded_stem_uses_scene_color_snapshot(
    base_stem: &str,
    permutation: ShaderPermutation,
) -> bool {
    EmbeddedStemQuery::for_stem(base_stem, permutation)
        .snapshot_requirements()
        .uses_scene_color
}

#[cfg(test)]
mod tests {
    use crate::materials::SHADER_PERM_MULTIVIEW_STEREO;
    use crate::materials::ShaderPermutation;

    use super::{embedded_stem_uses_scene_color_snapshot, embedded_stem_uses_scene_depth_snapshot};
    use crate::materials::embedded::stem_metadata::embedded_stem_requires_intersection_pass;

    #[test]
    fn metadata_flags_cover_snapshot_and_intersection_material_classes() {
        let mono = ShaderPermutation(0);

        assert!(embedded_stem_uses_scene_color_snapshot(
            "blur_default",
            mono
        ));
        assert!(embedded_stem_uses_scene_color_snapshot(
            "blur_default",
            SHADER_PERM_MULTIVIEW_STEREO
        ));
        assert!(!embedded_stem_requires_intersection_pass(
            "blur_default",
            mono
        ));
        assert!(embedded_stem_uses_scene_depth_snapshot(
            "blur_default",
            mono
        ));

        assert!(embedded_stem_uses_scene_color_snapshot(
            "refract_default",
            mono
        ));
        assert!(!embedded_stem_requires_intersection_pass(
            "refract_default",
            mono
        ));
        assert!(embedded_stem_uses_scene_depth_snapshot(
            "refract_default",
            mono
        ));

        assert!(embedded_stem_requires_intersection_pass(
            "pbsintersect_default",
            mono
        ));
        assert!(!embedded_stem_uses_scene_color_snapshot(
            "pbsintersect_default",
            mono
        ));
        assert!(embedded_stem_uses_scene_depth_snapshot(
            "pbsintersect_default",
            mono
        ));
    }
}
