//! Name-driven keyword inference and scalar default tables for embedded uniform packing.

use crate::assets::material::{MaterialPropertyLookupIds, MaterialPropertyStore};

use super::super::layout::{EmbeddedSharedKeywordIds, StemEmbeddedPropertyIds};
use super::helpers::{
    first_float_by_pids, is_keyword_like_field, keyword_float_enabled_any_pids,
    shader_writer_unescaped_field_name, texture_property_present_pids,
};

pub(super) fn inferred_keyword_float_f32(
    field_name: &str,
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    ids: &StemEmbeddedPropertyIds,
) -> Option<f32> {
    let field_name = shader_writer_unescaped_field_name(field_name);
    if let Some(probes) = ids.keyword_field_probe_ids.get(field_name) {
        if keyword_float_enabled_any_pids(store, lookup, probes) {
            return Some(1.0);
        }
    }

    let kw = ids.shared.as_ref();
    match field_name {
        "_ALPHATEST_ON" | "_ALPHATEST" | "ALPHATEST" | "_ALPHA_TEST" | "ALPHA_TEST"
        | "_ALPHACLIP" | "ALPHACLIP" | "_ALPHA_CLIP" | "ALPHA_CLIP" => {
            return Some(if material_mode_or_blend_mode_is(store, lookup, kw, 1) {
                1.0
            } else {
                0.0
            });
        }
        "_ALPHABLEND_ON" | "_ALPHABLEND" | "ALPHABLEND" | "_ALPHA_BLEND" | "ALPHA_BLEND" => {
            return Some(if material_mode_or_blend_mode_is(store, lookup, kw, 2) {
                1.0
            } else {
                0.0
            });
        }
        "_ALPHAPREMULTIPLY_ON"
        | "_ALPHAPREMULTIPLY"
        | "ALPHAPREMULTIPLY"
        | "_ALPHA_PREMULTIPLY"
        | "ALPHA_PREMULTIPLY" => {
            return Some(if material_mode_or_blend_mode_is(store, lookup, kw, 3) {
                1.0
            } else {
                0.0
            });
        }
        _ => {}
    }

    let inferred = match field_name {
        "_LERPTEX" => texture_property_present_pids(store, lookup, &[kw.lerp_tex]),
        "_ALBEDOTEX" => texture_property_present_pids(store, lookup, &[kw.main_tex, kw.main_tex1]),
        "_EMISSION" | "_EMISSIONTEX" => {
            texture_property_present_pids(store, lookup, &[kw.emission_map, kw.emission_map1])
        }
        "_NORMALMAP" => texture_property_present_pids(
            store,
            lookup,
            &[kw.normal_map, kw.normal_map1, kw.bump_map],
        ),
        "_SPECULARMAP" => texture_property_present_pids(
            store,
            lookup,
            &[kw.specular_map, kw.specular_map1, kw.spec_gloss_map],
        ),
        "_METALLICGLOSSMAP" => {
            texture_property_present_pids(store, lookup, &[kw.metallic_gloss_map])
        }
        "_METALLICMAP" => texture_property_present_pids(
            store,
            lookup,
            &[kw.metallic_map, kw.metallic_map1, kw.metallic_gloss_map],
        ),
        "_DETAIL_MULX2" => texture_property_present_pids(
            store,
            lookup,
            &[kw.detail_albedo_map, kw.detail_normal_map, kw.detail_mask],
        ),
        "_PARALLAXMAP" => texture_property_present_pids(store, lookup, &[kw.parallax_map]),
        "_OCCLUSION" => texture_property_present_pids(
            store,
            lookup,
            &[kw.occlusion, kw.occlusion1, kw.occlusion_map],
        ),
        _ if is_keyword_like_field(field_name) => false,
        _ => return None,
    };
    Some(if inferred { 1.0 } else { 0.0 })
}

fn material_mode_or_blend_mode_is(
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    kw: &EmbeddedSharedKeywordIds,
    mode_value: i32,
) -> bool {
    let mode = first_float_by_pids(store, lookup, &[kw.mode]).map(|v| v.round() as i32);
    let blend = first_float_by_pids(store, lookup, &[kw.blend_mode, kw.blend_mode_alt])
        .map(|v| v.round() as i32);
    mode == Some(mode_value) || blend == Some(mode_value)
}

pub(super) fn default_f32_for_field(
    field_name: &str,
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    ids: &StemEmbeddedPropertyIds,
) -> f32 {
    let field_name = shader_writer_unescaped_field_name(field_name);
    if let Some(v) = inferred_keyword_float_f32(field_name, store, lookup, ids) {
        return v;
    }
    match field_name {
        "_Lerp"
        | "_TextureLerp"
        | "_ProjectionLerp"
        | "_CubeLOD"
        | "_Metallic"
        | "_Metallic1"
        | "_UVSec"
        | "_Mode"
        | "_OffsetFactor"
        | "_OffsetUnits"
        | "_Stencil"
        | "_StencilOp"
        | "_StencilFail"
        | "_StencilZFail"
        | "_RimIntensity"
        | "_RimAlbedoTint"
        | "_RimCubemapTint"
        | "_SpecularIntensity"
        | "_ShadowRimAlbedoTint"
        | "_OutlineAlbedoTint"
        | "_OutlineLighting"
        | "_OutlineEmissive"
        | "_OutlineEmissiveues"
        | "_FadeDither"
        | "_FadeDitherDistance"
        | "_VertexColorAlbedo"
        | "_TilingMode"
        | "_UVSetAlbedo"
        | "_UVSetNormal"
        | "_UVSetDetNormal"
        | "_UVSetDetMask"
        | "_UVSetMetallic"
        | "_UVSetSpecular"
        | "_UVSetReflectivity"
        | "_UVSetThickness"
        | "_UVSetOcclusion"
        | "_UVSetEmission"
        | "_ClearCoat"
        | "_ReflectionBlendMode"
        | "_EmissionToDiffuse"
        | "_SpecMode"
        | "_SpecularStyle"
        | "_Offset" => 0.0,
        "_NormalScale"
        | "_NormalScale1"
        | "_BumpScale"
        | "_DetailNormalMapScale"
        | "_GlossMapScale"
        | "_OcclusionStrength"
        | "_SpecularHighlights"
        | "_GlossyReflections"
        | "_Exposure"
        | "_Gamma"
        | "_ZWrite"
        | "_Saturation"
        | "_Reflectivity"
        | "_ClearcoatStrength"
        | "_ScaleWithLight"
        | "_ScaleWithLightSensitivity"
        | "_RimAttenEffect"
        | "_SpecularAlbedoTint"
        | "_OutlineWidth"
        | "_SSDistortion"
        | "_SSPower"
        | "_SSScale"
        | "_SrcBlendBase"
        | "_SrcBlendAdd" => 1.0,
        "_Exp" | "_Exp0" | "_Exp1" | "_PolarPow" | "_LerpPolarPow" => 1.0,
        "_Distance" => 1.0,
        "_Transition" => 0.1,
        "_MaxIntensity" => 4.0,
        "_Parallax" => 0.02,
        "_GammaCurve" => 2.2,
        "_SrcBlend" => 1.0,
        "_DstBlend" | "_DstBlendBase" => 0.0,
        "_DstBlendAdd" => 1.0,
        "_ZTest" | "_Cull" | "_Culling" => 2.0,
        "_StencilComp" => 8.0,
        "_StencilWriteMask" | "_StencilReadMask" => 255.0,
        "_ColorMask" => 15.0,
        "_colormask" => 15.0,
        "_ReflectionMode" => 3.0,
        "_ClearcoatSmoothness" => 0.8,
        "_RimRange" | "_ShadowRimRange" => 0.7,
        "_RimThreshold" | "_ShadowRimThreshold" => 0.1,
        "_RimSharpness" => 0.1,
        "_ShadowRimSharpness" => 0.3,
        "_AnisotropicAX" => 0.25,
        "_AnisotropicAY" => 0.75,
        "_HalftoneDotSize" => 1.7,
        "_HalftoneDotAmount" => 50.0,
        "_HalftoneLineAmount" => 150.0,
        "_Cutoff" | "_AlphaClip" | "_Glossiness" | "_Glossiness1" => 0.5,
        _ => 0.5,
    }
}
