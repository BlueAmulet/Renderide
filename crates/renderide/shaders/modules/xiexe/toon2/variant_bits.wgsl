//! Variant-bit decoding for the Xiexe Toon 2.0 family.
//!
//! The Froox shader-internal-name suffix produces a `u32` bitmask whose bits index into
//! the alphabetically-sorted `UniqueKeywords` list built from the Unity `XSToon2.0` /
//! `XSToon2.0_Outlined` `#pragma multi_compile` groups. Built-in pragmas such as
//! `multi_compile_fog` / `_fwdbase` / `_fwdadd_fullshadows` / `_shadowcaster` are not in
//! `VariantGroups`, so they consume no bits.
//!
//! Two Unity tokens are composite per `XSDefines.cginc`:
//!   * `OCCLUSION_METALLIC` enables both `OCCLUSION_MAP` and `METALLICGLOSS_MAP`.
//!   * `RAMPMASK_OUTLINEMASK_THICKNESS` enables `RAMP_MASK`, `OUTLINE_MASK`, and `THICKNESS_MAP`.
//!
//! Variants without custom `multi_compile` directives (the dithered/fade/transparent/dashed-
//! outlined Unity references) ship a zero bitmask; the dispatchers pin their alpha mode
//! through `XIEE_ALPHA_MODE` and fall through to feature-disabled defaults.

#define_import_path renderide::xiexe::toon2::variant_bits

#import renderide::material::variant_bits as vb
#import renderide::xiexe::toon2::base as xb

/// `AlphaBlend` keyword bit (straight-alpha blending).
const XTOON_KW_ALPHABLEND: u32 = 1u << 0u;
/// `Cutout` keyword bit (alpha-test).
const XTOON_KW_CUTOUT: u32 = 1u << 1u;
/// `EMISSION_MAP` keyword bit.
const XTOON_KW_EMISSION_MAP: u32 = 1u << 2u;
/// `MATCAP` keyword bit.
const XTOON_KW_MATCAP: u32 = 1u << 3u;
/// `NORMAL_MAP` keyword bit.
const XTOON_KW_NORMAL_MAP: u32 = 1u << 4u;
/// `OCCLUSION_METALLIC` keyword bit (drives both metallic-gloss and occlusion maps).
const XTOON_KW_OCCLUSION_METALLIC: u32 = 1u << 5u;
/// `RAMPMASK_OUTLINEMASK_THICKNESS` keyword bit (drives ramp-mask, outline-mask, and thickness).
const XTOON_KW_RAMPMASK_OUTLINEMASK_THICKNESS: u32 = 1u << 6u;
/// `Transparent` keyword bit (premultiplied transparent blending).
const XTOON_KW_TRANSPARENT: u32 = 1u << 7u;
/// `VERTEX_COLOR_ALBEDO` keyword bit.
const XTOON_KW_VERTEX_COLOR_ALBEDO: u32 = 1u << 8u;
/// `VERTEXLIGHT_ON` keyword bit. Present for Froox-side parity; the clustered renderer
/// does not require this keyword to gate per-vertex point-light evaluation.
const XTOON_KW_VERTEXLIGHT_ON: u32 = 1u << 9u;

/// Tests one keyword bit against the material's `_RenderideVariantBits`.
fn xtoon_kw(mask: u32) -> bool {
    return vb::enabled(xb::mat._RenderideVariantBits, mask);
}

/// `AlphaBlend` keyword on.
fn kw_AlphaBlend() -> bool {
    return xtoon_kw(XTOON_KW_ALPHABLEND);
}

/// `Cutout` keyword on.
fn kw_Cutout() -> bool {
    return xtoon_kw(XTOON_KW_CUTOUT);
}

/// `EMISSION_MAP` keyword on.
fn kw_EMISSION_MAP() -> bool {
    return xtoon_kw(XTOON_KW_EMISSION_MAP);
}

/// `MATCAP` keyword on.
fn kw_MATCAP() -> bool {
    return xtoon_kw(XTOON_KW_MATCAP);
}

/// `NORMAL_MAP` keyword on.
fn kw_NORMAL_MAP() -> bool {
    return xtoon_kw(XTOON_KW_NORMAL_MAP);
}

/// `OCCLUSION_METALLIC` keyword on.
fn kw_OCCLUSION_METALLIC() -> bool {
    return xtoon_kw(XTOON_KW_OCCLUSION_METALLIC);
}

/// `RAMPMASK_OUTLINEMASK_THICKNESS` keyword on.
fn kw_RAMPMASK_OUTLINEMASK_THICKNESS() -> bool {
    return xtoon_kw(XTOON_KW_RAMPMASK_OUTLINEMASK_THICKNESS);
}

/// `Transparent` keyword on.
fn kw_Transparent() -> bool {
    return xtoon_kw(XTOON_KW_TRANSPARENT);
}

/// `VERTEX_COLOR_ALBEDO` keyword on.
fn kw_VERTEX_COLOR_ALBEDO() -> bool {
    return xtoon_kw(XTOON_KW_VERTEX_COLOR_ALBEDO);
}

/// `VERTEXLIGHT_ON` keyword on.
fn kw_VERTEXLIGHT_ON() -> bool {
    return xtoon_kw(XTOON_KW_VERTEXLIGHT_ON);
}

/// True when the normal map should be sampled and applied.
fn normal_map_enabled() -> bool {
    return kw_NORMAL_MAP();
}

/// True when the emission term should be evaluated for this material. Mirrors the
/// `EMISSION_MAP` keyword and falls back to a non-black `_EmissionColor` so materials that
/// drive emission purely through the color slider still light up.
fn emission_map_enabled() -> bool {
    if (kw_EMISSION_MAP()) {
        return true;
    }
    let c = xb::mat._EmissionColor.rgb;
    return dot(c, c) > 1e-8;
}

/// True when the metallic-gloss map should be sampled (expanded from `OCCLUSION_METALLIC`).
fn metallic_map_enabled() -> bool {
    return kw_OCCLUSION_METALLIC();
}

/// True when the occlusion map should be sampled (expanded from `OCCLUSION_METALLIC`).
fn occlusion_enabled() -> bool {
    return kw_OCCLUSION_METALLIC();
}

/// True when the ramp selection mask should be sampled (expanded from `RAMPMASK_OUTLINEMASK_THICKNESS`).
fn ramp_mask_enabled() -> bool {
    return kw_RAMPMASK_OUTLINEMASK_THICKNESS();
}

/// True when the outline mask should be sampled (expanded from `RAMPMASK_OUTLINEMASK_THICKNESS`).
fn outline_mask_enabled() -> bool {
    return kw_RAMPMASK_OUTLINEMASK_THICKNESS();
}

/// True when the thickness map should be sampled (expanded from `RAMPMASK_OUTLINEMASK_THICKNESS`).
fn thickness_enabled() -> bool {
    return kw_RAMPMASK_OUTLINEMASK_THICKNESS();
}

/// True when matcap mode is selected, either via the `MATCAP` keyword or `_ReflectionMode == 2`.
fn matcap_enabled() -> bool {
    return kw_MATCAP() || abs(xb::mat._ReflectionMode - 2.0) < 0.5;
}

/// True when the reflection mode explicitly disables indirect specular.
fn reflection_disabled() -> bool {
    return !kw_MATCAP() && abs(xb::mat._ReflectionMode - 3.0) < 0.5;
}

/// True when the reflection mode selects the per-material baked-cubemap branch.
fn baked_cubemap_enabled() -> bool {
    return !kw_MATCAP() && abs(xb::mat._ReflectionMode - 1.0) < 0.5;
}

/// True when the shader should use the skybox/PBR reflection branch (`_ReflectionMode == 0`).
fn reflection_uses_pbr() -> bool {
    return !reflection_disabled() && !matcap_enabled() && !baked_cubemap_enabled();
}

/// True when vertex-color albedo tinting is enabled via the variant keyword.
fn vertex_color_albedo_enabled() -> bool {
    return kw_VERTEX_COLOR_ALBEDO();
}

/// Resolves the runtime alpha mode for shaders whose dispatcher pins
/// `XIEE_ALPHA_MODE = ALPHA_OPAQUE` and defers the decision to the variant bitmask.
/// Mirrors the precedence of the upstream `Cutout AlphaBlend Transparent` multi_compile group:
/// cutout wins over transparent which wins over alpha-blend.
fn resolved_alpha_mode_from_bits(static_alpha_mode: u32) -> u32 {
    if (static_alpha_mode != xb::ALPHA_OPAQUE) {
        return static_alpha_mode;
    }
    if (kw_Cutout()) {
        return xb::ALPHA_CUTOUT;
    }
    if (kw_Transparent()) {
        return xb::ALPHA_TRANSPARENT;
    }
    if (kw_AlphaBlend()) {
        return xb::ALPHA_FADE;
    }
    return xb::ALPHA_OPAQUE;
}
