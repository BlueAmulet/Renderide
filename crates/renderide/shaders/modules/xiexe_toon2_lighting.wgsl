//! Direct + indirect lighting for the Xiexe Toon 2.0 BRDF.
//!
//! Keeps the XSToon2 ramp-diffuse surface model and clustered forward light walk, while
//! routing the specular lobes and PBR reflections through the same Filament-style energy
//! terms used by PBS materials. Matcaps, baked cubemaps, rim lighting, outlines, emission,
//! and reflection blend modes remain stylized Xiexe controls.

#define_import_path renderide::xiexe::toon2::lighting

#import renderide::xiexe::toon2::base as xb
#import renderide::cubemap_storage as cubemap_storage
#import renderide::globals as rg
#import renderide::pbs::cluster as pcls
#import renderide::pbs::brdf as brdf
#import renderide::birp::light as bl
#import renderide::reflection_probes as rprobe
#import renderide::uv_utils as uvu

/// SH-probe sample used for xiexe's uncoloured indirect-diffuse term.
fn indirect_diffuse(s: xb::SurfaceData, view_layer: u32) -> vec3<f32> {
    return rprobe::indirect_diffuse(s.normal, view_layer, true);
}

/// Scalar AO weight used when modern XSToon3 paths expect a single occlusion factor.
fn occlusion_scalar(s: xb::SurfaceData) -> f32 {
    return clamp(xb::grayscale(s.occlusion), 0.0, 1.0);
}

/// Reflection tint used by `_RimCubemapTint`. Falls back to white when no specular probe is bound so
/// the tint slider does not collapse the rim light to black.
fn environment_tint(s: xb::SurfaceData, view_dir: vec3<f32>, world_pos: vec3<f32>, view_layer: u32) -> vec3<f32> {
    if (!rprobe::has_indirect_specular(view_layer, true)) {
        return vec3<f32>(1.0);
    }
    return rprobe::raw_indirect_specular(world_pos, s.normal, view_dir, s.roughness, true, view_layer);
}

/// `UNITY_SPECCUBE_LOD_STEPS` on PC/console.
const SPECCUBE_LOD_STEPS: f32 = 6.0;

/// Resolves a single `rg::GpuLight` into a `LightSample` (direction toward the light,
/// color, attenuation, directional flag).
fn sample_light(light: rg::GpuLight, world_pos: vec3<f32>) -> xb::LightSample {
    if (light.light_type == 1u) {
        let dir_len_sq = dot(light.direction.xyz, light.direction.xyz);
        return xb::LightSample(
            select(vec3<f32>(0.0, 0.0, 1.0), normalize(-light.direction.xyz), dir_len_sq > 1e-16),
            light.color.xyz,
            bl::direct_light_intensity(light.intensity),
            true,
        );
    }

    let to_light = light.position.xyz - world_pos;
    let dist = length(to_light);
    let l = xb::safe_normalize(to_light, vec3<f32>(0.0, 1.0, 0.0));
    var attenuation = bl::punctual_attenuation(light.intensity, dist, light.range);
    if (light.light_type == 2u) {
        let spot_cos = dot(-l, xb::safe_normalize(light.direction.xyz, vec3<f32>(0.0, -1.0, 0.0)));
        let inner_cos = min(light.spot_cos_half_angle + 0.1, 1.0);
        attenuation = attenuation * smoothstep(light.spot_cos_half_angle, inner_cos, spot_cos);
    }
    return xb::LightSample(l, light.color.xyz, attenuation, false);
}

/// Toon ramp lookup. The half-Lambert remap (`NdotL * 0.5 + 0.5`) maps to the U axis;
/// the ramp-mask sample maps to the V axis. `_ShadowSharpness` sharpens the
/// attenuation before it multiplies half-Lambert so the banding stays on the shadow
/// transition rather than on the ramp itself.
fn ramp_for_ndl(ndl: f32, attenuation: f32, ramp_mask: f32) -> vec3<f32> {
    let att_sharp = mix(attenuation, round(attenuation), clamp(xb::mat._ShadowSharpness, 0.0, 1.0));
    let x = clamp((ndl * 0.5 + 0.5) * att_sharp, 0.0, 1.0);
    return textureSample(xb::_Ramp, xb::_Ramp_sampler, vec2<f32>(x, clamp(ramp_mask, 0.0, 1.0))).rgb;
}

/// XSToon-style remap used by `_SpecularArea` and clear-coat roughness inputs before they are
/// passed to the Filament/PBS GGX path as perceptual roughness.
fn remap_specular_area(area: f32) -> f32 {
    let remapped = max(0.01, area);
    return remapped * (1.7 - 0.7 * remapped);
}

/// Metallic workflow F0 used by Xiexe's PBS-grade specular paths.
fn xiexe_specular_reflectance(s: xb::SurfaceData) -> vec3<f32> {
    let reflectivity = clamp(s.reflectivity, 0.0, 1.0);
    let dielectric_reflectance = 0.16 * reflectivity * reflectivity;
    return vec3<f32>(dielectric_reflectance * (1.0 - s.metallic)) + s.diffuse_color * s.metallic;
}

/// Perceptual roughness for the primary Xiexe specular lobe.
fn primary_specular_roughness() -> f32 {
    return clamp(remap_specular_area(xb::mat._SpecularArea), 0.045, 1.0);
}

/// Perceptual roughness for the secondary clear-coat lobe.
fn clearcoat_roughness(s: xb::SurfaceData) -> f32 {
    return clamp(remap_specular_area(1.0 - s.clearcoat_smoothness), 0.045, 1.0);
}

/// Direct-specular inputs derived once per fragment for the primary lobe.
struct DirectSpecularTerms {
    /// Primary lobe F0.
    specular_reflectance: vec3<f32>,
    /// Primary lobe perceptual roughness.
    roughness: f32,
    /// Multiple-scattering energy compensation sampled from the frame DFG LUT.
    energy_compensation: vec3<f32>,
}

/// Resolves the primary direct-specular terms shared by every clustered light.
fn primary_direct_specular_terms(s: xb::SurfaceData, view_dir: vec3<f32>) -> DirectSpecularTerms {
    let specular_reflectance = xiexe_specular_reflectance(s);
    let roughness = primary_specular_roughness();
    let n_dot_v = clamp(dot(s.normal, view_dir), 0.0, 1.0);
    let dfg = brdf::sample_ibl_dfg_lut(roughness, n_dot_v);
    let energy_compensation = brdf::energy_compensation_from_dfg(dfg, specular_reflectance);
    return DirectSpecularTerms(specular_reflectance, roughness, energy_compensation);
}

/// Filament-style GGX direct-specular lobe evaluated against an arbitrary normal.
fn direct_specular_filament(
    normal: vec3<f32>,
    s: xb::SurfaceData,
    light: xb::LightSample,
    view_dir: vec3<f32>,
    perceptual_roughness: f32,
    specular_reflectance: vec3<f32>,
    energy_compensation: vec3<f32>,
    intensity: f32,
    albedo_tint: f32,
) -> vec3<f32> {
    if (intensity <= 1e-4) {
        return vec3<f32>(0.0);
    }

    let ndl = xb::saturate(dot(normal, light.direction));
    if (ndl <= 1e-4 || light.attenuation <= 1e-4) {
        return vec3<f32>(0.0);
    }

    let h = xb::safe_normalize(light.direction + view_dir, normal);
    let ndh = xb::saturate(dot(normal, h));
    let ndv = max(dot(normal, view_dir), 1e-4);
    let ldh = xb::saturate(dot(light.direction, h));

    let alpha = max(perceptual_roughness * perceptual_roughness, brdf::MIN_ALPHA);
    let d_term = brdf::d_ggx(ndh, alpha);
    let v_term = brdf::v_smith_ggx_correlated(ndv, ndl, alpha);
    let f_term = brdf::f_schlick(specular_reflectance, brdf::f90_from_f0(specular_reflectance), ldh);
    let radiance = light.color * light.attenuation * ndl;

    var specular = max(vec3<f32>(0.0), d_term * v_term * f_term * energy_compensation);
    specular = specular * radiance * intensity;
    specular = specular * mix(vec3<f32>(1.0), s.diffuse_color, clamp(albedo_tint, 0.0, 1.0));
    return specular;
}

/// Primary specular lobe driven by `_SpecularArea` and `_SpecularIntensity`.
fn direct_specular(
    s: xb::SurfaceData,
    light: xb::LightSample,
    view_dir: vec3<f32>,
    terms: DirectSpecularTerms,
) -> vec3<f32> {
    return direct_specular_filament(
        s.normal,
        s,
        light,
        view_dir,
        terms.roughness,
        terms.specular_reflectance,
        terms.energy_compensation,
        max(0.0, xb::mat._SpecularIntensity),
        xb::mat._SpecularAlbedoTint,
    );
}

/// Secondary clear-coat direct-specular lobe driven by the metallic-gloss map's `g/b` channels.
fn clearcoat_direct_specular(
    s: xb::SurfaceData,
    light: xb::LightSample,
    view_dir: vec3<f32>,
) -> vec3<f32> {
    if (!xb::clearcoat_enabled()) {
        return vec3<f32>(0.0);
    }

    let roughness = clearcoat_roughness(s);
    let specular_reflectance = vec3<f32>(brdf::DEFAULT_DIELECTRIC_F0);
    let n_dot_v = clamp(dot(s.raw_normal, view_dir), 0.0, 1.0);
    let dfg = brdf::sample_ibl_dfg_lut(roughness, n_dot_v);
    let energy_compensation = brdf::energy_compensation_from_dfg(dfg, specular_reflectance);
    return direct_specular_filament(
        s.raw_normal,
        s,
        light,
        view_dir,
        roughness,
        specular_reflectance,
        energy_compensation,
        s.clearcoat_strength,
        0.0,
    );
}

/// Rim contribution from the dominant light plus ambient probe lighting.
fn rim_light(
    s: xb::SurfaceData,
    light: xb::LightSample,
    view_dir: vec3<f32>,
    ambient: vec3<f32>,
    env_map: vec3<f32>,
) -> vec3<f32> {
    let ndl = xb::saturate(dot(s.normal, light.direction));
    let vdn = abs(dot(view_dir, s.normal));
    let sharp = max(xb::mat._RimSharpness, 0.001);
    var rim = xb::saturate(1.0 - vdn) * pow(ndl, max(xb::mat._RimThreshold, 0.0));
    rim = smoothstep(xb::mat._RimRange - sharp, xb::mat._RimRange + sharp, rim);

    var col = rim * xb::mat._RimIntensity * (light.color + ambient);
    col = col * mix(vec3<f32>(1.0), vec3<f32>(light.attenuation) + ambient, clamp(xb::mat._RimAttenEffect, 0.0, 1.0));
    col = col * xb::mat._RimColor.rgb;
    col = col * mix(vec3<f32>(1.0), s.diffuse_color, clamp(xb::mat._RimAlbedoTint, 0.0, 1.0));
    col = col * mix(vec3<f32>(1.0), env_map, clamp(xb::mat._RimCubemapTint, 0.0, 1.0));
    return col;
}

/// Shadow-rim multiplier from the dominant light plus a small ambient lift.
fn shadow_rim(
    s: xb::SurfaceData,
    view_dir: vec3<f32>,
    light: xb::LightSample,
    ambient: vec3<f32>,
) -> vec3<f32> {
    let ndl = xb::saturate(dot(s.normal, light.direction));
    let vdn = abs(dot(view_dir, s.normal));
    let sharp = max(xb::mat._ShadowRimSharpness, 0.001);
    var rim = xb::saturate(1.0 - vdn) * pow(xb::saturate(1.0 - ndl), max(xb::mat._ShadowRimThreshold * 2.0, 0.0));
    rim = smoothstep(xb::mat._ShadowRimRange - sharp, xb::mat._ShadowRimRange + sharp, rim);

    let tint = xb::mat._ShadowRim.rgb * mix(vec3<f32>(1.0), s.diffuse_color, clamp(xb::mat._ShadowRimAlbedoTint, 0.0, 1.0)) + ambient * 0.1;
    return mix(vec3<f32>(1.0), tint, rim);
}

/// Stylised subsurface scattering from XSToon3, preserving the XSToon2 property set.
fn subsurface(
    s: xb::SurfaceData,
    light: xb::LightSample,
    view_dir: vec3<f32>,
    ambient: vec3<f32>,
) -> vec3<f32> {
    if (dot(xb::mat._SSColor.rgb, xb::mat._SSColor.rgb) <= 1e-8) {
        return vec3<f32>(0.0);
    }

    let raw_ndl = dot(s.normal, light.direction);
    let ndl = xb::saturate(raw_ndl);
    if (ndl <= 1e-4 || light.attenuation <= 1e-4) {
        return vec3<f32>(0.0);
    }

    let attenuation = xb::saturate(light.attenuation * (raw_ndl * 0.5 + 0.5));
    let h = xb::safe_normalize(light.direction + s.normal * xb::mat._SSDistortion, s.normal);
    let vdh = pow(xb::saturate(dot(view_dir, -h)), max(xb::mat._SSPower, 0.001));
    let scatter = xb::mat._SSColor.rgb * (vdh + ambient) * attenuation * xb::mat._SSScale * s.thickness;
    return max(vec3<f32>(0.0), light.color * scatter * s.albedo.rgb) * ndl * light.attenuation;
}

/// View-space matcap UV. Projects `n` onto the camera's right and up basis vectors and remaps to
/// `[0, 1]`, matching Unity's `matcapSample`.
fn matcap_uv(view_dir: vec3<f32>, n: vec3<f32>) -> vec2<f32> {
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let view_up = xb::safe_normalize(up - view_dir * dot(view_dir, up), vec3<f32>(0.0, 1.0, 0.0));
    let view_right = xb::safe_normalize(cross(view_dir, view_up), vec3<f32>(1.0, 0.0, 0.0));
    return vec2<f32>(dot(view_right, n), dot(view_up, n)) * 0.5 + vec2<f32>(0.5);
}

/// Reflection blend-weight shared by the non-matcap indirect-specular blend modes.
fn reflection_blend_weight(s: xb::SurfaceData) -> f32 {
    if (xb::matcap_enabled()) {
        return 1.0;
    }
    return clamp(s.reflectivity * s.reflectivity_mask, 0.0, 1.0);
}

/// True when `_ReflectionBlendMode` selects the multiplicative branch for non-matcap reflections.
fn reflection_is_multiplicative() -> bool {
    return abs(xb::mat._ReflectionBlendMode - 1.0) < 0.5;
}

/// Samples one indirect-reflection branch using the current reflection mode.
///
/// Mode `0` ("PBR") routes through the renderer reflection-probe radiance, mode
/// `1` ("baked cubemap") samples the per-material `_BakedCubemap` directly with a roughness-LOD
/// reflection vector, and mode `2` ("matcap") samples `_Matcap` with the view-space matcap UV.
fn indirect_reflection_branch(
    s: xb::SurfaceData,
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    world_pos: vec3<f32>,
    view_layer: u32,
    perceptual_roughness: f32,
    specular_reflectance: vec3<f32>,
    intensity: f32,
    ambient: vec3<f32>,
    dominant_light_col_atten: vec3<f32>,
) -> vec3<f32> {
    if (xb::reflection_disabled()) {
        return vec3<f32>(0.0);
    }

    if (xb::matcap_enabled()) {
        let stereo_view_dir = rg::stereo_center_view_dir_for_world_pos(world_pos, view_layer);
        let uv = matcap_uv(stereo_view_dir, normal);
        let lod = clamp((1.0 - clamp(perceptual_roughness, 0.0, 1.0)) * SPECCUBE_LOD_STEPS, 0.0, SPECCUBE_LOD_STEPS);
        var spec = textureSampleLevel(xb::_Matcap, xb::_Matcap_sampler, uv, lod).rgb * xb::mat._MatcapTint.rgb;
        spec = spec * (ambient + dominant_light_col_atten * 0.5);
        return spec;
    }

    if (xb::baked_cubemap_enabled()) {
        let r = reflect(-view_dir, normal);
        let lod = clamp(
            (1.0 - clamp(perceptual_roughness, 0.0, 1.0)) * SPECCUBE_LOD_STEPS,
            0.0,
            SPECCUBE_LOD_STEPS,
        );
        let sample_r = cubemap_storage::sample_dir(r, xb::mat._BakedCubemap_StorageVInverted);
        var spec = textureSampleLevel(
            xb::_BakedCubemap,
            xb::_BakedCubemap_sampler,
            sample_r,
            lod,
        ).rgb
            * specular_reflectance
            * occlusion_scalar(s)
            * intensity;
        if (!reflection_is_multiplicative()) {
            spec = spec * (ambient + dominant_light_col_atten * 0.5);
        }
        return spec;
    }

    let roughness = clamp(perceptual_roughness, 0.045, 1.0);
    let n_dot_v = clamp(dot(normal, view_dir), 0.0, 1.0);
    let indirect_enabled = rprobe::has_indirect_specular(view_layer, xb::reflection_uses_pbr());
    let dfg = brdf::sample_ibl_dfg_lut(roughness, n_dot_v);
    let specular_energy = brdf::indirect_specular_energy_from_dfg(dfg, specular_reflectance, indirect_enabled);
    let specular_occlusion = brdf::specular_ao_lagarde(n_dot_v, occlusion_scalar(s), roughness);
    let spec = rprobe::indirect_specular_with_energy(
        world_pos,
        normal,
        view_dir,
        roughness,
        specular_energy,
        specular_occlusion,
        indirect_enabled,
        view_layer,
    ) * intensity;
    return spec;
}

/// Indirect-specular contribution including the clear-coat lobe.
fn indirect_specular(
    s: xb::SurfaceData,
    view_dir: vec3<f32>,
    world_pos: vec3<f32>,
    view_layer: u32,
    ambient: vec3<f32>,
    dominant_light_col_atten: vec3<f32>,
) -> vec3<f32> {
    let specular_reflectance = xiexe_specular_reflectance(s);

    var spec = indirect_reflection_branch(
        s,
        s.normal,
        view_dir,
        world_pos,
        view_layer,
        s.roughness,
        specular_reflectance,
        1.0,
        ambient,
        dominant_light_col_atten,
    );

    if (!xb::matcap_enabled() && xb::clearcoat_enabled()) {
        spec = spec + indirect_reflection_branch(
            s,
            s.raw_normal,
            view_dir,
            world_pos,
            view_layer,
            clearcoat_roughness(s),
            vec3<f32>(brdf::DEFAULT_DIELECTRIC_F0),
            s.clearcoat_strength,
            ambient,
            dominant_light_col_atten,
        );
    }

    return spec;
}

/// Applies reflection to the accumulated diffuse surface color.
///
/// XSToon2 exposes `_ReflectionBlendMode`, but its matcap composition always adds the sampled
/// reflection to the surface color.
fn apply_reflection_blend(surface: vec3<f32>, reflection: vec3<f32>, weight: f32) -> vec3<f32> {
    let clamped_weight = clamp(weight, 0.0, 1.0);
    if (clamped_weight <= 1e-4) {
        return surface;
    }

    if (xb::matcap_enabled()) {
        return surface + reflection;
    }

    if (reflection_is_multiplicative()) {
        return mix(surface, surface * reflection, clamped_weight);
    }
    if (abs(xb::mat._ReflectionBlendMode - 2.0) < 0.5) {
        return surface - reflection * clamped_weight;
    }
    return surface + reflection * clamped_weight;
}

/// Approximates XSToon3's scene-brightness measurement from the dominant light and SH ambient.
fn environment_brightness(ambient: vec3<f32>, dominant_light_col_atten: vec3<f32>) -> f32 {
    return (xb::grayscale(ambient) + xb::grayscale(dominant_light_col_atten)) * 0.5;
}

/// Base-pass emission contribution, including `_EmissionToDiffuse` and `_ScaleWithLight`.
fn emission_color(
    s: xb::SurfaceData,
    ambient: vec3<f32>,
    dominant_light_col_atten: vec3<f32>,
    base_pass: bool,
) -> vec3<f32> {
    if (!base_pass || !xb::emission_map_enabled()) {
        return vec3<f32>(0.0);
    }

    var emission = mix(s.emission, s.emission * s.diffuse_color, clamp(xb::mat._EmissionToDiffuse, 0.0, 1.0));
    emission = emission * xb::mat._EmissionColor.rgb;

    if (xb::scale_with_light_enabled()) {
        let sensitivity = clamp(xb::mat._ScaleWithLightSensitivity, 0.0, 1.0);
        let scale = xb::saturate(smoothstep(1.0 - sensitivity, 1.0 + sensitivity, 1.0 - environment_brightness(ambient, dominant_light_col_atten)));
        emission = emission * scale;
    }

    return emission;
}

/// Forward-pass clustered light walk.
///
/// Composition follows the XSToon3 accumulator order while preserving XSToon2's ramp-only diffuse
/// model and existing material properties:
///   `surface = diffuse`
///   `surface *= occlusionColor`
///   `surface = applyReflectionBlend(surface, indirectSpecular, reflectivityWeight)`
///   `surface += directSpecular * occlusion`
///   `surface += rim`
///   `surface += subsurface`
///   `surface *= shadowRim`
///   `surface += emission`
fn clustered_toon_lighting(
    frag_xy: vec2<f32>,
    s: xb::SurfaceData,
    world_pos: vec3<f32>,
    view_layer: u32,
    include_directional: bool,
    include_local: bool,
    base_pass: bool,
) -> vec3<f32> {
    let view_dir = rg::view_dir_for_world_pos(world_pos, view_layer);
    let ambient = indirect_diffuse(s, view_layer);
    let env = environment_tint(s, view_dir, world_pos, view_layer);
    let direct_specular_occlusion = occlusion_scalar(s);
    let primary_specular_terms = primary_direct_specular_terms(s, view_dir);

    let cluster_id = pcls::cluster_id_from_frag(
        frag_xy,
        world_pos,
        rg::frame.view_space_z_coeffs,
        rg::frame.view_space_z_coeffs_right,
        view_layer,
        rg::frame.viewport_width,
        rg::frame.viewport_height,
        rg::frame.cluster_count_x,
        rg::frame.cluster_count_y,
        rg::frame.cluster_count_z,
        rg::frame.near_clip,
        rg::frame.far_clip,
    );
    let count = pcls::cluster_light_count_at(cluster_id);
    let i_max = min(count, pcls::MAX_LIGHTS_PER_TILE);

    var direct_diffuse = vec3<f32>(0.0);
    var direct_spec = vec3<f32>(0.0);
    var sss = vec3<f32>(0.0);

    var dominant_light = xb::LightSample(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0), 0.0, true);
    var dominant_light_col_atten = vec3<f32>(0.0);
    var dominant_weight = -1.0;

    for (var i = 0u; i < i_max; i++) {
        let li = pcls::cluster_light_index_at(cluster_id, i);
        if (li >= rg::frame.light_count) {
            continue;
        }

        let light = sample_light(rg::lights[li], world_pos);
        if ((light.is_directional && !include_directional) || (!light.is_directional && !include_local)) {
            continue;
        }

        let ndl = dot(s.normal, light.direction);
        let ramp = ramp_for_ndl(ndl, light.attenuation, s.ramp_mask);
        let light_col_atten = light.color * light.attenuation;
        direct_diffuse = direct_diffuse + s.albedo.rgb * ramp * light_col_atten;
        direct_spec = direct_spec + direct_specular(s, light, view_dir, primary_specular_terms);
        direct_spec = direct_spec + clearcoat_direct_specular(s, light, view_dir);
        sss = sss + subsurface(s, light, view_dir, ambient);

        let weight = xb::grayscale(light_col_atten * vec3<f32>(xb::saturate(dot(s.normal, light.direction))));
        if (weight > dominant_weight) {
            dominant_weight = weight;
            dominant_light = light;
            dominant_light_col_atten = light_col_atten;
        }
    }

    var surface = direct_diffuse;
    if (base_pass) {
        surface = surface + s.albedo.rgb * ambient;
    }
    surface = surface * s.occlusion;

    if (base_pass) {
        let reflection = indirect_specular(s, view_dir, world_pos, view_layer, ambient, dominant_light_col_atten);
        surface = apply_reflection_blend(surface, reflection, reflection_blend_weight(s));
    }

    surface = surface + direct_spec * direct_specular_occlusion;
    surface = surface + sss;

    if (base_pass) {
        if (dominant_weight > 0.0) {
            surface = surface + rim_light(s, dominant_light, view_dir, ambient, env);
            surface = surface * shadow_rim(s, view_dir, dominant_light, ambient);
        }

        surface = surface + emission_color(s, ambient, dominant_light_col_atten, base_pass);
    }

    return max(surface, vec3<f32>(0.0));
}

/// Outline-pass clustered light walk for the "Lit" outline mode.
///
/// Uses the dominant direct-light term plus SH ambient rather than summing every light, matching
/// the newer XSToon3 "main light + ambient" outline response while preserving the existing shell
/// extrusion pass and property aliases.
fn clustered_outline_lighting(
    frag_xy: vec2<f32>,
    s: xb::SurfaceData,
    world_pos: vec3<f32>,
    view_layer: u32,
) -> vec3<f32> {
    let ambient = indirect_diffuse(s, view_layer);
    let cluster_id = pcls::cluster_id_from_frag(
        frag_xy,
        world_pos,
        rg::frame.view_space_z_coeffs,
        rg::frame.view_space_z_coeffs_right,
        view_layer,
        rg::frame.viewport_width,
        rg::frame.viewport_height,
        rg::frame.cluster_count_x,
        rg::frame.cluster_count_y,
        rg::frame.cluster_count_z,
        rg::frame.near_clip,
        rg::frame.far_clip,
    );
    let count = pcls::cluster_light_count_at(cluster_id);
    let i_max = min(count, pcls::MAX_LIGHTS_PER_TILE);

    var dominant_direct = vec3<f32>(0.0);
    var dominant_weight = -1.0;
    for (var i = 0u; i < i_max; i++) {
        let li = pcls::cluster_light_index_at(cluster_id, i);
        if (li >= rg::frame.light_count) {
            continue;
        }

        let light = sample_light(rg::lights[li], world_pos);
        let ndl = xb::saturate(dot(s.normal, light.direction));
        let direct = xb::saturate(light.attenuation * ndl) * light.color;
        let weight = xb::grayscale(direct);
        if (weight > dominant_weight) {
            dominant_weight = weight;
            dominant_direct = direct;
        }
    }

    return dominant_direct + ambient;
}
