// Renderide clustered forward (Forward+) bindings and helpers for UnityShaderConverter.
// Byte layout matches WGSL/Rust `SceneUniforms`, `GpuLight`, and PBR cluster iteration in
// `crates/renderide/src/gpu/pipeline/shaders/pbr.rs`.
//
// Emitted only when `RENDERIDE_CLUSTER_FORWARD_BINDINGS` is defined (see PassLightModeFilter).

#ifndef RENDERIDE_CLUSTER_FORWARD_SLANG_INCLUDED
#define RENDERIDE_CLUSTER_FORWARD_SLANG_INCLUDED

static const uint RENDERIDE_TILE_SIZE = 16u;
static const uint RENDERIDE_MAX_LIGHTS_PER_TILE = 32u;

/// Mirrors [`crate::gpu::pipeline::uniforms::SceneUniforms`] (64 bytes, std140).
struct RenderideSceneUniforms
{
    float3 view_position;
    float _pad0;
    float4 view_space_z_coeffs;
    uint cluster_count_x;
    uint cluster_count_y;
    uint cluster_count_z;
    float near_clip;
    float far_clip;
    uint light_count;
    uint viewport_width;
    uint viewport_height;
};

/// Mirrors [`crate::render::lights::GpuLight`] (96-byte stride in storage arrays).
struct RenderideGpuLight
{
    float3 position;
    float _pad0;
    float3 direction;
    float _pad1;
    float3 color;
    float intensity;
    float range;
    float spot_cos_half_angle;
    uint light_type;
    uint _pad_before_shadow_params;
    float shadow_strength;
    float shadow_near_plane;
    float shadow_bias;
    float shadow_normal_bias;
    uint shadow_type;
    uint3 _pad_trailing;
};

[[vk::binding(0, 1)]]
cbuffer RenderideSceneBlock
{
    RenderideSceneUniforms renderide_scene;
};

[[vk::binding(1, 1)]]
StructuredBuffer<RenderideGpuLight> renderide_lights;

[[vk::binding(2, 1)]]
StructuredBuffer<uint> renderide_cluster_light_counts;

[[vk::binding(3, 1)]]
StructuredBuffer<uint> renderide_cluster_light_indices;

float RenderidePow5(float x)
{
    float x2 = x * x;
    return x2 * x2 * x;
}

float3 RenderideFresnelSchlick(float cosTheta, float3 f0)
{
    return f0 + (1.0f - f0) * RenderidePow5(1.0f - cosTheta);
}

float RenderideDistributionGgx(float nDotH, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float denom = nDotH * nDotH * (a2 - 1.0f) + 1.0f;
    return a2 / max(denom * denom * 3.14159265f, 0.0001f);
}

float RenderideGeometrySchlickGgx(float nDotV, float roughness)
{
    float r = roughness + 1.0f;
    float k = r * r / 8.0f;
    return nDotV / max(nDotV * (1.0f - k) + k, 0.0001f);
}

float RenderideGeometrySmith(float nDotV, float nDotL, float roughness)
{
    return RenderideGeometrySchlickGgx(nDotV, roughness) * RenderideGeometrySchlickGgx(nDotL, roughness);
}

uint2 RenderideClusterXyFromFrag(float2 fragXy, uint viewportW, uint viewportH)
{
    float maxX = max(float(viewportW) - 0.5f, 0.5f);
    float maxY = max(float(viewportH) - 0.5f, 0.5f);
    float2 pxy = clamp(fragXy, float2(0.5f, 0.5f), float2(maxX, maxY));
    float2 tileF = (pxy - float2(0.5f, 0.5f)) / float2(float(RENDERIDE_TILE_SIZE));
    return uint2(uint(floor(tileF.x)), uint(floor(tileF.y)));
}

/// Cook-Torrance direct lighting from clustered lights (same model as builtin PBR WGSL).
/// Optional entry point for PBS shaders; Unity Standard paths can call this when rewired to cluster data.
float3 RenderideClusterForward_AccumulateLo(
    float3 worldPos,
    float3 worldNormal,
    float4 fragClipPosition,
    float3 baseColor,
    float metallic,
    float roughness)
{
    float3 n = normalize(worldNormal);
    float3 v = normalize(renderide_scene.view_position - worldPos);
    float3 f0 = lerp(float3(0.04f, 0.04f, 0.04f), baseColor, metallic);
    float3 lo = float3(0.0f, 0.0f, 0.0f);

    float viewZ = dot(renderide_scene.view_space_z_coeffs.xyz, worldPos) + renderide_scene.view_space_z_coeffs.w;
    float d = clamp(-viewZ, renderide_scene.near_clip, renderide_scene.far_clip);
    float logArg = renderide_scene.far_clip / renderide_scene.near_clip;
    float zNorm = (logArg > 1.00001f)
        ? (log(d / renderide_scene.near_clip) / log(logArg) * float(renderide_scene.cluster_count_z))
        : 0.0f;
    uint clusterZ = uint(clamp(zNorm, 0.0f, float(max(renderide_scene.cluster_count_z, 1u) - 1u)));

    uint2 clusterXy = RenderideClusterXyFromFrag(
        fragClipPosition.xy,
        renderide_scene.viewport_width,
        renderide_scene.viewport_height);
    uint cx = min(clusterXy.x, max(renderide_scene.cluster_count_x, 1u) - 1u);
    uint cy = min(clusterXy.y, max(renderide_scene.cluster_count_y, 1u) - 1u);
    uint clusterId = cx + renderide_scene.cluster_count_x * (cy + renderide_scene.cluster_count_y * clusterZ);

    uint count = renderide_cluster_light_counts[clusterId];
    uint baseIdx = clusterId * RENDERIDE_MAX_LIGHTS_PER_TILE;

    for (uint i = 0u; i < count; i++)
    {
        uint lightIdx = renderide_cluster_light_indices[baseIdx + i];
        if (lightIdx >= renderide_scene.light_count)
            continue;

        RenderideGpuLight light = renderide_lights[lightIdx];
        float3 lightPos = light.position;
        float3 lightDir = light.direction;
        float3 lightColor = light.color;

        float3 l;
        float attenuation;

        if (light.light_type == 0u)
        {
            float3 toLight = lightPos - worldPos;
            float dist = length(toLight);
            l = normalize(toLight);
            float att = light.intensity / max(dist * dist, 0.0001f);
            float rangeFade = 1.0f - smoothstep(light.range * 0.9f, light.range, dist);
            attenuation = (light.range > 0.0f) ? (att * rangeFade) : 0.0f;
        }
        else if (light.light_type == 1u)
        {
            float dirLenSq = dot(lightDir, lightDir);
            l = (dirLenSq > 1e-16f) ? normalize(-lightDir) : float3(0.0f, 0.0f, 1.0f);
            attenuation = light.intensity;
        }
        else
        {
            float3 toLight = lightPos - worldPos;
            float dist = length(toLight);
            l = normalize(toLight);
            float spotCos = dot(-l, normalize(lightDir));
            float spotAtten = smoothstep(light.spot_cos_half_angle, light.spot_cos_half_angle + 0.1f, spotCos);
            float rangeFade = 1.0f - smoothstep(light.range * 0.9f, light.range, dist);
            float att = light.intensity * spotAtten * rangeFade / max(dist * dist, 0.0001f);
            attenuation = (light.range > 0.0f) ? att : 0.0f;
        }

        float3 h = normalize(v + l);
        float nDotL = max(dot(n, l), 0.0f);
        float nDotV = max(dot(n, v), 0.0001f);
        float nDotH = max(dot(n, h), 0.0f);
        float3 radiance = lightColor * attenuation * nDotL;
        float3 f = RenderideFresnelSchlick(max(dot(h, v), 0.0f), f0);
        // RGB specular term (Fresnel is float3; D and G are scalar — product is float3).
        float3 spec = (RenderideDistributionGgx(nDotH, roughness) * RenderideGeometrySmith(nDotV, nDotL, roughness) * f) /
            max(4.0f * nDotV * nDotL, 0.0001f);
        lo += ((1.0f - f) * (1.0f - metallic) * baseColor / 3.14159265f + spec) * radiance;
    }

    return lo;
}

#endif // RENDERIDE_CLUSTER_FORWARD_SLANG_INCLUDED
