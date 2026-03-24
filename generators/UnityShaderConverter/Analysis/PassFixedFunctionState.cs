using UnityShaderParser.ShaderLab;

namespace UnityShaderConverter.Analysis;

/// <summary>Concrete ShaderLab fixed-function state for one pass when values are not property-driven.</summary>
public sealed class PassFixedFunctionState
{
    /// <summary>Default state when no pass commands were parsed.</summary>
    public static PassFixedFunctionState Empty { get; } = new();

    /// <summary>Cull mode when set; <c>null</c> means Unity default (Back) unless <see cref="CullReferencesProperty"/>.</summary>
    public CullMode? CullMode { get; init; }

    /// <summary>True when <c>Cull</c> references a material property.</summary>
    public bool CullReferencesProperty { get; init; }

    /// <summary>Depth write when set; <c>null</c> means default On.</summary>
    public bool? DepthWrite { get; init; }

    /// <summary>True when <c>ZWrite</c> uses a property reference.</summary>
    public bool DepthWriteReferencesProperty { get; init; }

    /// <summary>Depth test when set; <c>null</c> means default LEqual.</summary>
    public ComparisonMode? DepthTest { get; init; }

    /// <summary>True when <c>ZTest</c> uses a property reference.</summary>
    public bool DepthTestReferencesProperty { get; init; }

    /// <summary>RT0 blend when parsed; <c>null</c> if never set or not representable.</summary>
    public PassBlendStateRt0? BlendRt0 { get; init; }

    /// <summary>Stencil block when fully concrete; <c>null</c> if absent or any field references a property.</summary>
    public PassStencilConcrete? Stencil { get; init; }

    /// <summary>True if a stencil command referenced a property.</summary>
    public bool StencilReferencesProperty { get; init; }

    /// <summary><c>ColorMask</c> string when concrete (e.g. RGBA, 0).</summary>
    public string? ColorMask { get; init; }

    /// <summary>True when color mask uses a property.</summary>
    public bool ColorMaskReferencesProperty { get; init; }

    /// <summary><c>Offset</c> factor and units when concrete.</summary>
    public (float Factor, float Units)? DepthBias { get; init; }

    /// <summary>True if offset references a property.</summary>
    public bool DepthBiasReferencesProperty { get; init; }

    /// <summary><c>LOD</c> level when set.</summary>
    public int? Lod { get; init; }

    /// <summary>Merged <c>Queue</c>, <c>RenderType</c>, etc.: pass <c>Tags</c> override subshader tags.</summary>
    public IReadOnlyDictionary<string, string> EffectiveTags { get; init; } =
        new Dictionary<string, string>(StringComparer.Ordinal);

    /// <summary>ShaderLab uniform name when <see cref="CullReferencesProperty"/> is true (e.g. <c>_Cull</c>).</summary>
    public string? CullPropertyUniformName { get; init; }

    /// <summary>ShaderLab uniform when <see cref="DepthWriteReferencesProperty"/> is true.</summary>
    public string? DepthWritePropertyUniformName { get; init; }

    /// <summary>ShaderLab uniform when <see cref="DepthTestReferencesProperty"/> is true.</summary>
    public string? DepthTestPropertyUniformName { get; init; }

    /// <summary>ShaderLab uniform when <see cref="ColorMaskReferencesProperty"/> is true.</summary>
    public string? ColorMaskPropertyUniformName { get; init; }

    /// <summary>ShaderLab uniform for <c>Offset</c> factor when referenced.</summary>
    public string? DepthBiasFactorPropertyUniformName { get; init; }

    /// <summary>ShaderLab uniform for <c>Offset</c> units when referenced.</summary>
    public string? DepthBiasUnitsPropertyUniformName { get; init; }

    /// <summary>Stencil <c>Ref</c> property name when referenced.</summary>
    public string? StencilRefPropertyUniformName { get; init; }

    /// <summary>Stencil <c>Comp</c> property name when referenced.</summary>
    public string? StencilCompPropertyUniformName { get; init; }

    /// <summary>Stencil <c>Pass</c> property name when referenced.</summary>
    public string? StencilPassPropertyUniformName { get; init; }

    /// <summary>Stencil <c>ReadMask</c> property name when referenced.</summary>
    public string? StencilReadMaskPropertyUniformName { get; init; }

    /// <summary>Stencil <c>WriteMask</c> property name when referenced.</summary>
    public string? StencilWriteMaskPropertyUniformName { get; init; }

    /// <summary>True when any fixed-function field is driven by a material property and needs runtime pipeline state.</summary>
    public bool HasDynamicRenderState() =>
        CullReferencesProperty ||
        DepthWriteReferencesProperty ||
        DepthTestReferencesProperty ||
        (BlendRt0 is { HasPropertyReference: true }) ||
        StencilReferencesProperty ||
        ColorMaskReferencesProperty ||
        DepthBiasReferencesProperty;
}

/// <summary>Blend state for render target 0.</summary>
public sealed class PassBlendStateRt0
{
    /// <summary><c>Blend Off</c> or disabled.</summary>
    public bool BlendDisabled { get; init; }

    /// <summary>True if any factor is a property reference.</summary>
    public bool HasPropertyReference { get; init; }

    public BlendFactor? SourceRgb { get; init; }
    public BlendFactor? DestRgb { get; init; }
    public BlendFactor? SourceAlpha { get; init; }
    public BlendFactor? DestAlpha { get; init; }

    /// <summary>Material property for RGB source when <see cref="HasPropertyReference"/>.</summary>
    public string? SrcRgbPropertyUniformName { get; init; }

    /// <summary>Material property for RGB destination when referenced.</summary>
    public string? DstRgbPropertyUniformName { get; init; }

    /// <summary>Material property for alpha source when referenced.</summary>
    public string? SrcAlphaPropertyUniformName { get; init; }

    /// <summary>Material property for alpha destination when referenced.</summary>
    public string? DstAlphaPropertyUniformName { get; init; }
}

/// <summary>Stencil state when all fields are literal.</summary>
public sealed class PassStencilConcrete
{
    public byte Ref { get; init; }
    public byte ReadMask { get; init; }
    public byte WriteMask { get; init; }
    public ComparisonMode CompFront { get; init; }
    public StencilOp PassFront { get; init; }
    public StencilOp FailFront { get; init; }
    public StencilOp ZFailFront { get; init; }
    public ComparisonMode CompBack { get; init; }
    public StencilOp PassBack { get; init; }
    public StencilOp FailBack { get; init; }
    public StencilOp ZFailBack { get; init; }
}
