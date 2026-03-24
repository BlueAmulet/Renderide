using UnityShaderParser.ShaderLab;

namespace UnityShaderConverter.Analysis;

/// <summary>Builds <see cref="PassFixedFunctionState"/> from ShaderLab pass commands and subshader tags.</summary>
public static class RenderStateExtractor
{
    /// <summary>Walks pass commands (last wins) and merges tags with <paramref name="subShaderTags"/>.</summary>
    public static PassFixedFunctionState Extract(
        IReadOnlyList<ShaderLabCommandNode>? commands,
        IReadOnlyDictionary<string, string> subShaderTags)
    {
        var passTags = new Dictionary<string, string>(StringComparer.Ordinal);
        CullMode? cull = null;
        bool cullProp = false;
        string? cullPropName = null;
        bool? zwrite = null;
        bool zwriteProp = false;
        string? zwritePropName = null;
        ComparisonMode? ztest = null;
        bool ztestProp = false;
        string? ztestPropName = null;
        PassBlendStateRt0? blend = null;
        PassStencilConcrete? stencil = null;
        bool stencilProp = false;
        string? stencilRefProp = null;
        string? stencilCompProp = null;
        string? stencilPassProp = null;
        string? stencilReadMaskProp = null;
        string? stencilWriteMaskProp = null;
        string? colorMask = null;
        bool colorMaskProp = false;
        string? colorMaskPropName = null;
        (float, float)? offset = null;
        bool offsetProp = false;
        string? offsetFactorProp = null;
        string? offsetUnitsProp = null;
        int? lod = null;

        foreach (ShaderLabCommandNode cmd in commands ?? Array.Empty<ShaderLabCommandNode>())
        {
            switch (cmd)
            {
                case ShaderLabCommandTagsNode tn when tn.Tags is not null:
                    foreach (KeyValuePair<string, string> kv in tn.Tags)
                        passTags[kv.Key] = kv.Value;
                    break;
                case ShaderLabCommandCullNode c:
                    if (c.Mode.IsPropertyReference)
                    {
                        cullProp = true;
                        cull = null;
                        cullPropName = c.Mode.Property;
                    }
                    else
                    {
                        cullProp = false;
                        cull = c.Mode.Value;
                        cullPropName = null;
                    }

                    break;
                case ShaderLabCommandZWriteNode zw:
                    if (zw.Enabled.IsPropertyReference)
                    {
                        zwriteProp = true;
                        zwrite = null;
                        zwritePropName = zw.Enabled.Property;
                    }
                    else
                    {
                        zwriteProp = false;
                        zwrite = zw.Enabled.Value;
                        zwritePropName = null;
                    }

                    break;
                case ShaderLabCommandZTestNode zt:
                    if (zt.Mode.IsPropertyReference)
                    {
                        ztestProp = true;
                        ztest = null;
                        ztestPropName = zt.Mode.Property;
                    }
                    else
                    {
                        ztestProp = false;
                        ztest = zt.Mode.Value;
                        ztestPropName = null;
                    }

                    break;
                case ShaderLabCommandBlendNode b when b.RenderTarget == 0:
                    blend = ExtractBlendRt0(b);
                    break;
                case ShaderLabCommandColorMaskNode cm:
                    if (cm.Mask.IsPropertyReference)
                    {
                        colorMaskProp = true;
                        colorMask = null;
                        colorMaskPropName = cm.Mask.Property;
                    }
                    else
                    {
                        colorMaskProp = false;
                        colorMask = cm.Mask.Value;
                        colorMaskPropName = null;
                    }

                    break;
                case ShaderLabCommandOffsetNode off:
                    if (off.Factor.IsPropertyReference || off.Units.IsPropertyReference)
                    {
                        offsetProp = true;
                        offset = null;
                        offsetFactorProp = off.Factor.IsPropertyReference ? off.Factor.Property : null;
                        offsetUnitsProp = off.Units.IsPropertyReference ? off.Units.Property : null;
                    }
                    else
                    {
                        offsetProp = false;
                        offset = (off.Factor.Value, off.Units.Value);
                        offsetFactorProp = null;
                        offsetUnitsProp = null;
                    }

                    break;
                case ShaderLabCommandStencilNode st:
                    if (StencilAnyPropertyReference(st))
                    {
                        stencilProp = true;
                        stencil = null;
                        stencilRefProp = st.Ref.IsPropertyReference ? st.Ref.Property : null;
                        stencilCompProp = st.ComparisonOperationFront.IsPropertyReference ? st.ComparisonOperationFront.Property : null;
                        stencilPassProp = st.PassOperationFront.IsPropertyReference ? st.PassOperationFront.Property : null;
                        stencilReadMaskProp = st.ReadMask.IsPropertyReference ? st.ReadMask.Property : null;
                        stencilWriteMaskProp = st.WriteMask.IsPropertyReference ? st.WriteMask.Property : null;
                    }
                    else
                    {
                        stencilProp = false;
                        stencil = BuildStencilConcrete(st);
                        stencilRefProp = null;
                        stencilCompProp = null;
                        stencilPassProp = null;
                        stencilReadMaskProp = null;
                        stencilWriteMaskProp = null;
                    }

                    break;
                case ShaderLabCommandLodNode l:
                    lod = l.LodLevel;
                    break;
            }
        }

        var effective = new Dictionary<string, string>(subShaderTags, StringComparer.Ordinal);
        foreach (KeyValuePair<string, string> kv in passTags)
            effective[kv.Key] = kv.Value;

        return new PassFixedFunctionState
        {
            CullMode = cull,
            CullReferencesProperty = cullProp,
            CullPropertyUniformName = cullPropName,
            DepthWrite = zwrite,
            DepthWriteReferencesProperty = zwriteProp,
            DepthWritePropertyUniformName = zwritePropName,
            DepthTest = ztest,
            DepthTestReferencesProperty = ztestProp,
            DepthTestPropertyUniformName = ztestPropName,
            BlendRt0 = blend,
            Stencil = stencil,
            StencilReferencesProperty = stencilProp,
            StencilRefPropertyUniformName = stencilRefProp,
            StencilCompPropertyUniformName = stencilCompProp,
            StencilPassPropertyUniformName = stencilPassProp,
            StencilReadMaskPropertyUniformName = stencilReadMaskProp,
            StencilWriteMaskPropertyUniformName = stencilWriteMaskProp,
            ColorMask = colorMask,
            ColorMaskReferencesProperty = colorMaskProp,
            ColorMaskPropertyUniformName = colorMaskPropName,
            DepthBias = offset,
            DepthBiasReferencesProperty = offsetProp,
            DepthBiasFactorPropertyUniformName = offsetFactorProp,
            DepthBiasUnitsPropertyUniformName = offsetUnitsProp,
            Lod = lod,
            EffectiveTags = effective,
        };
    }

    private static PassBlendStateRt0 ExtractBlendRt0(ShaderLabCommandBlendNode b)
    {
        if (!b.Enabled)
        {
            return new PassBlendStateRt0 { BlendDisabled = true };
        }

        bool prop = IsBlendProp(b.SourceFactorRGB) || IsBlendProp(b.DestinationFactorRGB) ||
                    IsBlendProp(b.SourceFactorAlpha) || IsBlendProp(b.DestinationFactorAlpha);
        string? srcRgbP = BlendPropName(b.SourceFactorRGB);
        string? dstRgbP = BlendPropName(b.DestinationFactorRGB);
        string? srcAlphaP = BlendPropName(b.SourceFactorAlpha);
        string? dstAlphaP = BlendPropName(b.DestinationFactorAlpha);
        PassBlendStateRt0 state = new()
        {
            BlendDisabled = false,
            HasPropertyReference = prop,
            SourceRgb = FactorOrNull(b.SourceFactorRGB),
            DestRgb = FactorOrNull(b.DestinationFactorRGB),
            SourceAlpha = FactorOrNull(b.SourceFactorAlpha),
            DestAlpha = FactorOrNull(b.DestinationFactorAlpha),
            SrcRgbPropertyUniformName = srcRgbP,
            DstRgbPropertyUniformName = dstRgbP,
            SrcAlphaPropertyUniformName = srcAlphaP,
            DstAlphaPropertyUniformName = dstAlphaP,
        };
        return state;
    }

    private static string? BlendPropName(PropertyReferenceOr<BlendFactor>? f) =>
        f is { IsPropertyReference: true } p ? p.Property : null;

    private static bool IsBlendProp(PropertyReferenceOr<BlendFactor>? f) =>
        f is { IsPropertyReference: true };

    private static BlendFactor? FactorOrNull(PropertyReferenceOr<BlendFactor>? f)
    {
        if (f is null || f.Value.IsPropertyReference)
            return null;
        return f.Value.Value;
    }

    private static bool StencilAnyPropertyReference(ShaderLabCommandStencilNode st) =>
        st.Ref.IsPropertyReference || st.ReadMask.IsPropertyReference || st.WriteMask.IsPropertyReference ||
        st.ComparisonOperationFront.IsPropertyReference || st.PassOperationFront.IsPropertyReference ||
        st.FailOperationFront.IsPropertyReference || st.ZFailOperationFront.IsPropertyReference ||
        st.ComparisonOperationBack.IsPropertyReference || st.PassOperationBack.IsPropertyReference ||
        st.FailOperationBack.IsPropertyReference || st.ZFailOperationBack.IsPropertyReference;

    private static PassStencilConcrete BuildStencilConcrete(ShaderLabCommandStencilNode st) =>
        new()
        {
            Ref = st.Ref.Value,
            ReadMask = st.ReadMask.Value,
            WriteMask = st.WriteMask.Value,
            CompFront = st.ComparisonOperationFront.Value,
            PassFront = st.PassOperationFront.Value,
            FailFront = st.FailOperationFront.Value,
            ZFailFront = st.ZFailOperationFront.Value,
            CompBack = st.ComparisonOperationBack.Value,
            PassBack = st.PassOperationBack.Value,
            FailBack = st.FailOperationBack.Value,
            ZFailBack = st.ZFailOperationBack.Value,
        };
}
