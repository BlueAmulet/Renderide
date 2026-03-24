namespace UnityShaderConverter.Analysis;

/// <summary>
/// Filters ShaderLab code passes by merged <c>LightMode</c> tags for clustered-forward workflows
/// (skip deferred / meta passes) and determines when Renderide cluster-forward bindings are emitted.
/// </summary>
public static class PassLightModeFilter
{
    /// <summary>Unity <c>LightMode</c> values that are not needed for forward raster shading.</summary>
    public static readonly HashSet<string> DefaultNonForwardLightModes = new(StringComparer.OrdinalIgnoreCase)
    {
        "Deferred",
        "DeferredReflections",
        "Meta",
        "MotionVectors",
    };

    /// <summary>
    /// Returns a copy of <paramref name="document"/> whose <see cref="ShaderFileDocument.Passes"/> list
    /// excludes passes matching the filter options. Remaining passes are renumbered by <see cref="ShaderPassDocument.PassIndex"/>.
    /// <see cref="ShaderFileDocument.MultiCompilePragmas"/> is recomputed from the kept passes only.
    /// </summary>
    public static ShaderFileDocument ApplyDocumentFilters(ShaderFileDocument document, PassFilterOptions options)
    {
        if (!options.SkipNonForwardPasses && !options.SkipForwardAddPasses)
            return document;

        var kept = new List<ShaderPassDocument>();
        foreach (ShaderPassDocument pass in document.Passes)
        {
            if (ShouldDropPass(pass, options))
                continue;
            kept.Add(ClonePassWithIndex(pass, kept.Count));
        }

        if (kept.Count == document.Passes.Count)
            return document;

        var multiCompiles = new List<string>();
        foreach (ShaderPassDocument p in kept)
        {
            foreach (string line in ExtractMultiCompileLines(p.ProgramSource))
                multiCompiles.Add(line);
        }

        multiCompiles.Sort(StringComparer.Ordinal);
        return new ShaderFileDocument
        {
            SourcePath = document.SourcePath,
            ShaderName = document.ShaderName,
            Properties = document.Properties,
            SubShaderTags = document.SubShaderTags,
            Passes = kept,
            MultiCompilePragmas = DeduplicateSorted(multiCompiles),
            AnalyzerWarnings = document.AnalyzerWarnings,
            TotalSubShaderCount = document.TotalSubShaderCount,
        };
    }

    /// <summary>
    /// Copies <paramref name="variantDefines"/> and appends <c>RENDERIDE_CLUSTER_FORWARD_BINDINGS</c> when
    /// <see cref="PassNeedsRenderideClusterForwardBindings"/> is true for <paramref name="pass"/>.
    /// </summary>
    public static List<string> MergeVariantDefines(IReadOnlyList<string> variantDefines, ShaderPassDocument pass)
    {
        var copy = new List<string>();
        foreach (string d in variantDefines)
        {
            if (d.Length > 0)
                copy.Add(d);
        }

        if (PassNeedsRenderideClusterForwardBindings(pass))
            copy.Add("RENDERIDE_CLUSTER_FORWARD_BINDINGS");
        return copy;
    }

    /// <summary>
    /// When true, Slang should define <c>RENDERIDE_CLUSTER_FORWARD_BINDINGS</c> so
    /// <c>RenderideClusterForward.slang</c> declares group-1 resources matching Renderide PBR scene layout.
    /// </summary>
    public static bool PassNeedsRenderideClusterForwardBindings(ShaderPassDocument pass)
    {
        if (!pass.FixedFunctionState.EffectiveTags.TryGetValue("LightMode", out string? lm) ||
            string.IsNullOrWhiteSpace(lm))
            return false;

        lm = lm.Trim();
        if (lm.Equals("ShadowCaster", StringComparison.OrdinalIgnoreCase))
            return false;
        if (lm.Equals("Deferred", StringComparison.OrdinalIgnoreCase))
            return false;
        if (lm.Equals("DeferredReflections", StringComparison.OrdinalIgnoreCase))
            return false;
        if (lm.Equals("Meta", StringComparison.OrdinalIgnoreCase))
            return false;
        if (lm.Equals("MotionVectors", StringComparison.OrdinalIgnoreCase))
            return false;
        if (lm.Equals("DepthOnly", StringComparison.OrdinalIgnoreCase))
            return false;
        return true;
    }

    private static bool ShouldDropPass(ShaderPassDocument pass, PassFilterOptions options)
    {
        if (!pass.FixedFunctionState.EffectiveTags.TryGetValue("LightMode", out string? lm) ||
            string.IsNullOrWhiteSpace(lm))
            return false;

        lm = lm.Trim();
        if (options.SkipNonForwardPasses && DefaultNonForwardLightModes.Contains(lm))
            return true;
        return options.SkipForwardAddPasses &&
               lm.Equals("ForwardAdd", StringComparison.OrdinalIgnoreCase);
    }

    private static ShaderPassDocument ClonePassWithIndex(ShaderPassDocument p, int newIndex) =>
        new()
        {
            PassName = p.PassName,
            PassIndex = newIndex,
            ProgramSource = p.ProgramSource,
            Pragmas = p.Pragmas,
            VertexEntry = p.VertexEntry,
            FragmentEntry = p.FragmentEntry,
            RenderStateSummary = p.RenderStateSummary,
            FixedFunctionState = p.FixedFunctionState,
            PragmaShaderTarget = p.PragmaShaderTarget,
        };

    private static IReadOnlyList<string> DeduplicateSorted(List<string> lines)
    {
        var unique = new List<string>();
        string? prev = null;
        foreach (string line in lines)
        {
            if (line == prev)
                continue;
            unique.Add(line);
            prev = line;
        }

        return unique;
    }

    private static IEnumerable<string> ExtractMultiCompileLines(string program)
    {
        foreach (string line in program.Split('\n'))
        {
            string trimmed = line.TrimStart();
            if (trimmed.StartsWith("#pragma multi_compile", StringComparison.Ordinal) ||
                trimmed.StartsWith("#pragma shader_feature", StringComparison.Ordinal) ||
                trimmed.StartsWith("#pragma multi_compile_", StringComparison.Ordinal))
                yield return trimmed;
        }
    }
}

/// <summary>Controls optional pass dropping before Slang / Rust emission.</summary>
public readonly struct PassFilterOptions
{
    /// <summary>When true, drops passes whose <c>LightMode</c> is deferred, meta, motion vectors, etc.</summary>
    public bool SkipNonForwardPasses { get; init; }

    /// <summary>When true, drops <c>ForwardAdd</c> passes (use only after clustered base lighting).</summary>
    public bool SkipForwardAddPasses { get; init; }
}
