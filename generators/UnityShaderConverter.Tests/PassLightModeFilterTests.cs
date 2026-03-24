using UnityShaderConverter.Analysis;

namespace UnityShaderConverter.Tests;

/// <summary>Tests for <see cref="PassLightModeFilter"/>.</summary>
public sealed class PassLightModeFilterTests
{
    [Fact]
    public void ApplyDocumentFilters_SkipNonForward_RemovesDeferredPass()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TestData", "ForwardDeferredTwoPasses.shader");
        Assert.True(File.Exists(path), $"Missing test file: {path}");
        bool ok = ShaderLabAnalyzer.TryAnalyze(path, out var doc, out var diags, out var errors);
        Assert.True(ok, string.Join("; ", errors) + string.Join("; ", diags));
        Assert.NotNull(doc);
        Assert.Equal(2, doc!.Passes.Count);

        ShaderFileDocument filtered = PassLightModeFilter.ApplyDocumentFilters(
            doc,
            new PassFilterOptions { SkipNonForwardPasses = true, SkipForwardAddPasses = false });

        Assert.Single(filtered.Passes);
        Assert.Equal(0, filtered.Passes[0].PassIndex);
        Assert.Equal("ForwardBase", filtered.Passes[0].FixedFunctionState.EffectiveTags["LightMode"]);
    }

    [Fact]
    public void ApplyDocumentFilters_Disabled_KeepsAllPasses()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TestData", "ForwardDeferredTwoPasses.shader");
        bool ok = ShaderLabAnalyzer.TryAnalyze(path, out var doc, out _, out _);
        Assert.True(ok);
        ShaderFileDocument filtered = PassLightModeFilter.ApplyDocumentFilters(
            doc!,
            new PassFilterOptions { SkipNonForwardPasses = false, SkipForwardAddPasses = false });
        Assert.Same(doc, filtered);
    }

    [Fact]
    public void PassNeedsRenderideClusterForwardBindings_ForwardBase_True()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TestData", "ForwardDeferredTwoPasses.shader");
        bool ok = ShaderLabAnalyzer.TryAnalyze(path, out var doc, out _, out _);
        Assert.True(ok);
        Assert.True(PassLightModeFilter.PassNeedsRenderideClusterForwardBindings(doc!.Passes[0]));
        Assert.False(PassLightModeFilter.PassNeedsRenderideClusterForwardBindings(doc.Passes[1]));
    }

    [Fact]
    public void PassNeedsRenderideClusterForwardBindings_MinimalUnlit_NoLightMode_False()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TestData", "MinimalUnlit.shader");
        bool ok = ShaderLabAnalyzer.TryAnalyze(path, out var doc, out _, out _);
        Assert.True(ok);
        Assert.False(PassLightModeFilter.PassNeedsRenderideClusterForwardBindings(doc!.Passes[0]));
    }

    [Fact]
    public void MergeVariantDefines_AddsClusterBindingDefineForForwardBase()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TestData", "ForwardDeferredTwoPasses.shader");
        bool ok = ShaderLabAnalyzer.TryAnalyze(path, out var doc, out _, out _);
        Assert.True(ok);
        var merged = PassLightModeFilter.MergeVariantDefines(new[] { "FOO" }, doc!.Passes[0]);
        Assert.Equal(new[] { "FOO", "RENDERIDE_CLUSTER_FORWARD_BINDINGS" }, merged);
    }
}
