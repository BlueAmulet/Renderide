using UnityShaderConverter.Analysis;

namespace UnityShaderConverter.Tests;

/// <summary>Tests for <see cref="PragmaParser"/> surface detection and target extraction.</summary>
public sealed class PragmaParserTests
{
    [Fact]
    public void HasSurfacePragma_TrueWhenSurfacePragmaPresent()
    {
        const string program = """
            #pragma surface surf Standard fullforwardshadows addshadow
            void surf();
            """;
        Assert.True(PragmaParser.HasSurfacePragma(program));
    }

    [Fact]
    public void HasSurfacePragma_FalseForVertFragOnly()
    {
        const string program = """
            #pragma vertex vert
            #pragma fragment frag
            """;
        Assert.False(PragmaParser.HasSurfacePragma(program));
    }

    [Fact]
    public void TryGetShaderTarget_ParsesFloat()
    {
        const string program = "#pragma target 3.0\nvoid f();";
        Assert.True(PragmaParser.TryGetShaderTarget(program, out float t));
        Assert.Equal(3.0f, t);
    }
}
