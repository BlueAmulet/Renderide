namespace UnityShaderConverter;

/// <summary>
/// Central place for locating Unity built-in <c>.cginc</c> trees so ShaderLab parsing and <c>slangc</c> use the same search order.
/// </summary>
public static class UnityCgIncludesResolver
{
    /// <summary>
    /// Ordered search paths: optional CLI/env override, <c>UnityBuiltinCGIncludes</c> next to the app, then repo walk from <paramref name="shaderPath"/>.
    /// </summary>
    public static IReadOnlyList<string> GetSearchDirectories(string? cliOrEnvOverride, string shaderPath) =>
        GetSearchDirectories(cliOrEnvOverride, Array.Empty<string>(), shaderPath);

    /// <summary>
    /// Like <see cref="GetSearchDirectories(string?, string)"/> but appends extra include roots (e.g. <c>--extra-cg-include</c>) after the primary Unity tree.
    /// </summary>
    public static IReadOnlyList<string> GetSearchDirectories(
        string? cliOrEnvOverride,
        IEnumerable<string>? extraIncludeRoots,
        string shaderPath)
    {
        var paths = new List<string>();
        TryAddUnityCgRoot(paths, cliOrEnvOverride);
        TryAddUnityCgRoot(paths, TryBundledDirectory());
        TryAddUnityCgRoot(paths, TryWalkFromShader(shaderPath));
        if (extraIncludeRoots is not null)
        {
            foreach (string raw in extraIncludeRoots)
            {
                if (string.IsNullOrWhiteSpace(raw))
                    continue;
                TryAddExistingDirectory(paths, raw.Trim());
            }
        }

        return paths;
    }

    /// <summary>
    /// Returns ordered <c>-I</c> directories for <c>slangc</c>: all Unity CGIncludes candidates plus extra roots.
    /// </summary>
    public static IReadOnlyList<string> ResolveAllForSlang(
        string? cliOrEnvOverride,
        IEnumerable<string>? extraIncludeRoots,
        string shaderPath) =>
        GetSearchDirectories(cliOrEnvOverride, extraIncludeRoots, shaderPath);

    /// <summary>
    /// Returns the first directory that contains <c>UnityCG.cginc</c> (legacy single-<c>-I</c> helper).
    /// </summary>
    public static string? ResolveForSlang(string? cliOrEnvOverride, string shaderPath)
    {
        IReadOnlyList<string> dirs = GetSearchDirectories(cliOrEnvOverride, shaderPath);
        return dirs.Count > 0 ? dirs[0] : null;
    }

    /// <summary>Directory copied to build output as <c>UnityBuiltinCGIncludes</c> (contains <c>UnityCG.cginc</c>).</summary>
    public static string? TryBundledDirectory()
    {
        try
        {
            string candidate = Path.Combine(AppContext.BaseDirectory, "UnityBuiltinCGIncludes");
            return File.Exists(Path.Combine(candidate, "UnityCG.cginc")) ? candidate : null;
        }
        catch (ArgumentException)
        {
            return null;
        }
    }

    private static void TryAddUnityCgRoot(List<string> paths, string? directory)
    {
        if (string.IsNullOrWhiteSpace(directory))
            return;
        try
        {
            string full = Path.GetFullPath(directory.Trim());
            if (!File.Exists(Path.Combine(full, "UnityCG.cginc")))
                return;
            TryAddUniquePath(paths, full);
        }
        catch (ArgumentException)
        {
            // ignored
        }
    }

    private static void TryAddExistingDirectory(List<string> paths, string directory)
    {
        try
        {
            string full = Path.GetFullPath(directory);
            if (!Directory.Exists(full))
                return;
            TryAddUniquePath(paths, full);
        }
        catch (ArgumentException)
        {
        }
    }

    private static void TryAddUniquePath(List<string> paths, string full)
    {
        foreach (string existing in paths)
        {
            if (string.Equals(existing, full, StringComparison.OrdinalIgnoreCase))
                return;
        }

        paths.Add(full);
    }

    /// <summary>
    /// Walks parents of <paramref name="shaderPath"/> to find <c>third_party/UnityShaderParser/.../CGIncludes</c> containing <c>UnityCG.cginc</c>.
    /// </summary>
    private static string? TryWalkFromShader(string shaderPath)
    {
        try
        {
            string? dir = Path.GetDirectoryName(Path.GetFullPath(shaderPath));
            while (!string.IsNullOrEmpty(dir))
            {
                string candidate = Path.Combine(
                    dir,
                    "third_party",
                    "UnityShaderParser",
                    "UnityShaderParser.Tests",
                    "TestShaders",
                    "UnityBuiltinShaders",
                    "CGIncludes");
                if (File.Exists(Path.Combine(candidate, "UnityCG.cginc")))
                    return candidate;

                string nestedRenderide = Path.Combine(
                    dir,
                    "Renderide",
                    "third_party",
                    "UnityShaderParser",
                    "UnityShaderParser.Tests",
                    "TestShaders",
                    "UnityBuiltinShaders",
                    "CGIncludes");
                if (File.Exists(Path.Combine(nestedRenderide, "UnityCG.cginc")))
                    return nestedRenderide;

                dir = Path.GetDirectoryName(dir);
            }
        }
        catch (ArgumentException)
        {
            return null;
        }

        return null;
    }
}
