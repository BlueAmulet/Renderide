using System.Text.Json;
using System.Text.Json.Serialization;

namespace UnityShaderConverter;

/// <summary>
/// Skips <c>slangc</c> when the source shader file has not changed since the last successful compile (per-module manifest).
/// </summary>
/// <remarks>Does not track nested <c>#include</c> mtimes; use <c>--compile-cache</c> as a development accelerator only.</remarks>
public static class ShaderCompileCache
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = false,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };

    /// <summary>When manifest matches <paramref name="shaderPath"/> write time and every WGSL file exists, returns true.</summary>
    public static bool TrySkipCompile(
        string cacheDirectory,
        string shaderPath,
        IReadOnlyList<string> wgslOutputPaths,
        out string? reason)
    {
        reason = null;
        if (string.IsNullOrWhiteSpace(cacheDirectory) || !Directory.Exists(cacheDirectory))
            return false;
        if (wgslOutputPaths.Count == 0)
            return false;
        try
        {
            string modKey = SanitizeKey(Path.GetFileNameWithoutExtension(shaderPath));
            string manifestPath = Path.Combine(cacheDirectory, modKey + ".json");
            if (!File.Exists(manifestPath))
            {
                reason = "no manifest";
                return false;
            }

            long shaderTicks = File.GetLastWriteTimeUtc(shaderPath).Ticks;
            string json = File.ReadAllText(manifestPath);
            Manifest? m = JsonSerializer.Deserialize<Manifest>(json, JsonOptions);
            if (m is null || m.ShaderTicks != shaderTicks || m.ShaderFullPath != Path.GetFullPath(shaderPath))
            {
                reason = "manifest stale or path mismatch";
                return false;
            }

            foreach (string wgsl in wgslOutputPaths)
            {
                if (!File.Exists(wgsl) || new FileInfo(wgsl).Length == 0)
                {
                    reason = $"missing or empty {wgsl}";
                    return false;
                }
            }

            return true;
        }
        catch (IOException)
        {
            reason = "io error reading cache";
            return false;
        }
        catch (UnauthorizedAccessException)
        {
            reason = "access denied";
            return false;
        }
    }

    /// <summary>Writes a manifest after a successful compile of all passes.</summary>
    public static void WriteManifest(string cacheDirectory, string shaderPath)
    {
        if (string.IsNullOrWhiteSpace(cacheDirectory))
            return;
        try
        {
            Directory.CreateDirectory(cacheDirectory);
            string modKey = SanitizeKey(Path.GetFileNameWithoutExtension(shaderPath));
            var m = new Manifest
            {
                ShaderFullPath = Path.GetFullPath(shaderPath),
                ShaderTicks = File.GetLastWriteTimeUtc(shaderPath).Ticks,
            };
            string manifestPath = Path.Combine(cacheDirectory, modKey + ".json");
            File.WriteAllText(manifestPath, JsonSerializer.Serialize(m, JsonOptions));
        }
        catch
        {
            // ignored
        }
    }

    private static string SanitizeKey(string name)
    {
        foreach (char c in Path.GetInvalidFileNameChars())
            name = name.Replace(c, '_');
        return name.Length > 0 ? name : "shader";
    }

    private sealed class Manifest
    {
        public string ShaderFullPath { get; set; } = "";
        public long ShaderTicks { get; set; }
    }
}
