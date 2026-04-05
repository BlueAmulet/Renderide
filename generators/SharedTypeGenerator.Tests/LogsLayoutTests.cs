using Renderide.Generators.Logging;
using Xunit;

namespace SharedTypeGenerator.Tests;

/// <summary>Serializes env-mutating tests in this assembly.</summary>
[CollectionDefinition(nameof(LogsLayoutEnvCollection), DisableParallelization = true)]
public sealed class LogsLayoutEnvCollection;

/// <summary>Tests for log path layout parity with the Rust <c>logger</c> crate.</summary>
[Collection(nameof(LogsLayoutEnvCollection))]
public sealed class LogsLayoutTests
{
    [Fact]
    public void FormatLogFilenameTimestampUtc_matches_rust_length_and_separators()
    {
        string s = LogsLayout.FormatLogFilenameTimestampUtc();
        Assert.Equal(19, s.Length);
        Assert.Contains('_', s);
        string[] parts = s.Split('_', 2);
        Assert.Equal(10, parts[0].Length);
        Assert.Equal(8, parts[1].Length);
        Assert.Equal(2, parts[0].Count(c => c == '-'));
        Assert.Equal(2, parts[1].Count(c => c == '-'));
    }

    [Fact]
    public void ResolveLogsRoot_uses_RENDERIDE_LOGS_ROOT_when_set()
    {
        string temp = Path.Combine(Path.GetTempPath(), "logs_layout_test_" + Guid.NewGuid().ToString("N"));
        try
        {
            Directory.CreateDirectory(temp);
            string prev = Environment.GetEnvironmentVariable(LogsLayout.LogsRootEnvVar) ?? "";
            Environment.SetEnvironmentVariable(LogsLayout.LogsRootEnvVar, temp);
            try
            {
                string root = LogsLayout.ResolveLogsRoot(gitTopLevel: null);
                Assert.Equal(Path.GetFullPath(temp), root);
            }
            finally
            {
                Environment.SetEnvironmentVariable(LogsLayout.LogsRootEnvVar,
                    string.IsNullOrEmpty(prev) ? null : prev);
            }
        }
        finally
        {
            try
            {
                Directory.Delete(temp, true);
            }
            catch
            {
                // ignored
            }
        }
    }
}
