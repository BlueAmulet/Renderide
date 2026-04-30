//! Per-view encoder recording strategy (serial vs rayon-parallel).

use crate::labeled_enum;

labeled_enum! {
    /// Controls whether per-view encoder recording uses rayon for parallelism.
    ///
    /// The default [`RecordParallelism::PerViewParallel`] records per-view encoders on rayon
    /// workers for CPU-side speedup on stereo / multi-camera scenes. Switch to
    /// [`RecordParallelism::Serial`] only for debugging or when isolating regressions.
    pub enum RecordParallelism: "record parallelism" {
        default => PerViewParallel;

        /// Record each per-view encoder sequentially on the main thread. Safe and debuggable.
        ///
        /// Canonical persist string is `"serial"` (snake_case) to match the project-wide TOML
        /// convention. The historical `"Serial"` PascalCase token from the pre-`rename_all`
        /// derive is accepted as an alias so existing config files keep loading; the next
        /// HUD-triggered save migrates them to snake_case.
        Serial => {
            persist: "serial",
            label: "Serial",
            aliases: ["Serial"],
        },
        /// Record each per-view encoder on a rayon worker thread. Requires all per-view pass
        /// nodes to be `Send` (enforced at compile time by the trait bound on
        /// [`crate::render_graph::PassNode`]).
        ///
        /// Canonical persist string is `"per_view_parallel"` (snake_case); `"PerViewParallel"`
        /// (the historical PascalCase form from the pre-`rename_all` derive) is accepted as an
        /// alias for backwards-compat loading.
        PerViewParallel => {
            persist: "per_view_parallel",
            label: "Per-view parallel",
            aliases: ["PerViewParallel"],
        },
    }
}

#[cfg(test)]
mod tests {
    use super::RecordParallelism;
    use crate::config::types::RendererSettings;

    #[test]
    fn parses_legacy_pascal_case_and_snake_case() {
        for token in ["PerViewParallel", "per_view_parallel"] {
            let toml = format!("[rendering]\nrecord_parallelism = \"{token}\"\n");
            let parsed: RendererSettings = toml::from_str(&toml).expect(token);
            assert_eq!(
                parsed.rendering.record_parallelism,
                RecordParallelism::PerViewParallel,
                "token `{token}` must resolve to PerViewParallel"
            );
        }
        for token in ["Serial", "serial"] {
            let toml = format!("[rendering]\nrecord_parallelism = \"{token}\"\n");
            let parsed: RendererSettings = toml::from_str(&toml).expect(token);
            assert_eq!(
                parsed.rendering.record_parallelism,
                RecordParallelism::Serial,
                "token `{token}` must resolve to Serial"
            );
        }
    }

    #[test]
    fn serializes_in_canonical_snake_case() {
        let mut s = RendererSettings::default();
        s.rendering.record_parallelism = RecordParallelism::PerViewParallel;
        let toml = toml::to_string(&s).expect("serialize");
        assert!(
            toml.contains("record_parallelism = \"per_view_parallel\""),
            "expected snake_case persist string for project-wide convention parity, got:\n{toml}"
        );
    }
}
