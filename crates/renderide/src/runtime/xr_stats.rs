//! Runtime-owned OpenXR recoverable failure counters.

/// Cumulative recoverable OpenXR failure counts surfaced to diagnostics.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct RuntimeXrStats {
    /// Count of OpenXR `wait_frame` errors since startup.
    pub(super) wait_frame_failures: u64,
    /// Count of OpenXR `locate_views` errors when rendering was requested.
    pub(super) locate_views_failures: u64,
}

impl RuntimeXrStats {
    /// Increments the OpenXR `wait_frame` failure counter.
    pub(super) fn note_wait_frame_failed(&mut self) {
        self.wait_frame_failures = self.wait_frame_failures.saturating_add(1);
    }

    /// Increments the OpenXR `locate_views` failure counter.
    pub(super) fn note_locate_views_failed(&mut self) {
        self.locate_views_failures = self.locate_views_failures.saturating_add(1);
    }
}

#[cfg(test)]
mod tests {
    use super::RuntimeXrStats;

    #[test]
    fn xr_failure_counters_increment_independently() {
        let mut stats = RuntimeXrStats::default();

        stats.note_wait_frame_failed();
        stats.note_wait_frame_failed();
        stats.note_locate_views_failed();

        assert_eq!(stats.wait_frame_failures, 2);
        assert_eq!(stats.locate_views_failures, 1);
    }
}
