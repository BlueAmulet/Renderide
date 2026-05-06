//! Cooperative renderer shutdown hooks.

use super::RendererRuntime;

impl RendererRuntime {
    /// Starts nonblocking teardown for runtime-owned resources that can outlive the frame loop.
    pub(crate) fn begin_graceful_shutdown(&mut self) {
        profiling::scope!("runtime::begin_graceful_shutdown");
        self.backend.begin_video_shutdown();
    }

    /// Returns `true` once runtime-owned shutdown work has quiesced.
    pub(crate) fn graceful_shutdown_complete(&mut self) -> bool {
        profiling::scope!("runtime::graceful_shutdown_complete");
        self.backend.video_shutdown_complete()
    }
}
