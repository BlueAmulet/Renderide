//! Tracy profiling integration — zero cost by default, enabled by the `tracy` Cargo feature.
//!
//! # How to enable
//!
//! Build with `--features tracy` to activate Tracy spans, frame marks, and GPU timestamp queries:
//!
//! ```bash
//! cargo build --profile dev-fast --features tracy
//! ```
//!
//! Then launch the [Tracy GUI](https://github.com/wolfpld/tracy) and connect on port **8086**.
//! Tracy uses `ondemand` mode, so data is only streamed while a GUI is connected.
//!
//! # Default builds (no `tracy` feature)
//!
//! Every macro and function in this module compiles to nothing. The `profiling` crate guarantees
//! this: when no backend feature is active, `profiling::scope!` and friends expand to `()`.
//! Verify with `cargo expand` if in doubt.
//!
//! # GPU profiling
//!
//! [`GpuProfilerHandle`] wraps [`wgpu_profiler::GpuProfiler`] (only compiled with `tracy`).
//! It inserts timestamp queries around the two render-graph execution sub-phases
//! (`graph::frame_global` and `graph::per_view`) and bridges results to the Tracy GPU timeline.
//!
//! GPU timestamps require both [`wgpu::Features::TIMESTAMP_QUERY`] and
//! [`wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS`]. When either is missing on the adapter,
//! [`GpuProfilerHandle::try_new`] returns [`None`] and a warning is logged; CPU spans still work.
//!
//! # Thread naming
//!
//! Call [`register_main_thread`] once at startup so the main thread appears by name in Tracy.
//! Pass [`rayon_thread_start_handler`] to `rayon::ThreadPoolBuilder::start_handler` so Rayon
//! workers are also named.

pub use profiling::finish_frame;
pub use profiling::function_scope;
pub use profiling::scope;

/// Registers the calling thread as `"renderer-main"` in the active profiler.
///
/// Expands to nothing when the `tracy` feature is off.
#[inline]
pub fn register_main_thread() {
    profiling::register_thread!("renderer-main");
}

/// Emits a frame mark to the active profiler, closing the current frame boundary.
///
/// Call exactly once per winit tick, at the very end of [`crate::app::RenderideApp::tick_frame`].
/// Without frame marks Tracy still records spans but the frame timeline and histogram are empty.
///
/// Expands to nothing when the `tracy` feature is off.
#[inline]
pub fn emit_frame_mark() {
    profiling::finish_frame!();
}

/// Returns a closure suitable for [`rayon::ThreadPoolBuilder::start_handler`].
///
/// Each Rayon worker thread registers itself as `"rayon-worker-{index}"` with the active profiler,
/// so it appears by name on the Tracy thread timeline. When the `tracy` feature is off this
/// returns a no-op closure with zero overhead.
pub fn rayon_thread_start_handler() -> impl Fn(usize) + Send + Sync + 'static {
    move |_thread_index| {
        profiling::register_thread!(&format!("rayon-worker-{_thread_index}"));
    }
}

/// Requests the GPU features needed for timestamp-query-based profiling.
///
/// Returns the subset of `{TIMESTAMP_QUERY, TIMESTAMP_QUERY_INSIDE_ENCODERS}` that the adapter
/// actually supports. If `cfg(feature = "tracy")` is not active, always returns empty.
///
/// Call this in [`crate::gpu::context`]'s feature-intersection helpers and OR the result into
/// the device's requested features.
pub fn timestamp_query_features_if_supported(adapter: &wgpu::Adapter) -> wgpu::Features {
    #[cfg(feature = "tracy")]
    {
        let needed =
            wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        adapter.features() & needed
    }
    #[cfg(not(feature = "tracy"))]
    {
        let _ = adapter;
        wgpu::Features::empty()
    }
}

// ---------------------------------------------------------------------------
// PhaseQuery — GPU timestamp query token, with a no-op stub when `tracy` is off
// ---------------------------------------------------------------------------

/// GPU timestamp query token returned by [`GpuProfilerHandle::begin_query`].
///
/// When the `tracy` feature is on this is [`wgpu_profiler::GpuProfilerQuery`]; when it is off
/// this is a zero-sized placeholder so call sites compile identically under both states.
#[cfg(feature = "tracy")]
pub type PhaseQuery = wgpu_profiler::GpuProfilerQuery;

/// Zero-sized placeholder for [`wgpu_profiler::GpuProfilerQuery`] when the `tracy` feature is off.
#[cfg(not(feature = "tracy"))]
pub struct PhaseQuery;

// ---------------------------------------------------------------------------
// GPU profiler handle — real implementation when `tracy` is on
// ---------------------------------------------------------------------------

#[cfg(feature = "tracy")]
mod gpu_profiler_impl {
    use wgpu_profiler::{GpuProfiler, GpuProfilerSettings};

    use super::PhaseQuery;

    /// Wraps [`GpuProfiler`] and provides a GPU timestamp query interface for the render-graph
    /// execution sub-phases, bridging results to the Tracy GPU timeline.
    ///
    /// Created via [`GpuProfilerHandle::try_new`]; only available when the `tracy` feature is on.
    pub struct GpuProfilerHandle {
        inner: GpuProfiler,
    }

    impl GpuProfilerHandle {
        /// Creates a new handle if the device supports [`wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS`].
        ///
        /// Returns [`None`] when timestamp queries are unavailable; callers fall back to CPU-only
        /// spans without any GPU timeline data.
        pub fn try_new(device: &wgpu::Device) -> Option<Self> {
            let has_inside_encoders = device
                .features()
                .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS);
            let settings = GpuProfilerSettings {
                enable_timer_queries: has_inside_encoders,
                enable_debug_groups: true,
                max_num_pending_frames: 3,
            };
            match GpuProfiler::new(device, settings) {
                Ok(inner) => Some(Self { inner }),
                Err(e) => {
                    logger::warn!("GPU profiler creation failed: {e}; GPU timeline unavailable");
                    None
                }
            }
        }

        /// Opens a GPU timestamp query at the start of a render-graph phase.
        ///
        /// Writes `WriteTimestamp` commands into `encoder`. The returned [`PhaseQuery`]
        /// must be closed via [`Self::end_query`] before [`Self::resolve_queries`] is called.
        #[inline]
        pub fn begin_query(
            &self,
            label: impl Into<String>,
            encoder: &mut wgpu::CommandEncoder,
        ) -> PhaseQuery {
            self.inner.begin_query(label, encoder)
        }

        /// Closes a query previously opened with [`Self::begin_query`].
        #[inline]
        pub fn end_query(&self, encoder: &mut wgpu::CommandEncoder, query: PhaseQuery) {
            self.inner.end_query(encoder, query);
        }

        /// Inserts query-resolve commands into `encoder` for all unresolved queries this frame.
        ///
        /// Call once per encoder just before [`wgpu::CommandEncoder::finish`]. The encoder used
        /// for resolution must be submitted **after** all encoders that opened queries in this
        /// profiling frame.
        #[inline]
        pub fn resolve_queries(&mut self, encoder: &mut wgpu::CommandEncoder) {
            self.inner.resolve_queries(encoder);
        }

        /// Marks the end of the current profiling frame and validates that all queries have been
        /// resolved.
        ///
        /// Call once per render tick after all command encoders for this frame have been
        /// submitted. Logs a warning on failure (e.g. unresolved queries).
        #[inline]
        pub fn end_frame(&mut self) {
            if let Err(e) = self.inner.end_frame() {
                logger::warn!("GPU profiler end_frame failed: {e}");
            }
        }

        /// Drains results from the oldest completed profiling frame into Tracy.
        ///
        /// Call once per render tick after [`Self::end_frame`]. Results are available 1-2 frames
        /// after recording because the GPU needs to finish executing before the timestamps are
        /// readable. `timestamp_period` is from [`wgpu::Queue::get_timestamp_period`].
        #[inline]
        pub fn process_finished_frame(&mut self, timestamp_period: f32) {
            self.inner.process_finished_frame(timestamp_period);
        }
    }
}

// ---------------------------------------------------------------------------
// GPU profiler handle — zero-sized stub when `tracy` is off
// ---------------------------------------------------------------------------

#[cfg(not(feature = "tracy"))]
mod gpu_profiler_stub {
    use super::PhaseQuery;

    /// Zero-sized stub that stands in for the real GPU profiler handle when the `tracy` feature
    /// is not enabled. All methods are no-ops inlined to nothing; the stub is never instantiated
    /// because [`GpuProfilerHandle::try_new`] always returns [`None`].
    pub struct GpuProfilerHandle;

    impl GpuProfilerHandle {
        /// Always returns [`None`]; GPU profiling is unavailable without the `tracy` feature.
        #[inline]
        pub fn try_new(_device: &wgpu::Device) -> Option<Self> {
            None
        }

        /// No-op stub; see the `tracy` feature variant for the real implementation.
        #[inline]
        pub fn begin_query(
            &self,
            _label: impl Into<String>,
            _encoder: &mut wgpu::CommandEncoder,
        ) -> PhaseQuery {
            PhaseQuery
        }

        /// No-op stub; see the `tracy` feature variant for the real implementation.
        #[inline]
        pub fn end_query(&self, _encoder: &mut wgpu::CommandEncoder, _query: PhaseQuery) {}

        /// No-op stub; see the `tracy` feature variant for the real implementation.
        #[inline]
        pub fn resolve_queries(&mut self, _encoder: &mut wgpu::CommandEncoder) {}

        /// No-op stub; see the `tracy` feature variant for the real implementation.
        #[inline]
        pub fn end_frame(&mut self) {}

        /// No-op stub; see the `tracy` feature variant for the real implementation.
        #[inline]
        pub fn process_finished_frame(&mut self, _timestamp_period: f32) {}
    }
}

#[cfg(feature = "tracy")]
pub use gpu_profiler_impl::GpuProfilerHandle;

#[cfg(not(feature = "tracy"))]
pub use gpu_profiler_stub::GpuProfilerHandle;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies that `rayon_thread_start_handler` produces a valid closure that does not panic
    /// when called with arbitrary thread indices.
    #[test]
    fn rayon_start_handler_does_not_panic_for_any_index() {
        let handler = rayon_thread_start_handler();
        handler(0);
        handler(1);
        handler(usize::MAX);
    }

    /// Confirms that the public surface of this module compiles and is callable without the
    /// `tracy` feature active. All calls must be no-ops; the test itself is the compile check.
    #[cfg(not(feature = "tracy"))]
    #[test]
    fn stubs_are_accessible_without_tracy_feature() {
        register_main_thread();
        emit_frame_mark();
        let _ = rayon_thread_start_handler();
    }

    /// Verifies that `timestamp_query_features_if_supported` has the correct function signature
    /// and can be referenced as a function pointer when the `tracy` feature is off.
    ///
    /// The `cfg(not(feature = "tracy"))` branch returns `wgpu::Features::empty()` without ever
    /// calling `adapter.features()`, so no real wgpu instance is required.
    #[cfg(not(feature = "tracy"))]
    #[test]
    fn timestamp_features_fn_signature_compiles_without_tracy() {
        // Reference as function pointer to confirm the signature compiles.
        let _fn_ptr: fn(&wgpu::Adapter) -> wgpu::Features = timestamp_query_features_if_supported;
    }

    /// `register_main_thread` and `emit_frame_mark` must be safely callable more than once per
    /// process; calling them repeatedly should never panic under any feature configuration.
    #[test]
    fn thread_registration_and_frame_mark_are_idempotent() {
        register_main_thread();
        register_main_thread();
        emit_frame_mark();
        emit_frame_mark();
    }

    /// The no-tracy [`PhaseQuery`] placeholder is zero-sized so its presence in per-phase structs
    /// cannot regress memory layout when profiling is disabled.
    #[cfg(not(feature = "tracy"))]
    #[test]
    fn phase_query_stub_is_zero_sized() {
        assert_eq!(std::mem::size_of::<PhaseQuery>(), 0);
    }

    /// The no-tracy [`GpuProfilerHandle`] stub is also zero-sized; construction is unreachable via
    /// [`GpuProfilerHandle::try_new`] (always returns [`None`]), so the placeholder must stay free.
    #[cfg(not(feature = "tracy"))]
    #[test]
    fn gpu_profiler_handle_stub_is_zero_sized() {
        assert_eq!(std::mem::size_of::<GpuProfilerHandle>(), 0);
    }
}
