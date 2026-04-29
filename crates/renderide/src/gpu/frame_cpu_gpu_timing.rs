//! Frame timing for the debug HUD: CPU per-frame work, GPU per-frame work, and the wall-clock
//! roundtrip between consecutive frame starts.
//!
//! Three numbers are tracked, with the boundaries chosen so their meaning is unambiguous on a
//! pipelined renderer that submits GPU work from a driver thread:
//!
//! - **CPU frame ms** — wall-clock from the start of the winit tick (when the main thread begins
//!   per-frame work) to the moment [`wgpu::Queue::submit`] returns on the driver thread for the
//!   tick's *last* tracked submit. This captures everything the CPU does to prepare the frame,
//!   including the driver-thread submit overhead, but not GPU execution.
//! - **GPU frame ms** — wall-clock from that same `Queue::submit` return to the
//!   [`wgpu::Queue::on_submitted_work_done`] callback firing for that submit. This is the closest
//!   wgpu can give us to "GPU finished all of this frame's work" without timestamp queries.
//! - **Roundtrip ms** — wall-clock between consecutive winit ticks. Tracked outside this struct
//!   ([`crate::diagnostics::FrameTimingHudSnapshot::wall_frame_time_ms`]).
//!
//! Both CPU and GPU values are populated **on the driver thread**, which means they may arrive
//! after the originating winit tick has already ended its [`FrameCpuGpuTiming::end_frame`]. The
//! HUD reads [`FrameCpuGpuTiming::last_completed_cpu_frame_ms`] /
//! [`FrameCpuGpuTiming::last_completed_gpu_frame_ms`], which survive across [`Self::begin_frame`]
//! so the overlay can keep showing the most recent completed numbers without blocking.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Per-tick state for AAA-style CPU/GPU frame metrics (see [`GpuContext`](super::GpuContext)).
#[derive(Debug, Default)]
pub struct FrameCpuGpuTiming {
    /// Monotonic id; callbacks ignore stale generations after a new [`Self::begin_frame`].
    generation: u64,
    /// Start of the winit frame tick ([`crate::app::RenderideApp::tick_frame`]).
    frame_start: Option<Instant>,
    /// Number of tracked submits this tick (1-based).
    submit_seq: u32,
    /// Set in [`Self::end_frame`] to the last submit index for this tick.
    finalized_seq: Option<u32>,
    /// Driver-thread post-submit instants for this tick, keyed by submit index.
    pending_real_submit_by_seq: HashMap<u32, Instant>,
    /// GPU duration (ms) per submit index, filled asynchronously by wgpu callbacks.
    pending_gpu_by_seq: HashMap<u32, f64>,
    /// CPU ms (frame_start → real `Queue::submit` return) for the current tick when the driver
    /// thread reported the last submit before [`Self::end_frame`] picked it up.
    pub(crate) cpu_frame_ms: Option<f64>,
    /// GPU ms (real `Queue::submit` return → completion callback) for the current tick when the
    /// callback arrived in time for [`Self::end_frame`].
    pub(crate) gpu_frame_ms: Option<f64>,
    /// Most recent CPU frame ms reported by any driver-thread submit, regardless of generation.
    /// Survives [`Self::begin_frame`] so the HUD always has a value to render.
    pub(crate) last_completed_cpu_frame_ms: Option<f64>,
    /// Most recent GPU frame ms reported by any completion callback, regardless of generation.
    /// Survives [`Self::begin_frame`] so the HUD always has a value to render.
    pub(crate) last_completed_gpu_frame_ms: Option<f64>,
}

/// Identifying info for one tracked submit, attached to a driver-thread batch.
///
/// The driver thread uses the embedded [`FrameCpuGpuTimingHandle`] to record the post-submit
/// instant and to publish CPU/GPU frame ms back to the main thread.
#[derive(Clone)]
pub struct FrameTimingTrack {
    /// Shared handle to the [`FrameCpuGpuTiming`] state.
    pub handle: FrameCpuGpuTimingHandle,
    /// Generation captured at [`FrameCpuGpuTiming::on_before_tracked_submit`] time.
    pub generation: u64,
    /// 1-based submit index within the originating tick.
    pub seq: u32,
    /// Winit tick start instant, used to compute CPU frame ms once the real submit returns.
    pub frame_start: Instant,
}

impl FrameCpuGpuTiming {
    /// Starts tracking for a new winit tick; clears prior tick metrics.
    pub fn begin_frame(&mut self, frame_start: Instant) {
        self.generation = self.generation.wrapping_add(1);
        self.frame_start = Some(frame_start);
        self.submit_seq = 0;
        self.finalized_seq = None;
        self.pending_real_submit_by_seq.clear();
        self.pending_gpu_by_seq.clear();
        self.cpu_frame_ms = None;
        self.gpu_frame_ms = None;
        // Intentionally keep `last_completed_*_frame_ms` for HUD display without blocking.
    }

    /// Call after all render graph submits for this tick (last submit index is known).
    ///
    /// Picks up the per-tick CPU/GPU values when the driver thread already reported them; both
    /// numbers may still arrive later, in which case [`Self::last_completed_cpu_frame_ms`] /
    /// [`Self::last_completed_gpu_frame_ms`] are what the HUD renders.
    pub fn end_frame(&mut self) {
        if self.frame_start.is_none() {
            return;
        }
        self.finalized_seq = Some(self.submit_seq);
        if self.submit_seq > 0 {
            if let (Some(start), Some(real_submit_at)) = (
                self.frame_start,
                self.pending_real_submit_by_seq.remove(&self.submit_seq),
            ) {
                self.cpu_frame_ms =
                    Some(real_submit_at.duration_since(start).as_secs_f64() * 1000.0);
            }
            if let Some(ms) = self.pending_gpu_by_seq.remove(&self.submit_seq) {
                self.gpu_frame_ms = Some(ms);
            }
        }
    }

    /// Records that the driver thread has finished `Queue::submit` for `seq`.
    ///
    /// Updates [`Self::last_completed_cpu_frame_ms`] unconditionally so the HUD never goes blank,
    /// and folds the value into the per-tick `cpu_frame_ms` when the tick is still current.
    fn record_real_submit(
        &mut self,
        submitted_generation: u64,
        seq: u32,
        frame_start: Instant,
        real_submit_at: Instant,
    ) {
        let cpu_ms = real_submit_at
            .saturating_duration_since(frame_start)
            .as_secs_f64()
            * 1000.0;
        self.last_completed_cpu_frame_ms = Some(cpu_ms);
        if submitted_generation != self.generation {
            return;
        }
        self.pending_real_submit_by_seq.insert(seq, real_submit_at);
        if self.finalized_seq == Some(seq) {
            self.cpu_frame_ms = Some(cpu_ms);
        }
    }

    /// Records the GPU side of a tracked submit when its completion callback fires.
    fn record_gpu_done(&mut self, submitted_generation: u64, seq: u32, gpu_ms: f64) {
        self.last_completed_gpu_frame_ms = Some(gpu_ms);
        if submitted_generation != self.generation {
            return;
        }
        self.pending_gpu_by_seq.insert(seq, gpu_ms);
        if self.finalized_seq == Some(seq) {
            self.gpu_frame_ms = Some(gpu_ms);
        }
    }

    /// Reserves the next submit index for the current tick.
    ///
    /// Returns [`None`] before the first [`Self::begin_frame`] or after a [`Self::end_frame`]
    /// without a follow-up `begin_frame`.
    pub fn on_before_tracked_submit(&mut self) -> Option<(u64, u32, Instant)> {
        let frame_start = self.frame_start?;
        self.submit_seq = self.submit_seq.saturating_add(1);
        Some((self.generation, self.submit_seq, frame_start))
    }
}

/// Shared timing state held by [`super::GpuContext`].
pub type FrameCpuGpuTimingHandle = Arc<Mutex<FrameCpuGpuTiming>>;

/// Records the driver-thread post-submit instant for a tracked batch.
///
/// Call from the driver thread immediately after [`wgpu::Queue::submit`] returns so the captured
/// instant straddles "CPU finished preparing this frame" and "GPU started executing it." Returns
/// the same instant so callers can reuse it as the baseline for the GPU completion callback.
pub fn record_real_submit(track: &FrameTimingTrack) -> Instant {
    let real_submit_at = Instant::now();
    let mut g = track.handle.lock().unwrap_or_else(|e| e.into_inner());
    g.record_real_submit(
        track.generation,
        track.seq,
        track.frame_start,
        real_submit_at,
    );
    real_submit_at
}

/// Builds a callback that records GPU duration for `seq` when the queue finishes that submission.
///
/// `real_submit_at` must be captured on the driver thread after `Queue::submit` returns, so the
/// resulting `gpu_ms` measures `submit_returned → on_submitted_work_done` rather than including
/// driver-ring wait time.
pub fn make_gpu_done_callback(
    handle: FrameCpuGpuTimingHandle,
    generation: u64,
    seq: u32,
    real_submit_at: Instant,
) -> impl FnOnce() + Send + 'static {
    move || {
        let gpu_ms = real_submit_at.elapsed().as_secs_f64() * 1000.0;
        let mut g = handle.lock().unwrap_or_else(|e| e.into_inner());
        g.record_gpu_done(generation, seq, gpu_ms);
    }
}

#[cfg(test)]
mod tests {
    use super::FrameCpuGpuTiming;
    use std::time::{Duration, Instant};

    #[test]
    fn cpu_and_gpu_frame_ms_populated_when_callbacks_arrive_in_time() {
        let mut t = FrameCpuGpuTiming::default();
        let frame_start = Instant::now();
        t.begin_frame(frame_start);
        let (generation, seq, fs) = t.on_before_tracked_submit().expect("tracked");
        assert_eq!(seq, 1);
        assert_eq!(fs, frame_start);
        let real_submit_at = frame_start + Duration::from_millis(3);
        t.record_real_submit(generation, seq, frame_start, real_submit_at);
        t.record_gpu_done(generation, seq, 4.0);
        t.end_frame();
        let cpu = t.cpu_frame_ms.expect("cpu_frame_ms");
        assert!((2.5..3.5).contains(&cpu), "cpu={cpu}");
        assert_eq!(t.gpu_frame_ms, Some(4.0));
        assert_eq!(t.last_completed_gpu_frame_ms, Some(4.0));
    }

    #[test]
    fn last_completed_cpu_frame_ms_survives_begin_frame_and_late_submit_record() {
        let mut t = FrameCpuGpuTiming::default();
        let start = Instant::now();
        t.begin_frame(start);
        let (generation, seq, fs) = t.on_before_tracked_submit().expect("tracked");
        t.end_frame();
        // Late record: generation already advanced.
        t.begin_frame(start + Duration::from_millis(16));
        let real_submit_at = fs + Duration::from_millis(2);
        t.record_real_submit(generation, seq, fs, real_submit_at);
        assert!(t.cpu_frame_ms.is_none());
        let last = t.last_completed_cpu_frame_ms.expect("last cpu frame ms");
        assert!((1.5..2.5).contains(&last), "last={last}");
    }

    #[test]
    fn last_completed_gpu_frame_ms_survives_begin_frame_and_late_callback() {
        let mut t = FrameCpuGpuTiming::default();
        let start = Instant::now();
        t.begin_frame(start);
        let (generation, seq, _fs) = t.on_before_tracked_submit().expect("tracked");
        t.end_frame();
        // Late callback: generation already advanced.
        t.begin_frame(start + Duration::from_millis(16));
        t.record_gpu_done(generation, seq, 2.5);
        assert!(t.gpu_frame_ms.is_none());
        assert_eq!(t.last_completed_gpu_frame_ms, Some(2.5));
    }
}
