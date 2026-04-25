//! Renderer-side decoupling state machine driven by the host's
//! [`RenderDecouplingConfig`](crate::shared::RenderDecouplingConfig).
//!
//! Mirrors the Renderite.Unity `RenderingManager` behavior:
//! - Activation: when the renderer has been waiting for a [`crate::shared::FrameSubmitData`]
//!   longer than [`DecouplingState::activate_interval_seconds`], it goes "decoupled" so it can
//!   keep drawing the prior scene state instead of stalling.
//! - Re-couple: each arriving [`crate::shared::FrameSubmitData`] reports
//!   `frame_begin_to_submit` (time from sending the matching outgoing
//!   [`crate::shared::FrameStartData`] to receiving the host's submit). Sub-threshold submits
//!   advance a stable-frame counter; once it reaches [`DecouplingState::recouple_frame_count`]
//!   the decoupled flag clears. Submits at-or-above the threshold reset the counter.
//! - Asset integration budget: while decoupled, the per-tick budget is capped at
//!   [`DecouplingState::decoupled_max_asset_processing_seconds`] (the host-supplied ceiling)
//!   so the renderer can keep up with display while the host catches up.
//!
//! Pure state — no IPC, no winit, no GPU. Activation/recouple are driven by `Instant` inputs
//! supplied by callers so the logic is unit-testable without wiring the runtime.
//!
//! `decouple_activate_interval = 0.0` together with `recouple_frame_count = i32::MAX` is the
//! `ForceDecouple` mode emitted by FrooxEngine: activation triggers on the first tick with a
//! pending begin-frame and recouple progress can never reach the threshold, so the renderer
//! stays decoupled until a non-forced config arrives.
//!
//! Defaults match the Renderite.Unity `RenderingManager` initial values so the renderer is
//! sane before the host's first [`RenderDecouplingConfig`] arrives.
use std::time::{Duration, Instant};

use crate::shared::RenderDecouplingConfig;

/// Default activation threshold in seconds (1/15 s ≈ 66.67 ms). Matches Renderite.Unity.
const DEFAULT_ACTIVATE_INTERVAL_SECONDS: f32 = 1.0 / 15.0;
/// Default decoupled asset-processing ceiling in seconds (2 ms). Matches Renderite.Unity.
const DEFAULT_DECOUPLED_MAX_ASSET_PROCESSING_SECONDS: f32 = 0.002;
/// Default consecutive sub-threshold frames required to re-couple. Matches Renderite.Unity.
const DEFAULT_RECOUPLE_FRAME_COUNT: i32 = 10;

/// Renderer-side decoupling state machine.
#[derive(Debug, Clone)]
pub struct DecouplingState {
    /// Wait threshold in seconds before flipping `active` true.
    activate_interval_seconds: f32,
    /// Asset-integration ceiling in seconds while decoupled.
    decoupled_max_asset_processing_seconds: f32,
    /// Consecutive sub-threshold submits required to re-couple.
    recouple_frame_count: i32,
    /// Whether the renderer is currently running decoupled from host lock-step.
    active: bool,
    /// Number of consecutive sub-threshold submits seen while `active`.
    recouple_progress: i32,
    /// Wall-clock instant the most recent outgoing [`crate::shared::FrameStartData`] was sent.
    last_frame_start_sent_at: Option<Instant>,
    /// Most recent observed `FrameStartData → FrameSubmitData` round-trip duration.
    last_frame_begin_to_submit: Option<Duration>,
}

impl Default for DecouplingState {
    fn default() -> Self {
        Self {
            activate_interval_seconds: DEFAULT_ACTIVATE_INTERVAL_SECONDS,
            decoupled_max_asset_processing_seconds: DEFAULT_DECOUPLED_MAX_ASSET_PROCESSING_SECONDS,
            recouple_frame_count: DEFAULT_RECOUPLE_FRAME_COUNT,
            active: false,
            recouple_progress: 0,
            last_frame_start_sent_at: None,
            last_frame_begin_to_submit: None,
        }
    }
}

impl DecouplingState {
    /// Replaces the threshold/ceiling/recouple-count with the host's [`RenderDecouplingConfig`].
    /// Active state and recouple progress are preserved across config updates so a
    /// forced-decouple → restored cycle does not bounce the flag mid-tick.
    pub fn apply_config(&mut self, cfg: &RenderDecouplingConfig) {
        self.activate_interval_seconds = cfg.decouple_activate_interval.max(0.0);
        self.decoupled_max_asset_processing_seconds =
            cfg.decoupled_max_asset_processing_time.max(0.0);
        self.recouple_frame_count = cfg.recouple_frame_count;
    }

    /// Whether the renderer is currently running decoupled from host lock-step.
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Last observed wait between outgoing [`crate::shared::FrameStartData`] and the matching
    /// incoming [`crate::shared::FrameSubmitData`]; [`None`] until the first round-trip completes.
    pub fn last_frame_begin_to_submit(&self) -> Option<Duration> {
        self.last_frame_begin_to_submit
    }

    /// Activation threshold in seconds (host-controlled).
    pub fn activate_interval_seconds(&self) -> f32 {
        self.activate_interval_seconds
    }

    /// Asset-integration ceiling in seconds while decoupled (host-controlled).
    pub fn decoupled_max_asset_processing_seconds(&self) -> f32 {
        self.decoupled_max_asset_processing_seconds
    }

    /// Consecutive sub-threshold submits required to re-couple (host-controlled).
    pub fn recouple_frame_count(&self) -> i32 {
        self.recouple_frame_count
    }

    /// Records the wall-clock at which the most recent outgoing [`crate::shared::FrameStartData`]
    /// was sent. Called by `RendererFrontend::pre_frame` after a successful primary-queue send.
    pub fn record_frame_start_sent(&mut self, now: Instant) {
        self.last_frame_start_sent_at = Some(now);
    }

    /// Per-tick activation check. If the renderer is currently waiting on a [`crate::shared::FrameSubmitData`]
    /// (`awaiting_submit == true`, i.e. `last_frame_data_processed == false`) and the elapsed wait
    /// exceeds [`Self::activate_interval_seconds`], flip `active` and reset the recouple counter.
    pub fn update_activation_for_tick(&mut self, now: Instant, awaiting_submit: bool) {
        if !awaiting_submit || self.active {
            return;
        }
        let Some(sent_at) = self.last_frame_start_sent_at else {
            return;
        };
        let elapsed = now.saturating_duration_since(sent_at);
        if elapsed.as_secs_f32() >= self.activate_interval_seconds {
            self.active = true;
            self.recouple_progress = 0;
        }
    }

    /// Records the round-trip for the just-received [`crate::shared::FrameSubmitData`] and, when
    /// decoupled, advances or resets the recouple counter.
    ///
    /// Mirrors `RenderingManager.cs:400-415`: sub-threshold submits increment the counter and at
    /// `recouple_frame_count` the decoupled flag clears; at-or-above-threshold submits reset the
    /// counter.
    pub fn record_frame_submit_received(&mut self, now: Instant) {
        let elapsed = self
            .last_frame_start_sent_at
            .take()
            .map(|sent| now.saturating_duration_since(sent));
        self.last_frame_begin_to_submit = elapsed;

        if !self.active {
            return;
        }
        let Some(elapsed) = elapsed else {
            return;
        };
        if elapsed.as_secs_f32() >= self.activate_interval_seconds {
            self.recouple_progress = 0;
        } else {
            self.recouple_progress = self.recouple_progress.saturating_add(1);
            if self.recouple_progress >= self.recouple_frame_count {
                self.active = false;
                self.recouple_progress = 0;
            }
        }
    }

    /// Returns the wall-clock budget (in milliseconds) the runtime should pass to
    /// [`crate::backend::RenderBackend::drain_asset_tasks`] this tick. While decoupled the
    /// host-supplied ceiling replaces the local default; while coupled the local default is used.
    /// The returned value is always at least 1 ms so [`std::time::Duration::from_millis`] cannot
    /// produce a zero-length budget.
    pub fn effective_asset_integration_budget_ms(&self, coupled_default_ms: u32) -> u32 {
        if !self.active {
            return coupled_default_ms.max(1);
        }
        let ceiling_ms = (self.decoupled_max_asset_processing_seconds * 1000.0).round();
        if !ceiling_ms.is_finite() || ceiling_ms <= 0.0 {
            return 1;
        }
        let clamped = ceiling_ms.min(f32::from(u16::MAX)) as u32;
        clamped.max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::RenderDecouplingConfig;

    fn cfg(interval: f32, decoupled_max: f32, recouple: i32) -> RenderDecouplingConfig {
        RenderDecouplingConfig {
            decouple_activate_interval: interval,
            decoupled_max_asset_processing_time: decoupled_max,
            recouple_frame_count: recouple,
        }
    }

    #[test]
    fn defaults_match_unity_renderer() {
        let s = DecouplingState::default();
        assert!((s.activate_interval_seconds() - 1.0 / 15.0).abs() < 1e-6);
        assert!((s.decoupled_max_asset_processing_seconds() - 0.002).abs() < 1e-6);
        assert_eq!(s.recouple_frame_count(), 10);
        assert!(!s.is_active());
    }

    #[test]
    fn apply_config_updates_thresholds() {
        let mut s = DecouplingState::default();
        s.apply_config(&cfg(0.05, 0.004, 60));
        assert!((s.activate_interval_seconds() - 0.05).abs() < 1e-6);
        assert!((s.decoupled_max_asset_processing_seconds() - 0.004).abs() < 1e-6);
        assert_eq!(s.recouple_frame_count(), 60);
    }

    #[test]
    fn apply_config_clamps_negative_to_zero() {
        let mut s = DecouplingState::default();
        s.apply_config(&cfg(-1.0, -0.001, 5));
        assert_eq!(s.activate_interval_seconds(), 0.0);
        assert_eq!(s.decoupled_max_asset_processing_seconds(), 0.0);
    }

    #[test]
    fn activation_requires_pending_submit() {
        let mut s = DecouplingState::default();
        s.apply_config(&cfg(0.05, 0.004, 5));
        let t0 = Instant::now();
        s.record_frame_start_sent(t0);
        let later = t0 + Duration::from_secs_f32(0.1);
        s.update_activation_for_tick(later, /* awaiting_submit */ false);
        assert!(!s.is_active(), "no activation when not awaiting submit");
    }

    #[test]
    fn activation_triggers_when_wait_exceeds_threshold() {
        let mut s = DecouplingState::default();
        s.apply_config(&cfg(0.05, 0.004, 5));
        let t0 = Instant::now();
        s.record_frame_start_sent(t0);
        let later = t0 + Duration::from_secs_f32(0.06);
        s.update_activation_for_tick(later, true);
        assert!(s.is_active());
    }

    #[test]
    fn activation_holds_when_wait_under_threshold() {
        let mut s = DecouplingState::default();
        s.apply_config(&cfg(0.05, 0.004, 5));
        let t0 = Instant::now();
        s.record_frame_start_sent(t0);
        let later = t0 + Duration::from_secs_f32(0.01);
        s.update_activation_for_tick(later, true);
        assert!(!s.is_active());
    }

    #[test]
    fn activation_skipped_without_recorded_send() {
        let mut s = DecouplingState::default();
        s.apply_config(&cfg(0.05, 0.004, 5));
        s.update_activation_for_tick(Instant::now(), true);
        assert!(!s.is_active());
    }

    #[test]
    fn fast_submit_advances_recouple_counter() {
        let mut s = DecouplingState::default();
        s.apply_config(&cfg(0.05, 0.004, 3));
        // Activate first.
        let t0 = Instant::now();
        s.record_frame_start_sent(t0);
        s.update_activation_for_tick(t0 + Duration::from_secs_f32(0.06), true);
        assert!(s.is_active());

        // Three fast (sub-threshold) submits clear `active` exactly at the count.
        for n in 1..=3 {
            let send = Instant::now();
            s.record_frame_start_sent(send);
            s.record_frame_submit_received(send + Duration::from_secs_f32(0.01));
            if n < 3 {
                assert!(s.is_active(), "still decoupled after {n} fast submits");
            } else {
                assert!(!s.is_active(), "recoupled after {n} fast submits");
            }
        }
    }

    #[test]
    fn slow_submit_resets_recouple_counter() {
        let mut s = DecouplingState::default();
        s.apply_config(&cfg(0.05, 0.004, 3));
        let t0 = Instant::now();
        s.record_frame_start_sent(t0);
        s.update_activation_for_tick(t0 + Duration::from_secs_f32(0.06), true);
        assert!(s.is_active());

        // Two fast submits, then one slow, then two fast → still decoupled (counter reset).
        let pattern = [0.01, 0.01, 0.06, 0.01, 0.01];
        for delay_s in pattern {
            let send = Instant::now();
            s.record_frame_start_sent(send);
            s.record_frame_submit_received(send + Duration::from_secs_f32(delay_s));
        }
        assert!(s.is_active(), "slow submit must reset counter");
    }

    #[test]
    fn force_decouple_stays_active_indefinitely() {
        let mut s = DecouplingState::default();
        // FrooxEngine `ForceDecouple.Value`: interval=0, recouple=i32::MAX.
        s.apply_config(&cfg(0.0, 0.004, i32::MAX));
        let t0 = Instant::now();
        s.record_frame_start_sent(t0);
        s.update_activation_for_tick(t0, true);
        assert!(s.is_active(), "interval==0 activates immediately");

        for _ in 0..1000 {
            let send = Instant::now();
            s.record_frame_start_sent(send);
            // All submits are technically "≥ 0.0" so they reset the counter — but even if they
            // were sub-threshold, `i32::MAX` would never be reached. Either branch keeps `active`.
            s.record_frame_submit_received(send + Duration::from_secs_f32(0.001));
        }
        assert!(
            s.is_active(),
            "force-decouple must persist across many submits"
        );
    }

    #[test]
    fn budget_uses_coupled_default_when_inactive() {
        let s = DecouplingState::default();
        assert_eq!(s.effective_asset_integration_budget_ms(8), 8);
    }

    #[test]
    fn budget_uses_decoupled_ceiling_when_active() {
        let mut s = DecouplingState::default();
        s.apply_config(&cfg(0.05, 0.004, 5));
        let t0 = Instant::now();
        s.record_frame_start_sent(t0);
        s.update_activation_for_tick(t0 + Duration::from_secs_f32(0.06), true);
        assert!(s.is_active());
        assert_eq!(s.effective_asset_integration_budget_ms(32), 4);
    }

    #[test]
    fn budget_clamps_to_minimum_one_ms() {
        let mut s = DecouplingState::default();
        s.apply_config(&cfg(0.05, 0.0, 5));
        let t0 = Instant::now();
        s.record_frame_start_sent(t0);
        s.update_activation_for_tick(t0 + Duration::from_secs_f32(0.06), true);
        assert!(s.is_active());
        assert_eq!(s.effective_asset_integration_budget_ms(8), 1);
    }

    #[test]
    fn last_frame_begin_to_submit_recorded() {
        let mut s = DecouplingState::default();
        let t0 = Instant::now();
        s.record_frame_start_sent(t0);
        s.record_frame_submit_received(t0 + Duration::from_millis(20));
        let observed = s.last_frame_begin_to_submit().expect("recorded");
        assert!(observed >= Duration::from_millis(20));
    }
}
