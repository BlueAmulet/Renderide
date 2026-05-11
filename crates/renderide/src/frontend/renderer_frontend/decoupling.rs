//! Decoupling-state methods on [`RendererFrontend`]. Logging for the
//! `Activate`/`Recouple`/`AdvanceProgress`/`ResetProgress` decisions lives in
//! [`super::super::decoupling::logging`] so this file stays focused on
//! delegation.

use std::time::Instant;

use crate::shared::RenderDecouplingConfig;

use super::super::decoupling::DecouplingState;
use super::super::decoupling::decisions::DecouplingActivationDecision;
use super::super::decoupling::logging::log_activation;
use super::RendererFrontend;

impl RendererFrontend {
    /// Read-only handle to the host-driven decoupling state.
    pub fn decoupling_state(&self) -> &DecouplingState {
        &self.decoupling
    }

    /// Whether the renderer is currently running decoupled from host lock-step.
    #[cfg(test)]
    pub fn is_decoupled(&self) -> bool {
        self.decoupling.is_active()
    }

    /// Replaces renderer-side decoupling thresholds with the host's config.
    pub fn set_decoupling_config(&mut self, cfg: RenderDecouplingConfig) {
        self.decoupling.apply_config(&cfg);
    }

    /// Per-tick decoupling activation check.
    pub fn update_decoupling_activation(&mut self, now: Instant) {
        let decision = self
            .decoupling
            .update_activation_for_tick(now, self.lockstep.awaiting_submit());
        if decision == DecouplingActivationDecision::Activate {
            log_activation(&self.decoupling, self.lockstep.last_frame_index());
        }
    }
}
