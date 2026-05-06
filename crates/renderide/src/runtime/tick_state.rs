//! Runtime-owned per-tick scratch and phase gates.

use crate::scene::RenderSpaceId;

/// Per-tick gates and reusable view-planning scratch.
pub(super) struct RuntimeTickState {
    /// Set when asset integration completed for the current winit tick.
    did_integrate_this_tick: bool,
    /// Reusable per-frame scratch for secondary render-texture view collection.
    pub(super) secondary_view_tasks_scratch: Vec<(RenderSpaceId, f32, usize)>,
}

impl RuntimeTickState {
    /// Creates empty tick state.
    pub(super) fn new() -> Self {
        Self {
            did_integrate_this_tick: false,
            secondary_view_tasks_scratch: Vec::new(),
        }
    }

    /// Clears once-per-tick gates at the start of a new winit tick.
    pub(super) fn reset_for_tick(&mut self) {
        self.did_integrate_this_tick = false;
    }

    /// Whether asset integration already ran this tick.
    pub(super) fn did_integrate_assets_this_tick(&self) -> bool {
        self.did_integrate_this_tick
    }

    /// Marks asset integration as completed for this tick.
    pub(super) fn mark_integrated_assets_this_tick(&mut self) {
        self.did_integrate_this_tick = true;
    }
}

#[cfg(test)]
mod tests {
    use super::RuntimeTickState;

    #[test]
    fn asset_integration_gate_resets_per_tick() {
        let mut state = RuntimeTickState::new();

        assert!(!state.did_integrate_assets_this_tick());
        state.mark_integrated_assets_this_tick();
        assert!(state.did_integrate_assets_this_tick());
        state.reset_for_tick();
        assert!(!state.did_integrate_assets_this_tick());
    }
}
