//! Runtime-owned IPC scratch and counters.

use hashbrown::HashMap;

/// IPC scratch state that is not part of transport ownership.
pub(super) struct RuntimeIpcState {
    /// In-flight shader uploads whose resolution is running on the rayon pool.
    pub(super) pending_shader_resolutions: Vec<super::shader_material_ipc::PendingShaderResolution>,
    /// Running counts of post-init renderer command variants seen without a running handler.
    unhandled_ipc_command_counts: HashMap<&'static str, u64>,
}

impl RuntimeIpcState {
    /// Creates empty IPC scratch state.
    pub(super) fn new() -> Self {
        Self {
            pending_shader_resolutions: Vec::new(),
            unhandled_ipc_command_counts: HashMap::new(),
        }
    }

    /// Records one unhandled renderer command variant.
    pub(super) fn record_unhandled_renderer_command(&mut self, tag: &'static str) -> u64 {
        let count = self.unhandled_ipc_command_counts.entry(tag).or_insert(0);
        *count += 1;
        *count
    }

    /// Total number of unhandled post-handshake renderer commands.
    pub(super) fn unhandled_command_event_total(&self) -> u64 {
        self.unhandled_ipc_command_counts.values().copied().sum()
    }
}

#[cfg(test)]
mod tests {
    use super::RuntimeIpcState;

    #[test]
    fn unhandled_command_total_sums_variant_counts() {
        let mut state = RuntimeIpcState::new();

        state.record_unhandled_renderer_command("Foo");
        state.record_unhandled_renderer_command("Foo");
        state.record_unhandled_renderer_command("Bar");

        assert_eq!(state.unhandled_command_event_total(), 3);
    }
}
