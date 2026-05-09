//! Runtime-owned IPC scratch and counters.

use std::collections::VecDeque;

use hashbrown::HashMap;

use crate::shared::RendererCommand;

const DEFERRED_PRE_FINALIZE_WARN_THRESHOLD: usize = 1024;

/// IPC scratch state that is not part of transport ownership.
pub(super) struct RuntimeIpcState {
    /// In-flight shader uploads whose resolution is running on the rayon pool.
    pub(super) pending_shader_resolutions: Vec<super::shader_material_ipc::PendingShaderResolution>,
    /// Host commands received after init data but before init finalization.
    deferred_pre_finalize_commands: VecDeque<RendererCommand>,
    /// Running counts of post-init renderer command variants seen without a running handler.
    unhandled_ipc_command_counts: HashMap<&'static str, u64>,
}

impl RuntimeIpcState {
    /// Creates empty IPC scratch state.
    pub(super) fn new() -> Self {
        Self {
            pending_shader_resolutions: Vec::new(),
            deferred_pre_finalize_commands: VecDeque::new(),
            unhandled_ipc_command_counts: HashMap::new(),
        }
    }

    /// Defers a host command received before init finalization.
    pub(super) fn defer_pre_finalize_command(&mut self, cmd: RendererCommand) {
        self.deferred_pre_finalize_commands.push_back(cmd);
        let count = self.deferred_pre_finalize_commands.len();
        if count == DEFERRED_PRE_FINALIZE_WARN_THRESHOLD
            || (count > DEFERRED_PRE_FINALIZE_WARN_THRESHOLD
                && (count - DEFERRED_PRE_FINALIZE_WARN_THRESHOLD)
                    .is_multiple_of(DEFERRED_PRE_FINALIZE_WARN_THRESHOLD))
        {
            logger::warn!("IPC: {count} commands queued while waiting for init finalization");
        }
    }

    /// Drains deferred pre-finalize commands in host arrival order.
    pub(super) fn take_deferred_pre_finalize_commands(&mut self) -> VecDeque<RendererCommand> {
        std::mem::take(&mut self.deferred_pre_finalize_commands)
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

    /// Number of commands waiting for init finalization replay.
    #[cfg(test)]
    pub(super) fn deferred_pre_finalize_command_count(&self) -> usize {
        self.deferred_pre_finalize_commands.len()
    }
}

#[cfg(test)]
mod tests {
    use super::RuntimeIpcState;
    use crate::shared::{QualityConfig, RendererCommand};

    #[test]
    fn unhandled_command_total_sums_variant_counts() {
        let mut state = RuntimeIpcState::new();

        state.record_unhandled_renderer_command("Foo");
        state.record_unhandled_renderer_command("Foo");
        state.record_unhandled_renderer_command("Bar");

        assert_eq!(state.unhandled_command_event_total(), 3);
    }

    #[test]
    fn deferred_pre_finalize_commands_drain_fifo() {
        let mut state = RuntimeIpcState::new();

        state.defer_pre_finalize_command(RendererCommand::QualityConfig(QualityConfig {
            per_pixel_lights: 1,
            ..Default::default()
        }));
        state.defer_pre_finalize_command(RendererCommand::QualityConfig(QualityConfig {
            per_pixel_lights: 2,
            ..Default::default()
        }));

        assert_eq!(state.deferred_pre_finalize_command_count(), 2);
        let mut drained = state.take_deferred_pre_finalize_commands();
        assert_eq!(state.deferred_pre_finalize_command_count(), 0);

        match drained.pop_front() {
            Some(RendererCommand::QualityConfig(cfg)) => assert_eq!(cfg.per_pixel_lights, 1),
            other => panic!("unexpected first deferred command: {other:?}"),
        }
        match drained.pop_front() {
            Some(RendererCommand::QualityConfig(cfg)) => assert_eq!(cfg.per_pixel_lights, 2),
            other => panic!("unexpected second deferred command: {other:?}"),
        }
        assert!(drained.is_empty());
    }
}
