//! IPC-facing entry points on [`super::RendererRuntime`].
//!
//! Owns the per-tick command drain ([`RendererRuntime::poll_ipc`]). Incoming commands are decoded
//! by `crate::frontend::dispatch` and applied by `runtime::ipc_effects`, keeping frontend dispatch
//! independent of the runtime facade.

use crate::frontend::InitState;
use crate::frontend::dispatch::renderer_command_kind::renderer_command_variant_tag;
use crate::shared::RendererCommand;

use super::RendererRuntime;

impl RendererRuntime {
    /// Total number of post-handshake IPC commands logged as unhandled (sum of per-variant counters).
    pub fn unhandled_ipc_command_event_total(&self) -> u64 {
        self.ipc_state.unhandled_command_event_total()
    }

    /// Records one unhandled post-handshake renderer command for diagnostics.
    pub(crate) fn record_unhandled_renderer_command(&mut self, tag: &'static str) {
        self.ipc_state.record_unhandled_renderer_command(tag);
    }

    /// Drains IPC and dispatches commands. Each poll batch is ordered so `renderer_init_data` runs
    /// first, then frame submits, then the rest (see [`crate::frontend::RendererFrontend::poll_commands`]).
    pub fn poll_ipc(&mut self) {
        profiling::scope!("ipc::poll_batch");
        super::shader_material_ipc::drain_pending_shader_resolutions(
            &mut self.ipc_state.pending_shader_resolutions,
            &mut self.backend,
            &mut self.frontend,
        );
        let mut batch = self.frontend.poll_commands();
        trace_ipc_batch(
            &batch,
            self.frontend.init_state(),
            self.ipc_state.pending_shader_resolutions.len(),
        );
        for cmd in batch.drain(..) {
            let _tag = renderer_command_variant_tag(&cmd);
            profiling::scope!("ipc::dispatch", _tag);
            self.handle_ipc_command(cmd);
        }
        self.frontend.recycle_command_batch(batch);
    }
}

fn trace_ipc_batch(batch: &[RendererCommand], init_state: InitState, pending_shaders: usize) {
    if batch.is_empty() || !logger::enabled(logger::LogLevel::Trace) {
        return;
    }
    let mut counts: Vec<(&'static str, usize)> = Vec::new();
    for cmd in batch {
        let tag = renderer_command_variant_tag(cmd);
        if let Some((_, count)) = counts.iter_mut().find(|(existing, _)| *existing == tag) {
            *count += 1;
        } else {
            counts.push((tag, 1));
        }
    }
    let mut kinds = String::new();
    for (idx, (tag, count)) in counts.iter().enumerate() {
        if idx > 0 {
            kinds.push_str(", ");
        }
        kinds.push_str(tag);
        kinds.push('=');
        kinds.push_str(&count.to_string());
    }
    logger::trace!(
        "IPC poll batch: commands={} init_state={:?} pending_shader_resolutions={} kinds=[{}]",
        batch.len(),
        init_state,
        pending_shaders,
        kinds,
    );
}
