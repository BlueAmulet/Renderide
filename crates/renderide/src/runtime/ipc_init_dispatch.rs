//! IPC command routing by [`crate::frontend::InitState`]: init handshake vs running dispatch.

use crate::frontend::InitState;
use crate::runtime::RendererRuntime;
use crate::shared::RendererCommand;

/// Dispatches a single command according to the current init phase.
pub(crate) fn dispatch_ipc_command(runtime: &mut RendererRuntime, cmd: RendererCommand) {
    match runtime.frontend.init_state() {
        InitState::Uninitialized => match cmd {
            RendererCommand::KeepAlive(_) => {}
            RendererCommand::RendererInitData(d) => runtime.on_init_data(d),
            _ => {
                logger::error!("IPC: expected RendererInitData first");
                runtime.frontend.set_fatal_error(true);
            }
        },
        InitState::InitReceived => match cmd {
            RendererCommand::KeepAlive(_) => {}
            RendererCommand::RendererInitFinalizeData(_) => {
                runtime.frontend.set_init_state(InitState::Finalized);
            }
            RendererCommand::RendererInitProgressUpdate(_) => {}
            RendererCommand::RendererEngineReady(_) => {}
            _ => {
                logger::trace!("IPC: deferring command until init finalized (skeleton)");
            }
        },
        InitState::Finalized => runtime.handle_running_command(cmd),
    }
}
