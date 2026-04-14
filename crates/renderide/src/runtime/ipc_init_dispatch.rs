//! IPC command routing by [`crate::frontend::InitState`]: init handshake vs running dispatch.

use crate::assets::texture::supported_host_formats_for_init;
use crate::frontend::InitState;
use crate::ipc::DualQueueIpc;
use crate::output_device::head_output_device_wants_openxr;
use crate::runtime::RendererRuntime;
use crate::shared::{HeadOutputDevice, RendererCommand, RendererInitResult};

/// Sends [`RendererInitResult`] to the host after [`crate::shared::RendererInitData`] is applied.
pub(crate) fn send_renderer_init_result(ipc: &mut DualQueueIpc, output_device: HeadOutputDevice) {
    let stereo = if head_output_device_wants_openxr(output_device) {
        "OpenXR(multiview)"
    } else {
        "None"
    };
    let result = RendererInitResult {
        actual_output_device: output_device,
        renderer_identifier: Some("Renderide 0.1.0 (wgpu skeleton)".to_string()),
        main_window_handle_ptr: 0,
        stereo_rendering_mode: Some(stereo.to_string()),
        max_texture_size: 8192,
        is_gpu_texture_pot_byte_aligned: true,
        supported_texture_formats: supported_host_formats_for_init(),
    };
    ipc.send_primary(RendererCommand::RendererInitResult(result));
}

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
