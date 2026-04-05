//! [`RendererFrontend`] implementation.

use crate::connection::{ConnectionParams, InitError};
use crate::ipc::{DualQueueIpc, SharedMemoryAccessor};
use crate::shared::{FrameStartData, RendererCommand, RendererInitData};

/// Host init sequence state (replaces paired booleans such as `init_received` / `init_finalized`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum InitState {
    /// Waiting for [`RendererCommand::renderer_init_data`].
    #[default]
    Uninitialized,
    /// `renderer_init_data` received; waiting for [`RendererCommand::renderer_init_finalize_data`].
    InitReceived,
    /// Normal operation (or standalone mode).
    Finalized,
}

impl InitState {
    /// Whether host init handshake is complete.
    pub fn is_finalized(self) -> bool {
        matches!(self, InitState::Finalized)
    }
}

/// IPC, shared memory, init sequence, and lock-step fields. Does not own GPU pools or scene graph.
pub struct RendererFrontend {
    ipc: Option<DualQueueIpc>,
    params: Option<ConnectionParams>,
    init_state: InitState,
    pending_init: Option<RendererInitData>,
    shared_memory: Option<SharedMemoryAccessor>,
    /// After a successful frame submit application, host may expect another begin-frame.
    pub last_frame_data_processed: bool,
    pub last_frame_index: i32,
    sent_bootstrap_frame_start: bool,
    pub shutdown_requested: bool,
    pub fatal_error: bool,
}

impl RendererFrontend {
    /// Builds frontend; does not open IPC yet (see [`Self::connect_ipc`]).
    pub fn new(params: Option<ConnectionParams>) -> Self {
        let standalone = params.is_none();
        let init_state = if standalone {
            InitState::Finalized
        } else {
            InitState::default()
        };
        Self {
            ipc: None,
            params,
            init_state,
            pending_init: None,
            shared_memory: None,
            last_frame_data_processed: standalone,
            last_frame_index: -1,
            sent_bootstrap_frame_start: false,
            shutdown_requested: false,
            fatal_error: false,
        }
    }

    pub fn init_state(&self) -> InitState {
        self.init_state
    }

    pub fn set_init_state(&mut self, state: InitState) {
        self.init_state = state;
    }

    pub fn pending_init(&self) -> Option<&RendererInitData> {
        self.pending_init.as_ref()
    }

    pub fn set_pending_init(&mut self, data: RendererInitData) {
        self.pending_init = Some(data);
    }

    pub fn take_pending_init(&mut self) -> Option<RendererInitData> {
        self.pending_init.take()
    }

    pub fn shared_memory(&self) -> Option<&SharedMemoryAccessor> {
        self.shared_memory.as_ref()
    }

    pub fn shared_memory_mut(&mut self) -> Option<&mut SharedMemoryAccessor> {
        self.shared_memory.as_mut()
    }

    pub fn set_shared_memory(&mut self, shm: SharedMemoryAccessor) {
        self.shared_memory = Some(shm);
    }

    pub fn ipc_mut(&mut self) -> Option<&mut DualQueueIpc> {
        self.ipc.as_mut()
    }

    pub fn ipc(&self) -> Option<&DualQueueIpc> {
        self.ipc.as_ref()
    }

    /// Disjoint mutable handles for backends that need both shared memory and IPC in one call.
    pub fn transport_pair_mut(
        &mut self,
    ) -> (Option<&mut SharedMemoryAccessor>, Option<&mut DualQueueIpc>) {
        (self.shared_memory.as_mut(), self.ipc.as_mut())
    }

    /// Opens Primary/Background queues when connection parameters were provided at construction.
    pub fn connect_ipc(&mut self) -> Result<(), InitError> {
        let Some(ref p) = self.params.clone() else {
            return Ok(());
        };
        self.ipc = Some(DualQueueIpc::connect(p)?);
        Ok(())
    }

    pub fn is_ipc_connected(&self) -> bool {
        self.ipc.is_some()
    }

    /// Poll and sort commands so frame submits run first in each batch.
    pub fn poll_commands(&mut self) -> Vec<RendererCommand> {
        let Some(ref mut ipc) = self.ipc else {
            return Vec::new();
        };
        let mut batch = ipc.poll();
        batch.sort_by_key(|c| !matches!(c, RendererCommand::frame_submit_data(_)));
        batch
    }

    /// Lock-step begin-frame: send [`FrameStartData`] when allowed.
    ///
    /// The host primarily uses `last_frame_index` for lock-step; other [`FrameStartData`] fields are
    /// left empty until input/perf/reflection paths are wired.
    pub fn pre_frame(&mut self) {
        if !self.init_state.is_finalized() || self.fatal_error || self.ipc.is_none() {
            return;
        }

        let bootstrap = self.last_frame_index < 0 && !self.sent_bootstrap_frame_start;
        let should_send = self.last_frame_data_processed || bootstrap;
        if !should_send {
            return;
        }

        let frame_start = FrameStartData {
            last_frame_index: self.last_frame_index,
            ..Default::default()
        };
        if let Some(ref mut ipc) = self.ipc {
            ipc.send_primary(RendererCommand::frame_start_data(frame_start));
        }
        self.last_frame_data_processed = false;
        if bootstrap {
            self.sent_bootstrap_frame_start = true;
        }
    }

    /// Updates lock-step state after the host submits a frame.
    pub fn note_frame_submit_processed(&mut self, frame_index: i32) {
        self.last_frame_index = frame_index;
        self.last_frame_data_processed = true;
    }

    /// Marks init received after `renderer_init_data` (shared memory may be created here).
    pub fn on_init_received(&mut self) {
        self.init_state = InitState::InitReceived;
        self.last_frame_data_processed = true;
    }
}
