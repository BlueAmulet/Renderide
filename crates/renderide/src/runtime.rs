//! Renderer orchestration: IPC polling, init lifecycle, lock-step frame gating, mesh ingest.
//!
//! Phase order is aligned with `RenderingManager.HandleUpdate`: optionally send
//! [`FrameStartData`](crate::shared::FrameStartData), drain integration-style work (stub here), then
//! process incoming commands.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::assets::mesh::try_upload_mesh_from_raw;
use crate::assets::AssetSubsystem;
use crate::connection::{ConnectionParams, InitError};
use crate::ipc::{DualQueueIpc, SharedMemoryAccessor};
use crate::resources::MeshPool;
use crate::shared::{
    FrameStartData, FrameSubmitData, HeadOutputDevice, MeshUnload, MeshUploadData,
    MeshUploadResult, RendererCommand, RendererInitData, RendererInitResult, TextureFormat,
};

/// Max queued [`MeshUploadData`] when GPU is not ready yet (host data stays in shared memory).
const MAX_PENDING_MESH_UPLOADS: usize = 256;

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

/// Owns IPC (optional), lock-step flags, shared memory, and GPU mesh pool.
pub struct RendererRuntime {
    ipc: Option<DualQueueIpc>,
    params: Option<ConnectionParams>,
    init_state: InitState,
    /// After a successful [`FrameSubmitData`] application, host may expect another begin-frame.
    pub last_frame_data_processed: bool,
    pub last_frame_index: i32,
    sent_bootstrap_frame_start: bool,
    pub shutdown_requested: bool,
    pub fatal_error: bool,
    assets: AssetSubsystem,
    pending_init: Option<RendererInitData>,
    shared_memory: Option<SharedMemoryAccessor>,
    mesh_pool: MeshPool,
    gpu_device: Option<Arc<wgpu::Device>>,
    pending_mesh_uploads: VecDeque<MeshUploadData>,
}

impl RendererRuntime {
    /// Builds a runtime; does not open IPC yet (see [`Self::connect_ipc`]).
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
            last_frame_data_processed: standalone,
            last_frame_index: -1,
            sent_bootstrap_frame_start: false,
            shutdown_requested: false,
            fatal_error: false,
            assets: AssetSubsystem::default(),
            pending_init: None,
            shared_memory: None,
            mesh_pool: MeshPool::default_pool(),
            gpu_device: None,
            pending_mesh_uploads: VecDeque::new(),
        }
    }

    /// Opens Primary/Background queues when [`Self::new`] was given connection parameters.
    pub fn connect_ipc(&mut self) -> Result<(), InitError> {
        let Some(ref p) = self.params.clone() else {
            return Ok(());
        };
        self.ipc = Some(DualQueueIpc::connect(p)?);
        Ok(())
    }

    /// Whether IPC queues are open.
    pub fn is_ipc_connected(&self) -> bool {
        self.ipc.is_some()
    }

    pub fn init_state(&self) -> InitState {
        self.init_state
    }

    /// Mesh pool and VRAM accounting (draw prep, debugging).
    pub fn mesh_pool(&self) -> &MeshPool {
        &self.mesh_pool
    }

    /// Mutable mesh pool (eviction experiments).
    pub fn mesh_pool_mut(&mut self) -> &mut MeshPool {
        &mut self.mesh_pool
    }

    /// Exposes asset subsystem hooks (upload queues, handle table) for future workers.
    pub fn assets_mut(&mut self) -> &mut AssetSubsystem {
        &mut self.assets
    }

    /// Applies pending init once a GPU/window stack exists (e.g. window title).
    pub fn take_pending_init(&mut self) -> Option<RendererInitData> {
        self.pending_init.take()
    }

    /// Call after [`crate::gpu::GpuContext`] is created so mesh uploads can use [`wgpu::Device`].
    pub fn attach_gpu(&mut self, device: Arc<wgpu::Device>) {
        self.gpu_device = Some(device.clone());
        let pending: Vec<MeshUploadData> = self.pending_mesh_uploads.drain(..).collect();
        for data in pending {
            self.try_mesh_upload_with_device(&device, data);
        }
    }

    /// If connected and init is complete, sends [`FrameStartData`] when we are ready for the next
    /// host frame (Unity: `_lastFrameDataProcessed` or bootstrap), then clears the processed flag.
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

    /// Placeholder for bounded asset integration between begin-frame and frame processing (Unity:
    /// `RunAssetIntegration`).
    pub fn run_asset_integration_stub(&mut self, _budget: Duration) {
        let _ = self.assets.drain_pending_meta();
    }

    /// Drains IPC and dispatches commands. Frame submissions are sorted before other commands from
    /// the same [`Self::poll`] batch.
    pub fn poll_ipc(&mut self) {
        let Some(ref mut ipc) = self.ipc else {
            return;
        };
        let mut batch = ipc.poll();
        batch.sort_by_key(|c| !matches!(c, RendererCommand::frame_submit_data(_)));
        for cmd in batch {
            self.handle_command(cmd);
        }
    }

    fn handle_command(&mut self, cmd: RendererCommand) {
        match self.init_state {
            InitState::Uninitialized => match cmd {
                RendererCommand::keep_alive(_) => {}
                RendererCommand::renderer_init_data(d) => self.on_init_data(d),
                _ => {
                    logger::error!("IPC: expected RendererInitData first");
                    self.fatal_error = true;
                }
            },
            InitState::InitReceived => match cmd {
                RendererCommand::keep_alive(_) => {}
                RendererCommand::renderer_init_finalize_data(_) => {
                    self.init_state = InitState::Finalized;
                }
                RendererCommand::renderer_init_progress_update(_) => {}
                RendererCommand::renderer_engine_ready(_) => {}
                _ => {
                    logger::trace!("IPC: deferring command until init finalized (skeleton)");
                }
            },
            InitState::Finalized => self.handle_running_command(cmd),
        }
    }

    fn on_init_data(&mut self, d: RendererInitData) {
        if let Some(ref prefix) = d.shared_memory_prefix {
            self.shared_memory = Some(SharedMemoryAccessor::new(prefix.clone()));
            logger::info!("Shared memory prefix: {}", prefix);
        }
        self.pending_init = Some(d.clone());
        if let Some(ref mut ipc) = self.ipc {
            send_renderer_init_result(ipc, d.output_device);
        }
        self.init_state = InitState::InitReceived;
        self.last_frame_data_processed = true;
    }

    fn handle_running_command(&mut self, cmd: RendererCommand) {
        match cmd {
            RendererCommand::keep_alive(_) => {}
            RendererCommand::renderer_shutdown(_)
            | RendererCommand::renderer_shutdown_request(_) => {
                self.shutdown_requested = true;
            }
            RendererCommand::frame_submit_data(data) => self.on_frame_submit(data),
            RendererCommand::mesh_upload_data(d) => self.try_process_mesh_upload(d),
            RendererCommand::mesh_unload(u) => self.on_mesh_unload(u),
            RendererCommand::free_shared_memory_view(f) => {
                if let Some(shm) = self.shared_memory.as_mut() {
                    shm.release_view(f.buffer_id);
                }
            }
            _ => {
                logger::trace!("runtime: unhandled RendererCommand (expand handlers here)");
            }
        }
    }

    fn try_process_mesh_upload(&mut self, data: MeshUploadData) {
        if data.buffer.length <= 0 {
            return;
        }
        let Some(device) = self.gpu_device.clone() else {
            if self.pending_mesh_uploads.len() >= MAX_PENDING_MESH_UPLOADS {
                logger::warn!(
                    "mesh upload pending queue full; dropping asset {}",
                    data.asset_id
                );
                return;
            }
            self.pending_mesh_uploads.push_back(data);
            return;
        };
        self.try_mesh_upload_with_device(&device, data);
    }

    fn try_mesh_upload_with_device(&mut self, device: &Arc<wgpu::Device>, data: MeshUploadData) {
        let Some(shm) = self.shared_memory.as_mut() else {
            logger::warn!(
                "mesh {}: no shared memory accessor (standalone or missing prefix)",
                data.asset_id
            );
            return;
        };
        let upload_result = shm.with_read_bytes(&data.buffer, |raw| {
            try_upload_mesh_from_raw(device.as_ref(), raw, &data)
        });
        let Some(mesh) = upload_result else {
            logger::warn!("mesh {}: upload failed or rejected", data.asset_id);
            return;
        };
        let existed_before = self.mesh_pool.insert_mesh(mesh);
        if let Some(ref mut ipc) = self.ipc {
            ipc.send_background(RendererCommand::mesh_upload_result(MeshUploadResult {
                asset_id: data.asset_id,
                instance_changed: !existed_before,
            }));
        }
        logger::info!(
            "mesh {} uploaded (replaced={} resident_bytes≈{})",
            data.asset_id,
            existed_before,
            self.mesh_pool.accounting().total_resident_bytes()
        );
    }

    fn on_mesh_unload(&mut self, u: MeshUnload) {
        if self.mesh_pool.remove_mesh(u.asset_id) {
            logger::info!(
                "mesh {} unloaded (resident_bytes≈{})",
                u.asset_id,
                self.mesh_pool.accounting().total_resident_bytes()
            );
        }
    }

    fn on_frame_submit(&mut self, data: FrameSubmitData) {
        self.last_frame_index = data.frame_index;
        self.last_frame_data_processed = true;
        let start = Instant::now();
        self.run_asset_integration_stub(Duration::from_millis(2));
        logger::trace!(
            "frame_submit frame_index={} stub_integration_ms={:.3}",
            data.frame_index,
            start.elapsed().as_secs_f64() * 1000.0
        );
    }
}

fn send_renderer_init_result(ipc: &mut DualQueueIpc, output_device: HeadOutputDevice) {
    let result = RendererInitResult {
        actual_output_device: output_device,
        renderer_identifier: Some("Renderide 0.1.0 (wgpu skeleton)".to_string()),
        main_window_handle_ptr: 0,
        stereo_rendering_mode: Some("None".to_string()),
        max_texture_size: 8192,
        is_gpu_texture_pot_byte_aligned: true,
        supported_texture_formats: vec![TextureFormat::rgba32],
    };
    ipc.send_primary(RendererCommand::renderer_init_result(result));
}
