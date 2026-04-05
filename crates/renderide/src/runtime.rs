//! Renderer façade: orchestrates **frontend** (IPC / shared memory / lock-step), **scene** (host
//! logical state), and **backend** (GPU pools, material store, uploads).
//!
//! Phase order aligns with `RenderingManager.HandleUpdate`: optionally send
//! [`FrameStartData`](crate::shared::FrameStartData), drain integration-style work (stub here), then
//! process incoming commands.
//!
//! Lock-step is driven by the `last_frame_index` field of [`FrameStartData`](crate::shared::FrameStartData)
//! on the **outgoing** `frame_start_data` the renderer sends from [`RendererRuntime::pre_frame`].
//! If the host sends [`RendererCommand::frame_start_data`](crate::shared::RendererCommand::frame_start_data),
//! optional payloads are trace-logged until consumers exist.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::assets::resolve_shader_upload;
use crate::assets::shader::classify_shader;
use crate::assets::texture::supported_host_formats_for_init;
use crate::assets::AssetSubsystem;
use crate::backend::RenderBackend;
use crate::connection::{ConnectionParams, InitError};
use crate::frontend::RendererFrontend;

pub use crate::frontend::InitState;
use crate::ipc::SharedMemoryAccessor;
use crate::scene::SceneCoordinator;
use crate::shared::{
    FrameSubmitData, HeadOutputDevice, MaterialPropertyIdResult, MaterialsUpdateBatch,
    RendererCommand, RendererInitData, RendererInitResult, ShaderUnload, ShaderUpload,
    ShaderUploadResult,
};

/// Facade: [`RendererFrontend`] + [`SceneCoordinator`] + [`RenderBackend`] + ingestion helpers.
pub struct RendererRuntime {
    frontend: RendererFrontend,
    backend: RenderBackend,
    /// Render spaces and dense transform / mesh state from [`FrameSubmitData`](crate::shared::FrameSubmitData).
    pub scene: SceneCoordinator,
    assets: AssetSubsystem,
}

impl RendererRuntime {
    /// Builds a runtime; does not open IPC yet (see [`Self::connect_ipc`]).
    pub fn new(params: Option<ConnectionParams>) -> Self {
        Self {
            frontend: RendererFrontend::new(params),
            backend: RenderBackend::new(),
            scene: SceneCoordinator::new(),
            assets: AssetSubsystem::default(),
        }
    }

    /// Mesh deformation compute pipelines when GPU init succeeded.
    pub fn mesh_preprocess(&self) -> Option<&crate::gpu::MeshPreprocessPipelines> {
        self.backend.mesh_preprocess()
    }

    /// Opens Primary/Background queues when [`Self::new`] was given connection parameters.
    pub fn connect_ipc(&mut self) -> Result<(), InitError> {
        self.frontend.connect_ipc()
    }

    /// Whether IPC queues are open.
    pub fn is_ipc_connected(&self) -> bool {
        self.frontend.is_ipc_connected()
    }

    pub fn init_state(&self) -> InitState {
        self.frontend.init_state()
    }

    /// After a successful [`FrameSubmitData`] application, host may expect another begin-frame.
    pub fn last_frame_data_processed(&self) -> bool {
        self.frontend.last_frame_data_processed
    }

    /// Current lock-step frame index echoed to the host.
    pub fn last_frame_index(&self) -> i32 {
        self.frontend.last_frame_index
    }

    pub fn shutdown_requested(&self) -> bool {
        self.frontend.shutdown_requested
    }

    pub fn fatal_error(&self) -> bool {
        self.frontend.fatal_error
    }

    /// Mesh pool and VRAM accounting (draw prep, debugging).
    pub fn mesh_pool(&self) -> &crate::resources::MeshPool {
        self.backend.mesh_pool()
    }

    /// Mutable mesh pool (eviction experiments).
    pub fn mesh_pool_mut(&mut self) -> &mut crate::resources::MeshPool {
        self.backend.mesh_pool_mut()
    }

    /// Resident Texture2D table (bind-group prep).
    pub fn texture_pool(&self) -> &crate::resources::TexturePool {
        self.backend.texture_pool()
    }

    /// Mutable texture pool.
    pub fn texture_pool_mut(&mut self) -> &mut crate::resources::TexturePool {
        self.backend.texture_pool_mut()
    }

    /// Exposes asset subsystem hooks (upload queues, handle table) for future workers.
    pub fn assets_mut(&mut self) -> &mut AssetSubsystem {
        &mut self.assets
    }

    /// Material property store (host uniforms, textures, shader asset bindings).
    pub fn material_property_store(&self) -> &crate::assets::material::MaterialPropertyStore {
        self.backend.material_property_store()
    }

    /// Mutable store for tests and tooling.
    pub fn material_property_store_mut(
        &mut self,
    ) -> &mut crate::assets::material::MaterialPropertyStore {
        self.backend.material_property_store_mut()
    }

    /// Property name interning for material batches.
    pub fn property_id_registry(&self) -> &crate::assets::material::PropertyIdRegistry {
        self.backend.property_id_registry()
    }

    /// Registered material families and pipeline cache (after GPU attach).
    pub fn material_registry(&self) -> Option<&crate::materials::MaterialRegistry> {
        self.backend.material_registry()
    }

    /// Mutable registry (e.g. register custom [`crate::materials::MaterialPipelineFamily`]).
    pub fn material_registry_mut(&mut self) -> Option<&mut crate::materials::MaterialRegistry> {
        self.backend.material_registry_mut()
    }

    /// Applies pending init once a GPU/window stack exists (e.g. window title).
    pub fn take_pending_init(&mut self) -> Option<RendererInitData> {
        self.frontend.take_pending_init()
    }

    /// Call after [`crate::gpu::GpuContext`] is created so mesh/texture uploads can use the GPU.
    pub fn attach_gpu(&mut self, device: Arc<wgpu::Device>, queue: Arc<Mutex<wgpu::Queue>>) {
        let shm = self.frontend.shared_memory_mut();
        self.backend.attach(device, queue, shm);
    }

    /// If connected and init is complete, sends [`FrameStartData`] when we are ready for the next host frame.
    pub fn pre_frame(&mut self) {
        self.frontend.pre_frame();
    }

    /// Placeholder for bounded asset integration between begin-frame and frame processing (Unity:
    /// `RunAssetIntegration`).
    pub fn run_asset_integration_stub(&mut self, _budget: Duration) {
        let _ = self.assets.drain_pending_meta();
    }

    /// Drains IPC and dispatches commands. Frame submissions are sorted before other commands from
    /// the same poll batch.
    pub fn poll_ipc(&mut self) {
        let batch = self.frontend.poll_commands();
        for cmd in batch {
            self.handle_command(cmd);
        }
    }

    fn handle_command(&mut self, cmd: RendererCommand) {
        match self.frontend.init_state() {
            InitState::Uninitialized => match cmd {
                RendererCommand::keep_alive(_) => {}
                RendererCommand::renderer_init_data(d) => self.on_init_data(d),
                _ => {
                    logger::error!("IPC: expected RendererInitData first");
                    self.frontend.fatal_error = true;
                }
            },
            InitState::InitReceived => match cmd {
                RendererCommand::keep_alive(_) => {}
                RendererCommand::renderer_init_finalize_data(_) => {
                    self.frontend.set_init_state(InitState::Finalized);
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
            self.frontend
                .set_shared_memory(SharedMemoryAccessor::new(prefix.clone()));
            logger::info!("Shared memory prefix: {}", prefix);
            let (shm, ipc) = self.frontend.transport_pair_mut();
            if let (Some(shm), Some(ipc)) = (shm, ipc) {
                self.backend.flush_pending_material_batches(shm, ipc);
            }
        }
        self.frontend.set_pending_init(d.clone());
        if let Some(ref mut ipc) = self.frontend.ipc_mut() {
            send_renderer_init_result(ipc, d.output_device);
        }
        self.frontend.on_init_received();
    }

    fn handle_running_command(&mut self, cmd: RendererCommand) {
        match cmd {
            RendererCommand::keep_alive(_) => {}
            RendererCommand::renderer_shutdown(_)
            | RendererCommand::renderer_shutdown_request(_) => {
                self.frontend.shutdown_requested = true;
            }
            RendererCommand::frame_submit_data(data) => self.on_frame_submit(data),
            RendererCommand::mesh_upload_data(d) => {
                let (shm, ipc) = self.frontend.transport_pair_mut();
                if let Some(shm) = shm {
                    self.backend.try_process_mesh_upload(d, shm, ipc);
                } else {
                    logger::warn!("mesh upload: no shared memory (standalone?)");
                }
            }
            RendererCommand::mesh_unload(u) => self.backend.on_mesh_unload(u),
            RendererCommand::set_texture_2d_format(f) => {
                self.backend
                    .on_set_texture_2d_format(f, self.frontend.ipc_mut());
            }
            RendererCommand::set_texture_2d_properties(p) => {
                self.backend
                    .on_set_texture_2d_properties(p, self.frontend.ipc_mut());
            }
            RendererCommand::set_texture_2d_data(d) => {
                let (shm, ipc) = self.frontend.transport_pair_mut();
                self.backend.on_set_texture_2d_data(d, shm, ipc);
            }
            RendererCommand::unload_texture_2d(u) => self.backend.on_unload_texture_2d(u),
            RendererCommand::free_shared_memory_view(f) => {
                if let Some(shm) = self.frontend.shared_memory_mut() {
                    shm.release_view(f.buffer_id);
                }
            }
            RendererCommand::material_property_id_request(req) => {
                let property_ids: Vec<i32> = {
                    let reg = self.backend.property_id_registry_mut();
                    req.property_names
                        .iter()
                        .map(|n| reg.intern_for_host_request(n.as_deref().unwrap_or("")))
                        .collect()
                };
                if let Some(ref mut ipc) = self.frontend.ipc_mut() {
                    ipc.send_background(RendererCommand::material_property_id_result(
                        MaterialPropertyIdResult {
                            request_id: req.request_id,
                            property_ids,
                        },
                    ));
                }
            }
            RendererCommand::materials_update_batch(batch) => {
                self.on_materials_update_batch(batch);
            }
            RendererCommand::unload_material(u) => self.backend.on_unload_material(u.asset_id),
            RendererCommand::unload_material_property_block(u) => {
                self.backend.on_unload_material_property_block(u.asset_id);
            }
            RendererCommand::shader_upload(u) => self.on_shader_upload(u),
            RendererCommand::shader_unload(u) => self.on_shader_unload(u),
            RendererCommand::frame_start_data(fs) => {
                logger::trace!(
                    "host frame_start_data: last_frame_index={} has_performance={} has_inputs={} reflection_probes={} video_clock_errors={}",
                    fs.last_frame_index,
                    fs.performance.is_some(),
                    fs.inputs.is_some(),
                    fs.rendered_reflection_probes.len(),
                    fs.video_clock_errors.len(),
                );
            }
            _ => {
                logger::trace!("runtime: unhandled RendererCommand (expand handlers here)");
            }
        }
    }

    fn on_shader_upload(&mut self, upload: ShaderUpload) {
        let asset_id = upload.asset_id;
        let resolved = resolve_shader_upload(&upload);
        let kind = classify_shader(
            resolved.unity_shader_name.as_deref(),
            upload.file.as_deref(),
        );
        logger::info!(
            "shader_upload: asset_id={} unity_shader_name={:?} kind={kind:?} material_family={:?}",
            asset_id,
            resolved.unity_shader_name.as_deref(),
            resolved.family,
        );
        self.backend
            .register_shader_route(asset_id, resolved.family);
        if let Some(ref mut ipc) = self.frontend.ipc_mut() {
            ipc.send_background(RendererCommand::shader_upload_result(ShaderUploadResult {
                asset_id,
                instance_changed: true,
            }));
        }
    }

    fn on_shader_unload(&mut self, unload: ShaderUnload) {
        let id = unload.asset_id;
        self.backend.unregister_shader_route(id);
    }

    fn on_materials_update_batch(&mut self, batch: MaterialsUpdateBatch) {
        if self.frontend.shared_memory().is_none() {
            if !self.backend.enqueue_materials_batch_no_shm(batch) {
                // already logged
            }
            return;
        }
        let (shm, ipc) = self.frontend.transport_pair_mut();
        let (Some(shm), Some(ipc)) = (shm, ipc) else {
            return;
        };
        self.backend.apply_materials_update_batch(batch, shm, ipc);
    }

    fn on_frame_submit(&mut self, data: FrameSubmitData) {
        self.frontend.note_frame_submit_processed(data.frame_index);
        let start = Instant::now();
        self.run_asset_integration_stub(Duration::from_millis(2));

        if let Some(ref mut shm) = self.frontend.shared_memory_mut() {
            if let Err(e) = self.scene.apply_frame_submit(shm, &data) {
                logger::error!("scene apply_frame_submit failed: {e}");
            }
            if let Err(e) = self.scene.flush_world_caches() {
                logger::error!("scene flush_world_caches failed: {e}");
            }
        }

        logger::trace!(
            "frame_submit frame_index={} stub_integration_ms={:.3}",
            data.frame_index,
            start.elapsed().as_secs_f64() * 1000.0
        );
    }
}

fn send_renderer_init_result(ipc: &mut crate::ipc::DualQueueIpc, output_device: HeadOutputDevice) {
    let result = RendererInitResult {
        actual_output_device: output_device,
        renderer_identifier: Some("Renderide 0.1.0 (wgpu skeleton)".to_string()),
        main_window_handle_ptr: 0,
        stereo_rendering_mode: Some("None".to_string()),
        max_texture_size: 8192,
        is_gpu_texture_pot_byte_aligned: true,
        supported_texture_formats: supported_host_formats_for_init(),
    };
    ipc.send_primary(RendererCommand::renderer_init_result(result));
}
