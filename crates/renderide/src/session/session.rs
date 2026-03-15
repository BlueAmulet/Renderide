//! Session: orchestrates IPC, scene, assets, and frame flow.

use nalgebra::Matrix4;

use crate::assets::AssetRegistry;
use crate::config::RenderConfig;
use crate::ipc::mapper::{CommandMapper, TranslatedCommand};
use crate::ipc::receiver::CommandReceiver;
use crate::ipc::shared_memory::SharedMemoryAccessor;
use crate::render::batch::SpaceDrawBatch;
use crate::scene::{render_transform_to_matrix, SceneGraph};
use crate::session::init::{get_connection_parameters, take_singleton_init, InitError};
use crate::session::state::ViewState;
use crate::shared::{
    FrameStartData, FrameSubmitData, HeadOutputDevice, InputState, MeshUploadResult,
    RendererCommand, RendererInitResult, TextureFormat,
};

/// Main session: coordinates command ingest, translation, scene, and assets.
pub struct Session {
    receiver: CommandReceiver,
    mapper: CommandMapper,
    scene_graph: SceneGraph,
    asset_registry: AssetRegistry,
    view_state: ViewState,
    shared_memory: Option<SharedMemoryAccessor>,
    init_received: bool,
    init_finalized: bool,
    is_standalone: bool,
    shutdown: bool,
    fatal_error: bool,
    last_frame_index: i32,
    last_frame_data_processed: bool,
    sent_bootstrap_frame_start: bool,
    pending_input: Option<InputState>,
    pending_mesh_unloads: Vec<i32>,
    lock_cursor: bool,
    render_config: RenderConfig,
    pending_render_tasks: Vec<crate::shared::CameraRenderTask>,
    primary_camera_task: Option<crate::shared::CameraRenderTask>,
    primary_view_transform: Option<crate::shared::RenderTransform>,
    /// Space ID and override flag for the primary view (diagnostic).
    primary_view_space_id: Option<i32>,
    primary_view_override: Option<bool>,
    /// Whether view position comes from external source (e.g. VR head) — diagnostic.
    primary_view_position_is_external: Option<bool>,
    /// Root transform of primary space (for diagnostic: compare with view when override differs).
    primary_root_transform: Option<crate::shared::RenderTransform>,
}

impl Session {
    /// Creates a new session.
    pub fn new() -> Self {
        Self {
            receiver: CommandReceiver::new(),
            mapper: CommandMapper::default(),
            scene_graph: SceneGraph::new(),
            asset_registry: AssetRegistry::new(),
            view_state: ViewState::default(),
            shared_memory: None,
            init_received: false,
            init_finalized: false,
            is_standalone: false,
            shutdown: false,
            fatal_error: false,
            last_frame_index: -1,
            last_frame_data_processed: false,
            sent_bootstrap_frame_start: false,
            pending_input: None,
            pending_mesh_unloads: Vec::new(),
            lock_cursor: false,
            render_config: RenderConfig::default(),
            pending_render_tasks: Vec::new(),
            primary_camera_task: None,
            primary_view_transform: None,
            primary_view_space_id: None,
            primary_view_override: None,
            primary_view_position_is_external: None,
            primary_root_transform: None,
        }
    }

    /// Initializes the session. Call once at startup.
    pub fn init(&mut self) -> Result<(), InitError> {
        if !take_singleton_init() {
            return Err(InitError::SingletonAlreadyExists);
        }

        if get_connection_parameters().is_none() {
            self.is_standalone = true;
            self.init_finalized = true;
            return Ok(());
        }

        self.receiver.connect()?;
        if !self.receiver.is_connected() {
            self.is_standalone = true;
            self.init_finalized = true;
        }
        Ok(())
    }

    /// Per-frame update. Returns Some(exit_code) to request exit.
    pub fn update(&mut self) -> Option<i32> {
        if self.shutdown {
            return Some(0);
        }
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.handle_update())) {
            Ok(()) => None,
            Err(_e) => {
                self.fatal_error = true;
                Some(4)
            }
        }
    }

    fn handle_update(&mut self) {
        self.process_commands();

        if self.init_finalized && !self.fatal_error {
            let bootstrap = self.last_frame_index < 0 && !self.sent_bootstrap_frame_start;
            let should_send = self.last_frame_data_processed || bootstrap;
            if should_send && self.receiver.is_connected() {
                self.send_begin_frame();
                self.last_frame_data_processed = false;
                if bootstrap {
                    self.sent_bootstrap_frame_start = true;
                }
            }
        }
    }

    fn process_commands(&mut self) {
        let commands = self.receiver.poll();

        let (mesh_cmds, other_cmds): (Vec<_>, Vec<_>) = commands
            .into_iter()
            .partition(|c| matches!(c, RendererCommand::mesh_upload_data(_) | RendererCommand::mesh_unload(_)));

        for cmd in mesh_cmds {
            self.apply_command(cmd);
        }
        for cmd in other_cmds {
            self.apply_command(cmd);
        }
    }

    fn apply_command(&mut self, cmd: RendererCommand) {
        if !self.init_received {
            match &cmd {
                RendererCommand::renderer_init_data(_) => {
                    let translated = self.mapper.translate(cmd);
                    self.apply_translated(translated);
                    self.init_received = true;
                }
                _ => self.fatal_error = true,
            }
            return;
        }

        if !self.init_finalized {
            let translated = self.mapper.translate(cmd);
            match translated {
                TranslatedCommand::InitFinalize => {
                    self.init_finalized = true;
                }
                TranslatedCommand::MeshUpload(_) | TranslatedCommand::MeshUnload(_) => {
                    self.apply_translated(translated);
                }
                TranslatedCommand::FrameSubmit(data) => {
                    if self.shared_memory.is_some() {
                        if let Err(_e) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            self.process_frame_data(data);
                        })) {
                            self.fatal_error = true;
                        }
                    }
                }
                _ => {}
            }
            return;
        }

        let translated = self.mapper.translate(cmd);
        self.apply_translated(translated);
    }

    fn apply_translated(&mut self, cmd: TranslatedCommand) {
        match cmd {
            TranslatedCommand::SessionInit(config) => {
                if let Some(prefix) = config.shared_memory_prefix {
                    self.shared_memory = Some(SharedMemoryAccessor::new(prefix));
                }
                self.send_renderer_init_result();
            }
            TranslatedCommand::SessionShutdown => self.shutdown = true,
            TranslatedCommand::InitFinalize => self.init_finalized = true,
            TranslatedCommand::FrameSubmit(data) => {
                if self.init_finalized {
                    if let Err(_e) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        self.process_frame_data(data);
                    })) {
                        self.fatal_error = true;
                    } else {
                        self.last_frame_data_processed = true;
                    }
                }
            }
            TranslatedCommand::MeshUpload(data) => {
                let asset_id = data.asset_id;
                let (success, existed_before) = match &mut self.shared_memory {
                    Some(shm) => self.asset_registry.handle_mesh_upload(shm, data),
                    None => (false, false),
                };
                if success {
                    self.receiver.send_background(RendererCommand::mesh_upload_result(
                        MeshUploadResult {
                            asset_id,
                            instance_changed: !existed_before,
                        },
                    ));
                }
            }
            TranslatedCommand::MeshUnload(asset_id) => {
                self.asset_registry.handle_mesh_unload(asset_id);
                self.pending_mesh_unloads.push(asset_id);
            }
            TranslatedCommand::ConfigUpdate(config) => {
                self.view_state.near_clip = config.near_clip;
                self.view_state.far_clip = config.far_clip;
                self.view_state.desktop_fov = config.desktop_fov;
                self.render_config = config;
            }
            TranslatedCommand::NoOp | TranslatedCommand::Unimplemented(_) => {}
        }
    }

    fn process_frame_data(&mut self, data: FrameSubmitData) {
        self.last_frame_index = data.frame_index;
        self.view_state.near_clip = data.near_clip;
        self.view_state.far_clip = data.far_clip;
        self.view_state.desktop_fov = data.desktop_fov;
        self.primary_view_transform = None;
        self.primary_view_space_id = None;
        self.primary_view_override = None;
        self.primary_view_position_is_external = None;
        self.primary_root_transform = None;

        if let Some(ref output) = data.output_state {
            self.lock_cursor = output.lock_cursor;
        }

        if let Some(ref mut shm) = self.shared_memory {
            self.scene_graph.apply_frame_update(shm, &data);
        }

        let active_non_overlay: Vec<_> = data
            .render_spaces
            .iter()
            .filter(|u| u.is_active && !u.is_overlay)
            .collect();
        if active_non_overlay.len() > 1 {
            self.fatal_error = true;
            return;
        }
        if let Some(update) = active_non_overlay.first() {
            self.primary_view_space_id = Some(update.id);
            self.primary_view_override = Some(update.override_view_position);
            self.primary_view_position_is_external = Some(update.view_position_is_external);
            self.primary_root_transform = Some(update.root_transform);
            // View selection: override (freecam) → overriden_view_transform; else → root_transform.
            // When view_position_is_external is true (e.g. VR/third-person), view may need to come
            // from input/head state; we use root for now since the host does not send a separate head pose.
            self.primary_view_transform = Some(if update.override_view_position {
                update.overriden_view_transform
            } else {
                update.root_transform
            });
        }
        if self.primary_view_transform.is_none() {
            if let Some(first) = data.render_spaces.first() {
                self.primary_view_space_id = Some(first.id);
                self.primary_view_override = Some(first.override_view_position);
                self.primary_view_position_is_external = Some(first.view_position_is_external);
                self.primary_root_transform = Some(first.root_transform);
                self.primary_view_transform = Some(first.root_transform);
            }
        }

        self.pending_render_tasks = data.render_tasks;
        self.primary_camera_task = self.pending_render_tasks.first().cloned();

    }

    fn send_renderer_init_result(&mut self) {
        let result = RendererInitResult {
            actual_output_device: HeadOutputDevice::screen,
            renderer_identifier: Some("Renderide 0.1.0 (wgpu)".to_string()),
            main_window_handle_ptr: 0,
            stereo_rendering_mode: Some("None".to_string()),
            max_texture_size: 8192,
            is_gpu_texture_pot_byte_aligned: true,
            supported_texture_formats: vec![TextureFormat::rgba32],
            ..Default::default()
        };
        self.receiver.send(RendererCommand::renderer_init_result(result));
    }

    fn send_begin_frame(&mut self) {
        let frame_start = FrameStartData {
            last_frame_index: self.last_frame_index,
            performance: None,
            inputs: self.pending_input.take(),
            rendered_reflection_probes: Vec::new(),
            video_clock_errors: Vec::new(),
        };
        self.receiver.send(RendererCommand::frame_start_data(frame_start));
    }

    /// Processes render tasks (camera renders to buffers). Stub for now.
    pub fn process_render_tasks(&mut self) {
        self.pending_render_tasks.clear();
    }

    /// Drains mesh asset IDs unloaded this frame.
    pub fn drain_pending_mesh_unloads(&mut self) -> Vec<i32> {
        std::mem::take(&mut self.pending_mesh_unloads)
    }

    /// Sets input for next FrameStartData.
    pub fn set_pending_input(&mut self, input: InputState) {
        self.pending_input = Some(input);
    }

    /// Whether cursor lock was requested.
    pub fn cursor_lock_requested(&self) -> bool {
        self.lock_cursor
    }

    /// Returns the asset registry.
    /// Returns the scene graph for world matrix lookups (e.g. bone transforms).
    pub fn scene_graph(&self) -> &SceneGraph {
        &self.scene_graph
    }

    pub fn asset_registry(&self) -> &AssetRegistry {
        &self.asset_registry
    }

    /// Returns the primary camera task.
    pub fn primary_camera_task(&self) -> Option<&crate::shared::CameraRenderTask> {
        self.primary_camera_task.as_ref()
    }

    /// Returns the space ID for the primary view (diagnostic).
    pub fn primary_view_space_id(&self) -> Option<i32> {
        self.primary_view_space_id
    }

    /// Returns whether the primary view uses overridden position (diagnostic).
    pub fn primary_view_override(&self) -> Option<bool> {
        self.primary_view_override
    }

    /// Returns whether the primary view position is external (e.g. VR head) — diagnostic.
    pub fn primary_view_position_is_external(&self) -> Option<bool> {
        self.primary_view_position_is_external
    }

    /// Returns the primary space root transform (diagnostic: compare with view when override differs).
    pub fn primary_root_transform(&self) -> Option<&crate::shared::RenderTransform> {
        self.primary_root_transform.as_ref()
    }

    /// Returns the primary view transform.
    pub fn primary_view_transform(&self) -> Option<&crate::shared::RenderTransform> {
        self.primary_view_transform.as_ref()
    }

    /// Last frame index.
    pub fn last_frame_index(&self) -> i32 {
        self.last_frame_index
    }

    /// Near clip.
    pub fn near_clip(&self) -> f32 {
        self.view_state.near_clip
    }

    /// Far clip.
    pub fn far_clip(&self) -> f32 {
        self.view_state.far_clip
    }

    /// Desktop FOV.
    pub fn desktop_fov(&self) -> f32 {
        self.view_state.desktop_fov
    }

    /// Collects draw batches for rendering.
    pub fn collect_draw_batches(&mut self) -> Vec<SpaceDrawBatch> {
        let mut batches = Vec::new();
        let active_space_ids: Vec<i32> = self
            .scene_graph
            .scenes()
            .iter()
            .filter(|(_, s)| s.is_active && !s.is_overlay)
            .map(|(id, _)| *id)
            .collect();

        let mut draw_batch_samples: Option<(i32, usize, usize, usize, Vec<(i32, String)>)> = None;

        for space_id in active_space_ids {
            self.scene_graph.compute_world_matrices(space_id);
            let scene = match self.scene_graph.get_scene(space_id) {
                Some(s) => s,
                None => continue,
            };

            let mut draws = Vec::new();
            let mut samples = Vec::new();
            let frame_index = self.last_frame_index;
            for entry in &scene.drawables {
                if entry.node_id < 0 {
                    continue;
                }
                let idx = entry.node_id as usize;
                let world_matrix = match self.scene_graph.get_world_matrix(space_id, idx) {
                    Some(m) => m,
                    None => {
                        if idx >= scene.nodes.len() {
                            continue;
                        }
                        let local =
                            render_transform_to_matrix(&scene.nodes[idx]);
                        local
                    }
                };
                let material_id = entry.material_handle.unwrap_or(-1);
                if samples.len() < 3 {
                    let t = world_matrix.column(3);
                    samples.push((entry.node_id, format!("({:.2},{:.2},{:.2})", t.x, t.y, t.z)));
                }
                draws.push((world_matrix, entry.mesh_handle, false, material_id, None));
            }
            for entry in &scene.skinned_drawables {
                if entry.node_id < 0 {
                    continue;
                }
                let idx = entry.node_id as usize;
                let world_matrix = match self.scene_graph.get_world_matrix(space_id, idx) {
                    Some(m) => m,
                    None => {
                        if idx >= scene.nodes.len() {
                            continue;
                        }
                        let local =
                            render_transform_to_matrix(&scene.nodes[idx]);
                        local
                    }
                };
                let material_id = entry.material_handle.unwrap_or(-1);
                if samples.len() < 3 {
                    let t = world_matrix.column(3);
                    samples.push((entry.node_id, format!("({:.2},{:.2},{:.2})", t.x, t.y, t.z)));
                }
                draws.push((
                    world_matrix,
                    entry.mesh_handle,
                    true,
                    material_id,
                    entry.bone_transform_ids.clone(),
                ));
            }

            if !draws.is_empty() {
                if draw_batch_samples.is_none() && !samples.is_empty() {
                    draw_batch_samples = Some((
                        space_id,
                        scene.nodes.len(),
                        scene.drawables.len() + scene.skinned_drawables.len(),
                        draws.len(),
                        samples,
                    ));
                }
                batches.push(SpaceDrawBatch {
                    space_id,
                    is_overlay: scene.is_overlay,
                    view_transform: scene.view_transform,
                    draws,
                });
            }
        }

        batches.sort_by_key(|b| b.is_overlay);
        batches
    }
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}
