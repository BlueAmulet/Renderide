//! Runtime application of decoded IPC dispatch effects.

use crate::diagnostics::crash_context::{self, InitState as CrashInitState};
use crate::frontend::InitState;
use crate::frontend::dispatch::command_dispatch::RunningCommandEffect;
use crate::frontend::dispatch::ipc_init::{self, IpcDispatchEffect, RendererInitCapabilities};
use crate::frontend::output_device::head_output_device_wants_openxr;
use crate::ipc::SharedMemoryAccessor;
use crate::shared::{
    DesktopConfig, FrameStartData, MaterialPropertyIdRequest, MaterialPropertyIdResult,
    MeshUploadData, RenderDecouplingConfig, RendererCommand, RendererInitData, SetCubemapData,
    SetCubemapFormat, SetCubemapProperties, SetTexture2DData, SetTexture2DFormat,
    SetTexture2DProperties, SetTexture3DData, SetTexture3DFormat, SetTexture3DProperties,
};

use super::RendererRuntime;

impl RendererRuntime {
    /// Decodes and applies one IPC command according to the current init state.
    pub(crate) fn handle_ipc_command(&mut self, cmd: RendererCommand) {
        let effect = ipc_init::dispatch_ipc_command(self.frontend.init_state(), cmd);
        self.apply_ipc_dispatch_effect(effect);
    }

    /// Applies an init-routed command effect.
    pub(crate) fn apply_ipc_dispatch_effect(&mut self, effect: IpcDispatchEffect) {
        match effect {
            IpcDispatchEffect::Ignore => {}
            IpcDispatchEffect::ApplyInitData(d) => {
                self.apply_renderer_init_data(d);
                crash_context::set_init_state(CrashInitState::InitDataReceived);
            }
            IpcDispatchEffect::Finalize => {
                logger::info!("IPC init finalized; renderer entering running command dispatch");
                self.frontend.set_init_state(InitState::Finalized);
                crash_context::set_init_state(CrashInitState::Finalized);
                self.replay_deferred_pre_finalize_commands();
            }
            IpcDispatchEffect::DispatchRunning(effect) => {
                self.apply_running_command_effect(effect);
            }
            IpcDispatchEffect::DeferUntilFinalized(cmd) => {
                logger::trace!("IPC: deferring command until init finalized");
                self.ipc_state.defer_pre_finalize_command(*cmd);
            }
            IpcDispatchEffect::FatalExpectedInitData { actual_tag } => {
                logger::error!(
                    "IPC: expected RendererInitData first, received RendererCommand::{actual_tag}\n{}",
                    crash_context::format_snapshot()
                );
                self.frontend.set_fatal_error(true);
            }
        }
    }

    /// Replays commands that arrived after init data and before init finalization.
    pub(crate) fn replay_deferred_pre_finalize_commands(&mut self) {
        let mut deferred = self.ipc_state.take_deferred_pre_finalize_commands();
        if deferred.is_empty() {
            return;
        }
        logger::info!(
            "IPC init finalized; replaying {} deferred command(s) mix=[{}]",
            deferred.len(),
            super::ipc_state::summarize_renderer_command_mix(deferred.iter())
        );
        while let Some(cmd) = deferred.pop_front() {
            self.handle_ipc_command(cmd);
            if self.frontend.fatal_error() {
                break;
            }
        }
    }

    /// Applies a decoded post-init command effect to runtime-owned domains.
    pub(crate) fn apply_running_command_effect(&mut self, effect: RunningCommandEffect) {
        match effect {
            RunningCommandEffect::KeepAlive => {}
            RunningCommandEffect::RequestShutdown => self.frontend.set_shutdown_requested(true),
            RunningCommandEffect::FrameSubmit(data) => self.apply_frame_submit_data(data),
            RunningCommandEffect::MeshUpload(d) => self.process_mesh_upload(d),
            RunningCommandEffect::MeshUnload(u) => self.backend.on_mesh_unload(u),
            effect @ (RunningCommandEffect::SetTexture2DFormat(_)
            | RunningCommandEffect::SetTexture2DProperties(_)
            | RunningCommandEffect::SetTexture2DData(_)
            | RunningCommandEffect::UnloadTexture2D(_)
            | RunningCommandEffect::SetTexture3DFormat(_)
            | RunningCommandEffect::SetTexture3DProperties(_)
            | RunningCommandEffect::SetTexture3DData(_)
            | RunningCommandEffect::UnloadTexture3D(_)
            | RunningCommandEffect::SetCubemapFormat(_)
            | RunningCommandEffect::SetCubemapProperties(_)
            | RunningCommandEffect::SetCubemapData(_)
            | RunningCommandEffect::UnloadCubemap(_)
            | RunningCommandEffect::SetRenderTextureFormat(_)
            | RunningCommandEffect::UnloadRenderTexture(_)) => {
                self.apply_texture_asset_effect(effect);
            }
            effect @ (RunningCommandEffect::SetDesktopTextureProperties(_)
            | RunningCommandEffect::DesktopTexturePropertiesUpdate(_)
            | RunningCommandEffect::UnloadDesktopTexture(_)
            | RunningCommandEffect::PointRenderBufferUpload(_)
            | RunningCommandEffect::PointRenderBufferUnload(_)
            | RunningCommandEffect::TrailRenderBufferUpload(_)
            | RunningCommandEffect::TrailRenderBufferUnload(_)
            | RunningCommandEffect::GaussianSplatConfig(_)
            | RunningCommandEffect::GaussianSplatUploadRaw(_)
            | RunningCommandEffect::GaussianSplatUploadEncoded(_)
            | RunningCommandEffect::UnloadGaussianSplat(_)
            | RunningCommandEffect::PointRenderBufferConsumed
            | RunningCommandEffect::TrailRenderBufferConsumed
            | RunningCommandEffect::GaussianSplatResult) => {
                self.apply_auxiliary_asset_effect(effect);
            }
            effect @ (RunningCommandEffect::VideoTextureLoad(_)
            | RunningCommandEffect::VideoTextureUpdate(_)
            | RunningCommandEffect::VideoTextureProperties(_)
            | RunningCommandEffect::VideoTextureStartAudioTrack(_)
            | RunningCommandEffect::UnloadVideoTexture(_)) => {
                self.apply_video_texture_effect(effect);
            }
            RunningCommandEffect::FreeSharedMemoryView { buffer_id } => {
                self.release_shared_memory_view(buffer_id);
            }
            effect @ (RunningCommandEffect::MaterialPropertyIdRequest(_)
            | RunningCommandEffect::MaterialsUpdateBatch(_)
            | RunningCommandEffect::UnloadMaterial { .. }
            | RunningCommandEffect::UnloadMaterialPropertyBlock { .. }
            | RunningCommandEffect::ShaderUpload(_)
            | RunningCommandEffect::ShaderUnload(_)) => self.apply_material_shader_effect(effect),
            RunningCommandEffect::FrameStartData(fs) => log_frame_start_data_trace(fs.as_ref()),
            RunningCommandEffect::LightsBufferRendererSubmission(sub) => {
                self.apply_lights_buffer_renderer_submission(sub);
            }
            RunningCommandEffect::LightsBufferRendererConsumed => {
                logger::trace!("runtime: lights_buffer_renderer_consumed from host (ignored)");
            }
            RunningCommandEffect::RenderTextureResult => {
                logger::trace!(
                    "runtime: render_texture_result from host (ignored; renderer is source)"
                );
            }
            RunningCommandEffect::RendererEngineReady => {
                logger::trace!(
                    "runtime: renderer_engine_ready from host (post-init lifecycle ack; no action)"
                );
            }
            RunningCommandEffect::DesktopConfig(cfg) => self.apply_desktop_config(cfg),
            RunningCommandEffect::RenderDecouplingConfig(cfg) => {
                self.apply_render_decoupling_config(cfg);
            }
            RunningCommandEffect::Unhandled { tag } => self.note_unhandled_renderer_command(tag),
        }
    }

    fn note_unhandled_renderer_command(&mut self, tag: &'static str) {
        let count = self.record_unhandled_renderer_command(tag);
        if count == 1 {
            logger::warn!(
                "runtime: no handler for RendererCommand::{tag} (host sent unexpected command; further occurrences counted in diagnostics)"
            );
        } else {
            logger::trace!(
                "runtime: no handler for RendererCommand::{tag} occurrence_count={count}"
            );
        }
    }

    fn apply_texture_asset_effect(&mut self, effect: RunningCommandEffect) {
        match effect {
            RunningCommandEffect::SetTexture2DFormat(f) => self.dispatch_texture_2d_format(f),
            RunningCommandEffect::SetTexture2DProperties(p) => {
                self.dispatch_texture_2d_properties(p);
            }
            RunningCommandEffect::SetTexture2DData(d) => self.dispatch_texture_2d_data(d),
            RunningCommandEffect::UnloadTexture2D(u) => self.backend.on_unload_texture_2d(u),
            RunningCommandEffect::SetTexture3DFormat(f) => self.dispatch_texture_3d_format(f),
            RunningCommandEffect::SetTexture3DProperties(p) => {
                self.dispatch_texture_3d_properties(p);
            }
            RunningCommandEffect::SetTexture3DData(d) => self.dispatch_texture_3d_data(d),
            RunningCommandEffect::UnloadTexture3D(u) => self.backend.on_unload_texture_3d(u),
            RunningCommandEffect::SetCubemapFormat(f) => self.dispatch_cubemap_format(f),
            RunningCommandEffect::SetCubemapProperties(p) => self.dispatch_cubemap_properties(p),
            RunningCommandEffect::SetCubemapData(d) => self.dispatch_cubemap_data(d),
            RunningCommandEffect::UnloadCubemap(u) => self.backend.on_unload_cubemap(u),
            RunningCommandEffect::SetRenderTextureFormat(f) => self
                .backend
                .on_set_render_texture_format(f, self.frontend.ipc_mut()),
            RunningCommandEffect::UnloadRenderTexture(u) => {
                self.backend.on_unload_render_texture(u);
            }
            _ => {}
        }
    }

    fn apply_auxiliary_asset_effect(&mut self, effect: RunningCommandEffect) {
        match effect {
            RunningCommandEffect::SetDesktopTextureProperties(p) => self
                .backend
                .on_set_desktop_texture_properties(p, self.frontend.ipc_mut()),
            RunningCommandEffect::DesktopTexturePropertiesUpdate(u) => {
                self.backend.on_desktop_texture_properties_update(u);
            }
            RunningCommandEffect::UnloadDesktopTexture(u) => {
                self.backend.on_unload_desktop_texture(u);
            }
            RunningCommandEffect::PointRenderBufferUpload(u) => self
                .backend
                .on_point_render_buffer_upload(u, self.frontend.ipc_mut()),
            RunningCommandEffect::PointRenderBufferUnload(u) => {
                self.backend.on_point_render_buffer_unload(u);
            }
            RunningCommandEffect::TrailRenderBufferUpload(u) => self
                .backend
                .on_trail_render_buffer_upload(u, self.frontend.ipc_mut()),
            RunningCommandEffect::TrailRenderBufferUnload(u) => {
                self.backend.on_trail_render_buffer_unload(u);
            }
            RunningCommandEffect::GaussianSplatConfig(c) => {
                self.backend.on_gaussian_splat_config(c);
            }
            RunningCommandEffect::GaussianSplatUploadRaw(u) => self
                .backend
                .on_gaussian_splat_upload_raw(u, self.frontend.ipc_mut()),
            RunningCommandEffect::GaussianSplatUploadEncoded(u) => self
                .backend
                .on_gaussian_splat_upload_encoded(u, self.frontend.ipc_mut()),
            RunningCommandEffect::UnloadGaussianSplat(u) => {
                self.backend.on_unload_gaussian_splat(u);
            }
            RunningCommandEffect::PointRenderBufferConsumed => {
                logger::trace!(
                    "runtime: point_render_buffer_consumed from host (ignored; renderer is source)"
                );
            }
            RunningCommandEffect::TrailRenderBufferConsumed => {
                logger::trace!(
                    "runtime: trail_render_buffer_consumed from host (ignored; renderer is source)"
                );
            }
            RunningCommandEffect::GaussianSplatResult => {
                logger::trace!(
                    "runtime: gaussian_splat_result from host (ignored; renderer is source)"
                );
            }
            _ => {}
        }
    }

    fn apply_video_texture_effect(&mut self, effect: RunningCommandEffect) {
        match effect {
            RunningCommandEffect::VideoTextureLoad(l) => self.backend.on_video_texture_load(l),
            RunningCommandEffect::VideoTextureUpdate(u) => self.backend.on_video_texture_update(u),
            RunningCommandEffect::VideoTextureProperties(p) => {
                self.backend.on_video_texture_properties(p);
            }
            RunningCommandEffect::VideoTextureStartAudioTrack(s) => {
                self.backend.on_video_texture_start_audio_track(s);
            }
            RunningCommandEffect::UnloadVideoTexture(u) => self.backend.on_unload_video_texture(u),
            _ => {}
        }
    }

    fn apply_material_shader_effect(&mut self, effect: RunningCommandEffect) {
        match effect {
            RunningCommandEffect::MaterialPropertyIdRequest(req) => {
                self.material_property_id_request(req);
            }
            RunningCommandEffect::MaterialsUpdateBatch(batch) => {
                super::shader_material_ipc::on_materials_update_batch(
                    &mut self.frontend,
                    &mut self.backend,
                    batch,
                );
            }
            RunningCommandEffect::UnloadMaterial { asset_id } => {
                self.backend.on_unload_material(asset_id);
            }
            RunningCommandEffect::UnloadMaterialPropertyBlock { asset_id } => {
                self.backend.on_unload_material_property_block(asset_id);
            }
            RunningCommandEffect::ShaderUpload(u) => {
                super::shader_material_ipc::on_shader_upload(
                    &mut self.ipc_state.pending_shader_resolutions,
                    u,
                );
            }
            RunningCommandEffect::ShaderUnload(u) => {
                super::shader_material_ipc::on_shader_unload(&mut self.backend, u);
            }
            _ => {}
        }
    }

    fn apply_renderer_init_data(&mut self, d: RendererInitData) {
        logger::info!(
            "IPC init data received: output_device={:?} shared_memory_prefix_present={}",
            d.output_device,
            d.shared_memory_prefix.is_some(),
        );
        self.host_camera.output_device = d.output_device;
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
        let init_result = ipc_init::build_renderer_init_result(
            d.output_device,
            renderer_init_capabilities(d.output_device),
        );
        if let Some(ipc) = self.frontend.ipc_mut()
            && !ipc.send_primary(RendererCommand::RendererInitResult(init_result))
        {
            logger::error!(
                "IPC: RendererInitResult was not sent (primary queue full); stopping init handshake"
            );
            self.frontend.set_fatal_error(true);
            return;
        }
        self.frontend.on_init_received();
    }

    fn process_mesh_upload(&mut self, d: MeshUploadData) {
        let (shm, ipc) = self.frontend.transport_pair_mut();
        if let Some(shm) = shm {
            self.backend.try_process_mesh_upload(d, shm, ipc);
        } else {
            logger::warn!("mesh upload: no shared memory (standalone?)");
        }
    }

    fn release_shared_memory_view(&mut self, buffer_id: i32) {
        if let Some(shm) = self.frontend.shared_memory_mut() {
            shm.release_view(buffer_id);
        }
    }

    fn dispatch_texture_2d_format(&mut self, f: SetTexture2DFormat) {
        self.backend
            .on_set_texture_2d_format(f, self.frontend.ipc_mut());
    }

    fn dispatch_texture_2d_properties(&mut self, p: SetTexture2DProperties) {
        self.backend
            .on_set_texture_2d_properties(p, self.frontend.ipc_mut());
    }

    fn dispatch_texture_2d_data(&mut self, d: SetTexture2DData) {
        let (shm, ipc) = self.frontend.transport_pair_mut();
        self.backend.on_set_texture_2d_data(d, shm, ipc);
    }

    fn dispatch_texture_3d_format(&mut self, f: SetTexture3DFormat) {
        self.backend
            .on_set_texture_3d_format(f, self.frontend.ipc_mut());
    }

    fn dispatch_texture_3d_properties(&mut self, p: SetTexture3DProperties) {
        self.backend
            .on_set_texture_3d_properties(p, self.frontend.ipc_mut());
    }

    fn dispatch_texture_3d_data(&mut self, d: SetTexture3DData) {
        let (shm, ipc) = self.frontend.transport_pair_mut();
        self.backend.on_set_texture_3d_data(d, shm, ipc);
    }

    fn dispatch_cubemap_format(&mut self, f: SetCubemapFormat) {
        self.backend
            .on_set_cubemap_format(f, self.frontend.ipc_mut());
    }

    fn dispatch_cubemap_properties(&mut self, p: SetCubemapProperties) {
        self.backend
            .on_set_cubemap_properties(p, self.frontend.ipc_mut());
    }

    fn dispatch_cubemap_data(&mut self, d: SetCubemapData) {
        let (shm, ipc) = self.frontend.transport_pair_mut();
        self.backend.on_set_cubemap_data(d, shm, ipc);
    }

    fn apply_render_decoupling_config(&mut self, cfg: RenderDecouplingConfig) {
        logger::info!(
            "runtime: render_decoupling_config activate_interval_s={:.4} decoupled_max_asset_processing_s={:.4} recouple_frame_count={}",
            cfg.decouple_activate_interval,
            cfg.decoupled_max_asset_processing_time,
            cfg.recouple_frame_count
        );
        self.frontend.set_decoupling_config(cfg);
    }

    fn apply_desktop_config(&self, _cfg: DesktopConfig) {
        logger::trace!(
            "runtime: desktop_config ignored; renderer config owns desktop frame pacing"
        );
    }

    fn material_property_id_request(&mut self, req: MaterialPropertyIdRequest) {
        profiling::scope!("command::material_property_id_request");
        let property_ids: Vec<i32> = {
            let reg = self.backend.property_id_registry();
            req.property_names
                .iter()
                .map(|n| reg.intern_for_host_request(n.as_deref().unwrap_or("")))
                .collect()
        };
        if let Some(ipc) = self.frontend.ipc_mut() {
            let _ = ipc.send_background_reliable(RendererCommand::MaterialPropertyIdResult(
                MaterialPropertyIdResult {
                    request_id: req.request_id,
                    property_ids,
                },
            ));
        }
    }

    fn apply_lights_buffer_renderer_submission(
        &mut self,
        sub: crate::shared::LightsBufferRendererSubmission,
    ) {
        let buffer_id = sub.lights_buffer_unique_id;
        let (shm, ipc) = self.frontend.transport_pair_mut();
        let Some(shm) = shm else {
            logger::warn!("lights_buffer_renderer_submission: no shared memory (id={buffer_id})");
            return;
        };
        super::lights_ipc::apply_lights_buffer_submission(&mut self.scene, shm, ipc, sub);
    }
}

fn renderer_init_capabilities(
    output_device: crate::shared::HeadOutputDevice,
) -> RendererInitCapabilities {
    let stereo_rendering_mode = if head_output_device_wants_openxr(output_device) {
        "OpenXR(multiview)"
    } else {
        "None"
    };
    RendererInitCapabilities {
        stereo_rendering_mode: stereo_rendering_mode.to_string(),
        max_texture_size: crate::gpu::RENDERER_MAX_TEXTURE_DIMENSION_2D as i32,
        supported_texture_formats: crate::assets::texture::supported_host_formats_for_init(),
    }
}

/// Logs structured fields from a host [`FrameStartData`] payload (lock-step / diagnostics only).
fn log_frame_start_data_trace(fs: &FrameStartData) {
    logger::trace!(
        "host frame_start_data: last_frame_index={} has_performance={} has_inputs={} reflection_probes={} video_clock_errors={}",
        fs.last_frame_index,
        fs.performance.is_some(),
        fs.inputs.is_some(),
        fs.rendered_reflection_probes.len(),
        fs.video_clock_errors.len(),
    );
}
