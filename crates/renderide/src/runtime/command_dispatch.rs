//! Domain-grouped dispatch for [`super::commands::handle_running_command`].
//!
//! Keeps the runtime entrypoint as a thin wrapper while grouping related [`RendererCommand`] arms.

use crate::shared::{
    FrameStartData, MaterialPropertyIdRequest, MaterialPropertyIdResult, MeshUploadData,
    RendererCommand,
};

use super::renderer_command_kind::renderer_command_variant_tag;
use super::RendererRuntime;

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

/// Routes a post-handshake [`RendererCommand`] to the appropriate runtime / backend handler.
pub(super) fn dispatch_running_command(runtime: &mut RendererRuntime, cmd: RendererCommand) {
    match cmd {
        RendererCommand::KeepAlive(_) => {}
        RendererCommand::RendererShutdown(_) | RendererCommand::RendererShutdownRequest(_) => {
            request_shutdown(runtime);
        }
        RendererCommand::FrameSubmitData(data) => runtime.on_frame_submit(data),
        RendererCommand::MeshUploadData(d) => process_mesh_upload(runtime, d),
        RendererCommand::MeshUnload(u) => runtime.backend.on_mesh_unload(u),
        RendererCommand::SetTexture2DFormat(f) => {
            runtime
                .backend
                .on_set_texture_2d_format(f, runtime.frontend.ipc_mut());
        }
        RendererCommand::SetTexture2DProperties(p) => {
            runtime
                .backend
                .on_set_texture_2d_properties(p, runtime.frontend.ipc_mut());
        }
        RendererCommand::SetTexture2DData(d) => {
            let (shm, ipc) = runtime.frontend.transport_pair_mut();
            runtime.backend.on_set_texture_2d_data(d, shm, ipc);
        }
        RendererCommand::UnloadTexture2D(u) => runtime.backend.on_unload_texture_2d(u),
        RendererCommand::SetTexture3DFormat(f) => {
            runtime
                .backend
                .on_set_texture_3d_format(f, runtime.frontend.ipc_mut());
        }
        RendererCommand::SetTexture3DProperties(p) => {
            runtime
                .backend
                .on_set_texture_3d_properties(p, runtime.frontend.ipc_mut());
        }
        RendererCommand::SetTexture3DData(d) => {
            let (shm, ipc) = runtime.frontend.transport_pair_mut();
            runtime.backend.on_set_texture_3d_data(d, shm, ipc);
        }
        RendererCommand::UnloadTexture3D(u) => runtime.backend.on_unload_texture_3d(u),
        RendererCommand::SetCubemapFormat(f) => {
            runtime
                .backend
                .on_set_cubemap_format(f, runtime.frontend.ipc_mut());
        }
        RendererCommand::SetCubemapProperties(p) => {
            runtime
                .backend
                .on_set_cubemap_properties(p, runtime.frontend.ipc_mut());
        }
        RendererCommand::SetCubemapData(d) => {
            let (shm, ipc) = runtime.frontend.transport_pair_mut();
            runtime.backend.on_set_cubemap_data(d, shm, ipc);
        }
        RendererCommand::UnloadCubemap(u) => runtime.backend.on_unload_cubemap(u),
        RendererCommand::SetRenderTextureFormat(f) => {
            runtime
                .backend
                .on_set_render_texture_format(f, runtime.frontend.ipc_mut());
        }
        RendererCommand::UnloadRenderTexture(u) => runtime.backend.on_unload_render_texture(u),
        RendererCommand::FreeSharedMemoryView(f) => {
            release_shared_memory_view(runtime, f.buffer_id);
        }
        RendererCommand::MaterialPropertyIdRequest(req) => {
            material_property_id_request(runtime, req);
        }
        RendererCommand::MaterialsUpdateBatch(batch) => {
            runtime.on_materials_update_batch(batch);
        }
        RendererCommand::UnloadMaterial(u) => runtime.backend.on_unload_material(u.asset_id),
        RendererCommand::UnloadMaterialPropertyBlock(u) => {
            runtime
                .backend
                .on_unload_material_property_block(u.asset_id);
        }
        RendererCommand::ShaderUpload(u) => runtime.on_shader_upload(u),
        RendererCommand::ShaderUnload(u) => runtime.on_shader_unload(u),
        RendererCommand::FrameStartData(ref fs) => log_frame_start_data_trace(fs),
        RendererCommand::LightsBufferRendererSubmission(sub) => {
            runtime.on_lights_buffer_renderer_submission(sub);
        }
        RendererCommand::LightsBufferRendererConsumed(_) => {
            logger::trace!("runtime: lights_buffer_renderer_consumed from host (ignored)");
        }
        RendererCommand::RenderTextureResult(_) => {
            logger::trace!(
                "runtime: render_texture_result from host (ignored; renderer is source)"
            );
        }
        ref cmd => {
            let tag = renderer_command_variant_tag(cmd);
            runtime.record_unhandled_renderer_command(tag);
            logger::warn!(
                "runtime: no handler for RendererCommand::{tag} (host sent unexpected command)"
            );
        }
    }
}

fn request_shutdown(runtime: &mut RendererRuntime) {
    runtime.frontend.set_shutdown_requested(true);
}

fn process_mesh_upload(runtime: &mut RendererRuntime, d: MeshUploadData) {
    let (shm, ipc) = runtime.frontend.transport_pair_mut();
    if let Some(shm) = shm {
        runtime.backend.try_process_mesh_upload(d, shm, ipc);
    } else {
        logger::warn!("mesh upload: no shared memory (standalone?)");
    }
}

fn release_shared_memory_view(runtime: &mut RendererRuntime, buffer_id: i32) {
    if let Some(shm) = runtime.frontend.shared_memory_mut() {
        shm.release_view(buffer_id);
    }
}

fn material_property_id_request(runtime: &mut RendererRuntime, req: MaterialPropertyIdRequest) {
    let property_ids: Vec<i32> = {
        let reg = runtime.backend.property_id_registry();
        req.property_names
            .iter()
            .map(|n| reg.intern_for_host_request(n.as_deref().unwrap_or("")))
            .collect()
    };
    if let Some(ref mut ipc) = runtime.frontend.ipc_mut() {
        let _ = ipc.send_background(RendererCommand::MaterialPropertyIdResult(
            MaterialPropertyIdResult {
                request_id: req.request_id,
                property_ids,
            },
        ));
    }
}
