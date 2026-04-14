//! Shader routing and material batch IPC handlers.

use crate::assets::resolve_shader_upload;
use crate::backend::RenderBackend;
use crate::frontend::RendererFrontend;
use crate::shared::{
    MaterialsUpdateBatch, RendererCommand, ShaderUnload, ShaderUpload, ShaderUploadResult,
};

/// Registers a host shader upload and notifies the host.
pub(crate) fn on_shader_upload(
    frontend: &mut RendererFrontend,
    backend: &mut RenderBackend,
    upload: ShaderUpload,
) {
    let asset_id = upload.asset_id;
    let resolved = resolve_shader_upload(&upload);
    logger::info!(
        "shader_upload: asset_id={} unity_shader_name={:?} raster_pipeline={:?}",
        asset_id,
        resolved.unity_shader_name.as_deref(),
        resolved.pipeline,
    );
    let display_name = resolved
        .unity_shader_name
        .clone()
        .or_else(|| upload.file.clone().filter(|s| !s.is_empty()));
    backend.register_shader_route(asset_id, resolved.pipeline, display_name);
    if let Some(ref mut ipc) = frontend.ipc_mut() {
        ipc.send_background(RendererCommand::ShaderUploadResult(ShaderUploadResult {
            asset_id,
            instance_changed: true,
        }));
    }
}

pub(crate) fn on_shader_unload(backend: &mut RenderBackend, unload: ShaderUnload) {
    let id = unload.asset_id;
    backend.unregister_shader_route(id);
}

pub(crate) fn on_materials_update_batch(
    frontend: &mut RendererFrontend,
    backend: &mut RenderBackend,
    batch: MaterialsUpdateBatch,
) {
    if frontend.shared_memory().is_none() {
        if !backend.enqueue_materials_batch_no_shm(batch) {
            // already logged
        }
        return;
    }
    let (shm, ipc) = frontend.transport_pair_mut();
    let (Some(shm), Some(ipc)) = (shm, ipc) else {
        return;
    };
    backend.apply_materials_update_batch(batch, shm, ipc);
}
