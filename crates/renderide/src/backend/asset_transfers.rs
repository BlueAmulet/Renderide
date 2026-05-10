//! Mesh and texture upload queues, cooperative integration, CPU-side format/property tables, and resident pools.
//!
//! [`AssetTransferQueue`] is owned by [`crate::backend::RenderBackend`]. It handles shared-memory
//! ingestion paths that populate
//! [`crate::gpu_pools::MeshPool`], [`crate::gpu_pools::TexturePool`], [`crate::gpu_pools::Texture3dPool`],
//! and [`crate::gpu_pools::CubemapPool`].

mod catalogs;
mod cubemap_task;
mod cubemap_upload_plan;
mod gpu_runtime;
mod integrator;
mod mesh_task;
mod pending;
mod pools;
mod shared_memory_payload;
mod texture3d_task;
mod texture3d_upload_plan;
mod texture_task;
mod texture_task_common;
mod texture_upload_plan;
mod uploads;
mod video_runtime;

use std::sync::Arc;

use crate::gpu::GpuLimits;
use crate::gpu_pools::{
    CubemapPool, GpuVideoTexture, MeshPool, RenderTexturePool, Texture3dPool, TexturePool,
    VideoTexturePool,
};
use crate::render_graph::GraphAssetResources;
use crate::shared::{
    MeshUnload, PointRenderBufferUnload, TrailRenderBufferUnload, UnloadCubemap,
    UnloadDesktopTexture, UnloadGaussianSplat, UnloadRenderTexture, UnloadTexture2D,
    UnloadTexture3D, UnloadVideoTexture, VideoTextureClockErrorState,
};

use super::resource_scope::RenderSpaceAssetSet;
use catalogs::AssetCatalogs;
use gpu_runtime::AssetGpuRuntime;
pub use integrator::{
    AssetIntegrationDrainSummary, AssetIntegrator, AssetTask, AssetTaskLane, ShaderRouteTask,
    drain_asset_tasks, drain_asset_tasks_unbounded,
};
use pending::PendingAssetUploads;
use pools::ResidentAssetPools;
pub use uploads::{
    attach_flush_pending_asset_uploads, on_desktop_texture_properties_update,
    on_gaussian_splat_config, on_gaussian_splat_upload_encoded, on_gaussian_splat_upload_raw,
    on_mesh_unload, on_point_render_buffer_unload, on_point_render_buffer_upload,
    on_set_cubemap_data, on_set_cubemap_format, on_set_cubemap_properties,
    on_set_desktop_texture_properties, on_set_render_texture_format, on_set_texture_2d_data,
    on_set_texture_2d_format, on_set_texture_2d_properties, on_set_texture_3d_data,
    on_set_texture_3d_format, on_set_texture_3d_properties, on_trail_render_buffer_unload,
    on_trail_render_buffer_upload, on_unload_cubemap, on_unload_desktop_texture,
    on_unload_gaussian_splat, on_unload_render_texture, on_unload_texture_2d, on_unload_texture_3d,
    on_unload_video_texture, on_video_texture_load, on_video_texture_properties,
    on_video_texture_start_audio_track, on_video_texture_update, try_process_mesh_upload,
};
use video_runtime::VideoAssetRuntime;

/// Pending mesh/texture payloads, CPU texture tables, GPU device/queue, resident pools, and [`AssetIntegrator`].
pub struct AssetTransferQueue {
    /// GPU-resident pools.
    pub(crate) pools: ResidentAssetPools,
    /// Host descriptor/property catalogs.
    pub(crate) catalogs: AssetCatalogs,
    /// Upload commands deferred until formats, GPU resources, or shared memory are available.
    pub(crate) pending: PendingAssetUploads,
    /// GPU handles and upload settings captured during backend attach.
    pub(crate) gpu: AssetGpuRuntime,
    /// Active video players and per-frame video telemetry.
    pub(crate) video: VideoAssetRuntime,
    /// Cooperative uploads drained by [`drain_asset_tasks`] / [`drain_asset_tasks_unbounded`].
    pub(crate) integrator: AssetIntegrator,
}

/// Requested purge counts for zero-owner render-space assets.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct AssetPurgeSummary {
    /// Mesh purge requests.
    pub(crate) meshes: usize,
    /// Texture2D purge requests.
    pub(crate) texture_2d: usize,
    /// Texture3D purge requests.
    pub(crate) texture_3d: usize,
    /// Cubemap purge requests.
    pub(crate) cubemaps: usize,
    /// Render texture purge requests.
    pub(crate) render_textures: usize,
    /// Video texture purge requests.
    pub(crate) video_textures: usize,
    /// Desktop texture purge requests.
    pub(crate) desktop_textures: usize,
    /// Point render buffer purge requests.
    pub(crate) point_render_buffers: usize,
    /// Trail render buffer purge requests.
    pub(crate) trail_render_buffers: usize,
    /// Gaussian splat purge requests.
    pub(crate) gaussian_splats: usize,
}

impl AssetPurgeSummary {
    /// Total purge requests across all asset families.
    pub(crate) fn total(self) -> usize {
        self.meshes
            + self.texture_2d
            + self.texture_3d
            + self.cubemaps
            + self.render_textures
            + self.video_textures
            + self.desktop_textures
            + self.point_render_buffers
            + self.trail_render_buffers
            + self.gaussian_splats
    }
}

impl AssetTransferQueue {
    /// Mutably borrows the cooperative asset integrator.
    pub(crate) fn integrator_mut(&mut self) -> &mut AssetIntegrator {
        &mut self.integrator
    }

    /// Whether any upload work is queued or deferred on missing prerequisites.
    pub(crate) fn has_pending_asset_work(&self) -> bool {
        self.integrator.total_queued() > 0
            || !self.pending.pending_mesh_uploads.is_empty()
            || !self.pending.pending_texture_uploads.is_empty()
            || !self.pending.pending_texture3d_uploads.is_empty()
            || !self.pending.pending_cubemap_uploads.is_empty()
    }

    /// Locally unloads all zero-owner assets released by closed render spaces.
    pub(crate) fn purge_render_space_assets(
        &mut self,
        assets: &RenderSpaceAssetSet,
    ) -> AssetPurgeSummary {
        profiling::scope!("assets::purge_render_space_assets");
        let mut summary = AssetPurgeSummary::default();

        for &asset_id in &assets.meshes {
            on_mesh_unload(self, MeshUnload { asset_id });
            summary.meshes += 1;
        }
        for &asset_id in &assets.texture_2d {
            on_unload_texture_2d(self, UnloadTexture2D { asset_id });
            summary.texture_2d += 1;
        }
        for &asset_id in &assets.texture_3d {
            on_unload_texture_3d(self, UnloadTexture3D { asset_id });
            summary.texture_3d += 1;
        }
        for &asset_id in &assets.cubemaps {
            on_unload_cubemap(self, UnloadCubemap { asset_id });
            summary.cubemaps += 1;
        }
        for &asset_id in &assets.render_textures {
            on_unload_render_texture(self, UnloadRenderTexture { asset_id });
            summary.render_textures += 1;
        }
        for &asset_id in &assets.video_textures {
            on_unload_video_texture(self, UnloadVideoTexture { asset_id });
            summary.video_textures += 1;
        }
        for &asset_id in &assets.desktop_textures {
            on_unload_desktop_texture(self, UnloadDesktopTexture { asset_id });
            summary.desktop_textures += 1;
        }
        for &asset_id in &assets.point_render_buffers {
            on_point_render_buffer_unload(self, PointRenderBufferUnload { asset_id });
            summary.point_render_buffers += 1;
        }
        for &asset_id in &assets.trail_render_buffers {
            on_trail_render_buffer_unload(self, TrailRenderBufferUnload { asset_id });
            summary.trail_render_buffers += 1;
        }
        for &asset_id in &assets.gaussian_splats {
            on_unload_gaussian_splat(self, UnloadGaussianSplat { asset_id });
            summary.gaussian_splats += 1;
        }

        summary
    }

    /// Stores GPU handles and limits after backend attach.
    pub(crate) fn attach_gpu_runtime(
        &mut self,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        gate: crate::gpu::GpuQueueAccessGate,
        limits: Arc<GpuLimits>,
    ) {
        self.gpu.attach(device, queue, gate, limits);
    }

    /// Resident mesh pool.
    pub(crate) fn mesh_pool(&self) -> &MeshPool {
        &self.pools.mesh_pool
    }

    /// Mutable resident mesh pool.
    pub(crate) fn mesh_pool_mut(&mut self) -> &mut MeshPool {
        &mut self.pools.mesh_pool
    }

    /// Resident Texture2D pool.
    pub(crate) fn texture_pool(&self) -> &TexturePool {
        &self.pools.texture_pool
    }

    /// Resident Texture3D pool.
    pub(crate) fn texture3d_pool(&self) -> &Texture3dPool {
        &self.pools.texture3d_pool
    }

    /// Resident cubemap pool.
    pub(crate) fn cubemap_pool(&self) -> &CubemapPool {
        &self.pools.cubemap_pool
    }

    /// Resident render-texture pool.
    pub(crate) fn render_texture_pool(&self) -> &RenderTexturePool {
        &self.pools.render_texture_pool
    }

    /// Resident video-texture pool.
    pub(crate) fn video_texture_pool(&self) -> &VideoTexturePool {
        &self.pools.video_texture_pool
    }

    /// GPU limits snapshot after attach.
    pub(crate) fn gpu_limits(&self) -> Option<&Arc<GpuLimits>> {
        self.gpu.gpu_limits.as_ref()
    }

    /// Number of host Texture2D format rows known to the asset catalog.
    pub(crate) fn texture_format_registration_count(&self) -> usize {
        self.catalogs.texture_formats.len()
    }

    /// Drains the latest video clock-error samples for transmission to the host.
    ///
    /// The runtime calls this once per tick before [`crate::frontend::RendererFrontend::pre_frame`]
    /// so the next [`crate::shared::FrameStartData`] carries the latest drift snapshot per video
    /// asset.
    pub fn take_pending_video_clock_errors(&mut self) -> Vec<VideoTextureClockErrorState> {
        self.video.take_pending_clock_errors()
    }

    /// Starts cooperative shutdown for active video texture players.
    pub(crate) fn begin_video_shutdown(&mut self) {
        self.video.begin_shutdown();
    }

    /// Returns `true` once all video texture players have finished shutdown.
    pub(crate) fn video_shutdown_complete(&mut self) -> bool {
        self.video.shutdown_complete()
    }

    /// Ensures a GPU video texture placeholder exists and returns it for mutation.
    pub(crate) fn ensure_video_texture_with_props(
        &mut self,
        props: &crate::shared::VideoTextureProperties,
    ) -> Option<&mut GpuVideoTexture> {
        let asset_id = props.asset_id;
        if self.pools.video_texture_pool.get(asset_id).is_none() {
            let texture = {
                let device = self.gpu.gpu_device.as_deref()?;
                GpuVideoTexture::new(device, asset_id, props)
            };
            if self.pools.video_texture_pool.insert(texture) {
                logger::debug!("video texture {asset_id}: replaced placeholder during creation");
            }
        }
        self.pools.video_texture_pool.get_mut(asset_id)
    }
}

impl GraphAssetResources for AssetTransferQueue {
    fn mesh_pool(&self) -> &MeshPool {
        self.mesh_pool()
    }

    fn texture_pool(&self) -> &TexturePool {
        self.texture_pool()
    }

    fn texture3d_pool(&self) -> &Texture3dPool {
        self.texture3d_pool()
    }

    fn cubemap_pool(&self) -> &CubemapPool {
        self.cubemap_pool()
    }

    fn render_texture_pool(&self) -> &RenderTexturePool {
        self.render_texture_pool()
    }

    fn video_texture_pool(&self) -> &VideoTexturePool {
        self.video_texture_pool()
    }
}

impl Default for AssetTransferQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl AssetTransferQueue {
    /// Empty pools and tables; no GPU until the backend calls attach.
    pub fn new() -> Self {
        Self {
            pools: ResidentAssetPools::default(),
            catalogs: AssetCatalogs::default(),
            pending: PendingAssetUploads::default(),
            gpu: AssetGpuRuntime::default(),
            video: VideoAssetRuntime::default(),
            integrator: AssetIntegrator::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::{TextureFilterMode, TextureWrapMode, VideoTextureProperties};

    #[test]
    fn video_texture_properties_default_preserves_asset_id() {
        let queue = AssetTransferQueue::new();

        let props = queue.catalogs.video_texture_properties_or_default(42);

        assert_eq!(props.asset_id, 42);
        assert_eq!(props.filter_mode, TextureFilterMode::Point);
        assert_eq!(props.wrap_u, TextureWrapMode::Repeat);
        assert_eq!(props.wrap_v, TextureWrapMode::Repeat);
    }

    #[test]
    fn video_texture_properties_default_uses_cached_properties() {
        let mut queue = AssetTransferQueue::new();
        queue.catalogs.video_texture_properties.insert(
            7,
            VideoTextureProperties {
                asset_id: 7,
                filter_mode: TextureFilterMode::Trilinear,
                aniso_level: 8,
                wrap_u: TextureWrapMode::Mirror,
                wrap_v: TextureWrapMode::Clamp,
            },
        );

        let props = queue.catalogs.video_texture_properties_or_default(7);

        assert_eq!(props.asset_id, 7);
        assert_eq!(props.filter_mode, TextureFilterMode::Trilinear);
        assert_eq!(props.aniso_level, 8);
        assert_eq!(props.wrap_u, TextureWrapMode::Mirror);
        assert_eq!(props.wrap_v, TextureWrapMode::Clamp);
    }
}
