//! GPU-side source payloads queued for SH2 projection.

use std::sync::Arc;

use super::source_keys::Sh2ProjectParams;

/// GPU-projected source payload queued for scheduling.
#[derive(Clone, Debug)]
pub(in crate::reflection_probes) enum GpuSh2Source {
    /// Cubemap sampled from the cubemap pool.
    Cubemap {
        /// Cubemap asset id.
        asset_id: i32,
        /// Source cubemap storage orientation.
        storage_v_inverted: bool,
    },
    /// Equirectangular 2D texture sampled from the texture pool.
    EquirectTexture2D {
        /// Texture asset id.
        asset_id: i32,
        /// Projection360 sampling parameters.
        params: Box<Sh2ProjectParams>,
    },
    /// Parameter-only sky material evaluator.
    SkyParams {
        /// Sky material parameters.
        params: Box<Sh2ProjectParams>,
    },
    /// Renderer-captured OnChanges cubemap.
    RuntimeCubemap {
        /// Captured texture kept alive with the source view.
        texture: Arc<wgpu::Texture>,
        /// Cube view sampled by the SH2 projection shader.
        view: Arc<wgpu::TextureView>,
    },
}
