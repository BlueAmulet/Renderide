//! Mesh draw stats and resident-pool counts fragment of [`super::FrameDiagnosticsSnapshot`].

use crate::diagnostics::BackendDiagSnapshot;
use crate::world_mesh::{WorldMeshDrawStateRow, WorldMeshDrawStats};

/// Mesh draw / batching / culling stats plus resident pool counts captured for the **Stats** and
/// **Draw state** tabs.
#[derive(Clone, Debug, Default)]
pub struct MeshDrawFragment {
    /// World mesh forward pass draw batching stats for the frame.
    pub stats: WorldMeshDrawStats,
    /// Sorted draw rows with resolved material pipeline state for the **Draw state** tab.
    pub draw_state_rows: Vec<WorldMeshDrawStateRow>,
    /// Host [`crate::shared::FrameSubmitData::render_tasks`] count from the last applied submit.
    pub last_submit_render_task_count: usize,
    /// Textures with a registered [`crate::shared::SetTexture2DFormat`] on the backend.
    pub textures_cpu_registered: usize,
    /// GPU-resident textures with at least mip 0 resident (`mip_levels_resident > 0`).
    pub textures_cpu_mip0_ready: usize,
    /// Resident GPU textures in [`crate::gpu_pools::TexturePool`].
    pub textures_gpu_resident: usize,
    /// GPU-resident host render textures ([`crate::gpu_pools::RenderTexturePool`]).
    pub render_textures_gpu_resident: usize,
    /// Rows in [`crate::gpu_pools::MeshPool`] (resident GPU mesh entries).
    pub mesh_pool_entry_count: usize,
}

impl MeshDrawFragment {
    /// Builds the fragment from the backend snapshot plus the host's last-applied submit count.
    pub fn capture(backend: &BackendDiagSnapshot, last_submit_render_task_count: usize) -> Self {
        Self {
            stats: backend.last_world_mesh_draw_stats,
            draw_state_rows: backend.last_world_mesh_draw_state_rows.clone(),
            last_submit_render_task_count,
            textures_cpu_registered: backend.texture_format_registration_count,
            textures_cpu_mip0_ready: backend.texture_mip0_ready_count,
            textures_gpu_resident: backend.texture_pool_resident_count,
            render_textures_gpu_resident: backend.render_texture_pool_len,
            mesh_pool_entry_count: backend.mesh_pool_entry_count,
        }
    }
}
