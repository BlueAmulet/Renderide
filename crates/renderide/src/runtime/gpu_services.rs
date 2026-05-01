//! GPU-facing per-tick services on [`super::RendererRuntime`].
//!
//! These helpers run once per tick (from the render entry point, the tick prologue, or the
//! app driver) and forward into a backend pool/cache concern that needs the GPU device or a
//! [`GpuContext`]. Keeping them on one file groups the runtime's cross-cutting GPU duties.

use crate::gpu::GpuContext;

use super::RendererRuntime;

impl RendererRuntime {
    /// Drops transient-pool GPU textures for free-list entries whose MSAA sample count no longer
    /// matches the effective swapchain tier (avoids VRAM retention when toggling MSAA).
    pub(super) fn transient_evict_stale_msaa_tiers_if_changed(
        &mut self,
        prev_effective: u32,
        new_effective: u32,
    ) {
        if prev_effective == new_effective {
            return;
        }
        let eff = new_effective.max(1);
        self.backend
            .transient_pool_mut()
            .evict_texture_keys_where(|k| k.sample_count > 1 && k.sample_count != eff);
    }

    /// Drains completed Hi-Z `map_async` readbacks into CPU snapshots (once per tick).
    ///
    /// Call at the top of the render-views phase so both the HMD and desktop paths share one drain.
    pub fn drain_hi_z_readback(&mut self, device: &wgpu::Device) {
        profiling::scope!("tick::drain_hi_z_readback");
        self.backend.hi_z_begin_frame_readback(device);
    }

    /// Advances nonblocking GPU services that feed host-visible async results.
    pub fn maintain_nonblocking_gpu_jobs(&mut self, gpu: &mut GpuContext) {
        profiling::scope!("tick::maintain_nonblocking_gpu_jobs");
        self.backend.maintain_skybox_ibl_jobs(gpu, &self.scene);
        self.backend.maintain_reflection_probe_sh2_jobs(gpu);
    }
}
