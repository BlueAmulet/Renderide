//! Swapchain image acquire/release under the shared GPU queue access gate.

use crate::gpu::GpuContext;
use openxr as xr;

/// Acquires one OpenXR swapchain image while holding the shared Vulkan queue access gate.
pub(super) fn acquire_swapchain_image(
    gpu: &GpuContext,
    swapchain: &mut xr::Swapchain<xr::Vulkan>,
) -> Result<usize, xr::sys::Result> {
    let _gate = gpu.gpu_queue_access_gate().lock();
    swapchain.acquire_image().map(|i| i as usize)
}

/// Releases one OpenXR swapchain image while holding the shared Vulkan queue access gate.
pub(super) fn release_swapchain_image(
    gpu: &GpuContext,
    swapchain: &mut xr::Swapchain<xr::Vulkan>,
) -> Result<(), xr::sys::Result> {
    let _gate = gpu.gpu_queue_access_gate().lock();
    swapchain.release_image()
}
