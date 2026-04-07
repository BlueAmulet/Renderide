//! OpenXR session and Vulkan device bootstrap (Vulkan + `KHR_vulkan_enable2`).

#[cfg(feature = "openxr")]
mod bootstrap;
#[cfg(feature = "openxr")]
mod session;
#[cfg(feature = "openxr")]
mod swapchain;

#[cfg(feature = "openxr")]
pub use bootstrap::{init_wgpu_openxr, XrWgpuHandles};
#[cfg(feature = "openxr")]
pub use session::{view_projection_from_xr_view, XrSessionState};
#[cfg(feature = "openxr")]
pub use swapchain::{
    create_stereo_depth_texture, XrStereoSwapchain, XrSwapchainError, XR_COLOR_FORMAT,
    XR_VIEW_COUNT,
};
