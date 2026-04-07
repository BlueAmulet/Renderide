//! OpenXR session and Vulkan device bootstrap (Vulkan + `KHR_vulkan_enable2`).

mod bootstrap;
mod input;
mod session;
mod swapchain;

pub use bootstrap::{init_wgpu_openxr, XrWgpuHandles};
pub use input::OpenxrInput;
pub use session::{
    headset_pose_from_xr_view, openxr_pose_to_engine, view_projection_from_xr_view, XrSessionState,
};
pub use swapchain::{
    create_stereo_depth_texture, XrStereoSwapchain, XrSwapchainError, XR_COLOR_FORMAT,
    XR_VIEW_COUNT,
};
