//! OpenXR entry loader, instance creation with extension negotiation, and the OpenXR-shaped
//! `vkGetInstanceProcAddr` shim shared with the Vulkan instance/device creation paths.

use std::ffi::c_void;
use std::sync::OnceLock;

use ash::vk::{self, Handle};
use openxr as xr;

use super::super::input::ProfileExtensionGates;
use super::types::XrBootstrapError;

/// Cached `vkGetInstanceProcAddr` function pointer captured from the active [`ash::Entry`].
/// Installed by [`super::vulkan::create_openxr_vulkan_instance`] before calling
/// [`xr::Instance::create_vulkan_instance`]; consumed by [`vk_get_instance_proc_addr_shim`].
pub(super) static CACHED_VK_GET_INSTANCE_PROC_ADDR: OnceLock<vk::PFN_vkGetInstanceProcAddr> =
    OnceLock::new();

/// OpenXR-shaped wrapper around the cached `vkGetInstanceProcAddr`.
///
/// OpenXR's `xrCreateVulkanInstanceKHR` expects a function pointer typed
/// `unsafe extern "system" fn(*const c_void, *const c_char) -> Option<...>`, while ash exposes the
/// same loader entry typed `unsafe extern "system" fn(vk::Instance, *const c_char) -> Option<...>`.
/// `vk::Instance` is a `#[repr(transparent)]` pointer-sized handle so the calling convention is
/// identical at the ABI level, but transmuting between fn-pointer types is not language-blessed.
/// This shim instead receives the OpenXR-typed call and forwards through the cached, properly
/// typed ash entry -- no fn-pointer transmute is required.
pub(super) unsafe extern "system" fn vk_get_instance_proc_addr_shim(
    instance: *const c_void,
    name: *const std::os::raw::c_char,
) -> Option<unsafe extern "system" fn()> {
    let real = CACHED_VK_GET_INSTANCE_PROC_ADDR.get().copied()?;
    let handle = vk::Instance::from_raw(instance as usize as u64);
    // SAFETY: `real` is the live `vkGetInstanceProcAddr` pointer captured from a successfully
    // loaded `ash::Entry`; OpenXR forwards the original `instance` handle and `name` pointer
    // verbatim, so the contract for `vkGetInstanceProcAddr` is upheld.
    unsafe { real(handle, name) }
}

/// Loads the OpenXR API entry: tries [`super::super::openxr_loader_paths::openxr_loader_candidate_paths`]
/// with [`xr::Entry::load_from`], then falls back to [`xr::Entry::load`] (default library search).
pub(super) fn load_xr_entry() -> Result<xr::Entry, xr::LoadError> {
    let paths = super::super::openxr_loader_paths::openxr_loader_candidate_paths();
    for path in paths {
        // SAFETY: `xr::Entry::load_from` dynamically loads the OpenXR loader from `path`; the
        // crate requires callers to guarantee the library at that path is a valid, ABI-compatible
        // OpenXR loader. The candidate paths come from platform-known install locations.
        match unsafe { xr::Entry::load_from(&path) } {
            Ok(entry) => {
                logger::debug!("OpenXR loader loaded from {}", path.display());
                return Ok(entry);
            }
            Err(e) => {
                logger::trace!("OpenXR loader not loaded from {}: {e}", path.display());
            }
        }
    }
    // SAFETY: `xr::Entry::load()` uses the default dynamic-linker search for the OpenXR loader;
    // relies on the platform's standard library search path to resolve a valid loader.
    match unsafe { xr::Entry::load() } {
        Ok(entry) => {
            logger::debug!("OpenXR loader loaded via default library search");
            Ok(entry)
        }
        Err(e) => Err(e),
    }
}

/// Result of [`create_openxr_instance`] for [`super::init::init_wgpu_openxr`].
pub(super) struct OpenxrInstanceBundle {
    pub(super) xr_instance: xr::Instance,
    pub(super) profile_gates: ProfileExtensionGates,
}

/// Loads extension flags, validates `XR_KHR_vulkan_enable2`, and creates the OpenXR [`xr::Instance`].
///
/// Every controller-related extension the runtime advertises is enabled so the corresponding
/// vendor interaction profile can be suggested. The resulting [`ProfileExtensionGates`] tells
/// [`super::super::input::OpenxrInput`] which profile binding tables to attempt.
pub(super) fn create_openxr_instance(
    xr_entry: xr::Entry,
) -> Result<OpenxrInstanceBundle, XrBootstrapError> {
    let available_extensions = xr_entry
        .enumerate_extensions()
        .map_err(|e| XrBootstrapError::Message(format!("enumerate_extensions: {e}")))?;
    if !available_extensions.khr_vulkan_enable2 {
        return Err(XrBootstrapError::Message(
            "OpenXR runtime does not expose XR_KHR_vulkan_enable2 (need Vulkan rendering).".into(),
        ));
    }

    let mut enabled_extensions = xr::ExtensionSet::default();
    enabled_extensions.khr_vulkan_enable2 = true;
    enabled_extensions.khr_generic_controller = available_extensions.khr_generic_controller;
    enabled_extensions.bd_controller_interaction = available_extensions.bd_controller_interaction;
    enabled_extensions.ext_hp_mixed_reality_controller =
        available_extensions.ext_hp_mixed_reality_controller;
    enabled_extensions.ext_samsung_odyssey_controller =
        available_extensions.ext_samsung_odyssey_controller;
    enabled_extensions.htc_vive_cosmos_controller_interaction =
        available_extensions.htc_vive_cosmos_controller_interaction;
    enabled_extensions.htc_vive_focus3_controller_interaction =
        available_extensions.htc_vive_focus3_controller_interaction;
    enabled_extensions.fb_touch_controller_pro = available_extensions.fb_touch_controller_pro;
    enabled_extensions.meta_touch_controller_plus = available_extensions.meta_touch_controller_plus;
    if available_extensions.ext_debug_utils {
        enabled_extensions.ext_debug_utils = true;
    }
    #[cfg(target_os = "android")]
    {
        enabled_extensions.khr_android_create_instance = true;
    }

    let xr_instance = xr_entry.create_instance(
        &xr::ApplicationInfo {
            application_name: "Renderide",
            application_version: 1,
            engine_name: "Renderide",
            engine_version: 1,
            api_version: xr::Version::new(1, 0, 0),
        },
        &enabled_extensions,
        &[],
    )?;

    let profile_gates = ProfileExtensionGates {
        khr_generic_controller: enabled_extensions.khr_generic_controller,
        bd_controller: enabled_extensions.bd_controller_interaction,
        ext_hp_mixed_reality_controller: enabled_extensions.ext_hp_mixed_reality_controller,
        ext_samsung_odyssey_controller: enabled_extensions.ext_samsung_odyssey_controller,
        htc_vive_cosmos_controller_interaction: enabled_extensions
            .htc_vive_cosmos_controller_interaction,
        htc_vive_focus3_controller_interaction: enabled_extensions
            .htc_vive_focus3_controller_interaction,
        fb_touch_controller_pro: enabled_extensions.fb_touch_controller_pro,
        meta_touch_controller_plus: enabled_extensions.meta_touch_controller_plus,
    };
    logger::info!(
        "OpenXR instance created: enabled_extensions=[{}]",
        enabled_openxr_extension_summary(&enabled_extensions)
    );

    Ok(OpenxrInstanceBundle {
        xr_instance,
        profile_gates,
    })
}

fn enabled_openxr_extension_summary(enabled: &xr::ExtensionSet) -> String {
    let mut names = Vec::new();
    if enabled.khr_vulkan_enable2 {
        names.push("KHR_vulkan_enable2");
    }
    if enabled.ext_debug_utils {
        names.push("EXT_debug_utils");
    }
    if enabled.khr_generic_controller {
        names.push("KHR_generic_controller");
    }
    if enabled.bd_controller_interaction {
        names.push("BD_controller_interaction");
    }
    if enabled.ext_hp_mixed_reality_controller {
        names.push("EXT_hp_mixed_reality_controller");
    }
    if enabled.ext_samsung_odyssey_controller {
        names.push("EXT_samsung_odyssey_controller");
    }
    if enabled.htc_vive_cosmos_controller_interaction {
        names.push("HTC_vive_cosmos_controller_interaction");
    }
    if enabled.htc_vive_focus3_controller_interaction {
        names.push("HTC_vive_focus3_controller_interaction");
    }
    if enabled.fb_touch_controller_pro {
        names.push("FB_touch_controller_pro");
    }
    if enabled.meta_touch_controller_plus {
        names.push("META_touch_controller_plus");
    }
    names.join(",")
}

/// OpenXR-reported Vulkan API version range expected by `xrCreateVulkanInstanceKHR` / `xrCreateVulkanDeviceKHR`.
pub(super) type VulkanGraphicsRequirements = <xr::Vulkan as xr::Graphics>::Requirements;

/// HMD system, blend mode, and Vulkan requirements from OpenXR.
pub(super) fn probe_head_set_and_vulkan_requirements(
    xr_instance: &xr::Instance,
) -> Result<
    (
        xr::SystemId,
        xr::EnvironmentBlendMode,
        VulkanGraphicsRequirements,
    ),
    XrBootstrapError,
> {
    let xr_system_id = xr_instance.system(xr::FormFactor::HEAD_MOUNTED_DISPLAY)?;
    let environment_blend_mode = xr_instance.enumerate_environment_blend_modes(
        xr_system_id,
        xr::ViewConfigurationType::PRIMARY_STEREO,
    )?[0];
    let reqs = xr_instance.graphics_requirements::<xr::Vulkan>(xr_system_id)?;
    Ok((xr_system_id, environment_blend_mode, reqs))
}
