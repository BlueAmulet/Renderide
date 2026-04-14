//! Maps [`HeadOutputDevice`](crate::shared::HeadOutputDevice) to VR / OpenXR presentation intent.
//!
//! **OpenXR GPU path:** [`head_output_device_wants_openxr`] is `true` for VR-capable devices; the app
//! then runs [`crate::xr::init_wgpu_openxr`]. If init fails, the renderer falls back to desktop GPU.
//!
//! **VR IPC input:** [`crate::frontend::input::vr_inputs_for_session`] supplies headset pose,
//! OpenXR-sampled controllers when available, and a non-empty [`InputState::vr`](crate::shared::InputState)
//! when the session device is VR-capable, even when the GPU path fell back to desktop.

use crate::shared::HeadOutputDevice;

/// Returns `true` for SteamVR, Windows MR, Oculus, and Oculus Quest — aligned with FrooxEngine’s
/// `HeadOutputDeviceExtension.IsVR` (`(uint)(device - 6) <= 3`).
pub fn head_output_device_is_vr(device: HeadOutputDevice) -> bool {
    matches!(
        device,
        HeadOutputDevice::SteamVR
            | HeadOutputDevice::WindowsMR
            | HeadOutputDevice::Oculus
            | HeadOutputDevice::OculusQuest
    )
}

/// Whether to bootstrap the OpenXR Vulkan path (matches [`head_output_device_is_vr`] for now).
pub fn head_output_device_wants_openxr(device: HeadOutputDevice) -> bool {
    head_output_device_is_vr(device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_vr_matches_cs_table() {
        let cases = [
            (HeadOutputDevice::Autodetect, false),
            (HeadOutputDevice::Headless, false),
            (HeadOutputDevice::Screen, false),
            (HeadOutputDevice::Screen360, false),
            (HeadOutputDevice::StaticCamera, false),
            (HeadOutputDevice::StaticCamera360, false),
            (HeadOutputDevice::SteamVR, true),
            (HeadOutputDevice::WindowsMR, true),
            (HeadOutputDevice::Oculus, true),
            (HeadOutputDevice::OculusQuest, true),
            (HeadOutputDevice::UNKNOWN, false),
        ];
        for (d, want) in cases {
            assert_eq!(head_output_device_is_vr(d), want, "wrong IsVR for {d:?}");
        }
    }

    #[test]
    fn wants_openxr_matches_is_vr() {
        let vr = HeadOutputDevice::SteamVR;
        let non = HeadOutputDevice::Screen;
        assert_eq!(
            head_output_device_wants_openxr(vr),
            head_output_device_is_vr(vr)
        );
        assert_eq!(
            head_output_device_wants_openxr(non),
            head_output_device_is_vr(non)
        );
        assert!(!head_output_device_wants_openxr(non));
    }
}
