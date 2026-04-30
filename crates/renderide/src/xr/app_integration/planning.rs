//! Pure prerequisite checks for deciding whether the HMD multiview path can submit.

use crate::gpu::GpuContext;
use crate::xr::XrFrameRenderer;

use super::types::{OpenxrFrameTick, XrSessionBundle};

/// Returns `true` when the session/runtime/GPU/tick state can submit an HMD projection layer.
pub(super) fn multiview_submit_prereqs(
    gpu: &GpuContext,
    bundle: &XrSessionBundle,
    runtime: &impl XrFrameRenderer,
    tick: &OpenxrFrameTick,
) -> bool {
    let handles = &bundle.handles;
    if !handles.xr_session.session_running() {
        return false;
    }
    if !handles.xr_session.is_visible() {
        return false;
    }
    if !runtime.vr_active() {
        return false;
    }
    if !gpu.device().features().contains(wgpu::Features::MULTIVIEW) {
        return false;
    }
    if !tick.should_render || tick.views.len() < 2 {
        return false;
    }
    true
}
