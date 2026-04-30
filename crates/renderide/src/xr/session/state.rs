//! OpenXR session frame loop: wait, begin, locate views, end.
//!
//! Tracks the latest [`xr::SessionState`] from the runtime so submission gates (compositor
//! visibility, exit propagation) can react to lifecycle transitions, and maintains a `frame_open`
//! flag so every successful `xrBeginFrame` is matched by exactly one `xrEndFrame`. Entry points
//! that call the compositor (`xrEndFrame`) are wrapped with an
//! [`super::end_frame_watchdog::EndFrameWatchdog`] so runtime stalls surface as `logger::error!`
//! lines instead of silent freezes.

mod frame_loop;
mod lifecycle;

use openxr as xr;

pub use lifecycle::TrackedSessionState;
use lifecycle::is_visible_tracked;

/// Owns OpenXR session objects (constructed in [`super::super::bootstrap::init_wgpu_openxr`]).
pub struct XrSessionState {
    /// OpenXR instance (retained for the session lifetime).
    pub(super) xr_instance: xr::Instance,
    /// Dropped before [`Self::xr_instance`] so the messenger handle is destroyed first; held only
    /// for this Drop ordering, hence never read after construction.
    #[expect(dead_code, reason = "drop-order-only field; see doc comment above")]
    pub(super) openxr_debug_messenger: Option<super::super::debug_utils::OpenxrDebugUtilsMessenger>,
    /// Blend mode used for `xrEndFrame`.
    pub(super) environment_blend_mode: xr::EnvironmentBlendMode,
    /// Vulkan-backed session.
    pub(super) session: xr::Session<xr::Vulkan>,
    /// Whether `xrBeginSession` has been called and `xrEndSession` has not.
    pub(super) session_running: bool,
    /// Latest [`xr::SessionState`] observed via `SessionStateChanged`.
    pub(super) last_session_state: TrackedSessionState,
    /// `true` between a successful `frame_stream.begin()` and the matching `frame_stream.end()`;
    /// prevents orphaned frames on error paths.
    pub(super) frame_open: bool,
    /// Set when the runtime requests teardown (`EXITING` / `LOSS_PENDING` / instance loss);
    /// read by the app loop to trigger `event_loop.exit()`.
    pub(super) exit_requested: bool,
    /// Blocks until the compositor signals frame timing.
    pub(super) frame_wait: xr::FrameWaiter,
    /// Submits composition layers to the compositor.
    pub(super) frame_stream: xr::FrameStream<xr::Vulkan>,
    /// Stage reference space for view and controller pose location.
    pub(super) stage: xr::Space,
    /// Scratch buffer for `xrPollEvent`.
    pub(super) event_storage: xr::EventDataBuffer,
}

/// Bundle of values needed to construct [`XrSessionState`] - `new` takes this instead of seven
/// separate parameters to keep the bootstrap signature readable.
pub(in crate::xr) struct XrSessionStateDescriptor {
    /// OpenXR instance (retained for the session lifetime).
    pub(in crate::xr) xr_instance: xr::Instance,
    /// Debug-utils messenger; must drop before the instance. See [`XrSessionState`].
    pub(in crate::xr) openxr_debug_messenger:
        Option<super::super::debug_utils::OpenxrDebugUtilsMessenger>,
    /// Blend mode used for `xrEndFrame`.
    pub(in crate::xr) environment_blend_mode: xr::EnvironmentBlendMode,
    /// Vulkan-backed session.
    pub(in crate::xr) session: xr::Session<xr::Vulkan>,
    /// Frame waiter from the session tuple.
    pub(in crate::xr) frame_wait: xr::FrameWaiter,
    /// Frame stream from the session tuple.
    pub(in crate::xr) frame_stream: xr::FrameStream<xr::Vulkan>,
    /// Stage reference space used for view + controller pose location.
    pub(in crate::xr) stage: xr::Space,
}

impl XrSessionState {
    /// Constructed only from [`crate::xr::bootstrap::init_wgpu_openxr`].
    pub(in crate::xr) fn new(desc: XrSessionStateDescriptor) -> Self {
        Self {
            xr_instance: desc.xr_instance,
            openxr_debug_messenger: desc.openxr_debug_messenger,
            environment_blend_mode: desc.environment_blend_mode,
            session: desc.session,
            session_running: false,
            last_session_state: TrackedSessionState::Unknown,
            frame_open: false,
            exit_requested: false,
            frame_wait: desc.frame_wait,
            frame_stream: desc.frame_stream,
            stage: desc.stage,
            event_storage: xr::EventDataBuffer::new(),
        }
    }

    /// Whether the OpenXR session is running (`xrBeginSession` called, `xrEndSession` not yet).
    pub fn session_running(&self) -> bool {
        self.session_running
    }

    /// Latest [`TrackedSessionState`] observed from the runtime; [`TrackedSessionState::Unknown`]
    /// until the first `SessionStateChanged` event.
    pub fn last_session_state(&self) -> TrackedSessionState {
        self.last_session_state
    }

    /// Whether the compositor is currently displaying this app's frames
    /// ([`TrackedSessionState::Visible`] or [`TrackedSessionState::Focused`]). Used to gate real
    /// projection-layer submission; the empty-frame path still runs to satisfy the OpenXR
    /// begin/end frame contract.
    pub fn is_visible(&self) -> bool {
        is_visible_tracked(self.last_session_state)
    }

    /// Whether the runtime has asked the renderer to exit (EXITING / LOSS_PENDING / instance
    /// loss). Checked by the app loop after each `poll_events`.
    pub fn exit_requested(&self) -> bool {
        self.exit_requested
    }

    /// Whether a frame scope is currently open (`xrBeginFrame` called without matching
    /// `xrEndFrame`).
    pub fn frame_open(&self) -> bool {
        self.frame_open
    }

    /// OpenXR instance handle (swapchain creation, view enumeration).
    pub fn xr_instance(&self) -> &xr::Instance {
        &self.xr_instance
    }

    /// Underlying Vulkan session (swapchain lifetime).
    pub fn xr_vulkan_session(&self) -> &xr::Session<xr::Vulkan> {
        &self.session
    }

    /// Stage reference space used for [`Self::locate_views`] and controller [`xr::Space`] location.
    pub fn stage_space(&self) -> &xr::Space {
        &self.stage
    }
}
