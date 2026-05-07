//! Renderer facade: orchestrates **frontend** (IPC / shared memory / lock-step), **scene** (host
//! logical state), and **backend** (GPU pools, material store, uploads).
//!
//! [`RendererRuntime`] *composes* a [`RendererFrontend`], a [`SceneCoordinator`], and a
//! [`RenderBackend`]; it does **not** own IPC queue state, scene tables, or GPU resources directly.
//! Each layer keeps its state private; runtime code calls through the layer's API in a fixed
//! per-tick order. Adding new logic here usually means a new method on the right layer plus a
//! short call from the orchestration site, not a new field on [`RendererRuntime`].
//!
//! # Per-tick phase order
//!
//! The authoritative call site is the app driver's redraw tick; this
//! module's methods correspond to the named phases:
//!
//! 1. **Wall-clock prologue** -- [`RendererRuntime::tick_frame_wall_clock_begin`]; resets per-tick flags.
//! 2. **IPC poll** -- [`RendererRuntime::poll_ipc`]; drains incoming `RendererCommand`s before any work runs.
//! 3. **Asset integration** -- [`RendererRuntime::run_asset_integration`]; time-sliced cooperative
//!    mesh/texture/material uploads via [`crate::backend::RenderBackend::drain_asset_tasks`].
//! 4. **Optional XR begin** -- `xr_begin_tick` in `app/`; OpenXR `wait_frame` / `locate_views` so the
//!    same view snapshot is visible to lock-step input.
//! 5. **Lock-step exchange** -- [`RendererRuntime::pre_frame`] emits
//!    [`FrameStartData`](crate::shared::FrameStartData) when allowed; the gating predicate
//!    [`RendererFrontend::should_send_begin_frame`] keeps the lock-step *state* in
//!    [`RendererFrontend`] (this module owns no lock-step counters).
//! 6. **Render** -- desktop multi-view or HMD path through [`crate::render_graph`].
//! 7. **Present + HUD** -- present surface, blit VR mirror, capture ImGui debug snapshots.
//!
//! Lock-step is driven by the `last_frame_index` field of [`FrameStartData`](crate::shared::FrameStartData)
//! on the **outgoing** `frame_start_data` the renderer sends from [`RendererRuntime::pre_frame`].
//! If the host sends [`RendererCommand::FrameStartData`](crate::shared::RendererCommand::FrameStartData),
//! optional payloads are trace-logged until consumers exist.
//!
//! `runtime/lockstep.rs` is a pure debug helper (duplicate-frame-index trace logging only); the
//! decision predicate and the counters live in [`crate::frontend`].
//!
//! # Submodule layout
//!
//! Per-tick logic is split by concern; every submodule extends [`RendererRuntime`] through its
//! own `impl` block:
//!
//! - [`accessors`] -- thin facade pass-throughs to the frontend, backend, scene, and settings.
//! - [`asset_integration`] -- cooperative asset-integration phase + once-per-tick gating.
//! - [`debug_hud_frame`] -- per-tick wiring for the diagnostics ImGui overlay.
//! - [`frame_extract`] -- immutable per-tick view extraction, draw collection, submit packet.
//! - [`frame_render`] -- render-mode dispatch, MSAA prep, frame-extract entry.
//! - [`frame_submit`] -- runtime-side application of host frame-submit payloads.
//! - [`frame_view_plan`] -- per-view CPU intent (target, clear, viewport, host camera).
//! - [`gpu_services`] -- GPU-facing helpers run once per tick (Hi-Z drain, async jobs, transient eviction).
//! - [`ipc_effects`] -- applies decoded frontend dispatch effects to runtime-owned domains.
//! - [`ipc_entry`] -- IPC poll and command-effect decode/apply entrypoints.
//! - [`lights_ipc`] -- applies host light-buffer IPC payloads to scene light caches.
//! - [`lockstep`] -- diagnostic helper for duplicate frame indices.
//! - [`shader_material_ipc`] -- applies shader route and material-batch IPC payloads.
//! - [`tick`] -- tick prologue, lock-step / output forwards, the two `tick_one_frame*` orchestrators.
//! - [`view_planning`] -- collection of HMD / secondary RT / main swapchain plans.
//! - [`xr_glue`] -- `XrHostCameraSync` and `XrFrameRenderer` impls for [`RendererRuntime`].
//!
//! IPC dispatch in `crate::frontend::dispatch` is decode-only. [`ipc_entry`] polls queue commands,
//! `frontend::dispatch` classifies them into domain effects, and [`ipc_effects`] is the single
//! runtime-owned application point for frontend, scene, backend, host camera, settings, and IPC
//! scratch mutations.

mod accessors;
mod asset_integration;
mod config_state;
mod debug_hud_frame;
mod diagnostics_state;
mod frame_extract;
pub(crate) mod frame_render;
mod frame_submit;
mod frame_view_plan;
mod gpu_services;
mod ipc_effects;
mod ipc_entry;
mod ipc_state;
mod lights_ipc;
mod lockstep;
mod shader_material_ipc;
mod shutdown;
mod tick;
mod tick_state;
mod view_planning;
mod xr_glue;
mod xr_stats;

use std::path::PathBuf;

use crate::backend::RenderBackend;
use crate::camera::HostCameraFrame;
use crate::config::RendererSettingsHandle;
use crate::connection::ConnectionParams;
use crate::frontend::RendererFrontend;
use crate::render_graph::GraphExecuteError;
use crate::scene::SceneCoordinator;

use config_state::RuntimeConfigState;
use diagnostics_state::RuntimeDiagnosticsState;
use ipc_state::RuntimeIpcState;
use tick_state::RuntimeTickState;
use xr_stats::RuntimeXrStats;

/// Result of one [`RendererRuntime::tick_one_frame`] call.
///
/// `shutdown_requested` lets the calling driver exit its event loop; `fatal_error` triggers a
/// non-zero process exit. `graph_error` carries any failure from [`RendererRuntime::render_frame`]
/// for the caller to decide whether to log + continue or escalate.
#[derive(Debug, Default)]
pub struct TickOutcome {
    /// Host requested an orderly shutdown via IPC during this tick.
    pub shutdown_requested: bool,
    /// IPC reported a fatal error during this tick (e.g. init dispatch protocol violation).
    pub fatal_error: bool,
    /// Render-graph execution error for this tick, if any.
    pub graph_error: Option<GraphExecuteError>,
}

/// Facade: [`RendererFrontend`] + [`SceneCoordinator`] + [`RenderBackend`] + ingestion helpers.
pub struct RendererRuntime {
    frontend: RendererFrontend,
    backend: RenderBackend,
    /// Render spaces and dense transform / mesh state from [`crate::shared::FrameSubmitData`].
    scene: SceneCoordinator,
    /// Last host clip / FOV / VR / ortho task state for [`crate::render_graph::GraphPassFrame`].
    host_camera: HostCameraFrame,
    /// Settings handle, config path, and disk-write suppression.
    config: RuntimeConfigState,
    /// Runtime-side diagnostics accumulation.
    diagnostics: RuntimeDiagnosticsState,
    /// IPC scratch and unhandled-command counters.
    ipc_state: RuntimeIpcState,
    /// Per-tick gates and reusable view-planning scratch.
    tick_state: RuntimeTickState,
    /// Cumulative recoverable OpenXR failure counts.
    xr_stats: RuntimeXrStats,
}

impl RendererRuntime {
    /// Builds a runtime; does not open IPC yet (see [`Self::connect_ipc`]).
    pub fn new(
        params: Option<ConnectionParams>,
        settings: RendererSettingsHandle,
        config_save_path: PathBuf,
    ) -> Self {
        Self {
            frontend: RendererFrontend::new(params),
            backend: RenderBackend::new(),
            scene: SceneCoordinator::new(),
            host_camera: HostCameraFrame::default(),
            config: RuntimeConfigState::new(settings, config_save_path),
            diagnostics: RuntimeDiagnosticsState::new(),
            ipc_state: RuntimeIpcState::new(),
            tick_state: RuntimeTickState::new(),
            xr_stats: RuntimeXrStats::default(),
        }
    }
}

#[cfg(test)]
mod orchestration_tests;
