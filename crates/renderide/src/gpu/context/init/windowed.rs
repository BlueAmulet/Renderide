//! Windowed [`GpuContext::new`] constructor.

use std::sync::Arc;

use winit::window::Window;

use super::shared::{
    GpuContextParts, GpuRuntimeHandles, WindowAdapterLogFields, assemble_context,
    log_device_capability_summary, log_windowed_gpu_selection_summary,
    log_windowed_gpu_startup_request, select_window_adapter_with_fallback,
};
use super::super::super::adapter::device::{request_device_for_adapter, try_gpu_profiler};
use super::super::super::adapter::features::adapter_render_features_intersection;
use super::super::super::adapter::msaa_support::MsaaSupport;
use super::super::super::limits::GpuLimits;
use super::super::super::sync::mapped_buffer_health::GpuMappedBufferHealth;
use super::super::{GpuContext, GpuError};
use crate::config::{GraphicsApiSetting, VsyncMode};
use crate::gpu::submission_state::GpuSubmissionState;

impl GpuContext {
    /// Asynchronously builds GPU state for `window`.
    ///
    /// `gpu_validation_layers` selects whether to request backend validation before `WGPU_*` env
    /// overrides; see [`crate::gpu::instance_flags_for_gpu_init`]. `power_preference` is sourced
    /// from [`crate::config::DebugSettings::power_preference`] and used to rank enumerated
    /// adapters (discrete first when [`wgpu::PowerPreference::HighPerformance`], integrated first
    /// when [`wgpu::PowerPreference::LowPower`]).
    ///
    /// `vsync` is resolved against the surface's actual present-mode capabilities via
    /// [`VsyncMode::resolve_present_mode`] (so e.g. [`VsyncMode::On`] picks `FifoRelaxed` when
    /// available, then falls back to plain `Fifo`).
    ///
    /// `max_frame_latency` is the initial fixed value for
    /// [`wgpu::SurfaceConfiguration::desired_maximum_frame_latency`]. The renderer uses `2`,
    /// allowing CPU recording for frame N+1 to overlap with GPU work for frame N without adding
    /// another queued frame.
    ///
    /// `graphics_api` chooses the first backend set used for instance and adapter selection. An
    /// explicit API is retried with [`GraphicsApiSetting::Auto`] when it finds no compatible
    /// adapter. The final backend set may still be overridden by `WGPU_BACKEND`.
    pub async fn new(
        window: Arc<dyn Window>,
        vsync: VsyncMode,
        max_frame_latency: u32,
        gpu_validation_layers: bool,
        power_preference: wgpu::PowerPreference,
        graphics_api: GraphicsApiSetting,
    ) -> Result<Self, GpuError> {
        log_windowed_gpu_startup_request(
            window.as_ref(),
            vsync,
            max_frame_latency,
            gpu_validation_layers,
            power_preference,
            graphics_api,
        );
        let selection = select_window_adapter_with_fallback(
            &window,
            graphics_api,
            gpu_validation_layers,
            power_preference,
        )
        .await?;
        let selection_log = WindowAdapterLogFields {
            graphics_api: selection.graphics_api,
            active_backends: selection.active_backends,
            instance_flags: selection.instance_flags,
        };
        let surface_safe = selection.surface;
        let adapter = selection.adapter;

        let mapped_buffer_health = Arc::new(GpuMappedBufferHealth::new());
        let required_features = adapter_render_features_intersection(&adapter);
        let (device, queue) = request_device_for_adapter(
            &adapter,
            required_features,
            Arc::clone(&mapped_buffer_health),
        )
        .await?;

        let limits = GpuLimits::try_new(device.as_ref(), &adapter)?;
        let size = window.surface_size();
        let supported_present_modes = surface_safe.get_capabilities(&adapter).present_modes;
        let mut config = surface_safe
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .ok_or(GpuError::SurfaceUnsupported)?;
        config.present_mode = vsync.resolve_present_mode(&supported_present_modes);
        config.desired_maximum_frame_latency = max_frame_latency;
        surface_safe.configure(&device, &config);

        let adapter_info = adapter.get_info();
        let depth_stencil_format = crate::gpu::main_forward_depth_stencil_format(required_features);
        let msaa = MsaaSupport::discover(
            &adapter,
            config.format,
            depth_stencil_format,
            required_features,
            "GPU",
        );
        log_windowed_gpu_selection_summary(
            &adapter_info,
            selection_log,
            &config,
            vsync,
            &supported_present_modes,
            &msaa,
        );
        log_device_capability_summary("GPU", device.as_ref());

        let gpu_profiler = try_gpu_profiler(
            &adapter,
            device.as_ref(),
            &queue,
            "GPU profiler unavailable: adapter lacks TIMESTAMP_QUERY; \
             Tracy GPU timeline will be empty (CPU spans still work)",
        );
        let runtime = GpuRuntimeHandles::new(
            Arc::clone(&device),
            Arc::new(queue),
            Arc::clone(&mapped_buffer_health),
        )?;
        let submission = GpuSubmissionState::new(
            runtime.driver_thread,
            runtime.frame_timing,
            runtime.frame_bracket,
            gpu_profiler,
            runtime.latest_gpu_pass_timings,
        );
        Ok(assemble_context(GpuContextParts {
            submission,
            adapter_info,
            msaa,
            limits,
            device,
            queue: runtime.queue,
            gpu_queue_access_gate: runtime.gpu_queue_access_gate,
            mapped_buffer_health,
            surface: Some(surface_safe),
            config,
            supported_present_modes,
            window: Some(window),
        }))
    }
}
