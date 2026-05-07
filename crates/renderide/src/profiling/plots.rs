/// Records the FPS cap currently applied by
/// the app driver's `about_to_wait` handler -- either
/// [`crate::config::DisplaySettings::focused_fps_cap`] or
/// [`crate::config::DisplaySettings::unfocused_fps_cap`], whichever matches the current focus
/// state. Zero means uncapped (winit is told `ControlFlow::Poll`); a VR tick emits zero because
/// the XR runtime paces the session independently.
///
/// Call once per winit iteration so the Tracy plot sits adjacent to the frame-mark timeline and
/// the value-per-frame is an exact reading rather than an interpolation. Expands to nothing when
/// the `tracy` feature is off.
#[inline]
pub fn plot_fps_cap_active(cap: u32) {
    #[cfg(feature = "tracy")]
    tracy_client::plot!("fps_cap_active", f64::from(cap));
    #[cfg(not(feature = "tracy"))]
    let _ = cap;
}

/// Records window focus (`1.0` focused, `0.0` unfocused) as a Tracy plot so focus-driven cap
/// switches in the app driver's `about_to_wait` handler are visible at a glance.
///
/// Intended to be plotted next to [`plot_fps_cap_active`]: a drop from `1.0` to `0.0` should line
/// up with the cap changing from `focused_fps_cap` to `unfocused_fps_cap` (or vice versa), which
/// is the usual cause of a sudden frame-time change while profiling.
///
/// Expands to nothing when the `tracy` feature is off.
#[inline]
pub fn plot_window_focused(focused: bool) {
    #[cfg(feature = "tracy")]
    tracy_client::plot!("window_focused", if focused { 1.0 } else { 0.0 });
    #[cfg(not(feature = "tracy"))]
    let _ = focused;
}

/// Records, in milliseconds, how long
/// the app driver's `about_to_wait` handler asked winit to park before the next
/// `RedrawRequested`. Emit the [`std::time::Duration`] between `now` and the
/// [`winit::event_loop::ControlFlow::WaitUntil`] deadline when the capped branch is taken, and
/// `0.0` when the handler returns with [`winit::event_loop::ControlFlow::Poll`].
///
/// The gap between Tracy frames that no [`profiling::scope`] can cover (because the main thread
/// is parked inside winit) shows up on this plot as a non-zero value, attributing the idle time
/// to the CPU-side frame-pacing cap rather than missing instrumentation. Expands to nothing when
/// the `tracy` feature is off.
#[inline]
pub fn plot_event_loop_wait_ms(ms: f64) {
    #[cfg(feature = "tracy")]
    tracy_client::plot!("event_loop_wait_ms", ms);
    #[cfg(not(feature = "tracy"))]
    let _ = ms;
}

/// Records the driver-thread submit backlog (`submits_pushed - submits_done`) as a Tracy
/// plot.
///
/// Call once per tick from the frame epilogue. A steady-state value of `0` or `1` is
/// healthy (one frame in flight on the driver matches the ring's nominal pipelining
/// depth); a sustained value at the ring capacity means the producer is back-pressured
/// by the driver and CPU/GPU pacing is bound by submit throughput. Useful next to
/// [`plot_event_loop_idle_ms`] when diagnosing why the main thread is sleeping.
///
/// Expands to nothing when the `tracy` feature is off.
#[inline]
pub fn plot_driver_submit_backlog(count: u64) {
    #[cfg(feature = "tracy")]
    tracy_client::plot!("driver_submit_backlog", count as f64);
    #[cfg(not(feature = "tracy"))]
    let _ = count;
}

/// Records, in milliseconds, the wall-clock gap between the end of the previous
/// app-driver redraw tick and the start of the current one.
///
/// Complements [`plot_event_loop_wait_ms`] (the *requested* wait) by showing the *actual* slept
/// duration -- divergence between the two points at additional blocking outside the pacing cap
/// (for example compositor vsync via `surface.get_current_texture`, which is itself already
/// covered by a dedicated `gpu::get_current_texture` scope).
///
/// Expands to nothing when the `tracy` feature is off.
#[inline]
pub fn plot_event_loop_idle_ms(ms: f64) {
    #[cfg(feature = "tracy")]
    tracy_client::plot!("event_loop_idle_ms", ms);
    #[cfg(not(feature = "tracy"))]
    let _ = ms;
}

/// Records the result of a swapchain acquire attempt as one-hot Tracy plots.
///
/// These samples explain CPU frames that have a frame mark but no render-graph GPU markers: a
/// timeout or occluded surface intentionally skips graph recording for that tick, while a
/// reconfigure means the graph will resume on a later acquire.
#[inline]
pub fn plot_surface_acquire_outcome(acquired: bool, skipped: bool, reconfigured: bool) {
    #[cfg(feature = "tracy")]
    {
        tracy_client::plot!(
            "surface_acquire::acquired",
            if acquired { 1.0 } else { 0.0 }
        );
        tracy_client::plot!("surface_acquire::skipped", if skipped { 1.0 } else { 0.0 });
        tracy_client::plot!(
            "surface_acquire::reconfigured",
            if reconfigured { 1.0 } else { 0.0 }
        );
    }
    #[cfg(not(feature = "tracy"))]
    {
        let _ = (acquired, skipped, reconfigured);
    }
}

/// Records, per call to `crate::passes::world_mesh_forward::encode::draw_subset`,
/// how many instance batches and how many input draws were submitted in that subpass.
///
/// One sample lands on the Tracy timeline per opaque or intersection subpass record, so the
/// plot trace shows fragmentation visually: when batches ~= draws, the merge isn't compressing;
/// when batches << draws, instancing is collapsing same-mesh runs as intended. Pair with
/// [`crate::world_mesh::WorldMeshDrawStats::gpu_instances_emitted`] in the HUD for a
/// per-frame integral. Expands to nothing when the `tracy` feature is off.
#[inline]
pub fn plot_world_mesh_subpass(batches: usize, draws: usize) {
    #[cfg(feature = "tracy")]
    {
        tracy_client::plot!("world_mesh::subpass_batches", batches as f64);
        tracy_client::plot!("world_mesh::subpass_draws", draws as f64);
    }
    #[cfg(not(feature = "tracy"))]
    {
        let _ = (batches, draws);
    }
}

/// Records deferred queue-write traffic for one frame.
#[inline]
pub fn plot_frame_upload_batch(writes: usize, bytes: usize) {
    #[cfg(feature = "tracy")]
    {
        tracy_client::plot!("frame_upload::writes", writes as f64);
        tracy_client::plot!("frame_upload::bytes", bytes as f64);
    }
    #[cfg(not(feature = "tracy"))]
    {
        let _ = (writes, bytes);
    }
}

/// CPU timings and counts for one render-graph command-encoding slice.
#[derive(Clone, Copy, Debug, Default)]
pub struct CommandEncodingProfileSample {
    /// Number of views encoded by the graph.
    pub view_count: usize,
    /// Number of command buffers submitted in the batch.
    pub command_buffers: usize,
    /// Frame-global pass count in the compiled schedule.
    pub frame_global_passes: usize,
    /// Per-view pass count in the compiled schedule.
    pub per_view_passes: usize,
    /// Declared transient texture handles in the compiled graph.
    pub transient_textures: usize,
    /// Physical transient texture slots after aliasing.
    pub transient_texture_slots: usize,
    /// Transient texture allocation misses during this frame.
    pub transient_texture_misses: usize,
    /// Transient buffer allocation misses during this frame.
    pub transient_buffer_misses: usize,
    /// Deferred upload writes drained before submit.
    pub upload_writes: usize,
    /// Deferred upload payload bytes drained before submit.
    pub upload_bytes: usize,
    /// Upload bytes staged through persistent arena slots.
    pub upload_persistent_staging_bytes: u64,
    /// Persistent arena slot reuse count.
    pub upload_persistent_slot_reuses: usize,
    /// Persistent arena slot allocation or growth count.
    pub upload_persistent_slot_grows: usize,
    /// Upload bytes staged through temporary fallback buffers.
    pub upload_temporary_staging_bytes: u64,
    /// Temporary staging fallback count caused by all persistent slots being unavailable.
    pub upload_temporary_staging_fallbacks: usize,
    /// Staged writes replayed through queue writes because no staging buffer fit.
    pub upload_oversized_queue_fallback_writes: usize,
    /// Bytes allocated across persistent upload arena slots.
    pub upload_arena_capacity_bytes: u64,
    /// Persistent upload arena slots mapped and available for writes.
    pub upload_arena_free_slots: usize,
    /// Persistent upload arena slots currently in flight.
    pub upload_arena_in_flight_slots: usize,
    /// Persistent upload arena slots waiting for remap completion.
    pub upload_arena_remapping_slots: usize,
    /// CPU time spent resolving transient resources for all views.
    pub pre_resolve_ms: f64,
    /// CPU time spent preparing shared/per-view resources before recording.
    pub prepare_resources_ms: f64,
    /// CPU time spent encoding frame-global work before `CommandEncoder::finish`.
    pub frame_global_encode_ms: f64,
    /// CPU time spent inside frame-global `CommandEncoder::finish`.
    pub frame_global_finish_ms: f64,
    /// CPU time spent encoding per-view work before `CommandEncoder::finish`.
    pub per_view_encode_ms: f64,
    /// Total CPU time spent inside per-view `CommandEncoder::finish` calls.
    pub per_view_finish_ms: f64,
    /// CPU time spent draining deferred uploads.
    pub upload_drain_ms: f64,
    /// CPU time spent inside the upload encoder `CommandEncoder::finish`.
    pub upload_finish_ms: f64,
    /// CPU time spent allocating and assembling the final command-buffer batch.
    pub command_batch_assembly_ms: f64,
    /// CPU time spent enqueueing the submit batch to the GPU driver thread.
    pub submit_enqueue_ms: f64,
    /// Largest single encoder finish observed in this frame.
    pub max_encoder_finish_ms: f64,
    /// World-mesh draw items visible to the command recorder.
    pub world_mesh_draws: usize,
    /// World-mesh indexed draw groups emitted by the command recorder.
    pub world_mesh_instance_batches: usize,
    /// World-mesh pipeline-pass draw submissions after multi-pass material expansion.
    pub world_mesh_pipeline_pass_submits: usize,
}

/// Records command-encoding timings and pressure counters for the current frame.
#[inline]
pub fn plot_command_encoding(sample: CommandEncodingProfileSample) {
    #[cfg(feature = "tracy")]
    plot_command_encoding_tracy(sample);
    #[cfg(not(feature = "tracy"))]
    consume_command_encoding_sample(sample);
}

#[cfg(feature = "tracy")]
fn plot_command_encoding_tracy(sample: CommandEncodingProfileSample) {
    tracy_client::plot!("command_encoding::views", sample.view_count as f64);
    tracy_client::plot!(
        "command_encoding::command_buffers",
        sample.command_buffers as f64
    );
    tracy_client::plot!(
        "command_encoding::frame_global_passes",
        sample.frame_global_passes as f64
    );
    tracy_client::plot!(
        "command_encoding::per_view_passes",
        sample.per_view_passes as f64
    );
    tracy_client::plot!(
        "command_encoding::transient_textures",
        sample.transient_textures as f64
    );
    tracy_client::plot!(
        "command_encoding::transient_texture_slots",
        sample.transient_texture_slots as f64
    );
    tracy_client::plot!(
        "command_encoding::transient_texture_misses",
        sample.transient_texture_misses as f64
    );
    tracy_client::plot!(
        "command_encoding::transient_buffer_misses",
        sample.transient_buffer_misses as f64
    );
    tracy_client::plot!(
        "command_encoding::upload_writes",
        sample.upload_writes as f64
    );
    tracy_client::plot!("command_encoding::upload_bytes", sample.upload_bytes as f64);
    plot_command_encoding_upload_arena_tracy(&sample);
    tracy_client::plot!("command_encoding::pre_resolve_ms", sample.pre_resolve_ms);
    tracy_client::plot!(
        "command_encoding::prepare_resources_ms",
        sample.prepare_resources_ms
    );
    tracy_client::plot!(
        "command_encoding::frame_global_encode_ms",
        sample.frame_global_encode_ms
    );
    tracy_client::plot!(
        "command_encoding::frame_global_finish_ms",
        sample.frame_global_finish_ms
    );
    tracy_client::plot!(
        "command_encoding::per_view_encode_ms",
        sample.per_view_encode_ms
    );
    tracy_client::plot!(
        "command_encoding::per_view_finish_ms",
        sample.per_view_finish_ms
    );
    tracy_client::plot!("command_encoding::upload_drain_ms", sample.upload_drain_ms);
    tracy_client::plot!(
        "command_encoding::upload_finish_ms",
        sample.upload_finish_ms
    );
    tracy_client::plot!(
        "command_encoding::command_batch_assembly_ms",
        sample.command_batch_assembly_ms
    );
    tracy_client::plot!(
        "command_encoding::submit_enqueue_ms",
        sample.submit_enqueue_ms
    );
    tracy_client::plot!(
        "command_encoding::max_encoder_finish_ms",
        sample.max_encoder_finish_ms
    );
    tracy_client::plot!(
        "command_encoding::world_mesh_draws",
        sample.world_mesh_draws as f64
    );
    tracy_client::plot!(
        "command_encoding::world_mesh_instance_batches",
        sample.world_mesh_instance_batches as f64
    );
    tracy_client::plot!(
        "command_encoding::world_mesh_pipeline_pass_submits",
        sample.world_mesh_pipeline_pass_submits as f64
    );
}

#[cfg(feature = "tracy")]
fn plot_command_encoding_upload_arena_tracy(sample: &CommandEncodingProfileSample) {
    tracy_client::plot!(
        "command_encoding::upload_persistent_staging_bytes",
        sample.upload_persistent_staging_bytes as f64
    );
    tracy_client::plot!(
        "command_encoding::upload_persistent_slot_reuses",
        sample.upload_persistent_slot_reuses as f64
    );
    tracy_client::plot!(
        "command_encoding::upload_persistent_slot_grows",
        sample.upload_persistent_slot_grows as f64
    );
    tracy_client::plot!(
        "command_encoding::upload_temporary_staging_bytes",
        sample.upload_temporary_staging_bytes as f64
    );
    tracy_client::plot!(
        "command_encoding::upload_temporary_staging_fallbacks",
        sample.upload_temporary_staging_fallbacks as f64
    );
    tracy_client::plot!(
        "command_encoding::upload_oversized_queue_fallback_writes",
        sample.upload_oversized_queue_fallback_writes as f64
    );
    tracy_client::plot!(
        "command_encoding::upload_arena_capacity_bytes",
        sample.upload_arena_capacity_bytes as f64
    );
    tracy_client::plot!(
        "command_encoding::upload_arena_free_slots",
        sample.upload_arena_free_slots as f64
    );
    tracy_client::plot!(
        "command_encoding::upload_arena_in_flight_slots",
        sample.upload_arena_in_flight_slots as f64
    );
    tracy_client::plot!(
        "command_encoding::upload_arena_remapping_slots",
        sample.upload_arena_remapping_slots as f64
    );
}

#[cfg(not(feature = "tracy"))]
fn consume_command_encoding_sample(sample: CommandEncodingProfileSample) {
    let CommandEncodingProfileSample {
        view_count,
        command_buffers,
        frame_global_passes,
        per_view_passes,
        transient_textures,
        transient_texture_slots,
        transient_texture_misses,
        transient_buffer_misses,
        upload_writes,
        upload_bytes,
        upload_persistent_staging_bytes,
        upload_persistent_slot_reuses,
        upload_persistent_slot_grows,
        upload_temporary_staging_bytes,
        upload_temporary_staging_fallbacks,
        upload_oversized_queue_fallback_writes,
        upload_arena_capacity_bytes,
        upload_arena_free_slots,
        upload_arena_in_flight_slots,
        upload_arena_remapping_slots,
        pre_resolve_ms,
        prepare_resources_ms,
        frame_global_encode_ms,
        frame_global_finish_ms,
        per_view_encode_ms,
        per_view_finish_ms,
        upload_drain_ms,
        upload_finish_ms,
        command_batch_assembly_ms,
        submit_enqueue_ms,
        max_encoder_finish_ms,
        world_mesh_draws,
        world_mesh_instance_batches,
        world_mesh_pipeline_pass_submits,
    } = sample;
    let _ = (
        view_count,
        command_buffers,
        frame_global_passes,
        per_view_passes,
        transient_textures,
        transient_texture_slots,
        transient_texture_misses,
        transient_buffer_misses,
        upload_writes,
        upload_bytes,
        upload_persistent_staging_bytes,
        upload_persistent_slot_reuses,
        upload_persistent_slot_grows,
        upload_temporary_staging_bytes,
        upload_temporary_staging_fallbacks,
        upload_oversized_queue_fallback_writes,
        upload_arena_capacity_bytes,
        upload_arena_free_slots,
        upload_arena_in_flight_slots,
        upload_arena_remapping_slots,
        pre_resolve_ms,
        prepare_resources_ms,
        frame_global_encode_ms,
        frame_global_finish_ms,
        per_view_encode_ms,
        per_view_finish_ms,
        upload_drain_ms,
        upload_finish_ms,
        command_batch_assembly_ms,
        submit_enqueue_ms,
        max_encoder_finish_ms,
        world_mesh_draws,
        world_mesh_instance_batches,
        world_mesh_pipeline_pass_submits,
    );
}

/// Asset-integration backlog and budget-exhaustion counters for one drain.
#[derive(Clone, Copy, Debug, Default)]
pub struct AssetIntegrationProfileSample {
    /// High-priority tasks still queued after the drain.
    pub high_priority_queued: usize,
    /// Normal-priority tasks still queued after the drain.
    pub normal_priority_queued: usize,
    /// Whether the high-priority emergency ceiling stopped the drain.
    pub high_priority_budget_exhausted: bool,
    /// Whether the normal-priority frame budget stopped the drain.
    pub normal_priority_budget_exhausted: bool,
}

/// Records asset-integration backlog and budget pressure for the current frame.
#[inline]
pub fn plot_asset_integration(sample: AssetIntegrationProfileSample) {
    #[cfg(feature = "tracy")]
    {
        tracy_client::plot!(
            "asset_integration::high_priority_queued",
            sample.high_priority_queued as f64
        );
        tracy_client::plot!(
            "asset_integration::normal_priority_queued",
            sample.normal_priority_queued as f64
        );
        tracy_client::plot!(
            "asset_integration::high_priority_budget_exhausted",
            if sample.high_priority_budget_exhausted {
                1.0
            } else {
                0.0
            }
        );
        tracy_client::plot!(
            "asset_integration::normal_priority_budget_exhausted",
            if sample.normal_priority_budget_exhausted {
                1.0
            } else {
                0.0
            }
        );
    }
    #[cfg(not(feature = "tracy"))]
    {
        let AssetIntegrationProfileSample {
            high_priority_queued,
            normal_priority_queued,
            high_priority_budget_exhausted,
            normal_priority_budget_exhausted,
        } = sample;
        let _ = (
            high_priority_queued,
            normal_priority_queued,
            high_priority_budget_exhausted,
            normal_priority_budget_exhausted,
        );
    }
}

/// Mesh-deform workload and cache pressure counters for one frame.
#[derive(Clone, Copy, Debug, Default)]
pub struct MeshDeformProfileSample {
    /// Deform work items collected for this frame.
    pub work_items: u64,
    /// Compute passes opened while recording mesh deformation.
    pub compute_passes: u64,
    /// Bind groups created while recording mesh deformation.
    pub bind_groups_created: u64,
    /// Bind groups reused from mesh-deform caches.
    pub bind_group_cache_reuses: u64,
    /// Encoder copy operations recorded by mesh deformation.
    pub copy_ops: u64,
    /// Sparse blendshape compute dispatches recorded.
    pub blend_dispatches: u64,
    /// Skinning compute dispatches recorded.
    pub skin_dispatches: u64,
    /// Work items skipped because their deform inputs were stable.
    pub stable_skips: u64,
    /// Scratch-buffer grow operations triggered by this frame.
    pub scratch_buffer_grows: u64,
    /// Work items skipped because the skin cache could not allocate safely.
    pub skipped_allocations: u64,
    /// Skin-cache entries reused.
    pub cache_reuses: u64,
    /// Skin-cache entries allocated.
    pub cache_allocations: u64,
    /// Skin-cache arena growth operations.
    pub cache_grows: u64,
    /// Prior-frame skin-cache entries evicted.
    pub cache_evictions: u64,
    /// Allocation attempts where all evictable entries were current-frame entries.
    pub cache_current_frame_eviction_refusals: u64,
}

/// Records mesh-deform workload and cache pressure counters for the current frame.
#[inline]
pub fn plot_mesh_deform(sample: MeshDeformProfileSample) {
    #[cfg(feature = "tracy")]
    {
        tracy_client::plot!("mesh_deform::work_items", sample.work_items as f64);
        tracy_client::plot!("mesh_deform::compute_passes", sample.compute_passes as f64);
        tracy_client::plot!(
            "mesh_deform::bind_groups_created",
            sample.bind_groups_created as f64
        );
        tracy_client::plot!(
            "mesh_deform::bind_group_cache_reuses",
            sample.bind_group_cache_reuses as f64
        );
        tracy_client::plot!("mesh_deform::copy_ops", sample.copy_ops as f64);
        tracy_client::plot!(
            "mesh_deform::blend_dispatches",
            sample.blend_dispatches as f64
        );
        tracy_client::plot!(
            "mesh_deform::skin_dispatches",
            sample.skin_dispatches as f64
        );
        tracy_client::plot!("mesh_deform::stable_skips", sample.stable_skips as f64);
        tracy_client::plot!(
            "mesh_deform::scratch_buffer_grows",
            sample.scratch_buffer_grows as f64
        );
        tracy_client::plot!(
            "mesh_deform::skipped_allocations",
            sample.skipped_allocations as f64
        );
        tracy_client::plot!("mesh_deform::cache_reuses", sample.cache_reuses as f64);
        tracy_client::plot!(
            "mesh_deform::cache_allocations",
            sample.cache_allocations as f64
        );
        tracy_client::plot!("mesh_deform::cache_grows", sample.cache_grows as f64);
        tracy_client::plot!(
            "mesh_deform::cache_evictions",
            sample.cache_evictions as f64
        );
        tracy_client::plot!(
            "mesh_deform::cache_current_frame_eviction_refusals",
            sample.cache_current_frame_eviction_refusals as f64
        );
    }
    #[cfg(not(feature = "tracy"))]
    {
        let MeshDeformProfileSample {
            work_items,
            compute_passes,
            bind_groups_created,
            bind_group_cache_reuses,
            copy_ops,
            blend_dispatches,
            skin_dispatches,
            stable_skips,
            scratch_buffer_grows,
            skipped_allocations,
            cache_reuses,
            cache_allocations,
            cache_grows,
            cache_evictions,
            cache_current_frame_eviction_refusals,
        } = sample;
        let _ = (
            work_items,
            compute_passes,
            bind_groups_created,
            bind_group_cache_reuses,
            copy_ops,
            blend_dispatches,
            skin_dispatches,
            stable_skips,
            scratch_buffer_grows,
            skipped_allocations,
            cache_reuses,
            cache_allocations,
            cache_grows,
            cache_evictions,
            cache_current_frame_eviction_refusals,
        );
    }
}
