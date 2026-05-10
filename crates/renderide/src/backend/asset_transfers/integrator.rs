//! Cooperative asset-integration queues and wall-clock-bounded draining.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::gpu::GpuLimits;
use crate::ipc::{DualQueueIpc, SharedMemoryAccessor};
use crate::materials::{MaterialSystem, RasterPipelineKind};
use crate::profiling::{AssetIntegrationProfileSample, plot_asset_integration};
use crate::shared::{
    MaterialsUpdateBatch, PointRenderBufferConsumed, PointRenderBufferUpload, RendererCommand,
    ShaderUploadResult, TrailRenderBufferConsumed, TrailRenderBufferUpload,
};

use super::AssetTransferQueue;
use super::cubemap_task::CubemapUploadTask;
use super::mesh_task::MeshUploadTask;
use super::texture_task::TextureUploadTask;
use super::texture3d_task::Texture3dUploadTask;

mod retired;

pub use retired::RetiredAssetResource;

/// Combined queued integration task count that emits queue-pressure diagnostics.
pub const ASSET_INTEGRATION_QUEUE_WARN_THRESHOLD: usize = 2048;

/// Queue-pressure log stride after [`ASSET_INTEGRATION_QUEUE_WARN_THRESHOLD`] is exceeded.
const ASSET_INTEGRATION_QUEUE_WARN_STRIDE: usize = 1024;

/// Number of integration updates a removed GPU resource is retained before drop.
const DELAYED_REMOVAL_UPDATES: usize = 3;

/// Minimum extra wall-clock slice granted to high-priority integration before yielding.
const MIN_HIGH_PRIORITY_EMERGENCY_BUDGET: Duration = Duration::from_millis(1);

/// Queue and budget state observed during one cooperative asset-integration drain.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AssetIntegrationDrainSummary {
    /// Main-thread tasks queued before the drain.
    pub main_before: usize,
    /// High-priority tasks queued before the drain.
    pub high_priority_before: usize,
    /// Normal-priority tasks queued before the drain.
    pub normal_priority_before: usize,
    /// Render-lane tasks queued before the drain.
    pub render_before: usize,
    /// Particle/dynamic-buffer tasks queued before the drain.
    pub particle_before: usize,
    /// Main-thread tasks queued after the drain.
    pub main_after: usize,
    /// High-priority tasks queued after the drain.
    pub high_priority_after: usize,
    /// Normal-priority tasks queued after the drain.
    pub normal_priority_after: usize,
    /// Render-lane tasks queued after the drain.
    pub render_after: usize,
    /// Particle/dynamic-buffer tasks queued after the drain.
    pub particle_after: usize,
    /// Whether the drain had GPU handles needed to execute upload work.
    pub gpu_ready: bool,
    /// Number of queue steps processed during the drain.
    pub processed_tasks: u32,
    /// Number of main-lane queue steps processed during the drain.
    pub processed_main_tasks: u32,
    /// Number of high-priority queue steps processed during the drain.
    pub processed_high_priority_tasks: u32,
    /// Number of normal-priority queue steps processed during the drain.
    pub processed_normal_priority_tasks: u32,
    /// Number of render-lane queue steps processed during the drain.
    pub processed_render_tasks: u32,
    /// Number of particle-lane queue steps processed during the drain.
    pub processed_particle_tasks: u32,
    /// Whether high-priority work exceeded the emergency budget.
    pub high_priority_budget_exhausted: bool,
    /// Whether normal-priority work exceeded the frame budget.
    pub normal_priority_budget_exhausted: bool,
    /// Whether render-lane work exceeded the frame budget.
    pub render_budget_exhausted: bool,
    /// Whether particle-lane work exceeded its separate post-main budget.
    pub particle_budget_exhausted: bool,
    /// Wall-clock time spent in non-particle integration lanes.
    pub elapsed: Duration,
    /// Wall-clock time spent in the particle lane.
    pub particle_elapsed: Duration,
    /// Highest combined queued task count observed since startup.
    pub peak_queued: usize,
}

impl AssetIntegrationDrainSummary {
    /// Captures queue state before integration starts.
    fn start(asset: &AssetTransferQueue) -> Self {
        Self {
            main_before: asset.integrator.main.len(),
            high_priority_before: asset.integrator.high_priority.len(),
            normal_priority_before: asset.integrator.normal_priority.len(),
            render_before: asset.integrator.render.len(),
            particle_before: asset.integrator.particle.len(),
            ..Self::default()
        }
    }

    /// Completes the summary from the queue state after integration ends.
    fn finish(mut self, asset: &AssetTransferQueue, finish: DrainFinishState) -> Self {
        self.main_after = asset.integrator.main.len();
        self.high_priority_after = asset.integrator.high_priority.len();
        self.normal_priority_after = asset.integrator.normal_priority.len();
        self.render_after = asset.integrator.render.len();
        self.particle_after = asset.integrator.particle.len();
        self.gpu_ready = finish.gpu_ready;
        self.high_priority_budget_exhausted = finish.budgets.high_priority;
        self.normal_priority_budget_exhausted = finish.budgets.normal_priority;
        self.render_budget_exhausted = finish.budgets.render;
        self.particle_budget_exhausted = finish.budgets.particle;
        self.processed_tasks = finish.processed.total();
        self.processed_main_tasks = finish.processed.main;
        self.processed_high_priority_tasks = finish.processed.high_priority;
        self.processed_normal_priority_tasks = finish.processed.normal_priority;
        self.processed_render_tasks = finish.processed.render;
        self.processed_particle_tasks = finish.processed.particle;
        self.elapsed = finish.elapsed;
        self.particle_elapsed = finish.particle_elapsed;
        self.peak_queued = asset.integrator.peak_queued();
        self
    }

    /// Combined queued work before the drain.
    pub fn total_before(self) -> usize {
        self.main_before
            + self.high_priority_before
            + self.render_before
            + self.normal_priority_before
            + self.particle_before
    }

    /// Combined queued work after the drain.
    pub fn total_after(self) -> usize {
        self.main_after
            + self.high_priority_after
            + self.render_after
            + self.normal_priority_after
            + self.particle_after
    }

    /// Whether any budget ceiling was reached while work remained queued.
    pub fn budget_exhausted(self) -> bool {
        self.high_priority_budget_exhausted
            || self.normal_priority_budget_exhausted
            || self.render_budget_exhausted
            || self.particle_budget_exhausted
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct BudgetExhaustion {
    high_priority: bool,
    normal_priority: bool,
    render: bool,
    particle: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct DrainFinishState {
    gpu_ready: bool,
    budgets: BudgetExhaustion,
    processed: ProcessedLaneCounts,
    particle_elapsed: Duration,
    elapsed: Duration,
}

/// Shader-route registration plus host acknowledgement produced by the async shader resolver.
#[derive(Debug)]
pub struct ShaderRouteTask {
    /// Host shader asset id.
    pub asset_id: i32,
    /// Resolved raster pipeline.
    pub pipeline: RasterPipelineKind,
    /// Resolved AssetBundle shader asset name, when available.
    pub shader_asset_name: Option<String>,
    /// Froox shader variant bitmask parsed from the serialized Shader name suffix.
    pub shader_variant_bits: Option<u32>,
}

/// One cooperative integration task.
#[derive(Debug)]
pub enum AssetTask {
    /// Renderer-main-thread material batch application.
    MaterialUpdate(MaterialsUpdateBatch),
    /// Renderer-main-thread shader route registration.
    ShaderRoute(ShaderRouteTask),
    /// Placeholder point render-buffer ingestion and acknowledgement.
    PointRenderBuffer(PointRenderBufferUpload),
    /// Placeholder trail render-buffer ingestion and acknowledgement.
    TrailRenderBuffer(TrailRenderBufferUpload),
    /// Host mesh payload integration.
    Mesh(MeshUploadTask),
    /// Host Texture2D mip integration.
    Texture(TextureUploadTask),
    /// Host Texture3D mip integration.
    Texture3d(Texture3dUploadTask),
    /// Host cubemap face/mip integration.
    Cubemap(CubemapUploadTask),
}

/// Whether a task needs another [`AssetTask::step`] call in a later drain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepResult {
    /// More work remains for this logical upload.
    Continue,
    /// Upload finished (success or logged failure; host callbacks sent when applicable).
    Done,
    /// Task is waiting for a background thread to finish; push to the back of the queue.
    YieldBackground,
}

/// Logical scheduler lane for an [`AssetTask`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AssetTaskLane {
    /// Renderer-main-thread tasks drained before other lanes.
    Main,
    /// Urgent upload lane.
    HighPriority,
    /// Standard upload lane.
    NormalPriority,
    /// Wgpu-native render-thread-adjacent work.
    Render,
    /// Dynamic-buffer / particle lane with a separate post-main budget.
    Particle,
}

/// Priority-separated cooperative upload queues.
#[derive(Debug, Default)]
pub struct AssetIntegrator {
    /// Renderer-main-thread tasks run before the priority lanes.
    pub main: VecDeque<AssetTask>,
    /// [`MeshUploadData::high_priority`] / texture data `high_priority` tasks.
    pub high_priority: VecDeque<AssetTask>,
    /// Standard-priority tasks.
    pub normal_priority: VecDeque<AssetTask>,
    /// Wgpu render-lane tasks.
    pub render: VecDeque<AssetTask>,
    /// Dynamic-buffer / particle tasks.
    pub particle: VecDeque<AssetTask>,
    /// Removed resources held alive for the delayed-removal window.
    delayed_removals: VecDeque<RetiredAssetResource>,
    /// Per-bucket delayed-removal counts.
    delayed_removal_counts: [usize; DELAYED_REMOVAL_UPDATES],
    /// Current delayed-removal bucket.
    delayed_removal_bucket_index: usize,
    /// Highest combined queue depth observed since startup.
    max_total_queued: usize,
}

impl AssetIntegrator {
    /// Total queued tasks.
    pub fn total_queued(&self) -> usize {
        self.main.len()
            + self.high_priority.len()
            + self.render.len()
            + self.normal_priority.len()
            + self.particle.len()
    }

    /// Highest combined queued task count observed since startup.
    pub fn peak_queued(&self) -> usize {
        self.max_total_queued
    }

    /// Pops the next task, preferring the high-priority queue.
    #[cfg(test)]
    pub fn pop_next(&mut self) -> Option<AssetTask> {
        self.main
            .pop_front()
            .or_else(|| self.high_priority.pop_front())
            .or_else(|| self.render.pop_front())
            .or_else(|| self.normal_priority.pop_front())
            .or_else(|| self.particle.pop_front())
    }

    /// Pushes a task to the front of the requested lane.
    pub fn push_front_lane(&mut self, task: AssetTask, lane: AssetTaskLane) {
        self.lane_mut(lane).push_front(task);
    }

    /// Pushes a task to the back of the requested lane.
    pub fn push_back_lane(&mut self, task: AssetTask, lane: AssetTaskLane) {
        self.lane_mut(lane).push_back(task);
    }

    /// Pops a task from the requested lane.
    pub fn pop_front_lane(&mut self, lane: AssetTaskLane) -> Option<AssetTask> {
        self.lane_mut(lane).pop_front()
    }

    /// Returns the queued count for `lane`.
    pub fn lane_len(&self, lane: AssetTaskLane) -> usize {
        match lane {
            AssetTaskLane::Main => self.main.len(),
            AssetTaskLane::HighPriority => self.high_priority.len(),
            AssetTaskLane::NormalPriority => self.normal_priority.len(),
            AssetTaskLane::Render => self.render.len(),
            AssetTaskLane::Particle => self.particle.len(),
        }
    }

    /// Whether `lane` has no queued work.
    pub fn lane_is_empty(&self, lane: AssetTaskLane) -> bool {
        self.lane_len(lane) == 0
    }

    fn lane_mut(&mut self, lane: AssetTaskLane) -> &mut VecDeque<AssetTask> {
        match lane {
            AssetTaskLane::Main => &mut self.main,
            AssetTaskLane::HighPriority => &mut self.high_priority,
            AssetTaskLane::NormalPriority => &mut self.normal_priority,
            AssetTaskLane::Render => &mut self.render,
            AssetTaskLane::Particle => &mut self.particle,
        }
    }

    /// Pushes a task to the front of the appropriate queue (resume after a [`StepResult::Continue`]).
    #[cfg(test)]
    pub fn push_front(&mut self, task: AssetTask, high_priority: bool) {
        if high_priority {
            self.push_front_lane(task, AssetTaskLane::HighPriority);
        } else {
            self.push_front_lane(task, AssetTaskLane::NormalPriority);
        }
    }

    /// Enqueues an upload task at the back of its priority lane.
    pub fn enqueue(&mut self, task: AssetTask, high_priority: bool) {
        if high_priority {
            self.push_back_lane(task, AssetTaskLane::HighPriority);
        } else {
            self.push_back_lane(task, AssetTaskLane::NormalPriority);
        }
        self.record_queue_depth();
    }

    /// Enqueues a task in a specific scheduler lane.
    pub fn enqueue_lane(&mut self, task: AssetTask, lane: AssetTaskLane) {
        self.push_back_lane(task, lane);
        self.record_queue_depth();
    }

    /// Enqueues a removed GPU resource for delayed drop.
    pub fn enqueue_delayed_removal(&mut self, resource: RetiredAssetResource) {
        self.delayed_removals.push_back(resource);
        self.delayed_removal_counts[self.delayed_removal_bucket_index] += 1;
    }

    /// Drops the delayed-removal bucket that has aged through the configured update window.
    pub fn process_delayed_removals(&mut self) -> usize {
        let index = (self.delayed_removal_bucket_index + (DELAYED_REMOVAL_UPDATES - 1))
            % DELAYED_REMOVAL_UPDATES;
        let count = self.delayed_removal_counts[index];
        let mut released_bytes = 0;
        for _ in 0..count {
            if let Some(resource) = self.delayed_removals.pop_front() {
                released_bytes += resource.resident_bytes();
            }
        }
        if count > 0 {
            logger::trace!(
                "asset integrator delayed removals released: count={count} bytes={released_bytes}"
            );
        }
        self.delayed_removal_counts[index] = 0;
        self.delayed_removal_bucket_index =
            (self.delayed_removal_bucket_index + 1) % DELAYED_REMOVAL_UPDATES;
        count
    }

    fn record_queue_depth(&mut self) {
        let queued = self.total_queued();
        self.max_total_queued = self.max_total_queued.max(queued);
        if should_log_asset_integration_queue_pressure(queued) {
            logger::warn!(
                "asset integrator backlog high: queued={} main={} high_priority={} render={} normal_priority={} particle={} threshold={}",
                queued,
                self.main.len(),
                self.high_priority.len(),
                self.render.len(),
                self.normal_priority.len(),
                self.particle.len(),
                ASSET_INTEGRATION_QUEUE_WARN_THRESHOLD
            );
        }
    }
}

fn should_log_asset_integration_queue_pressure(queued: usize) -> bool {
    queued == ASSET_INTEGRATION_QUEUE_WARN_THRESHOLD
        || (queued > ASSET_INTEGRATION_QUEUE_WARN_THRESHOLD
            && queued.is_multiple_of(ASSET_INTEGRATION_QUEUE_WARN_STRIDE))
}

/// Returns a stable tag for [`AssetTask`] variants, used as Tracy zone data.
#[cfg_attr(
    not(feature = "tracy"),
    expect(dead_code, reason = "tag only consumed by Tracy zones")
)]
fn asset_task_kind_tag(task: &AssetTask) -> &'static str {
    match task {
        AssetTask::MaterialUpdate(_) => "MaterialUpdate",
        AssetTask::ShaderRoute(_) => "ShaderRoute",
        AssetTask::PointRenderBuffer(_) => "PointRenderBuffer",
        AssetTask::TrailRenderBuffer(_) => "TrailRenderBuffer",
        AssetTask::Mesh(_) => "Mesh",
        AssetTask::Texture(_) => "Texture",
        AssetTask::Texture3d(_) => "Texture3d",
        AssetTask::Cubemap(_) => "Cubemap",
    }
}

/// GPU handles shared across all [`step_asset_task`] invocations in one drain.
struct AssetUploadGpuContext<'a> {
    /// Device for resource creation and format capability queries.
    device: &'a Arc<wgpu::Device>,
    /// GPU adapter limits shared with mesh upload paths.
    gpu_limits: &'a Arc<GpuLimits>,
    /// Queue for [`wgpu::Queue::write_texture`] / [`wgpu::Queue::write_buffer`] uploads.
    queue: &'a Arc<wgpu::Queue>,
    /// Shared GPU queue access gate for [`wgpu::Queue::write_texture`]; see
    /// [`crate::gpu::GpuQueueAccessGate`].
    gpu_queue_access_gate: &'a crate::gpu::GpuQueueAccessGate,
}

fn step_asset_task(
    asset: &mut AssetTransferQueue,
    materials: &mut MaterialSystem,
    gpu: Option<&AssetUploadGpuContext<'_>>,
    shm: &mut SharedMemoryAccessor,
    ipc: &mut Option<&mut DualQueueIpc>,
    task: &mut AssetTask,
) -> StepResult {
    profiling::scope!("asset::upload", asset_task_kind_tag(task));
    match task {
        AssetTask::MaterialUpdate(batch) => step_material_update_task(materials, shm, ipc, batch),
        AssetTask::ShaderRoute(route) => step_shader_route_task(materials, ipc, route),
        AssetTask::PointRenderBuffer(upload) => step_point_render_buffer_task(asset, ipc, upload),
        AssetTask::TrailRenderBuffer(upload) => step_trail_render_buffer_task(asset, ipc, upload),
        AssetTask::Mesh(task) => step_mesh_upload_task(asset, gpu, shm, ipc, task),
        AssetTask::Texture(task) => step_texture_upload_task(asset, gpu, shm, ipc, task),
        AssetTask::Texture3d(task) => step_texture3d_upload_task(asset, gpu, shm, ipc, task),
        AssetTask::Cubemap(task) => step_cubemap_upload_task(asset, gpu, shm, ipc, task),
    }
}

fn step_material_update_task(
    materials: &mut MaterialSystem,
    shm: &mut SharedMemoryAccessor,
    ipc: &mut Option<&mut DualQueueIpc>,
    batch: &mut MaterialsUpdateBatch,
) -> StepResult {
    let batch = std::mem::take(batch);
    if let Some(ipc) = ipc.as_deref_mut() {
        materials.apply_materials_update_batch(batch, shm, ipc);
    } else {
        logger::warn!(
            "materials update batch {}: IPC unavailable during integration; applying without ack",
            batch.update_batch_id
        );
        materials.apply_materials_update_batch_no_ack(batch, shm);
    }
    StepResult::Done
}

fn step_shader_route_task(
    materials: &mut MaterialSystem,
    ipc: &mut Option<&mut DualQueueIpc>,
    route: &mut ShaderRouteTask,
) -> StepResult {
    let shader_asset_name = route.shader_asset_name.take();
    materials.register_shader_route(
        route.asset_id,
        route.pipeline.clone(),
        shader_asset_name,
        route.shader_variant_bits,
    );
    if let Some(ipc) = ipc.as_deref_mut() {
        let _ =
            ipc.send_background_reliable(RendererCommand::ShaderUploadResult(ShaderUploadResult {
                asset_id: route.asset_id,
                instance_changed: true,
            }));
    }
    StepResult::Done
}

fn step_point_render_buffer_task(
    asset: &mut AssetTransferQueue,
    ipc: &mut Option<&mut DualQueueIpc>,
    upload: &mut PointRenderBufferUpload,
) -> StepResult {
    let upload = std::mem::take(upload);
    let asset_id = upload.asset_id;
    let count = upload.count;
    asset
        .catalogs
        .point_render_buffer_uploads
        .insert(asset_id, upload);
    if let Some(ipc) = ipc.as_deref_mut() {
        let _ = ipc.send_background_reliable(RendererCommand::PointRenderBufferConsumed(
            PointRenderBufferConsumed { asset_id },
        ));
    }
    logger::debug!("point render buffer {asset_id}: consumed placeholder upload count={count}");
    StepResult::Done
}

fn step_trail_render_buffer_task(
    asset: &mut AssetTransferQueue,
    ipc: &mut Option<&mut DualQueueIpc>,
    upload: &mut TrailRenderBufferUpload,
) -> StepResult {
    let upload = std::mem::take(upload);
    let asset_id = upload.asset_id;
    let trails_count = upload.trails_count;
    let trail_point_count = upload.trail_point_count;
    asset
        .catalogs
        .trail_render_buffer_uploads
        .insert(asset_id, upload);
    if let Some(ipc) = ipc.as_deref_mut() {
        let _ = ipc.send_background_reliable(RendererCommand::TrailRenderBufferConsumed(
            TrailRenderBufferConsumed { asset_id },
        ));
    }
    logger::debug!(
        "trail render buffer {asset_id}: consumed placeholder upload trails={trails_count} points_per_trail={trail_point_count}"
    );
    StepResult::Done
}

fn step_mesh_upload_task(
    asset: &mut AssetTransferQueue,
    gpu: Option<&AssetUploadGpuContext<'_>>,
    shm: &mut SharedMemoryAccessor,
    ipc: &mut Option<&mut DualQueueIpc>,
    task: &mut MeshUploadTask,
) -> StepResult {
    let Some(gpu) = gpu else {
        return StepResult::YieldBackground;
    };
    task.step(asset, gpu.device, gpu.gpu_limits, gpu.queue, shm, ipc)
}

fn step_texture_upload_task(
    asset: &mut AssetTransferQueue,
    gpu: Option<&AssetUploadGpuContext<'_>>,
    shm: &mut SharedMemoryAccessor,
    ipc: &mut Option<&mut DualQueueIpc>,
    task: &mut TextureUploadTask,
) -> StepResult {
    let Some(gpu) = gpu else {
        return StepResult::YieldBackground;
    };
    task.step(
        asset,
        gpu.device,
        gpu.queue.as_ref(),
        gpu.gpu_queue_access_gate,
        shm,
        ipc,
    )
}

fn step_texture3d_upload_task(
    asset: &mut AssetTransferQueue,
    gpu: Option<&AssetUploadGpuContext<'_>>,
    shm: &mut SharedMemoryAccessor,
    ipc: &mut Option<&mut DualQueueIpc>,
    task: &mut Texture3dUploadTask,
) -> StepResult {
    let Some(gpu) = gpu else {
        return StepResult::YieldBackground;
    };
    task.step(
        asset,
        gpu.device,
        gpu.queue.as_ref(),
        gpu.gpu_queue_access_gate,
        shm,
        ipc,
    )
}

fn step_cubemap_upload_task(
    asset: &mut AssetTransferQueue,
    gpu: Option<&AssetUploadGpuContext<'_>>,
    shm: &mut SharedMemoryAccessor,
    ipc: &mut Option<&mut DualQueueIpc>,
    task: &mut CubemapUploadTask,
) -> StepResult {
    let Some(gpu) = gpu else {
        return StepResult::YieldBackground;
    };
    task.step(
        asset,
        gpu.device,
        gpu.queue.as_ref(),
        gpu.gpu_queue_access_gate,
        shm,
        ipc,
    )
}

/// Returns the emergency ceiling for high-priority tasks in a bounded drain.
fn high_priority_emergency_deadline(start: Instant, normal_deadline: Instant) -> Instant {
    let normal_budget = match normal_deadline.checked_duration_since(start) {
        Some(duration) => duration,
        None => Duration::ZERO,
    };
    let emergency_budget = normal_budget.max(MIN_HIGH_PRIORITY_EMERGENCY_BUDGET);
    let base_deadline = normal_deadline.max(start);
    match base_deadline.checked_add(emergency_budget) {
        Some(deadline) => deadline,
        None => base_deadline,
    }
}

/// Emits current asset integration queue pressure to the profiler.
fn plot_asset_integrator_backlog(
    asset: &AssetTransferQueue,
    high_priority_budget_exhausted: bool,
    normal_priority_budget_exhausted: bool,
) {
    plot_asset_integration(AssetIntegrationProfileSample {
        high_priority_queued: asset.integrator.high_priority.len(),
        normal_priority_queued: asset.integrator.normal_priority.len(),
        high_priority_budget_exhausted,
        normal_priority_budget_exhausted,
    });
}

/// Drains urgent upload tasks until empty, background-yielded, or the emergency ceiling is hit.
fn drain_high_priority_asset_tasks(
    asset: &mut AssetTransferQueue,
    materials: &mut MaterialSystem,
    gpu: Option<&AssetUploadGpuContext<'_>>,
    shm: &mut SharedMemoryAccessor,
    ipc: &mut Option<&mut DualQueueIpc>,
    high_priority_deadline: Instant,
) -> LaneDrainOutcome {
    profiling::scope!("asset::high_priority_drain");
    drain_lane(
        asset,
        materials,
        gpu,
        shm,
        ipc,
        high_priority_deadline,
        AssetTaskLane::HighPriority,
    )
}

/// Drains normal upload tasks until empty, background-yielded, or the frame budget is hit.
fn drain_normal_priority_asset_tasks(
    asset: &mut AssetTransferQueue,
    materials: &mut MaterialSystem,
    gpu: Option<&AssetUploadGpuContext<'_>>,
    shm: &mut SharedMemoryAccessor,
    ipc: &mut Option<&mut DualQueueIpc>,
    normal_deadline: Instant,
) -> LaneDrainOutcome {
    profiling::scope!("asset::normal_priority_drain");
    drain_lane(
        asset,
        materials,
        gpu,
        shm,
        ipc,
        normal_deadline,
        AssetTaskLane::NormalPriority,
    )
}

/// Drains renderer-main-thread tasks until empty.
fn drain_main_asset_tasks(
    asset: &mut AssetTransferQueue,
    materials: &mut MaterialSystem,
    gpu: Option<&AssetUploadGpuContext<'_>>,
    shm: &mut SharedMemoryAccessor,
    ipc: &mut Option<&mut DualQueueIpc>,
) -> LaneDrainOutcome {
    profiling::scope!("asset::main_drain");
    drain_lane(
        asset,
        materials,
        gpu,
        shm,
        ipc,
        Instant::now() + Duration::from_secs(3600),
        AssetTaskLane::Main,
    )
}

/// Drains wgpu-native render-lane tasks until empty or the frame budget is hit.
fn drain_render_asset_tasks(
    asset: &mut AssetTransferQueue,
    materials: &mut MaterialSystem,
    gpu: Option<&AssetUploadGpuContext<'_>>,
    shm: &mut SharedMemoryAccessor,
    ipc: &mut Option<&mut DualQueueIpc>,
    render_deadline: Instant,
) -> LaneDrainOutcome {
    profiling::scope!("asset::render_drain");
    drain_lane(
        asset,
        materials,
        gpu,
        shm,
        ipc,
        render_deadline,
        AssetTaskLane::Render,
    )
}

/// Drains particle/dynamic-buffer tasks until empty or the particle budget is hit.
fn drain_particle_asset_tasks(
    asset: &mut AssetTransferQueue,
    materials: &mut MaterialSystem,
    gpu: Option<&AssetUploadGpuContext<'_>>,
    shm: &mut SharedMemoryAccessor,
    ipc: &mut Option<&mut DualQueueIpc>,
    particle_deadline: Instant,
) -> LaneDrainOutcome {
    profiling::scope!("asset::particle_drain");
    drain_lane(
        asset,
        materials,
        gpu,
        shm,
        ipc,
        particle_deadline,
        AssetTaskLane::Particle,
    )
}

/// Iteration cadence between [`Instant::now`] deadline polls in [`drain_priority_lane`].
///
/// `Instant::now` is a syscall on Windows (`QueryPerformanceCounter`) and on Linux variants where
/// `clock_gettime(CLOCK_MONOTONIC)` is not vDSO-accelerated. Tasks that complete in well under a
/// microsecond (texture mip step, zero-byte mesh layout fingerprint) make the per-iteration poll
/// dominate the loop. Polling every fourth iteration cuts the syscall rate ~4x while keeping the
/// deadline-overshoot bounded by `~3 * task_step_cost` plus the cost of one task spawn.
const DEADLINE_POLL_STRIDE: u32 = 4;

/// Result of draining one scheduler lane.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct LaneDrainOutcome {
    /// Whether the lane still had queued work when the drain ended.
    pending: bool,
    /// Queue steps processed by this drain.
    processed: u32,
}

/// Per-lane processed-step counters for one full drain.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ProcessedLaneCounts {
    /// Main-lane queue steps.
    main: u32,
    /// High-priority queue steps.
    high_priority: u32,
    /// Normal-priority queue steps.
    normal_priority: u32,
    /// Render-lane queue steps.
    render: u32,
    /// Particle-lane queue steps.
    particle: u32,
}

impl ProcessedLaneCounts {
    /// Total processed queue steps.
    fn total(self) -> u32 {
        self.main
            .saturating_add(self.high_priority)
            .saturating_add(self.normal_priority)
            .saturating_add(self.render)
            .saturating_add(self.particle)
    }
}

/// Shared inner loop for scheduler lane drains.
///
/// Returns `true` when the named lane still has work pending after the call (the deadline
/// expired before the queue drained or every yielded task tail-rotated without progress).
/// The two outer functions remain as thin wrappers so tracy zone names stay distinct between
/// priority lanes.
fn drain_lane(
    asset: &mut AssetTransferQueue,
    materials: &mut MaterialSystem,
    gpu: Option<&AssetUploadGpuContext<'_>>,
    shm: &mut SharedMemoryAccessor,
    ipc: &mut Option<&mut DualQueueIpc>,
    deadline: Instant,
    lane: AssetTaskLane,
) -> LaneDrainOutcome {
    let mut yielded: usize = 0;
    let mut iter_count: u32 = 0;
    let mut processed: u32 = 0;
    loop {
        // Coarse deadline check: every `DEADLINE_POLL_STRIDE` iterations rather than
        // every iteration, so cheap task steps (e.g. texture mip progression) do not pay
        // the `Instant::now` syscall on every pop.
        if iter_count.is_multiple_of(DEADLINE_POLL_STRIDE) && Instant::now() >= deadline {
            return LaneDrainOutcome {
                pending: !asset.integrator.lane_is_empty(lane),
                processed,
            };
        }
        iter_count = iter_count.wrapping_add(1);
        let task_opt = asset.integrator.pop_front_lane(lane);
        let Some(mut task) = task_opt else {
            return LaneDrainOutcome {
                pending: false,
                processed,
            };
        };
        let step_result = step_asset_task(asset, materials, gpu, shm, ipc, &mut task);
        processed = processed.saturating_add(1);
        match step_result {
            StepResult::Continue => {
                asset.integrator.push_front_lane(task, lane);
                yielded = 0;
            }
            StepResult::YieldBackground => {
                asset.integrator.push_back_lane(task, lane);
                let lane_len = asset.integrator.lane_len(lane);
                yielded += 1;
                if yielded >= lane_len {
                    return LaneDrainOutcome {
                        pending: false,
                        processed,
                    };
                }
            }
            StepResult::Done => {
                yielded = 0;
            }
        }
    }
}

/// Polls video texture players after upload integration.
///
/// Also samples each player's clock error against the host's last-applied playback request and
/// records the latest result so the runtime can flush it into the next
/// [`crate::shared::FrameStartData`].
fn poll_video_texture_events(asset: &mut AssetTransferQueue, ipc: &mut Option<&mut DualQueueIpc>) {
    profiling::scope!("asset::video_texture_poll_events");
    let mut video_textures = std::mem::take(&mut asset.video.video_players);
    {
        profiling::scope!("video::sample_clock_errors");
        for player in video_textures.values_mut() {
            player.process_events(asset, ipc);
            if let Some(state) = player.sample_clock_error() {
                asset.video.record_pending_clock_error(state);
            }
        }
    }
    asset.video.video_players = video_textures;
}

/// Runs integration steps: high-priority tasks get an emergency ceiling, then normal-priority tasks
/// run until `normal_deadline`.
pub fn drain_asset_tasks(
    asset: &mut AssetTransferQueue,
    materials: &mut MaterialSystem,
    shm: &mut SharedMemoryAccessor,
    ipc: &mut Option<&mut DualQueueIpc>,
    normal_deadline: Instant,
    particle_deadline: Instant,
) -> AssetIntegrationDrainSummary {
    profiling::scope!("asset::drain_tasks");
    let drain_start = Instant::now();
    let dropped_delayed_removals = asset.integrator.process_delayed_removals();
    if dropped_delayed_removals > 0 {
        logger::trace!(
            "asset integrator: dropped {} delayed GPU resource removal(s)",
            dropped_delayed_removals
        );
    }
    let summary = AssetIntegrationDrainSummary::start(asset);
    let high_priority_deadline = high_priority_emergency_deadline(drain_start, normal_deadline);
    let gpu_handles = match (
        asset.gpu.gpu_device.clone(),
        asset.gpu.gpu_limits.clone(),
        asset.gpu.gpu_queue.clone(),
        asset.gpu.gpu_queue_access_gate.clone(),
    ) {
        (Some(device), Some(gpu_limits), Some(queue_arc), Some(gate)) => {
            Some((device, gpu_limits, queue_arc, gate))
        }
        _ => None,
    };
    let gpu =
        gpu_handles.as_ref().map(
            |(device, gpu_limits, queue_arc, gate)| AssetUploadGpuContext {
                device,
                gpu_limits,
                queue: queue_arc,
                gpu_queue_access_gate: gate,
            },
        );
    let gpu = gpu.as_ref();

    let main_outcome = drain_main_asset_tasks(asset, materials, gpu, shm, ipc);

    let render_outcome = drain_render_asset_tasks(asset, materials, gpu, shm, ipc, normal_deadline);
    if render_outcome.pending {
        logger::trace!(
            "asset integrator: render-lane budget exhausted with {} task(s) pending",
            asset.integrator.render.len()
        );
    }

    let high_priority_outcome =
        drain_high_priority_asset_tasks(asset, materials, gpu, shm, ipc, high_priority_deadline);
    if high_priority_outcome.pending {
        logger::trace!(
            "asset integrator: high-priority emergency budget exhausted with {} task(s) pending",
            asset.integrator.high_priority.len()
        );
    }

    let normal_priority_outcome =
        drain_normal_priority_asset_tasks(asset, materials, gpu, shm, ipc, normal_deadline);
    if normal_priority_outcome.pending {
        // Tasks pending after wall-clock deadline. Not necessarily a bug -- asset arrival can
        // outpace integration on busy frames -- but persistent backlog growth indicates the
        // budget is too tight or a task is stuck. Per-frame at trace level so it does not
        // spam the default-level log.
        logger::trace!(
            "asset integrator: normal-priority budget exhausted with {} task(s) pending",
            asset.integrator.normal_priority.len()
        );
    }

    let integration_elapsed = drain_start.elapsed();
    let particle_start = Instant::now();
    let particle_outcome =
        drain_particle_asset_tasks(asset, materials, gpu, shm, ipc, particle_deadline);
    let particle_elapsed = particle_start.elapsed();
    if particle_outcome.pending {
        logger::trace!(
            "asset integrator: particle budget exhausted with {} task(s) pending",
            asset.integrator.particle.len()
        );
    }

    plot_asset_integrator_backlog(
        asset,
        high_priority_outcome.pending,
        normal_priority_outcome.pending,
    );

    poll_video_texture_events(asset, ipc);
    let processed = ProcessedLaneCounts {
        main: main_outcome.processed,
        high_priority: high_priority_outcome.processed,
        normal_priority: normal_priority_outcome.processed,
        render: render_outcome.processed,
        particle: particle_outcome.processed,
    };
    summary.finish(
        asset,
        DrainFinishState {
            gpu_ready: gpu.is_some(),
            budgets: BudgetExhaustion {
                high_priority: high_priority_outcome.pending,
                normal_priority: normal_priority_outcome.pending,
                render: render_outcome.pending,
                particle: particle_outcome.pending,
            },
            processed,
            particle_elapsed,
            elapsed: integration_elapsed,
        },
    )
}

/// Drains all queued tasks without a time limit (used on GPU attach before first frame).
pub fn drain_asset_tasks_unbounded(
    asset: &mut AssetTransferQueue,
    materials: &mut MaterialSystem,
    shm: &mut SharedMemoryAccessor,
    ipc: &mut Option<&mut DualQueueIpc>,
) {
    let far_future = Instant::now() + Duration::from_secs(3600);
    let _ = drain_asset_tasks(asset, materials, shm, ipc, far_future, far_future);
}

#[cfg(test)]
mod tests;
