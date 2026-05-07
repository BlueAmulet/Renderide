use std::collections::VecDeque;

use glam::Vec3;
use hashbrown::{HashMap, HashSet};

use crate::gpu::GpuContext;
use crate::ipc::SharedMemoryAccessor;
use crate::profiling;
use crate::scene::SceneCoordinator;
use crate::shared::{ComputeResult, FrameSubmitData, ReflectionProbeSH2Tasks, RenderSH2};
use crate::skybox::params::{SkyboxEvaluatorParams, SkyboxParamMode};
use crate::skybox::specular::SkyboxIblSource;

use super::projection_pipeline::{
    ProjectionBinding, ProjectionPipeline, ProjectionPipelineBuildOutcome, ProjectionPipelineKind,
    encode_projection_job, spawn_projection_pipeline_build,
};
use super::readback_jobs::{Sh2ReadbackJobs, SubmittedGpuSh2Job};
use super::sh2_math::{constant_color_sh2, f32x4_bits};
use super::source_resolution::{Sh2ResolvedSource, resolve_task_source};
use super::task_rows::{
    TaskAnswer, TaskHeader, debug_assert_no_scheduled_rows, read_task_header, task_stride,
    write_task_answer,
};

/// Skybox projection sample resolution per cube face.
pub(super) const DEFAULT_SAMPLE_SIZE: u32 = crate::skybox::params::DEFAULT_SKYBOX_SAMPLE_SIZE;
/// Maximum pending GPU jobs kept alive at once.
const MAX_IN_FLIGHT_JOBS: usize = 6;
/// Number of renderer ticks before a pending GPU readback is treated as failed.
pub(super) const MAX_PENDING_JOB_AGE_FRAMES: u32 = 120;
/// Bytes copied back from the compute output buffer.
pub(super) const SH2_OUTPUT_BYTES: u64 = (9 * 16) as u64;
/// Uniform payload shared by SH2 projection compute kernels.
pub(super) type Sh2ProjectParams = SkyboxEvaluatorParams;

/// Hashable `Projection360` equirectangular sampling state used by SH2 cache keys.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(crate) struct Projection360EquirectKey {
    /// `_FOV` bit pattern.
    fov_bits: [u32; 4],
    /// `_MainTex_ST` bit pattern.
    main_tex_st_bits: [u32; 4],
    /// `_MainTex_StorageVInverted` bit pattern.
    storage_v_inverted_bits: u32,
}

impl Projection360EquirectKey {
    /// Builds a cache-key fragment from the packed projection parameters.
    pub(super) fn from_params(params: &Sh2ProjectParams) -> Self {
        Self {
            fov_bits: f32x4_bits(params.color0),
            main_tex_st_bits: f32x4_bits(params.color1),
            storage_v_inverted_bits: params.scalars[0].to_bits(),
        }
    }
}

/// Parameter-only sky evaluator mode used by `sh2_project_sky_params`.
pub(super) type SkyParamMode = SkyboxParamMode;

/// Hashable description of the source projected into SH2.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub(crate) enum Sh2SourceKey {
    /// Analytic constant-color source.
    ConstantColor {
        /// Render-space id that owns the probe.
        render_space_id: i32,
        /// RGBA color bit pattern.
        color_bits: [u32; 4],
    },
    /// Resident cubemap source.
    Cubemap {
        /// Render-space id that owns the probe.
        render_space_id: i32,
        /// Cubemap asset id.
        asset_id: i32,
        /// Face size.
        size: u32,
        /// Contiguous resident mip count.
        resident_mips: u32,
        /// Source cubemap content generation.
        content_generation: u64,
        /// Projection sample grid edge per cube face.
        sample_size: u32,
        /// Host material generation mixed into skybox sources.
        material_generation: u64,
    },
    /// Resident equirectangular texture source.
    EquirectTexture2D {
        /// Render-space id that owns the probe.
        render_space_id: i32,
        /// Texture asset id.
        asset_id: i32,
        /// Mip0 width.
        width: u32,
        /// Mip0 height.
        height: u32,
        /// Contiguous resident mip count.
        resident_mips: u32,
        /// Source texture content generation.
        content_generation: u64,
        /// Projection sample grid edge per cube face.
        sample_size: u32,
        /// Host material generation.
        material_generation: u64,
        /// Projection360 equirectangular sampling state.
        projection: Projection360EquirectKey,
    },
    /// Parameter-only sky material source.
    SkyParams {
        /// Render-space id that owns the probe.
        render_space_id: i32,
        /// Skybox material asset id.
        material_asset_id: i32,
        /// Host material generation.
        material_generation: u64,
        /// Projection sample grid edge per cube face.
        sample_size: u32,
        /// Shader route discriminator.
        route_hash: u64,
    },
}

/// GPU-projected source payload queued for scheduling.
#[derive(Clone, Debug)]
pub(super) enum GpuSh2Source {
    /// Cubemap sampled from the cubemap pool.
    Cubemap { asset_id: i32 },
    /// Equirectangular 2D texture sampled from the texture pool.
    EquirectTexture2D {
        /// Texture asset id.
        asset_id: i32,
        /// Projection360 sampling parameters.
        params: Box<Sh2ProjectParams>,
    },
    /// Parameter-only sky material evaluator.
    SkyParams { params: Box<Sh2ProjectParams> },
}

/// Nonblocking SH2 projection cache and GPU-job scheduler.
pub struct ReflectionProbeSh2System {
    /// Completed projection results keyed by source identity.
    completed: HashMap<Sh2SourceKey, RenderSH2>,
    /// In-flight GPU readback jobs keyed by source identity.
    readback_jobs: Sh2ReadbackJobs,
    /// Sources that failed recently.
    failed: HashSet<Sh2SourceKey>,
    /// Source payloads awaiting an in-flight slot.
    queued_sources: HashMap<Sh2SourceKey, GpuSh2Source>,
    /// FIFO ordering for [`Self::queued_sources`].
    queue_order: VecDeque<Sh2SourceKey>,
    /// Lazily-created cubemap pipeline.
    cubemap_pipeline: Option<ProjectionPipeline>,
    /// Lazily-created equirectangular 2D pipeline.
    equirect_pipeline: Option<ProjectionPipeline>,
    /// Lazily-created parameter sky pipeline.
    sky_params_pipeline: Option<ProjectionPipeline>,
    /// Source keys touched by the current task pass.
    touched_this_pass: HashSet<Sh2SourceKey>,
    /// Completed asynchronous projection-pipeline builds.
    pipeline_build_rx: crossbeam_channel::Receiver<ProjectionPipelineBuildOutcome>,
    /// Submission side of [`Self::pipeline_build_rx`].
    pipeline_build_tx: crossbeam_channel::Sender<ProjectionPipelineBuildOutcome>,
    /// Projection pipelines currently compiling on worker threads.
    pending_pipeline_builds: HashSet<ProjectionPipelineKind>,
    /// Projection pipeline kinds that failed to build.
    failed_pipeline_builds: HashSet<ProjectionPipelineKind>,
}

enum ScheduleSourceOutcome {
    Submitted(SubmittedGpuSh2Job),
    RetryLater(GpuSh2Source),
}

impl Default for ReflectionProbeSh2System {
    fn default() -> Self {
        Self::new()
    }
}

impl ReflectionProbeSh2System {
    /// Creates an empty SH2 system.
    pub fn new() -> Self {
        let (pipeline_build_tx, pipeline_build_rx) = crossbeam_channel::unbounded();
        Self {
            completed: HashMap::new(),
            readback_jobs: Sh2ReadbackJobs::new(),
            failed: HashSet::new(),
            queued_sources: HashMap::new(),
            queue_order: VecDeque::new(),
            cubemap_pipeline: None,
            equirect_pipeline: None,
            sky_params_pipeline: None,
            touched_this_pass: HashSet::new(),
            pipeline_build_rx,
            pipeline_build_tx,
            pending_pipeline_builds: HashSet::new(),
            failed_pipeline_builds: HashSet::new(),
        }
    }

    /// Answers every SH2 task row in a frame submit without blocking for GPU readback.
    pub fn answer_frame_submit_tasks(
        &mut self,
        shm: &mut SharedMemoryAccessor,
        scene: &SceneCoordinator,
        materials: &crate::materials::MaterialSystem,
        assets: &crate::backend::AssetTransferQueue,
        data: &FrameSubmitData,
    ) {
        profiling::scope!("reflection_probe_sh2::answer_frame_submit_tasks");
        self.touched_this_pass.clear();
        for update in &data.render_spaces {
            let Some(tasks) = update.reflection_probe_sh2_taks.as_ref() else {
                continue;
            };
            self.answer_task_buffer(shm, scene, materials, assets, update.id, tasks);
        }
        self.prune_untouched_failures();
    }

    /// Advances GPU callbacks, maps completed buffers, and schedules queued work.
    pub fn maintain_gpu_jobs(
        &mut self,
        gpu: &mut GpuContext,
        assets: &crate::backend::AssetTransferQueue,
    ) {
        profiling::scope!("reflection_probe_sh2::maintain_gpu_jobs");
        let _ = gpu.device().poll(wgpu::PollType::Poll);
        let outcomes = self.readback_jobs.maintain();
        for (key, sh) in outcomes.completed {
            self.failed.remove(&key);
            self.completed.insert(key, sh);
        }
        for (key, reason) in outcomes.failed {
            logger::warn!("reflection_probe_sh2: GPU SH2 readback failed for {key:?}: {reason:?}");
            self.failed.insert(key);
        }
        self.drain_completed_pipeline_builds();
        self.schedule_queued_sources(gpu, assets);
    }

    /// Ensures an SH2 projection exists for a renderer-owned reflection-probe IBL source.
    ///
    /// Returns [`Some`] only after the source has a completed CPU or GPU projection. GPU-backed
    /// sources are queued on cache misses and complete through [`Self::maintain_gpu_jobs`].
    pub(crate) fn ensure_ibl_source(
        &mut self,
        render_space_id: i32,
        source: &SkyboxIblSource,
    ) -> Option<RenderSH2> {
        let (key, source) = sh2_source_from_ibl_source(render_space_id, source);
        self.ensure_resolved_source(key, source)
    }

    fn ensure_resolved_source(
        &mut self,
        key: Sh2SourceKey,
        source: Sh2ResolvedSource,
    ) -> Option<RenderSH2> {
        self.touched_this_pass.insert(key.clone());
        if let Some(sh) = self.completed.get(&key) {
            return Some(*sh);
        }
        if self.readback_jobs.contains_key(&key) || self.failed.contains(&key) {
            return None;
        }
        match source {
            Sh2ResolvedSource::Cpu(sh) => {
                let sh = *sh;
                self.completed.insert(key, sh);
                Some(sh)
            }
            Sh2ResolvedSource::Gpu(gpu_source) => {
                self.queue_source(key, gpu_source);
                None
            }
            Sh2ResolvedSource::Postpone => None,
        }
    }

    /// Answers all rows in one shared-memory task descriptor.
    fn answer_task_buffer(
        &mut self,
        shm: &mut SharedMemoryAccessor,
        scene: &SceneCoordinator,
        materials: &crate::materials::MaterialSystem,
        assets: &crate::backend::AssetTransferQueue,
        render_space_id: i32,
        tasks: &ReflectionProbeSH2Tasks,
    ) {
        profiling::scope!("reflection_probe_sh2::answer_task_buffer");
        if tasks.tasks.length <= 0 {
            return;
        }

        let ok = shm.access_mut_bytes(&tasks.tasks, |bytes| {
            profiling::scope!("reflection_probe_sh2::task_buffer_scan");
            let mut offset = 0usize;
            while offset + task_stride() <= bytes.len() {
                let Some(task) = read_task_header(bytes, offset) else {
                    break;
                };
                if task.renderable_index < 0 {
                    break;
                }
                let answer = self.answer_for_task(scene, materials, assets, render_space_id, task);
                write_task_answer(bytes, offset, answer);
                offset += task_stride();
            }
            debug_assert_no_scheduled_rows(bytes);
        });

        if !ok {
            logger::warn!(
                "reflection_probe_sh2: could not write SH2 task results (shared memory buffer)"
            );
        }
    }

    /// Resolves one host task into an immediate answer.
    fn answer_for_task(
        &mut self,
        scene: &SceneCoordinator,
        materials: &crate::materials::MaterialSystem,
        assets: &crate::backend::AssetTransferQueue,
        render_space_id: i32,
        task: TaskHeader,
    ) -> TaskAnswer {
        let Some((key, source)) =
            resolve_task_source(scene, materials, assets, render_space_id, task)
        else {
            return TaskAnswer::status(ComputeResult::Failed);
        };

        let key_failed = self.failed.contains(&key);
        match self.ensure_resolved_source(key, source) {
            Some(sh) => TaskAnswer::computed(sh),
            None if key_failed => TaskAnswer::status(ComputeResult::Failed),
            None => TaskAnswer::status(ComputeResult::Postpone),
        }
    }

    /// Queues a source for later GPU scheduling.
    fn queue_source(&mut self, key: Sh2SourceKey, source: GpuSh2Source) {
        if self.queued_sources.contains_key(&key) {
            return;
        }
        self.queue_order.push_back(key.clone());
        self.queued_sources.insert(key, source);
    }

    /// Drops failed keys that are no longer present in host task rows.
    fn prune_untouched_failures(&mut self) {
        self.failed
            .retain(|key| self.touched_this_pass.contains(key));
    }

    /// Schedules queued sources until the in-flight cap is reached.
    fn schedule_queued_sources(
        &mut self,
        gpu: &mut GpuContext,
        assets: &crate::backend::AssetTransferQueue,
    ) {
        profiling::scope!("reflection_probe_sh2::schedule_queued_sources");
        let attempts = self.queue_order.len();
        for _ in 0..attempts {
            if self.readback_jobs.len() >= MAX_IN_FLIGHT_JOBS {
                break;
            }
            let Some(key) = self.queue_order.pop_front() else {
                break;
            };
            let Some(source) = self.queued_sources.remove(&key) else {
                continue;
            };
            if self.completed.contains_key(&key)
                || self.readback_jobs.contains_key(&key)
                || self.failed.contains(&key)
            {
                continue;
            }
            match self.schedule_source(gpu, assets, key.clone(), source) {
                Ok(ScheduleSourceOutcome::Submitted(job)) => {
                    self.readback_jobs.insert(key, job);
                }
                Ok(ScheduleSourceOutcome::RetryLater(source)) => {
                    self.queue_order.push_back(key.clone());
                    self.queued_sources.insert(key, source);
                }
                Err(e) => {
                    logger::warn!("reflection_probe_sh2: GPU SH2 schedule failed: {e}");
                    self.failed.insert(key);
                }
            }
        }
    }

    /// Encodes and submits one source projection.
    fn schedule_source(
        &mut self,
        gpu: &mut GpuContext,
        assets: &crate::backend::AssetTransferQueue,
        key: Sh2SourceKey,
        source: GpuSh2Source,
    ) -> Result<ScheduleSourceOutcome, String> {
        profiling::scope!("reflection_probe_sh2::schedule_source");
        let pipeline_kind = projection_pipeline_kind(&source);
        if !self.ensure_projection_pipeline_ready(gpu.device(), pipeline_kind)? {
            return Ok(ScheduleSourceOutcome::RetryLater(source));
        }
        let pipeline = self.projection_pipeline(pipeline_kind).ok_or_else(|| {
            format!(
                "projection pipeline {} missing after build",
                pipeline_kind.stem()
            )
        })?;
        match source {
            GpuSh2Source::Cubemap { asset_id } => {
                profiling::scope!("reflection_probe_sh2::schedule_cubemap");
                let tex = assets
                    .cubemap_pool()
                    .get(asset_id)
                    .filter(|t| t.mip_levels_resident > 0)
                    .ok_or_else(|| format!("cubemap {asset_id} not resident"))?;
                let sampler = gpu.device().create_sampler(&wgpu::SamplerDescriptor {
                    label: Some("SH2 cubemap sampler"),
                    address_mode_u: wgpu::AddressMode::ClampToEdge,
                    address_mode_v: wgpu::AddressMode::ClampToEdge,
                    address_mode_w: wgpu::AddressMode::ClampToEdge,
                    mag_filter: wgpu::FilterMode::Linear,
                    min_filter: wgpu::FilterMode::Linear,
                    mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                    ..Default::default()
                });
                let view = tex.view.clone();
                let submit_done_tx = self.readback_jobs.submit_done_sender();
                Ok(ScheduleSourceOutcome::Submitted(encode_projection_job(
                    gpu,
                    key,
                    pipeline,
                    &[
                        ProjectionBinding::TextureView(view.as_ref()),
                        ProjectionBinding::Sampler(&sampler),
                    ],
                    &Sh2ProjectParams::empty(SkyParamMode::Procedural),
                    &submit_done_tx,
                    "reflection_probe_sh2::project_cubemap",
                )?))
            }
            GpuSh2Source::EquirectTexture2D { asset_id, params } => {
                profiling::scope!("reflection_probe_sh2::schedule_equirect");
                let tex = assets
                    .texture_pool()
                    .get(asset_id)
                    .filter(|t| t.mip_levels_resident > 0)
                    .ok_or_else(|| format!("texture2d {asset_id} not resident"))?;
                let sampler = gpu.device().create_sampler(&wgpu::SamplerDescriptor {
                    label: Some("SH2 equirect sampler"),
                    address_mode_u: wgpu::AddressMode::Repeat,
                    address_mode_v: wgpu::AddressMode::ClampToEdge,
                    mag_filter: wgpu::FilterMode::Linear,
                    min_filter: wgpu::FilterMode::Linear,
                    mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                    ..Default::default()
                });
                let view = tex.view.clone();
                let submit_done_tx = self.readback_jobs.submit_done_sender();
                Ok(ScheduleSourceOutcome::Submitted(encode_projection_job(
                    gpu,
                    key,
                    pipeline,
                    &[
                        ProjectionBinding::TextureView(view.as_ref()),
                        ProjectionBinding::Sampler(&sampler),
                    ],
                    params.as_ref(),
                    &submit_done_tx,
                    "reflection_probe_sh2::project_equirect",
                )?))
            }
            GpuSh2Source::SkyParams { params } => {
                profiling::scope!("reflection_probe_sh2::schedule_sky_params");
                let submit_done_tx = self.readback_jobs.submit_done_sender();
                Ok(ScheduleSourceOutcome::Submitted(encode_projection_job(
                    gpu,
                    key,
                    pipeline,
                    &[],
                    params.as_ref(),
                    &submit_done_tx,
                    "reflection_probe_sh2::project_sky_params",
                )?))
            }
        }
    }

    fn drain_completed_pipeline_builds(&mut self) {
        while let Ok(outcome) = self.pipeline_build_rx.try_recv() {
            self.pending_pipeline_builds.remove(&outcome.kind);
            match outcome.result {
                Ok(pipeline) => {
                    *self.projection_pipeline_slot_mut(outcome.kind) = Some(pipeline);
                }
                Err(e) => {
                    logger::warn!(
                        "reflection_probe_sh2: projection pipeline build failed for {}: {e}",
                        outcome.kind.stem()
                    );
                    self.failed_pipeline_builds.insert(outcome.kind);
                }
            }
        }
    }

    fn ensure_projection_pipeline_ready(
        &mut self,
        device: &std::sync::Arc<wgpu::Device>,
        kind: ProjectionPipelineKind,
    ) -> Result<bool, String> {
        if self.projection_pipeline(kind).is_some() {
            return Ok(true);
        }
        if self.failed_pipeline_builds.contains(&kind) {
            return Err(format!(
                "projection pipeline {} previously failed to build",
                kind.stem()
            ));
        }
        if self.pending_pipeline_builds.contains(&kind) {
            return Ok(false);
        }
        spawn_projection_pipeline_build(kind, device.clone(), self.pipeline_build_tx.clone())?;
        self.pending_pipeline_builds.insert(kind);
        Ok(false)
    }

    fn projection_pipeline(&self, kind: ProjectionPipelineKind) -> Option<&ProjectionPipeline> {
        match kind {
            ProjectionPipelineKind::Cubemap => self.cubemap_pipeline.as_ref(),
            ProjectionPipelineKind::Equirect => self.equirect_pipeline.as_ref(),
            ProjectionPipelineKind::SkyParams => self.sky_params_pipeline.as_ref(),
        }
    }

    fn projection_pipeline_slot_mut(
        &mut self,
        kind: ProjectionPipelineKind,
    ) -> &mut Option<ProjectionPipeline> {
        match kind {
            ProjectionPipelineKind::Cubemap => &mut self.cubemap_pipeline,
            ProjectionPipelineKind::Equirect => &mut self.equirect_pipeline,
            ProjectionPipelineKind::SkyParams => &mut self.sky_params_pipeline,
        }
    }
}

fn projection_pipeline_kind(source: &GpuSh2Source) -> ProjectionPipelineKind {
    match source {
        GpuSh2Source::Cubemap { .. } => ProjectionPipelineKind::Cubemap,
        GpuSh2Source::EquirectTexture2D { .. } => ProjectionPipelineKind::Equirect,
        GpuSh2Source::SkyParams { .. } => ProjectionPipelineKind::SkyParams,
    }
}

fn sh2_source_from_ibl_source(
    render_space_id: i32,
    source: &SkyboxIblSource,
) -> (Sh2SourceKey, Sh2ResolvedSource) {
    match source {
        SkyboxIblSource::Analytic(src) => {
            let mut params = src.params;
            params.sample_size = DEFAULT_SAMPLE_SIZE;
            (
                Sh2SourceKey::SkyParams {
                    render_space_id,
                    material_asset_id: src.material_asset_id,
                    material_generation: src.material_generation,
                    sample_size: DEFAULT_SAMPLE_SIZE,
                    route_hash: src.route_hash,
                },
                Sh2ResolvedSource::Gpu(GpuSh2Source::SkyParams {
                    params: Box::new(params),
                }),
            )
        }
        SkyboxIblSource::Cubemap(src) => (
            Sh2SourceKey::Cubemap {
                render_space_id,
                asset_id: src.asset_id,
                size: src.face_size,
                resident_mips: src.mip_levels_resident,
                content_generation: src.content_generation,
                sample_size: DEFAULT_SAMPLE_SIZE,
                material_generation: 0,
            },
            Sh2ResolvedSource::Gpu(GpuSh2Source::Cubemap {
                asset_id: src.asset_id,
            }),
        ),
        SkyboxIblSource::Equirect(src) => {
            let mut params = Sh2ProjectParams::empty(SkyParamMode::Procedural);
            params.color0 = src.equirect_fov;
            params.color1 = src.equirect_st;
            params.scalars[0] = if src.storage_v_inverted { 1.0 } else { 0.0 };
            (
                Sh2SourceKey::EquirectTexture2D {
                    render_space_id,
                    asset_id: src.asset_id,
                    width: src.width,
                    height: src.height,
                    resident_mips: src.mip_levels_resident,
                    content_generation: src.content_generation,
                    sample_size: DEFAULT_SAMPLE_SIZE,
                    material_generation: 0,
                    projection: Projection360EquirectKey::from_params(&params),
                },
                Sh2ResolvedSource::Gpu(GpuSh2Source::EquirectTexture2D {
                    asset_id: src.asset_id,
                    params: Box::new(params),
                }),
            )
        }
        SkyboxIblSource::SolidColor(src) => (
            Sh2SourceKey::ConstantColor {
                render_space_id,
                color_bits: f32x4_bits(src.color),
            },
            Sh2ResolvedSource::Cpu(Box::new(constant_color_sh2(Vec3::new(
                src.color[0],
                src.color[1],
                src.color[2],
            )))),
        ),
    }
}
