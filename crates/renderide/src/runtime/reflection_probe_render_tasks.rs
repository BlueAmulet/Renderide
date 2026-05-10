//! Host reflection-probe cubemap bake tasks, offscreen face rendering, readback, and IPC results.

use std::sync::Arc;

use glam::{Mat4, Vec3};
use hashbrown::HashSet;

use crate::backend::RenderBackend;
use crate::camera::{
    CameraClipPlanes, CameraPose, CameraProjectionKind, EyeView, HostCameraFrame, ViewId, Viewport,
};
use crate::gpu::{CUBEMAP_ARRAY_LAYERS, GpuContext};
use crate::ipc::{DualQueueIpc, SharedMemoryAccessor};
use crate::reflection_probes::specular::{
    RuntimeReflectionProbeCapture, RuntimeReflectionProbeCaptureKey,
};
use crate::render_graph::{FrameViewClear, GraphExecuteError, ViewPostProcessing};
use crate::scene::{
    ReflectionProbeOnChangesRenderRequest, RenderSpaceId, SceneCoordinator,
    changed_probe_completion, reflection_probe_skybox_only,
};
use crate::shared::{
    CameraClearMode, FrameSubmitData, ReflectionProbeClear, ReflectionProbeRenderResult,
    ReflectionProbeRenderTask, ReflectionProbeState, ReflectionProbeTimeSlicingMode,
    RendererCommand, RenderingContext,
};
use crate::skybox::ibl_cache::{SkyboxIblConvolver, mip_levels_for_edge};
use crate::world_mesh::{CameraTransformDrawFilter, WorldMeshDrawCollectParallelism};

mod readback;

use readback::{
    compute_probe_readback_layout, readback_reflection_probe_cube, write_probe_task_result,
    zero_probe_task_result,
};

use super::RendererRuntime;
use super::frame_extract::{ExtractedFrame, PreparedViews};
use super::frame_view_plan::{FrameViewPlan, FrameViewPlanTarget, OffscreenRtHandles};
use super::tick_state::QueuedReflectionProbeRenderTask;

const CUBE_FACE_COUNT: usize = CUBEMAP_ARRAY_LAYERS as usize;
const RGBA16F_BYTES_PER_PIXEL: usize = 8;
const RGBA8_BYTES_PER_PIXEL: usize = 4;
const PROBE_TASK_COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ProbeOutputFormat {
    Rgba8,
    Rgba16Float,
}

impl ProbeOutputFormat {
    const fn from_hdr(hdr: bool) -> Self {
        if hdr { Self::Rgba16Float } else { Self::Rgba8 }
    }

    const fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Rgba8 => RGBA8_BYTES_PER_PIXEL,
            Self::Rgba16Float => RGBA16F_BYTES_PER_PIXEL,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ProbeCubeFace {
    PosX,
    NegX,
    PosY,
    NegY,
    PosZ,
    NegZ,
}

impl ProbeCubeFace {
    const ALL: [Self; CUBE_FACE_COUNT] = [
        Self::PosX,
        Self::NegX,
        Self::PosY,
        Self::NegY,
        Self::PosZ,
        Self::NegZ,
    ];
    const ALL_MASK: u8 = 0b0011_1111;

    const fn index(self) -> usize {
        match self {
            Self::PosX => 0,
            Self::NegX => 1,
            Self::PosY => 2,
            Self::NegY => 3,
            Self::PosZ => 4,
            Self::NegZ => 5,
        }
    }

    const fn layer(self) -> u32 {
        self.index() as u32
    }

    const fn view_id_face_index(self) -> u8 {
        self.index() as u8
    }

    const fn bit(self) -> u8 {
        1 << self.index()
    }

    const fn basis(self) -> ProbeFaceBasis {
        match self {
            Self::PosX => ProbeFaceBasis {
                forward: Vec3::X,
                right: Vec3::NEG_Z,
                up: Vec3::Y,
            },
            Self::NegX => ProbeFaceBasis {
                forward: Vec3::NEG_X,
                right: Vec3::Z,
                up: Vec3::Y,
            },
            Self::PosY => ProbeFaceBasis {
                forward: Vec3::Y,
                right: Vec3::X,
                up: Vec3::NEG_Z,
            },
            Self::NegY => ProbeFaceBasis {
                forward: Vec3::NEG_Y,
                right: Vec3::X,
                up: Vec3::Z,
            },
            Self::PosZ => ProbeFaceBasis {
                forward: Vec3::Z,
                right: Vec3::X,
                up: Vec3::Y,
            },
            Self::NegZ => ProbeFaceBasis {
                forward: Vec3::NEG_Z,
                right: Vec3::NEG_X,
                up: Vec3::Y,
            },
        }
    }

    #[cfg(test)]
    fn direction_for_uv(self, u: f32, v: f32) -> Vec3 {
        let x = 2.0 * u - 1.0;
        let y = 1.0 - 2.0 * v;
        let basis = self.basis();
        (basis.forward + basis.right * x + basis.up * y).normalize()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ProbeFaceBasis {
    forward: Vec3,
    right: Vec3,
    up: Vec3,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ProbeTaskExtent {
    size: u32,
    mip_levels: u32,
}

impl ProbeTaskExtent {
    fn from_task(task: &ReflectionProbeRenderTask) -> Result<Self, ReflectionProbeBakeError> {
        Self::from_size(task.size)
    }

    fn from_size(size: i32) -> Result<Self, ReflectionProbeBakeError> {
        let size =
            u32::try_from(size).map_err(|_err| ReflectionProbeBakeError::InvalidSize { size })?;
        if size == 0 {
            return Err(ReflectionProbeBakeError::InvalidSize { size: 0 });
        }
        Ok(Self {
            size,
            mip_levels: mip_levels_for_edge(size),
        })
    }

    const fn tuple(self) -> (u32, u32) {
        (self.size, self.size)
    }
}

#[derive(Clone, Debug)]
struct ProbeMipReadback {
    face: ProbeCubeFace,
    mip: u32,
    extent: u32,
    bytes_per_row_tight: u32,
    bytes_per_row_padded: u32,
    buffer_offset: u64,
    host_origin: usize,
    host_byte_count: usize,
}

#[derive(Clone, Debug)]
struct ProbeReadbackLayout {
    subresources: Vec<ProbeMipReadback>,
    buffer_size: u64,
    output_format: ProbeOutputFormat,
}

#[derive(Debug, thiserror::Error)]
enum ReflectionProbeBakeError {
    #[error("ReflectionProbeRenderTask render space {0} is missing")]
    MissingRenderSpace(i32),
    #[error("ReflectionProbeRenderTask render space {0} is inactive")]
    InactiveRenderSpace(i32),
    #[error("ReflectionProbeRenderTask renderable_index {0} is invalid")]
    InvalidRenderableIndex(i32),
    #[error("ReflectionProbeRenderTask renderable_index {0} was not found")]
    MissingProbe(i32),
    #[error("ReflectionProbeRenderTask probe transform_id {0} is invalid")]
    InvalidProbeTransform(i32),
    #[error("ReflectionProbeRenderTask probe transform_id {0} has no world matrix")]
    MissingProbeTransform(i32),
    #[error("ReflectionProbeRenderTask size {size} is invalid")]
    InvalidSize { size: i32 },
    #[error("ReflectionProbeRenderTask size {size} exceeds max_texture_dimension_2d={max}")]
    SizeExceedsLimit { size: u32, max: u32 },
    #[error(
        "ReflectionProbeRenderTask requires 6 texture array layers but max_texture_array_layers={max}"
    )]
    CubemapArrayLayersUnsupported { max: u32 },
    #[error("ReflectionProbeRenderTask mip_origins has {actual} faces; expected 6")]
    InvalidMipOriginFaces { actual: usize },
    #[error("ReflectionProbeRenderTask mip_origins[{face}] has {actual} mips; expected {expected}")]
    InvalidMipOriginCount {
        face: usize,
        expected: usize,
        actual: usize,
    },
    #[error("ReflectionProbeRenderTask mip_origins[{face}][{mip}] is negative: {origin}")]
    NegativeMipOrigin {
        face: usize,
        mip: usize,
        origin: i32,
    },
    #[error(
        "ReflectionProbeRenderTask result shared-memory descriptor is too small: need {required} bytes, got {actual}"
    )]
    ResultDescriptorTooSmall { required: usize, actual: usize },
    #[error("ReflectionProbeRenderTask byte count overflow")]
    OutputByteCountOverflow,
    #[error(
        "ReflectionProbeRenderTask readback buffer {size} bytes exceeds device max_buffer_size={max}"
    )]
    ReadbackBufferTooLarge { size: u64, max: u64 },
    #[error(
        "ReflectionProbeRenderTask mapped readback is too small: need {required} bytes, got {actual}"
    )]
    MappedReadbackTooSmall { required: usize, actual: usize },
    #[error("ReflectionProbeRenderTask result shared-memory descriptor could not be mapped")]
    SharedMemoryMapFailed,
    #[error("ReflectionProbeRenderTask render graph failed: {0}")]
    Graph(#[from] GraphExecuteError),
    #[error("ReflectionProbeRenderTask IBL convolve failed: {0}")]
    Convolve(String),
    #[error("device lost during ReflectionProbeRenderTask readback poll: {0}")]
    DeviceLost(String),
    #[error("ReflectionProbeRenderTask map_async timed out")]
    ReadbackTimeout,
    #[error("ReflectionProbeRenderTask map_async failed: {0}")]
    Map(String),
}

struct ProbeTaskTargets {
    cube_texture: Arc<wgpu::Texture>,
    face_color_views: [Arc<wgpu::TextureView>; CUBE_FACE_COUNT],
    face_depth_textures: [Arc<wgpu::Texture>; CUBE_FACE_COUNT],
    face_depth_views: [Arc<wgpu::TextureView>; CUBE_FACE_COUNT],
    color_format: wgpu::TextureFormat,
    extent: ProbeTaskExtent,
}

/// Active OnChanges probe capture, retained across ticks when host time slicing requests it.
pub(super) struct ActiveOnChangesReflectionProbeCapture {
    request: ReflectionProbeOnChangesRenderRequest,
    generation: u64,
    extent: ProbeTaskExtent,
    targets: ProbeTaskTargets,
    rendered_faces: u8,
    queued_unique_id: Option<i32>,
}

impl ProbeTaskTargets {
    fn create(gpu: &GpuContext, extent: ProbeTaskExtent) -> Result<Self, ReflectionProbeBakeError> {
        let max_dim = gpu.limits().max_texture_dimension_2d();
        if extent.size > max_dim {
            return Err(ReflectionProbeBakeError::SizeExceedsLimit {
                size: extent.size,
                max: max_dim,
            });
        }
        if !gpu.limits().array_layers_fit(CUBEMAP_ARRAY_LAYERS) {
            return Err(ReflectionProbeBakeError::CubemapArrayLayersUnsupported {
                max: gpu.limits().max_texture_array_layers(),
            });
        }
        let size = wgpu::Extent3d {
            width: extent.size,
            height: extent.size,
            depth_or_array_layers: CUBEMAP_ARRAY_LAYERS,
        };
        let cube_texture = Arc::new(gpu.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("renderide-reflection-probe-task-cube"),
            size,
            mip_level_count: extent.mip_levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: PROBE_TASK_COLOR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        }));
        let face_color_views = std::array::from_fn(|i| {
            Arc::new(cube_texture.create_view(&face_view_desc(
                "renderide-reflection-probe-task-face-color",
                i as u32,
                PROBE_TASK_COLOR_FORMAT,
                wgpu::TextureUsages::RENDER_ATTACHMENT,
            )))
        });
        crate::profiling::note_resource_churn!(
            TextureView,
            "runtime::reflection_probe_task_face_color_views"
        );

        let depth_format = crate::gpu::main_forward_depth_stencil_format(gpu.device().features());
        let depth_size = wgpu::Extent3d {
            width: extent.size,
            height: extent.size,
            depth_or_array_layers: 1,
        };
        let face_depth_textures = std::array::from_fn(|_i| {
            Arc::new(gpu.device().create_texture(&wgpu::TextureDescriptor {
                label: Some("renderide-reflection-probe-task-face-depth"),
                size: depth_size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: depth_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            }))
        });
        let face_depth_views = std::array::from_fn(|i| {
            Arc::new(
                face_depth_textures[i].create_view(&wgpu::TextureViewDescriptor {
                    label: Some("renderide-reflection-probe-task-face-depth"),
                    format: Some(depth_format),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    usage: Some(wgpu::TextureUsages::RENDER_ATTACHMENT),
                    aspect: wgpu::TextureAspect::All,
                    ..Default::default()
                }),
            )
        });
        crate::profiling::note_resource_churn!(
            TextureView,
            "runtime::reflection_probe_task_face_depth_views"
        );

        Ok(Self {
            cube_texture,
            face_color_views,
            face_depth_textures,
            face_depth_views,
            color_format: PROBE_TASK_COLOR_FORMAT,
            extent,
        })
    }

    fn to_offscreen_handles(&self, face: ProbeCubeFace) -> OffscreenRtHandles {
        OffscreenRtHandles {
            rt_id: -1,
            color_view: Arc::clone(&self.face_color_views[face.index()]),
            depth_texture: Arc::clone(&self.face_depth_textures[face.index()]),
            depth_view: Arc::clone(&self.face_depth_views[face.index()]),
            color_format: self.color_format,
        }
    }

    fn cube_sample_view(&self) -> Arc<wgpu::TextureView> {
        Arc::new(self.cube_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("renderide-reflection-probe-onchanges-cube-view"),
            format: Some(PROBE_TASK_COLOR_FORMAT),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(CUBEMAP_ARRAY_LAYERS),
        }))
    }
}

fn face_view_desc(
    label: &'static str,
    face: u32,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
) -> wgpu::TextureViewDescriptor<'static> {
    wgpu::TextureViewDescriptor {
        label: Some(label),
        format: Some(format),
        dimension: Some(wgpu::TextureViewDimension::D2),
        usage: Some(usage),
        aspect: wgpu::TextureAspect::All,
        base_mip_level: 0,
        mip_level_count: Some(1),
        base_array_layer: face,
        array_layer_count: Some(1),
    }
}

struct PlannedReflectionProbeTask {
    plans: Vec<FrameViewPlan<'static>>,
    targets: ProbeTaskTargets,
    readback_layout: ProbeReadbackLayout,
}

impl RendererRuntime {
    /// Queues host OnChanges reflection-probe render requests that need scene capture.
    pub(super) fn queue_onchanges_reflection_probe_requests(
        &mut self,
        requests: Vec<ReflectionProbeOnChangesRenderRequest>,
    ) {
        profiling::scope!("reflection_probe_onchanges::queue");
        for request in requests {
            if let Some(active) = self
                .tick_state
                .active_onchanges_reflection_probe_captures
                .iter_mut()
                .find(|active| same_onchanges_probe(active.request, request))
            {
                active.queued_unique_id = Some(request.unique_id);
                continue;
            }
            if let Some(pending) = self
                .tick_state
                .pending_onchanges_reflection_probe_requests
                .iter_mut()
                .find(|pending| {
                    pending.render_space_id == request.render_space_id
                        && pending.renderable_index == request.renderable_index
                })
            {
                pending.unique_id = request.unique_id;
                continue;
            }
            self.tick_state
                .pending_onchanges_reflection_probe_requests
                .push(request);
        }
    }

    /// Appends host reflection-probe cubemap bake tasks to the pre-begin-frame GPU readback queue.
    pub(super) fn queue_reflection_probe_render_tasks(&mut self, data: &FrameSubmitData) {
        profiling::scope!("reflection_probe_task::queue");
        let initial = self.tick_state.pending_reflection_probe_render_tasks.len();
        for space in &data.render_spaces {
            let render_space_id = RenderSpaceId(space.id);
            self.tick_state
                .pending_reflection_probe_render_tasks
                .extend(
                    space
                        .reflection_probe_render_tasks
                        .iter()
                        .cloned()
                        .map(|task| QueuedReflectionProbeRenderTask {
                            render_space_id,
                            task,
                        }),
                );
        }
        let added = self
            .tick_state
            .pending_reflection_probe_render_tasks
            .len()
            .saturating_sub(initial);
        if added > 0 {
            logger::debug!(
                "queued {} ReflectionProbeRenderTask bake(s); pending={}",
                added,
                self.tick_state.pending_reflection_probe_render_tasks.len()
            );
        }
    }

    /// Queues failure results for every reflection-probe bake task in a rejected frame submit.
    pub(super) fn queue_failed_reflection_probe_render_task_results(
        &mut self,
        data: &FrameSubmitData,
    ) {
        profiling::scope!("reflection_probe_task::queue_failed_results");
        for space in &data.render_spaces {
            self.tick_state
                .pending_reflection_probe_render_results
                .extend(space.reflection_probe_render_tasks.iter().map(|task| {
                    ReflectionProbeRenderResult {
                        render_task_id: task.render_task_id,
                        success: false,
                    }
                }));
        }
    }

    /// Attempts to flush queued reflection-probe bake results to the background IPC queue.
    pub(super) fn flush_reflection_probe_render_results(&mut self) {
        let RendererRuntime {
            frontend,
            tick_state,
            ..
        } = self;
        let mut ipc = frontend.ipc_mut();
        flush_reflection_probe_render_results_to_ipc(tick_state, &mut ipc);
    }

    /// Drains queued reflection-probe bake tasks before the next host begin-frame is sent.
    pub fn drain_reflection_probe_render_tasks(&mut self, gpu: &mut GpuContext) {
        profiling::scope!("reflection_probe_task::drain");
        let mut tasks = std::mem::take(&mut self.tick_state.pending_reflection_probe_render_tasks);
        self.flush_reflection_probe_render_results();
        if !tasks.is_empty() {
            let base_camera = self.host_camera;
            let RendererRuntime {
                frontend,
                backend,
                scene,
                tick_state,
                ..
            } = self;
            let (shm, mut ipc) = frontend.transport_pair_mut();
            flush_reflection_probe_render_results_to_ipc(tick_state, &mut ipc);
            if let Some(shm) = shm {
                let render_context = RenderingContext::RenderToAsset;
                let mut convolver = SkyboxIblConvolver::new();
                let mut completed = 0u64;
                let mut failed = 0u64;
                for queued in tasks.drain(..) {
                    match render_reflection_probe_task(ReflectionProbeTaskRenderCtx {
                        gpu: &mut *gpu,
                        backend: &mut *backend,
                        scene,
                        base_camera,
                        shm: &mut *shm,
                        convolver: &mut convolver,
                        render_context,
                        queued: &queued,
                    }) {
                        Ok(()) => {
                            completed = completed.saturating_add(1);
                            tick_state.pending_reflection_probe_render_results.push(
                                ReflectionProbeRenderResult {
                                    render_task_id: queued.task.render_task_id,
                                    success: true,
                                },
                            );
                        }
                        Err(error) => {
                            failed = failed.saturating_add(1);
                            logger::warn!(
                                "ReflectionProbeRenderTask bake failed for render_space_id={} render_task_id={}: {error}",
                                queued.render_space_id.0,
                                queued.task.render_task_id
                            );
                            zero_probe_task_result(shm, &queued.task);
                            tick_state.pending_reflection_probe_render_results.push(
                                ReflectionProbeRenderResult {
                                    render_task_id: queued.task.render_task_id,
                                    success: false,
                                },
                            );
                        }
                    }
                    flush_reflection_probe_render_results_to_ipc(tick_state, &mut ipc);
                }
                logger::debug!(
                    "drained ReflectionProbeRenderTask bakes: completed={} failed={}",
                    completed,
                    failed
                );
            } else {
                logger::warn!(
                    "dropping {} ReflectionProbeRenderTask bake(s): shared memory is unavailable",
                    tasks.len()
                );
                queue_reflection_probe_failures(
                    tick_state,
                    tasks.iter().map(|queued| &queued.task),
                );
                flush_reflection_probe_render_results_to_ipc(tick_state, &mut ipc);
            }
        }
        self.drain_onchanges_reflection_probe_captures(gpu);
    }
}

struct ReflectionProbeTaskRenderCtx<'a> {
    gpu: &'a mut GpuContext,
    backend: &'a mut RenderBackend,
    scene: &'a SceneCoordinator,
    base_camera: HostCameraFrame,
    shm: &'a mut SharedMemoryAccessor,
    convolver: &'a mut SkyboxIblConvolver,
    render_context: RenderingContext,
    queued: &'a QueuedReflectionProbeRenderTask,
}

fn render_reflection_probe_task(
    ctx: ReflectionProbeTaskRenderCtx<'_>,
) -> Result<(), ReflectionProbeBakeError> {
    profiling::scope!("reflection_probe_task::render_one");
    let planned = plan_reflection_probe_task(ctx.gpu, ctx.scene, ctx.base_camera, ctx.queued)?;
    let view_ids = planned
        .plans
        .iter()
        .map(|plan| plan.view_id)
        .collect::<Vec<_>>();
    let render_result = render_reflection_probe_faces_offscreen(
        ctx.gpu,
        ctx.backend,
        ctx.scene,
        ctx.render_context,
        planned.plans,
    );
    if let Err(error) = render_result {
        ctx.backend.retire_one_shot_views(&view_ids);
        return Err(error);
    }
    let mapped = match readback_reflection_probe_cube(
        ctx.gpu,
        ctx.convolver,
        planned.targets.cube_texture.as_ref(),
        planned.targets.extent,
        &planned.readback_layout,
    ) {
        Ok(mapped) => mapped,
        Err(error) => {
            ctx.backend.retire_one_shot_views(&view_ids);
            return Err(error);
        }
    };
    ctx.backend.retire_one_shot_views(&view_ids);
    write_probe_task_result(ctx.shm, &ctx.queued.task, &planned.readback_layout, &mapped)
}

impl RendererRuntime {
    fn drain_onchanges_reflection_probe_captures(&mut self, gpu: &mut GpuContext) {
        profiling::scope!("reflection_probe_onchanges::drain");
        self.start_pending_onchanges_reflection_probe_captures(gpu);
        if self
            .tick_state
            .active_onchanges_reflection_probe_captures
            .is_empty()
        {
            return;
        }

        let base_camera = self.host_camera;
        let render_context = RenderingContext::RenderToAsset;
        let mut active =
            std::mem::take(&mut self.tick_state.active_onchanges_reflection_probe_captures);
        let mut still_active = Vec::with_capacity(active.len());
        let mut completed_results = Vec::new();
        for mut capture in active.drain(..) {
            match step_onchanges_reflection_probe_capture(OnChangesCaptureStepCtx {
                gpu,
                backend: &mut self.backend,
                scene: &self.scene,
                base_camera,
                render_context,
                capture: &mut capture,
            }) {
                Ok(OnChangesCaptureStep::Pending) => still_active.push(capture),
                Ok(OnChangesCaptureStep::Complete) => {
                    let completed_request = capture.request;
                    let follow_up = capture.queued_unique_id.take().map(|unique_id| {
                        ReflectionProbeOnChangesRenderRequest {
                            unique_id,
                            ..completed_request
                        }
                    });
                    self.backend
                        .register_runtime_reflection_probe_capture(capture.into_runtime_capture());
                    completed_results.push(changed_probe_completion(
                        completed_request.render_space_id,
                        completed_request.unique_id,
                        false,
                    ));
                    if let Some(request) = follow_up {
                        self.tick_state
                            .pending_onchanges_reflection_probe_requests
                            .push(request);
                    }
                }
                Err(error) => {
                    logger::warn!(
                        "OnChanges reflection probe capture failed for render_space_id={} renderable_index={} unique_id={}: {error}",
                        capture.request.render_space_id,
                        capture.request.renderable_index,
                        capture.request.unique_id
                    );
                    completed_results.push(changed_probe_completion(
                        capture.request.render_space_id,
                        capture.request.unique_id,
                        true,
                    ));
                }
            }
        }
        self.tick_state.active_onchanges_reflection_probe_captures = still_active;
        self.frontend
            .enqueue_rendered_reflection_probes(completed_results);
    }

    fn start_pending_onchanges_reflection_probe_captures(&mut self, gpu: &GpuContext) {
        profiling::scope!("reflection_probe_onchanges::start_pending");
        let mut pending =
            std::mem::take(&mut self.tick_state.pending_onchanges_reflection_probe_requests);
        if pending.is_empty() {
            return;
        }
        let mut completed_results = Vec::new();
        for request in pending.drain(..) {
            let generation = self.tick_state.next_onchanges_reflection_probe_generation;
            self.tick_state.next_onchanges_reflection_probe_generation = self
                .tick_state
                .next_onchanges_reflection_probe_generation
                .saturating_add(1);
            match start_onchanges_reflection_probe_capture(gpu, &self.scene, request, generation) {
                Ok(OnChangesCaptureStart::ImmediateComplete) => {
                    completed_results.push(changed_probe_completion(
                        request.render_space_id,
                        request.unique_id,
                        false,
                    ));
                }
                Ok(OnChangesCaptureStart::Capture(capture)) => {
                    self.tick_state
                        .active_onchanges_reflection_probe_captures
                        .push(*capture);
                }
                Err(error) => {
                    logger::warn!(
                        "OnChanges reflection probe capture could not start for render_space_id={} renderable_index={} unique_id={}: {error}",
                        request.render_space_id,
                        request.renderable_index,
                        request.unique_id
                    );
                    completed_results.push(changed_probe_completion(
                        request.render_space_id,
                        request.unique_id,
                        true,
                    ));
                }
            }
        }
        self.frontend
            .enqueue_rendered_reflection_probes(completed_results);
    }
}

enum OnChangesCaptureStart {
    ImmediateComplete,
    Capture(Box<ActiveOnChangesReflectionProbeCapture>),
}

enum OnChangesCaptureStep {
    Pending,
    Complete,
}

struct OnChangesCaptureStepCtx<'a> {
    gpu: &'a mut GpuContext,
    backend: &'a mut RenderBackend,
    scene: &'a SceneCoordinator,
    base_camera: HostCameraFrame,
    render_context: RenderingContext,
    capture: &'a mut ActiveOnChangesReflectionProbeCapture,
}

fn start_onchanges_reflection_probe_capture(
    gpu: &GpuContext,
    scene: &SceneCoordinator,
    request: ReflectionProbeOnChangesRenderRequest,
    generation: u64,
) -> Result<OnChangesCaptureStart, ReflectionProbeBakeError> {
    let space_id = RenderSpaceId(request.render_space_id);
    let space = scene
        .space(space_id)
        .ok_or(ReflectionProbeBakeError::MissingRenderSpace(
            request.render_space_id,
        ))?;
    if !space.is_active() {
        return Err(ReflectionProbeBakeError::InactiveRenderSpace(
            request.render_space_id,
        ));
    }
    let probe_index = usize::try_from(request.renderable_index).map_err(|_err| {
        ReflectionProbeBakeError::InvalidRenderableIndex(request.renderable_index)
    })?;
    let probe = space.reflection_probes().get(probe_index).ok_or(
        ReflectionProbeBakeError::MissingProbe(request.renderable_index),
    )?;
    if probe.state.clear_flags == ReflectionProbeClear::Color
        || reflection_probe_skybox_only(probe.state.flags)
    {
        return Ok(OnChangesCaptureStart::ImmediateComplete);
    }
    if probe.state.r#type != crate::shared::ReflectionProbeType::OnChanges {
        return Err(ReflectionProbeBakeError::MissingProbe(
            request.renderable_index,
        ));
    }
    let extent = ProbeTaskExtent::from_size(probe.state.resolution)?;
    let targets = ProbeTaskTargets::create(gpu, extent)?;
    Ok(OnChangesCaptureStart::Capture(Box::new(
        ActiveOnChangesReflectionProbeCapture {
            request,
            generation,
            extent,
            targets,
            rendered_faces: 0,
            queued_unique_id: None,
        },
    )))
}

fn step_onchanges_reflection_probe_capture(
    ctx: OnChangesCaptureStepCtx<'_>,
) -> Result<OnChangesCaptureStep, ReflectionProbeBakeError> {
    profiling::scope!("reflection_probe_onchanges::step");
    let state = onchanges_capture_state(ctx.scene, ctx.capture.request)?;
    let faces = onchanges_faces_for_step(state.time_slicing_mode, ctx.capture.rendered_faces);
    if faces.is_empty() {
        return Ok(OnChangesCaptureStep::Complete);
    }
    let plans = plan_onchanges_reflection_probe_faces(
        ctx.scene,
        ctx.base_camera,
        ctx.capture,
        state,
        &faces,
    )?;
    let view_ids = plans.iter().map(|plan| plan.view_id).collect::<Vec<_>>();
    let render_result = render_reflection_probe_faces_offscreen(
        ctx.gpu,
        ctx.backend,
        ctx.scene,
        ctx.render_context,
        plans,
    );
    ctx.backend.retire_one_shot_views(&view_ids);
    render_result?;
    for face in faces {
        ctx.capture.rendered_faces |= face.bit();
    }
    if ctx.capture.rendered_faces == ProbeCubeFace::ALL_MASK {
        Ok(OnChangesCaptureStep::Complete)
    } else {
        Ok(OnChangesCaptureStep::Pending)
    }
}

fn onchanges_capture_state(
    scene: &SceneCoordinator,
    request: ReflectionProbeOnChangesRenderRequest,
) -> Result<ReflectionProbeState, ReflectionProbeBakeError> {
    let space_id = RenderSpaceId(request.render_space_id);
    let space = scene
        .space(space_id)
        .ok_or(ReflectionProbeBakeError::MissingRenderSpace(
            request.render_space_id,
        ))?;
    if !space.is_active() {
        return Err(ReflectionProbeBakeError::InactiveRenderSpace(
            request.render_space_id,
        ));
    }
    let probe_index = usize::try_from(request.renderable_index).map_err(|_err| {
        ReflectionProbeBakeError::InvalidRenderableIndex(request.renderable_index)
    })?;
    let probe = space.reflection_probes().get(probe_index).ok_or(
        ReflectionProbeBakeError::MissingProbe(request.renderable_index),
    )?;
    if probe.state.r#type != crate::shared::ReflectionProbeType::OnChanges {
        return Err(ReflectionProbeBakeError::MissingProbe(
            request.renderable_index,
        ));
    }
    Ok(probe.state)
}

fn plan_onchanges_reflection_probe_faces(
    scene: &SceneCoordinator,
    base_camera: HostCameraFrame,
    capture: &ActiveOnChangesReflectionProbeCapture,
    state: ReflectionProbeState,
    faces: &[ProbeCubeFace],
) -> Result<Vec<FrameViewPlan<'static>>, ReflectionProbeBakeError> {
    let space_id = RenderSpaceId(capture.request.render_space_id);
    let space = scene
        .space(space_id)
        .ok_or(ReflectionProbeBakeError::MissingRenderSpace(space_id.0))?;
    let probe = space
        .reflection_probes()
        .get(capture.request.renderable_index as usize)
        .ok_or(ReflectionProbeBakeError::MissingProbe(
            capture.request.renderable_index,
        ))?;
    let transform_index = usize::try_from(probe.transform_id)
        .map_err(|_err| ReflectionProbeBakeError::InvalidProbeTransform(probe.transform_id))?;
    let probe_world = scene
        .world_matrix_for_render_context(
            space_id,
            transform_index,
            RenderingContext::RenderToAsset,
            base_camera.head_output_transform,
        )
        .ok_or(ReflectionProbeBakeError::MissingProbeTransform(
            probe.transform_id,
        ))?;
    let filter = draw_filter_from_reflection_probe_state(&state);
    let probe_position = probe_world.col(3).truncate();
    Ok(faces
        .iter()
        .copied()
        .map(|face| FrameViewPlan {
            host_camera: host_camera_frame_for_probe_face(
                base_camera,
                state,
                capture.extent.tuple(),
                probe_position,
                face,
            ),
            render_space_filter: Some(space_id),
            draw_filter: Some(filter.clone()),
            view_id: ViewId::reflection_probe_render_task(
                space_id,
                capture.request.unique_id,
                face.view_id_face_index(),
            ),
            viewport_px: capture.extent.tuple(),
            clear: clear_from_reflection_probe_state(state),
            post_processing: reflection_probe_bake_post_processing(),
            target: FrameViewPlanTarget::SecondaryRt(capture.targets.to_offscreen_handles(face)),
        })
        .collect())
}

fn onchanges_faces_for_step(
    mode: ReflectionProbeTimeSlicingMode,
    rendered_faces: u8,
) -> Vec<ProbeCubeFace> {
    let remaining = ProbeCubeFace::ALL
        .into_iter()
        .filter(|face| rendered_faces & face.bit() == 0);
    if mode == ReflectionProbeTimeSlicingMode::IndividualFaces {
        remaining.take(1).collect()
    } else {
        remaining.collect()
    }
}

fn same_onchanges_probe(
    a: ReflectionProbeOnChangesRenderRequest,
    b: ReflectionProbeOnChangesRenderRequest,
) -> bool {
    a.render_space_id == b.render_space_id && a.renderable_index == b.renderable_index
}

impl ActiveOnChangesReflectionProbeCapture {
    fn into_runtime_capture(self) -> RuntimeReflectionProbeCapture {
        RuntimeReflectionProbeCapture {
            key: RuntimeReflectionProbeCaptureKey {
                space_id: RenderSpaceId(self.request.render_space_id),
                renderable_index: self.request.renderable_index,
            },
            generation: self.generation,
            face_size: self.extent.size,
            mip_levels: self.extent.mip_levels,
            texture: Arc::clone(&self.targets.cube_texture),
            view: self.targets.cube_sample_view(),
        }
    }
}

fn plan_reflection_probe_task(
    gpu: &GpuContext,
    scene: &SceneCoordinator,
    base_camera: HostCameraFrame,
    queued: &QueuedReflectionProbeRenderTask,
) -> Result<PlannedReflectionProbeTask, ReflectionProbeBakeError> {
    profiling::scope!("reflection_probe_task::plan");
    let task = &queued.task;
    let extent = ProbeTaskExtent::from_task(task)?;
    let output_format = ProbeOutputFormat::from_hdr(task.hdr);
    let readback_layout =
        compute_probe_readback_layout(task, extent, output_format, gpu.limits().max_buffer_size())?;
    let space =
        scene
            .space(queued.render_space_id)
            .ok_or(ReflectionProbeBakeError::MissingRenderSpace(
                queued.render_space_id.0,
            ))?;
    if !space.is_active() {
        return Err(ReflectionProbeBakeError::InactiveRenderSpace(
            queued.render_space_id.0,
        ));
    }
    let probe_index = usize::try_from(task.renderable_index)
        .map_err(|_err| ReflectionProbeBakeError::InvalidRenderableIndex(task.renderable_index))?;
    let probe = space.reflection_probes().get(probe_index).ok_or(
        ReflectionProbeBakeError::MissingProbe(task.renderable_index),
    )?;
    let transform_index = usize::try_from(probe.transform_id)
        .map_err(|_err| ReflectionProbeBakeError::InvalidProbeTransform(probe.transform_id))?;
    let probe_world = scene
        .world_matrix_for_render_context(
            queued.render_space_id,
            transform_index,
            RenderingContext::RenderToAsset,
            base_camera.head_output_transform,
        )
        .ok_or(ReflectionProbeBakeError::MissingProbeTransform(
            probe.transform_id,
        ))?;
    let targets = ProbeTaskTargets::create(gpu, extent)?;
    let filter = draw_filter_from_reflection_probe_task(task, &probe.state);
    let probe_position = probe_world.col(3).truncate();
    let plans = ProbeCubeFace::ALL
        .iter()
        .copied()
        .map(|face| FrameViewPlan {
            host_camera: host_camera_frame_for_probe_face(
                base_camera,
                probe.state,
                extent.tuple(),
                probe_position,
                face,
            ),
            draw_filter: Some(filter.clone()),
            render_space_filter: Some(queued.render_space_id),
            view_id: ViewId::reflection_probe_render_task(
                queued.render_space_id,
                task.render_task_id,
                face.view_id_face_index(),
            ),
            viewport_px: extent.tuple(),
            clear: clear_from_reflection_probe_state(probe.state),
            post_processing: reflection_probe_bake_post_processing(),
            target: FrameViewPlanTarget::SecondaryRt(targets.to_offscreen_handles(face)),
        })
        .collect();
    Ok(PlannedReflectionProbeTask {
        plans,
        targets,
        readback_layout,
    })
}

fn draw_filter_from_reflection_probe_task(
    task: &ReflectionProbeRenderTask,
    state: &ReflectionProbeState,
) -> CameraTransformDrawFilter {
    if reflection_probe_skybox_only(state.flags) {
        CameraTransformDrawFilter {
            only: Some(HashSet::new()),
            exclude: HashSet::new(),
        }
    } else {
        CameraTransformDrawFilter {
            only: None,
            exclude: task.exclude_transform_ids.iter().copied().collect(),
        }
    }
}

fn draw_filter_from_reflection_probe_state(
    state: &ReflectionProbeState,
) -> CameraTransformDrawFilter {
    if reflection_probe_skybox_only(state.flags) {
        CameraTransformDrawFilter {
            only: Some(HashSet::new()),
            exclude: HashSet::new(),
        }
    } else {
        CameraTransformDrawFilter {
            only: None,
            exclude: HashSet::new(),
        }
    }
}

fn clear_from_reflection_probe_state(state: ReflectionProbeState) -> FrameViewClear {
    if state.clear_flags == ReflectionProbeClear::Color {
        FrameViewClear {
            mode: CameraClearMode::Color,
            color: state.background_color,
        }
    } else {
        FrameViewClear::skybox()
    }
}

fn reflection_probe_bake_post_processing() -> ViewPostProcessing {
    ViewPostProcessing::disabled()
}

fn host_camera_frame_for_probe_face(
    base: HostCameraFrame,
    state: ReflectionProbeState,
    viewport_px: (u32, u32),
    position: Vec3,
    face: ProbeCubeFace,
) -> HostCameraFrame {
    let clip = reflection_probe_clip(state);
    let world_matrix = probe_face_world_matrix(position, face);
    let pose = CameraPose::from_world_matrix(world_matrix);
    let viewport = Viewport::from_tuple(viewport_px);
    HostCameraFrame {
        frame_index: base.frame_index,
        clip,
        desktop_fov_degrees: 90.0,
        vr_active: false,
        output_device: base.output_device,
        projection_kind: CameraProjectionKind::Perspective,
        primary_ortho_task: None,
        stereo: None,
        head_output_transform: base.head_output_transform,
        explicit_view: Some(EyeView::from_pose_projection(
            pose,
            if viewport.is_empty() {
                Mat4::IDENTITY
            } else {
                EyeView::perspective_from_pose(pose, viewport, 90.0, clip).proj
            },
        )),
        eye_world_position: Some(position),
        suppress_occlusion_temporal: true,
    }
}

fn reflection_probe_clip(state: ReflectionProbeState) -> CameraClipPlanes {
    let near = finite_positive_or(state.near_clip, CameraClipPlanes::default().near).max(0.01);
    let far_default = CameraClipPlanes::default().far;
    let far = finite_positive_or(state.far_clip, far_default).max(near + 0.01);
    CameraClipPlanes::new(near, far)
}

fn finite_positive_or(value: f32, fallback: f32) -> f32 {
    if value.is_finite() && value > 0.0 {
        value
    } else {
        fallback
    }
}

fn probe_face_world_matrix(position: Vec3, face: ProbeCubeFace) -> Mat4 {
    let basis = face.basis();
    Mat4::from_cols(
        basis.right.extend(0.0),
        basis.up.extend(0.0),
        basis.forward.extend(0.0),
        position.extend(1.0),
    )
}

fn render_reflection_probe_faces_offscreen(
    gpu: &mut GpuContext,
    backend: &mut RenderBackend,
    scene: &SceneCoordinator,
    render_context: RenderingContext,
    plans: Vec<FrameViewPlan<'static>>,
) -> Result<(), ReflectionProbeBakeError> {
    profiling::scope!("reflection_probe_task::offscreen_render");
    let prepared_views = PreparedViews::new(plans, None);
    backend.prepare_lights_for_views(
        scene,
        prepared_views
            .plans()
            .iter()
            .map(|plan| plan.light_view_desc(render_context)),
    );
    let view_perms = prepared_views
        .plans()
        .iter()
        .map(FrameViewPlan::shader_permutation)
        .collect::<Vec<_>>();
    let shared = backend.extract_frame_shared(
        scene,
        render_context,
        WorldMeshDrawCollectParallelism::Full,
        view_perms,
    );
    let submit_frame = ExtractedFrame::new(prepared_views, shared)
        .prepare_draws()
        .into_submit_frame();
    submit_frame.execute(gpu, scene, backend)?;
    Ok(())
}

fn queue_reflection_probe_failures<'a>(
    tick_state: &mut super::tick_state::RuntimeTickState,
    tasks: impl IntoIterator<Item = &'a ReflectionProbeRenderTask>,
) {
    tick_state
        .pending_reflection_probe_render_results
        .extend(tasks.into_iter().map(|task| ReflectionProbeRenderResult {
            render_task_id: task.render_task_id,
            success: false,
        }));
}

fn flush_reflection_probe_render_results_to_ipc(
    tick_state: &mut super::tick_state::RuntimeTickState,
    ipc: &mut Option<&mut DualQueueIpc>,
) {
    if tick_state
        .pending_reflection_probe_render_results
        .is_empty()
    {
        return;
    }
    let Some(ipc) = ipc.as_mut() else {
        return;
    };
    let pending = std::mem::take(&mut tick_state.pending_reflection_probe_render_results);
    let mut sent = 0usize;
    for result in &pending {
        if !ipc
            .send_background_reliable(RendererCommand::ReflectionProbeRenderResult(result.clone()))
        {
            break;
        }
        sent = sent.saturating_add(1);
    }
    tick_state.pending_reflection_probe_render_results = pending.into_iter().skip(sent).collect();
}

pub(super) fn reflection_probe_render_task_count(data: &FrameSubmitData) -> usize {
    data.render_spaces
        .iter()
        .map(|space| space.reflection_probe_render_tasks.len())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix_direction_for_uv(face: ProbeCubeFace, u: f32, v: f32) -> Vec3 {
        let x = 2.0 * u - 1.0;
        let y = 1.0 - 2.0 * v;
        probe_face_world_matrix(Vec3::ZERO, face)
            .transform_vector3(Vec3::new(x, y, 1.0))
            .normalize()
    }

    #[test]
    fn cubemap_face_directions_match_bitmap_cube_order() {
        let samples = [
            (ProbeCubeFace::PosX, Vec3::new(1.0, 1.0, 1.0)),
            (ProbeCubeFace::NegX, Vec3::new(-1.0, 1.0, -1.0)),
            (ProbeCubeFace::PosY, Vec3::new(-1.0, 1.0, -1.0)),
            (ProbeCubeFace::NegY, Vec3::new(-1.0, -1.0, 1.0)),
            (ProbeCubeFace::PosZ, Vec3::new(-1.0, 1.0, 1.0)),
            (ProbeCubeFace::NegZ, Vec3::new(1.0, 1.0, -1.0)),
        ];
        for (face, expected) in samples {
            let actual = face.direction_for_uv(0.0, 0.0);
            assert!((actual - expected.normalize()).length() < 1e-6);
        }
    }

    #[test]
    fn probe_face_world_matrices_match_bitmap_cube_directions() {
        for face in ProbeCubeFace::ALL {
            assert!(
                (matrix_direction_for_uv(face, 0.5, 0.5) - face.basis().forward).length() < 1e-6
            );
            for (u, v) in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)] {
                let actual = matrix_direction_for_uv(face, u, v);
                let expected = face.direction_for_uv(u, v);
                assert!(
                    (actual - expected).length() < 1e-6,
                    "{face:?} uv=({u}, {v}) actual={actual:?} expected={expected:?}"
                );
            }
        }
    }

    #[test]
    fn onchanges_individual_faces_steps_one_remaining_face() {
        let faces =
            onchanges_faces_for_step(ReflectionProbeTimeSlicingMode::IndividualFaces, 0b0000_0011);

        assert_eq!(faces, vec![ProbeCubeFace::PosY]);
    }

    #[test]
    fn onchanges_all_faces_at_once_steps_all_remaining_faces() {
        let faces =
            onchanges_faces_for_step(ReflectionProbeTimeSlicingMode::AllFacesAtOnce, 0b0011_0001);

        assert_eq!(
            faces,
            vec![
                ProbeCubeFace::NegX,
                ProbeCubeFace::PosY,
                ProbeCubeFace::NegY
            ]
        );
    }

    #[test]
    fn probe_face_projection_is_square_ninety_degrees() {
        let frame = host_camera_frame_for_probe_face(
            HostCameraFrame::default(),
            ReflectionProbeState {
                near_clip: 0.1,
                far_clip: 100.0,
                ..Default::default()
            },
            (256, 256),
            Vec3::ZERO,
            ProbeCubeFace::PosZ,
        );
        let view = frame
            .explicit_view
            .expect("probe face should use explicit camera view");

        assert!((view.proj.x_axis.x - 1.0).abs() < 1e-6);
        assert!((view.proj.y_axis.y - 1.0).abs() < 1e-6);
    }

    #[test]
    fn reflection_probe_bake_views_disable_post_processing() {
        let policy = reflection_probe_bake_post_processing();

        assert!(!policy.is_enabled());
        assert!(!policy.screen_space_reflections);
        assert!(!policy.motion_blur);
    }

    #[test]
    fn skybox_only_probe_uses_empty_selective_filter() {
        let task = ReflectionProbeRenderTask {
            exclude_transform_ids: vec![1, 2],
            ..Default::default()
        };
        let state = ReflectionProbeState {
            flags: 0b001,
            ..Default::default()
        };

        let filter = draw_filter_from_reflection_probe_task(&task, &state);

        assert!(filter.only.as_ref().is_some_and(HashSet::is_empty));
        assert!(filter.exclude.is_empty());
    }
}
