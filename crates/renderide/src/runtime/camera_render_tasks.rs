//! Host camera render-task queue, offscreen render, GPU readback, and IPC writeback.

use std::sync::Arc;
use std::time::Duration;

use hashbrown::HashSet;
use rayon::prelude::*;

use crate::backend::RenderBackend;
use crate::camera::{ViewId, camera_render_task_world_matrix, host_camera_frame_for_render_task};
use crate::gpu::GpuContext;
use crate::ipc::SharedMemoryAccessor;
use crate::render_graph::{FrameViewClear, GraphExecuteError, ViewPostProcessing};
use crate::scene::{RenderSpaceId, SceneCoordinator};
use crate::shared::{CameraRenderParameters, CameraRenderTask, RenderingContext, TextureFormat};
use crate::world_mesh::{CameraTransformDrawFilter, WorldMeshDrawCollectParallelism};

use super::RendererRuntime;
use super::frame_extract::{ExtractedFrame, PreparedViews};
use super::frame_view_plan::{FrameViewPlan, FrameViewPlanTarget, OffscreenRtHandles};

/// Buffers at or above this size are filled through rayon.
const PAR_FILL_THRESHOLD: usize = 128 * 1024;
/// Per-thread fill chunk for large shared-memory result buffers.
const PAR_FILL_CHUNK: usize = 64 * 1024;
/// Maximum time to wait for the blocking task readback callback after `device.poll`.
const CAMERA_READBACK_TIMEOUT: Duration = Duration::from_secs(5);
/// Color attachment format used for CPU camera readback tasks.
const CAMERA_TASK_COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
/// Bytes per texel copied from the readback color target.
const RGBA8_BYTES_PER_PIXEL: usize = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CameraTaskOutputFormat {
    Argb32,
    Rgba32,
    Bgra32,
    Rgb24,
}

impl CameraTaskOutputFormat {
    fn from_texture_format(format: TextureFormat) -> Option<Self> {
        match format {
            TextureFormat::ARGB32 => Some(Self::Argb32),
            TextureFormat::RGBA32 => Some(Self::Rgba32),
            TextureFormat::BGRA32 => Some(Self::Bgra32),
            TextureFormat::RGB24 => Some(Self::Rgb24),
            _ => None,
        }
    }

    const fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Argb32 | Self::Rgba32 | Self::Bgra32 => 4,
            Self::Rgb24 => 3,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CameraTaskExtent {
    width: u32,
    height: u32,
}

impl CameraTaskExtent {
    fn from_parameters(parameters: &CameraRenderParameters) -> Result<Self, CameraReadbackError> {
        let width = u32::try_from(parameters.resolution.x).map_err(|_err| {
            CameraReadbackError::InvalidExtent {
                width: parameters.resolution.x,
                height: parameters.resolution.y,
            }
        })?;
        let height = u32::try_from(parameters.resolution.y).map_err(|_err| {
            CameraReadbackError::InvalidExtent {
                width: parameters.resolution.x,
                height: parameters.resolution.y,
            }
        })?;
        if width == 0 || height == 0 {
            return Err(CameraReadbackError::InvalidExtent {
                width: parameters.resolution.x,
                height: parameters.resolution.y,
            });
        }
        Ok(Self { width, height })
    }

    const fn tuple(self) -> (u32, u32) {
        (self.width, self.height)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ReadbackLayout {
    width: u32,
    height: u32,
    bytes_per_row_tight: u32,
    bytes_per_row_padded: u32,
    buffer_size: u64,
}

#[derive(Debug, thiserror::Error)]
enum CameraReadbackError {
    #[error("CameraRenderTask missing parameters")]
    MissingParameters,
    #[error("CameraRenderTask render space {0} is missing")]
    MissingRenderSpace(i32),
    #[error("CameraRenderTask render space {0} is inactive")]
    InactiveRenderSpace(i32),
    #[error("CameraRenderTask extent {width}x{height} is invalid")]
    InvalidExtent { width: i32, height: i32 },
    #[error("CameraRenderTask extent {width}x{height} exceeds max_texture_dimension_2d={max}")]
    ExtentExceedsLimit { width: u32, height: u32, max: u32 },
    #[error("CameraRenderTask format {0:?} is not supported for readback")]
    UnsupportedFormat(TextureFormat),
    #[error("CameraRenderTask fov {0} requests deferred equirectangular capture")]
    EquirectangularDeferred(f32),
    #[error("CameraRenderTask readback buffer {size} bytes exceeds device max_buffer_size={max}")]
    ReadbackBufferTooLarge { size: u64, max: u64 },
    #[error("CameraRenderTask output byte count overflow")]
    OutputByteCountOverflow,
    #[error("CameraRenderTask mapped readback is too small: need {required} bytes, got {actual}")]
    MappedReadbackTooSmall { required: usize, actual: usize },
    #[error(
        "CameraRenderTask result shared-memory descriptor is too small: need {required} bytes, got {actual}"
    )]
    ResultDescriptorTooSmall { required: usize, actual: usize },
    #[error("CameraRenderTask result shared-memory descriptor could not be mapped")]
    SharedMemoryMapFailed,
    #[error("CameraRenderTask render graph failed: {0}")]
    Graph(#[from] GraphExecuteError),
    #[error("device lost during CameraRenderTask readback poll: {0}")]
    DeviceLost(String),
    #[error("CameraRenderTask map_async timed out")]
    ReadbackTimeout,
    #[error("CameraRenderTask map_async failed: {0}")]
    Map(String),
}

#[derive(Default)]
struct CameraReadbackDrainStats {
    completed: u64,
    failed: u64,
}

struct CameraTaskTargets {
    color_texture: Arc<wgpu::Texture>,
    color_view: Arc<wgpu::TextureView>,
    depth_texture: Arc<wgpu::Texture>,
    depth_view: Arc<wgpu::TextureView>,
    color_format: wgpu::TextureFormat,
    extent: CameraTaskExtent,
}

impl CameraTaskTargets {
    fn create(gpu: &GpuContext, extent: CameraTaskExtent) -> Result<Self, CameraReadbackError> {
        let max_dim = gpu.limits().max_texture_dimension_2d();
        if extent.width > max_dim || extent.height > max_dim {
            return Err(CameraReadbackError::ExtentExceedsLimit {
                width: extent.width,
                height: extent.height,
                max: max_dim,
            });
        }
        let size = wgpu::Extent3d {
            width: extent.width,
            height: extent.height,
            depth_or_array_layers: 1,
        };
        let color_texture = Arc::new(gpu.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("renderide-camera-task-color"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: CAMERA_TASK_COLOR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        }));
        let color_view =
            Arc::new(color_texture.create_view(&wgpu::TextureViewDescriptor::default()));
        crate::profiling::note_resource_churn!(TextureView, "runtime::camera_task_color_view");

        let depth_texture = Arc::new(gpu.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("renderide-camera-task-depth"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: crate::gpu::main_forward_depth_stencil_format(gpu.device().features()),
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        }));
        let depth_view =
            Arc::new(depth_texture.create_view(&wgpu::TextureViewDescriptor::default()));
        crate::profiling::note_resource_churn!(TextureView, "runtime::camera_task_depth_view");

        Ok(Self {
            color_texture,
            color_view,
            depth_texture,
            depth_view,
            color_format: CAMERA_TASK_COLOR_FORMAT,
            extent,
        })
    }

    fn to_offscreen_handles(&self) -> OffscreenRtHandles {
        OffscreenRtHandles {
            rt_id: -1,
            color_view: Arc::clone(&self.color_view),
            depth_texture: Arc::clone(&self.depth_texture),
            depth_view: Arc::clone(&self.depth_view),
            color_format: self.color_format,
        }
    }
}

impl RendererRuntime {
    /// Appends host camera render tasks to the pre-begin-frame GPU readback queue.
    pub(super) fn queue_camera_render_tasks(&mut self, tasks: &[CameraRenderTask]) {
        profiling::scope!("camera_task::queue");
        if tasks.is_empty() {
            return;
        }
        self.tick_state
            .pending_camera_render_tasks
            .extend(tasks.iter().cloned());
        self.set_pending_camera_readbacks(self.tick_state.pending_camera_render_tasks.len());
        logger::debug!(
            "queued {} CameraRenderTask readback(s); pending={}",
            tasks.len(),
            self.tick_state.pending_camera_render_tasks.len()
        );
    }

    /// Drains queued camera readback tasks before the next host begin-frame is sent.
    pub fn drain_camera_render_tasks(&mut self, gpu: &mut GpuContext) {
        profiling::scope!("camera_task::drain");
        let mut tasks = std::mem::take(&mut self.tick_state.pending_camera_render_tasks);
        if tasks.is_empty() {
            self.set_pending_camera_readbacks(0);
            return;
        }
        self.set_pending_camera_readbacks(tasks.len());
        let mut stats = CameraReadbackDrainStats::default();
        let RendererRuntime {
            frontend,
            backend,
            scene,
            host_camera,
            diagnostics,
            ..
        } = self;
        let Some(shm) = frontend.shared_memory_mut() else {
            logger::warn!(
                "dropping {} CameraRenderTask readback(s): shared memory is unavailable",
                tasks.len()
            );
            stats.failed = tasks.len() as u64;
            diagnostics.set_pending_camera_readbacks(0);
            diagnostics.note_camera_readback_results(stats.completed, stats.failed);
            return;
        };

        backend.prepare_lights_from_scene(scene);
        let render_context = RenderingContext::RenderToAsset;
        let base_camera = *host_camera;
        let total = tasks.len();
        for (index, task) in tasks.drain(..).enumerate() {
            let task_index = i32::try_from(index).unwrap_or(i32::MAX);
            match render_camera_task(CameraTaskRenderCtx {
                gpu: &mut *gpu,
                backend: &mut *backend,
                scene,
                base_camera,
                shm: &mut *shm,
                render_context,
                task_index,
                task: &task,
            }) {
                Ok(()) => stats.completed = stats.completed.saturating_add(1),
                Err(error) => {
                    stats.failed = stats.failed.saturating_add(1);
                    logger::warn!(
                        "CameraRenderTask readback failed for render_space_id={} task_index={task_index}: {error}",
                        task.render_space_id
                    );
                    zero_task_result(shm, &task);
                }
            }
            let remaining = total.saturating_sub(index.saturating_add(1));
            diagnostics.set_pending_camera_readbacks(remaining);
        }
        diagnostics.set_pending_camera_readbacks(0);
        diagnostics.note_camera_readback_results(stats.completed, stats.failed);
        logger::debug!(
            "drained CameraRenderTask readbacks: completed={} failed={}",
            stats.completed,
            stats.failed
        );
    }
}

struct CameraTaskRenderCtx<'a> {
    gpu: &'a mut GpuContext,
    backend: &'a mut RenderBackend,
    scene: &'a SceneCoordinator,
    base_camera: crate::camera::HostCameraFrame,
    shm: &'a mut SharedMemoryAccessor,
    render_context: RenderingContext,
    task_index: i32,
    task: &'a CameraRenderTask,
}

fn render_camera_task(ctx: CameraTaskRenderCtx<'_>) -> Result<(), CameraReadbackError> {
    profiling::scope!("camera_task::render_one");
    let planned = plan_camera_task(
        ctx.gpu,
        ctx.scene,
        ctx.base_camera,
        ctx.task_index,
        ctx.task,
    )?;
    let view_id = planned.plan.view_id;
    render_camera_task_offscreen(
        ctx.gpu,
        ctx.backend,
        ctx.scene,
        ctx.render_context,
        planned.plan,
    )?;
    let rgba = match readback_camera_task_texture(ctx.gpu, planned.targets.color_texture.as_ref()) {
        Ok(rgba) => rgba,
        Err(error) => {
            ctx.backend.retire_one_shot_views(&[view_id]);
            return Err(error);
        }
    };
    ctx.backend.retire_one_shot_views(&[view_id]);
    write_camera_task_result(
        ctx.shm,
        ctx.task,
        planned.output_format,
        planned.targets.extent,
        &rgba,
    )
}

struct PlannedCameraTask {
    plan: FrameViewPlan<'static>,
    targets: CameraTaskTargets,
    output_format: CameraTaskOutputFormat,
}

fn plan_camera_task(
    gpu: &GpuContext,
    scene: &SceneCoordinator,
    base_camera: crate::camera::HostCameraFrame,
    task_index: i32,
    task: &CameraRenderTask,
) -> Result<PlannedCameraTask, CameraReadbackError> {
    profiling::scope!("camera_task::plan");
    let parameters = task
        .parameters
        .as_ref()
        .ok_or(CameraReadbackError::MissingParameters)?;
    if parameters.fov >= 180.0 {
        return Err(CameraReadbackError::EquirectangularDeferred(parameters.fov));
    }
    let output_format = CameraTaskOutputFormat::from_texture_format(parameters.texture_format)
        .ok_or(CameraReadbackError::UnsupportedFormat(
            parameters.texture_format,
        ))?;
    let extent = CameraTaskExtent::from_parameters(parameters)?;
    let required = output_byte_count(extent, output_format)?;
    let actual = usize::try_from(task.result_data.length).unwrap_or(0);
    if actual < required {
        return Err(CameraReadbackError::ResultDescriptorTooSmall { required, actual });
    }
    let render_space_id = RenderSpaceId(task.render_space_id);
    let Some(space) = scene.space(render_space_id) else {
        return Err(CameraReadbackError::MissingRenderSpace(
            task.render_space_id,
        ));
    };
    if !space.is_active() {
        return Err(CameraReadbackError::InactiveRenderSpace(
            task.render_space_id,
        ));
    }
    let targets = CameraTaskTargets::create(gpu, extent)?;
    let camera_world_matrix = camera_render_task_world_matrix(task.position, task.rotation);
    let host_camera = host_camera_frame_for_render_task(
        &base_camera,
        parameters,
        extent.tuple(),
        camera_world_matrix,
    );
    let filter = draw_filter_from_camera_render_task(task);
    Ok(PlannedCameraTask {
        plan: FrameViewPlan {
            host_camera,
            draw_filter: Some(filter),
            view_id: ViewId::camera_render_task(render_space_id, task_index),
            viewport_px: extent.tuple(),
            clear: FrameViewClear::from_camera_render_parameters(parameters),
            post_processing: ViewPostProcessing::from_camera_render_parameters(parameters),
            target: FrameViewPlanTarget::SecondaryRt(targets.to_offscreen_handles()),
        },
        targets,
        output_format,
    })
}

fn draw_filter_from_camera_render_task(task: &CameraRenderTask) -> CameraTransformDrawFilter {
    if task.only_render_list.is_empty() {
        CameraTransformDrawFilter {
            only: None,
            exclude: task.exclude_render_list.iter().copied().collect(),
        }
    } else {
        CameraTransformDrawFilter {
            only: Some(
                task.only_render_list
                    .iter()
                    .copied()
                    .collect::<HashSet<_>>(),
            ),
            exclude: HashSet::new(),
        }
    }
}

fn render_camera_task_offscreen(
    gpu: &mut GpuContext,
    backend: &mut RenderBackend,
    scene: &SceneCoordinator,
    render_context: RenderingContext,
    plan: FrameViewPlan<'static>,
) -> Result<(), CameraReadbackError> {
    profiling::scope!("camera_task::offscreen_render");
    let view_id = plan.view_id;
    let prepared_views = PreparedViews::new(vec![plan], None);
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
    let result = submit_frame.execute(gpu, scene, backend);
    if result.is_err() {
        backend.retire_one_shot_views(&[view_id]);
    }
    result.map_err(CameraReadbackError::Graph)
}

fn readback_camera_task_texture(
    gpu: &GpuContext,
    color_texture: &wgpu::Texture,
) -> Result<Vec<u8>, CameraReadbackError> {
    profiling::scope!("camera_task::gpu_copy_and_map");
    let layout = compute_readback_layout(color_texture.size(), gpu.limits().max_buffer_size())?;
    let readback = create_readback_buffer(gpu, &layout);
    submit_texture_to_buffer_copy(gpu, color_texture, &layout, &readback);
    let slice = readback.slice(..);
    await_buffer_map(slice, gpu.device())?;
    let tight = {
        profiling::scope!("camera_task::copy_padded_rows");
        let view = slice.get_mapped_range();
        copy_padded_rows_to_tight(&view, &layout)?
    };
    readback.unmap();
    Ok(tight)
}

fn compute_readback_layout(
    extent: wgpu::Extent3d,
    max_buffer_size: u64,
) -> Result<ReadbackLayout, CameraReadbackError> {
    let width = extent.width;
    let height = extent.height;
    if width == 0 || height == 0 {
        return Err(CameraReadbackError::InvalidExtent {
            width: i32::try_from(width).unwrap_or(i32::MAX),
            height: i32::try_from(height).unwrap_or(i32::MAX),
        });
    }
    let bytes_per_row_tight = width
        .checked_mul(RGBA8_BYTES_PER_PIXEL as u32)
        .ok_or(CameraReadbackError::OutputByteCountOverflow)?;
    let alignment = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let bytes_per_row_padded = bytes_per_row_tight
        .div_ceil(alignment)
        .checked_mul(alignment)
        .ok_or(CameraReadbackError::OutputByteCountOverflow)?;
    let buffer_size = u64::from(bytes_per_row_padded) * u64::from(height);
    if buffer_size > max_buffer_size {
        return Err(CameraReadbackError::ReadbackBufferTooLarge {
            size: buffer_size,
            max: max_buffer_size,
        });
    }
    Ok(ReadbackLayout {
        width,
        height,
        bytes_per_row_tight,
        bytes_per_row_padded,
        buffer_size,
    })
}

fn create_readback_buffer(gpu: &GpuContext, layout: &ReadbackLayout) -> wgpu::Buffer {
    let buffer = gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("renderide-camera-task-readback"),
        size: layout.buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    crate::profiling::note_resource_churn!(Buffer, "runtime::camera_task_readback_buffer");
    buffer
}

fn submit_texture_to_buffer_copy(
    gpu: &GpuContext,
    color_texture: &wgpu::Texture,
    layout: &ReadbackLayout,
    readback: &wgpu::Buffer,
) {
    profiling::scope!("camera_task::gpu_copy");
    gpu.flush_driver();
    let mut encoder = gpu
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("renderide-camera-task-readback"),
        });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: color_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(layout.bytes_per_row_padded),
                rows_per_image: Some(layout.height),
            },
        },
        wgpu::Extent3d {
            width: layout.width,
            height: layout.height,
            depth_or_array_layers: 1,
        },
    );
    let command_buffer = {
        profiling::scope!("CommandEncoder::finish::camera_task_readback");
        encoder.finish()
    };
    gpu.queue().submit(std::iter::once(command_buffer));
}

fn await_buffer_map(
    slice: wgpu::BufferSlice<'_>,
    device: &wgpu::Device,
) -> Result<(), CameraReadbackError> {
    profiling::scope!("camera_task::map_readback");
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| CameraReadbackError::DeviceLost(format!("{e:?}")))?;
    match receiver.recv_timeout(CAMERA_READBACK_TIMEOUT) {
        Ok(result) => result.map_err(|e| CameraReadbackError::Map(format!("{e:?}"))),
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
            Err(CameraReadbackError::ReadbackTimeout)
        }
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => Err(CameraReadbackError::Map(
            "map_async callback disconnected".to_owned(),
        )),
    }
}

fn copy_padded_rows_to_tight(
    bytes: &[u8],
    layout: &ReadbackLayout,
) -> Result<Vec<u8>, CameraReadbackError> {
    let required = usize::try_from(layout.buffer_size)
        .map_err(|_err| CameraReadbackError::OutputByteCountOverflow)?;
    if bytes.len() < required {
        return Err(CameraReadbackError::MappedReadbackTooSmall {
            required,
            actual: bytes.len(),
        });
    }
    let tight_len =
        usize::try_from(u64::from(layout.bytes_per_row_tight) * u64::from(layout.height))
            .map_err(|_err| CameraReadbackError::OutputByteCountOverflow)?;
    let mut tight = vec![0u8; tight_len];
    for row in 0..(layout.height as usize) {
        let src_start = row * layout.bytes_per_row_padded as usize;
        let src_end = src_start + layout.bytes_per_row_tight as usize;
        let dst_start = row * layout.bytes_per_row_tight as usize;
        let dst_end = dst_start + layout.bytes_per_row_tight as usize;
        tight[dst_start..dst_end].copy_from_slice(&bytes[src_start..src_end]);
    }
    Ok(tight)
}

fn write_camera_task_result(
    shm: &mut SharedMemoryAccessor,
    task: &CameraRenderTask,
    output_format: CameraTaskOutputFormat,
    extent: CameraTaskExtent,
    rgba: &[u8],
) -> Result<(), CameraReadbackError> {
    profiling::scope!("camera_task::shared_memory_write");
    let required = output_byte_count(extent, output_format)?;
    let mut result = Err(CameraReadbackError::SharedMemoryMapFailed);
    let mapped = shm.access_mut_bytes(&task.result_data, |bytes| {
        zero_bytes_simd(bytes);
        if bytes.len() < required {
            result = Err(CameraReadbackError::ResultDescriptorTooSmall {
                required,
                actual: bytes.len(),
            });
            return;
        }
        result = pack_rgba8_to_host_buffer(rgba, extent, output_format, &mut bytes[..required]);
    });
    if mapped {
        result
    } else {
        Err(CameraReadbackError::SharedMemoryMapFailed)
    }
}

fn output_byte_count(
    extent: CameraTaskExtent,
    output_format: CameraTaskOutputFormat,
) -> Result<usize, CameraReadbackError> {
    (extent.width as usize)
        .checked_mul(extent.height as usize)
        .and_then(|pixels| pixels.checked_mul(output_format.bytes_per_pixel()))
        .ok_or(CameraReadbackError::OutputByteCountOverflow)
}

fn pack_rgba8_to_host_buffer(
    rgba: &[u8],
    extent: CameraTaskExtent,
    output_format: CameraTaskOutputFormat,
    dst: &mut [u8],
) -> Result<(), CameraReadbackError> {
    let src_required = output_byte_count(extent, CameraTaskOutputFormat::Rgba32)?;
    let dst_required = output_byte_count(extent, output_format)?;
    if rgba.len() < src_required {
        return Err(CameraReadbackError::ResultDescriptorTooSmall {
            required: src_required,
            actual: rgba.len(),
        });
    }
    if dst.len() < dst_required {
        return Err(CameraReadbackError::ResultDescriptorTooSmall {
            required: dst_required,
            actual: dst.len(),
        });
    }

    let width = extent.width as usize;
    let height = extent.height as usize;
    let src_row_bytes = width * RGBA8_BYTES_PER_PIXEL;
    let dst_pixel_bytes = output_format.bytes_per_pixel();
    let dst_row_bytes = width * dst_pixel_bytes;
    // The host bitmap for render tasks carries FlipY metadata, so keep the
    // raw readback row order and only repack channels here.
    for dst_row in 0..height {
        let src_row_start = dst_row * src_row_bytes;
        let dst_row_start = dst_row * dst_row_bytes;
        for x in 0..width {
            let src = src_row_start + x * RGBA8_BYTES_PER_PIXEL;
            let dst_i = dst_row_start + x * dst_pixel_bytes;
            let r = rgba[src];
            let g = rgba[src + 1];
            let b = rgba[src + 2];
            let a = rgba[src + 3];
            match output_format {
                CameraTaskOutputFormat::Argb32 => {
                    dst[dst_i] = a;
                    dst[dst_i + 1] = r;
                    dst[dst_i + 2] = g;
                    dst[dst_i + 3] = b;
                }
                CameraTaskOutputFormat::Rgba32 => {
                    dst[dst_i] = r;
                    dst[dst_i + 1] = g;
                    dst[dst_i + 2] = b;
                    dst[dst_i + 3] = a;
                }
                CameraTaskOutputFormat::Bgra32 => {
                    dst[dst_i] = b;
                    dst[dst_i + 1] = g;
                    dst[dst_i + 2] = r;
                    dst[dst_i + 3] = a;
                }
                CameraTaskOutputFormat::Rgb24 => {
                    dst[dst_i] = r;
                    dst[dst_i + 1] = g;
                    dst[dst_i + 2] = b;
                }
            }
        }
    }
    Ok(())
}

pub(super) fn zero_camera_render_task_results(
    shm: &mut SharedMemoryAccessor,
    tasks: &[CameraRenderTask],
) -> usize {
    profiling::scope!("camera_task::zero_results");
    tasks
        .iter()
        .filter(|task| !zero_task_result(shm, task))
        .count()
}

fn zero_task_result(shm: &mut SharedMemoryAccessor, task: &CameraRenderTask) -> bool {
    profiling::scope!("camera_task::zero_result");
    let ok = shm.access_mut_bytes(&task.result_data, zero_bytes_simd);
    if !ok {
        logger::warn!(
            "CameraRenderTask zero-fill failed for result buffer_id={} offset={} length={}",
            task.result_data.buffer_id,
            task.result_data.offset,
            task.result_data.length
        );
    }
    ok
}

fn zero_bytes_simd(bytes: &mut [u8]) {
    if bytes.len() >= PAR_FILL_THRESHOLD {
        bytes
            .par_chunks_mut(PAR_FILL_CHUNK)
            .for_each(|chunk| chunk.fill(0));
    } else {
        bytes.fill(0);
    }
}

#[cfg(test)]
mod tests {
    use glam::IVec2;

    use super::*;

    #[test]
    fn readback_layout_removes_row_padding_contract() {
        let layout = compute_readback_layout(
            wgpu::Extent3d {
                width: 17,
                height: 3,
                depth_or_array_layers: 1,
            },
            4096,
        )
        .expect("layout");

        assert_eq!(layout.bytes_per_row_tight, 68);
        assert_eq!(
            layout.bytes_per_row_padded,
            wgpu::COPY_BYTES_PER_ROW_ALIGNMENT
        );
        assert_eq!(
            layout.buffer_size,
            u64::from(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * 3
        );
    }

    #[test]
    fn copy_padded_rows_to_tight_strips_padding() {
        let layout = ReadbackLayout {
            width: 2,
            height: 2,
            bytes_per_row_tight: 8,
            bytes_per_row_padded: 12,
            buffer_size: 24,
        };
        let padded = [
            1, 2, 3, 4, 5, 6, 7, 8, 99, 99, 99, 99, 10, 11, 12, 13, 14, 15, 16, 17, 88, 88, 88, 88,
        ];

        let tight = copy_padded_rows_to_tight(&padded, &layout).expect("copy rows");

        assert_eq!(
            tight,
            vec![1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17]
        );
    }

    #[test]
    fn pack_rgba8_preserves_rows_and_converts_formats() {
        let extent = CameraTaskExtent {
            width: 2,
            height: 2,
        };
        let rgba = [
            10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33, 40, 41, 42, 43,
        ];

        let mut argb = vec![0; 16];
        pack_rgba8_to_host_buffer(&rgba, extent, CameraTaskOutputFormat::Argb32, &mut argb)
            .expect("argb pack");
        assert_eq!(
            argb,
            vec![
                13, 10, 11, 12, 23, 20, 21, 22, 33, 30, 31, 32, 43, 40, 41, 42
            ]
        );

        let mut rgba_out = vec![0; 16];
        pack_rgba8_to_host_buffer(&rgba, extent, CameraTaskOutputFormat::Rgba32, &mut rgba_out)
            .expect("rgba pack");
        assert_eq!(
            rgba_out,
            vec![
                10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33, 40, 41, 42, 43
            ]
        );

        let mut bgra = vec![0; 16];
        pack_rgba8_to_host_buffer(&rgba, extent, CameraTaskOutputFormat::Bgra32, &mut bgra)
            .expect("bgra pack");
        assert_eq!(
            bgra,
            vec![
                12, 11, 10, 13, 22, 21, 20, 23, 32, 31, 30, 33, 42, 41, 40, 43
            ]
        );

        let mut rgb = vec![0; 12];
        pack_rgba8_to_host_buffer(&rgba, extent, CameraTaskOutputFormat::Rgb24, &mut rgb)
            .expect("rgb pack");
        assert_eq!(rgb, vec![10, 11, 12, 20, 21, 22, 30, 31, 32, 40, 41, 42]);
    }

    #[test]
    fn pack_rgba8_rejects_small_destination_without_writing_past_end() {
        let extent = CameraTaskExtent {
            width: 2,
            height: 1,
        };
        let rgba = [1, 2, 3, 4, 5, 6, 7, 8];
        let mut dst = [9u8; 7];

        let error =
            pack_rgba8_to_host_buffer(&rgba, extent, CameraTaskOutputFormat::Rgba32, &mut dst)
                .expect_err("small dst must fail");

        assert!(matches!(
            error,
            CameraReadbackError::ResultDescriptorTooSmall {
                required: 8,
                actual: 7
            }
        ));
        assert_eq!(dst, [9u8; 7]);
    }

    #[test]
    fn task_extent_rejects_invalid_dimensions() {
        let parameters = CameraRenderParameters {
            resolution: IVec2::new(-1, 4),
            ..Default::default()
        };

        assert!(matches!(
            CameraTaskExtent::from_parameters(&parameters),
            Err(CameraReadbackError::InvalidExtent {
                width: -1,
                height: 4
            })
        ));
    }

    #[test]
    fn draw_filter_prefers_only_render_list_over_excludes() {
        let task = CameraRenderTask {
            only_render_list: vec![1, 2],
            exclude_render_list: vec![3],
            ..Default::default()
        };

        let filter = draw_filter_from_camera_render_task(&task);

        assert!(filter.only.as_ref().is_some_and(|only| only.contains(&1)));
        assert!(filter.exclude.is_empty());
    }

    #[test]
    fn output_format_accepts_initial_cpu_formats() {
        assert_eq!(
            CameraTaskOutputFormat::from_texture_format(TextureFormat::ARGB32),
            Some(CameraTaskOutputFormat::Argb32)
        );
        assert_eq!(
            CameraTaskOutputFormat::from_texture_format(TextureFormat::RGBA32),
            Some(CameraTaskOutputFormat::Rgba32)
        );
        assert_eq!(
            CameraTaskOutputFormat::from_texture_format(TextureFormat::BGRA32),
            Some(CameraTaskOutputFormat::Bgra32)
        );
        assert_eq!(
            CameraTaskOutputFormat::from_texture_format(TextureFormat::RGB24),
            Some(CameraTaskOutputFormat::Rgb24)
        );
        assert_eq!(
            CameraTaskOutputFormat::from_texture_format(TextureFormat::RGBAHalf),
            None
        );
    }

    #[test]
    fn camera_render_task_post_processing_policy_matches_host_parameters() {
        let disabled = CameraRenderParameters {
            post_processing: false,
            screen_space_reflections: true,
            ..Default::default()
        };
        let disabled_policy = ViewPostProcessing::from_camera_render_parameters(&disabled);

        assert!(!disabled_policy.is_enabled());
        assert!(!disabled_policy.screen_space_reflections);
        assert!(!disabled_policy.motion_blur);

        let enabled = CameraRenderParameters {
            post_processing: true,
            screen_space_reflections: false,
            ..Default::default()
        };
        let enabled_policy = ViewPostProcessing::from_camera_render_parameters(&enabled);

        assert!(enabled_policy.is_enabled());
        assert!(!enabled_policy.screen_space_reflections);
        assert!(!enabled_policy.motion_blur);
    }
}
