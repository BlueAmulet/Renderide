//! [`CompiledRenderGraph`] execution: multi-view scheduling, resource resolution, and submits.

use std::collections::HashMap;

use winit::window::Window;

use crate::backend::RenderBackend;
use crate::gpu::GpuContext;
use crate::scene::SceneCoordinator;

use super::super::context::{
    GraphRasterPassContext, GraphResolvedResources, RenderPassContext, ResolvedGraphBuffer,
    ResolvedGraphTexture, ResolvedImportedBuffer, ResolvedImportedTexture,
};
use super::super::error::GraphExecuteError;
use super::super::frame_params::{HostCameraFrame, OcclusionViewId};
use super::super::pass::PassPhase;
use super::super::resources::{
    BackendFrameBufferKind, BufferImportSource, FrameTargetRole, ImportSource,
    ImportedBufferHandle, ImportedTextureHandle, TextureHandle,
};
use super::super::transient_pool::{BufferKey, TextureKey};
use super::helpers;
use super::{
    CompiledRenderGraph, ExternalFrameTargets, FrameView, FrameViewTarget,
    MultiViewExecutionContext, OffscreenSingleViewExecuteSpec, ResolvedView,
};

impl CompiledRenderGraph {
    /// Ordered pass count.
    pub fn pass_count(&self) -> usize {
        self.passes.len()
    }

    /// Whether this graph targets the swapchain this frame.
    pub fn needs_surface_acquire(&self) -> bool {
        self.needs_surface_acquire
    }

    /// Desktop single-view entry: delegates to [`Self::execute_multi_view`] (one swapchain view).
    ///
    /// Submit count follows the multi-view rules (optional frame-global encoder + submit, then
    /// per-view encoder + submit). Matches [`crate::present::present_clear_frame`] recovery behavior
    /// for surface acquire (timeout/occluded skip, validation reconfigure).
    pub fn execute(
        &mut self,
        gpu: &mut GpuContext,
        window: &Window,
        scene: &SceneCoordinator,
        backend: &mut RenderBackend,
        host_camera: HostCameraFrame,
    ) -> Result<(), GraphExecuteError> {
        self.execute_multi_view(
            gpu,
            window,
            scene,
            backend,
            vec![FrameView {
                host_camera,
                target: FrameViewTarget::Swapchain,
                draw_filter: None,
                prefetched_world_mesh_draws: None,
            }],
        )
    }

    /// Records passes against pre-built multiview array targets (OpenXR swapchain path).
    pub fn execute_external_multiview(
        &mut self,
        gpu: &mut GpuContext,
        window: &Window,
        scene: &SceneCoordinator,
        backend: &mut RenderBackend,
        host_camera: HostCameraFrame,
        external: ExternalFrameTargets<'_>,
    ) -> Result<(), GraphExecuteError> {
        self.execute_multi_view(
            gpu,
            window,
            scene,
            backend,
            vec![FrameView {
                host_camera,
                target: FrameViewTarget::ExternalMultiview(external),
                draw_filter: None,
                prefetched_world_mesh_draws: None,
            }],
        )
    }

    /// Renders the graph to a single-view offscreen color/depth target (secondary camera → render texture).
    pub fn execute_offscreen_single_view(
        &mut self,
        gpu: &mut GpuContext,
        backend: &mut RenderBackend,
        spec: OffscreenSingleViewExecuteSpec<'_>,
    ) -> Result<(), GraphExecuteError> {
        let window = spec.window;
        let scene = spec.scene;
        let host_camera = spec.host_camera;
        let external = spec.external;
        let transform_filter = spec.transform_filter;
        let prefetched_world_mesh_draws = spec.prefetched_world_mesh_draws;
        self.execute_multi_view(
            gpu,
            window,
            scene,
            backend,
            vec![FrameView {
                host_camera,
                target: FrameViewTarget::OffscreenRt(external),
                draw_filter: transform_filter,
                prefetched_world_mesh_draws,
            }],
        )
    }

    /// Records all views: one encoder + submit for frame-global work, then one encoder + submit per view.
    ///
    /// Per-view passes use [`wgpu::Queue::write_buffer`] for camera uniforms, per-draw slabs, and
    /// cluster params. Those writes are ordered **before** the next `queue.submit`; a single submit
    /// for all views would leave only the last view’s uploads visible to every view’s GPU commands,
    /// so each view is isolated in its own submit.
    ///
    /// Frame-global passes ([`PassPhase::FrameGlobal`]) run once in the first encoder; per-view
    /// passes run for each [`FrameView`] in order.
    pub fn execute_multi_view(
        &mut self,
        gpu: &mut GpuContext,
        window: &Window,
        scene: &SceneCoordinator,
        backend: &mut RenderBackend,
        mut views: Vec<FrameView<'_>>,
    ) -> Result<(), GraphExecuteError> {
        if views.is_empty() {
            return Ok(());
        }

        let needs_swapchain = views
            .iter()
            .any(|v| matches!(v.target, FrameViewTarget::Swapchain));

        let (frame, backbuffer_view_holder): (
            Option<wgpu::SurfaceTexture>,
            Option<wgpu::TextureView>,
        ) = match helpers::acquire_swapchain_for_multi_view_if_needed(
            needs_swapchain,
            self.needs_surface_acquire,
            gpu,
            window,
        )? {
            helpers::MultiViewSwapchainAcquire::NotNeeded => (None, None),
            helpers::MultiViewSwapchainAcquire::SkipPresent => return Ok(()),
            helpers::MultiViewSwapchainAcquire::Acquired {
                frame,
                backbuffer_view,
            } => (Some(frame), Some(backbuffer_view)),
        };

        let device_arc = gpu.device().clone();
        let queue_arc = gpu.queue().clone();
        let device = device_arc.as_ref();
        let gpu_limits_owned = gpu.limits().clone();
        let gpu_limits = gpu_limits_owned.as_ref();

        backend.transient_pool_mut().begin_generation();

        let mut mv_ctx = MultiViewExecutionContext {
            gpu,
            scene,
            backend,
            device,
            gpu_limits,
            queue_arc: &queue_arc,
            backbuffer_view_holder: &backbuffer_view_holder,
        };

        self.execute_multi_view_frame_global_passes(&mut mv_ctx, &views)?;

        // Per-view: separate encoder + submit so queue writes before each submit apply only to this view.
        for view in &mut views {
            self.execute_multi_view_submit_for_one_view(&mut mv_ctx, view)?;
        }

        mv_ctx.backend.transient_pool_mut().gc_tick(120);

        if let Some(f) = frame {
            f.present();
        }
        Ok(())
    }

    /// One per-view encoder, per-view [`PassPhase::PerView`] passes, submit, and Hi-Z bookkeeping.
    fn execute_multi_view_submit_for_one_view(
        &mut self,
        mv_ctx: &mut MultiViewExecutionContext<'_>,
        view: &mut FrameView<'_>,
    ) -> Result<(), GraphExecuteError> {
        let MultiViewExecutionContext {
            gpu,
            scene,
            backend,
            device,
            gpu_limits,
            queue_arc,
            backbuffer_view_holder,
        } = mv_ctx;

        let prefetched = view.prefetched_world_mesh_draws.take();
        let draw_filter = view.draw_filter.clone();
        let host_camera = view.host_camera;
        let target_is_swapchain = matches!(view.target, FrameViewTarget::Swapchain);
        let resolved = Self::resolve_view_from_target(&view.target, gpu, backbuffer_view_holder)?;
        let graph_resources = self.resolve_graph_resources_for_view(device, backend, &resolved);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render-graph-per-view"),
        });
        {
            let mut frame_params = helpers::frame_render_params_from_resolved(
                scene,
                backend,
                &resolved,
                host_camera,
                draw_filter,
                prefetched,
            );
            for (pass_idx, pass) in self.passes.iter_mut().enumerate() {
                if pass.phase() == PassPhase::PerView {
                    if pass.graph_managed_raster() {
                        let template =
                            helpers::pass_info_raster_template(&self.pass_info, pass_idx)?;
                        let mut ctx = GraphRasterPassContext {
                            device,
                            gpu_limits,
                            queue: queue_arc,
                            backbuffer: resolved.backbuffer,
                            depth_view: Some(resolved.depth_view),
                            frame: Some(&mut frame_params),
                            graph_resources: Some(&graph_resources),
                        };
                        helpers::execute_graph_managed_raster_pass(
                            pass.as_mut(),
                            &template,
                            &graph_resources,
                            &mut encoder,
                            &mut ctx,
                        )?;
                    } else {
                        let mut ctx = RenderPassContext {
                            device,
                            gpu_limits,
                            queue: queue_arc,
                            encoder: &mut encoder,
                            backbuffer: resolved.backbuffer,
                            depth_view: Some(resolved.depth_view),
                            frame: Some(&mut frame_params),
                            graph_resources: Some(&graph_resources),
                        };
                        pass.execute(&mut ctx)?;
                    }
                }
            }
        }
        graph_resources.release_to_pool(backend.transient_pool_mut());

        if target_is_swapchain {
            let Some(bb) = backbuffer_view_holder.as_ref() else {
                return Err(GraphExecuteError::MissingSwapchainView);
            };
            let viewport_px = gpu.surface_extent_px();
            let mut queue_lock = queue_arc.lock().expect("queue mutex poisoned");
            if let Err(e) =
                backend.encode_debug_hud_overlay(device, &queue_lock, &mut encoder, bb, viewport_px)
            {
                logger::warn!("debug HUD overlay: {e}");
            }
            let cmd = encoder.finish();
            gpu.submit_tracked_frame_commands_with_queue(&mut queue_lock, cmd);
        } else {
            let cmd = encoder.finish();
            gpu.submit_tracked_frame_commands(cmd);
        }

        let occlusion_view = view.occlusion_view_id();
        let mut post_ctx = PostSubmitContext {
            device,
            backend,
            occlusion_view,
            host_camera: view.host_camera,
        };
        for pass in self.passes.iter_mut() {
            if pass.phase() == PassPhase::PerView {
                pass.post_submit(&mut post_ctx)
                    .map_err(GraphExecuteError::Pass)?;
            }
        }
        Ok(())
    }

    /// Runs [`PassPhase::FrameGlobal`] passes once per tick using the first view for host/scene context.
    fn execute_multi_view_frame_global_passes(
        &mut self,
        mv_ctx: &mut MultiViewExecutionContext<'_>,
        views: &[FrameView<'_>],
    ) -> Result<(), GraphExecuteError> {
        let MultiViewExecutionContext {
            gpu,
            scene,
            backend,
            device,
            gpu_limits,
            queue_arc,
            backbuffer_view_holder,
        } = mv_ctx;

        let has_frame_global = self
            .passes
            .iter()
            .any(|p| p.phase() == PassPhase::FrameGlobal);
        if !has_frame_global {
            return Ok(());
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render-graph-frame-global"),
        });
        // Frame-global phase (e.g. mesh deform): use first view for host camera / scene context.
        {
            let first = views.first().expect("views non-empty");
            let resolved =
                Self::resolve_view_from_target(&first.target, gpu, backbuffer_view_holder)?;
            let graph_resources = self.resolve_graph_resources_for_view(device, backend, &resolved);
            {
                let mut frame_params = helpers::frame_render_params_from_resolved(
                    scene,
                    backend,
                    &resolved,
                    first.host_camera,
                    first.draw_filter.clone(),
                    None,
                );
                for (pass_idx, pass) in self.passes.iter_mut().enumerate() {
                    if pass.phase() == PassPhase::FrameGlobal {
                        if pass.graph_managed_raster() {
                            let template =
                                helpers::pass_info_raster_template(&self.pass_info, pass_idx)?;
                            let mut ctx = GraphRasterPassContext {
                                device,
                                gpu_limits,
                                queue: queue_arc,
                                backbuffer: None,
                                depth_view: None,
                                frame: Some(&mut frame_params),
                                graph_resources: Some(&graph_resources),
                            };
                            helpers::execute_graph_managed_raster_pass(
                                pass.as_mut(),
                                &template,
                                &graph_resources,
                                &mut encoder,
                                &mut ctx,
                            )?;
                        } else {
                            let mut ctx = RenderPassContext {
                                device,
                                gpu_limits,
                                queue: queue_arc,
                                encoder: &mut encoder,
                                backbuffer: None,
                                depth_view: None,
                                frame: Some(&mut frame_params),
                                graph_resources: Some(&graph_resources),
                            };
                            pass.execute(&mut ctx)?;
                        }
                    }
                }
            }
            graph_resources.release_to_pool(backend.transient_pool_mut());
        }
        let cmd = encoder.finish();
        gpu.submit_tracked_frame_commands(cmd);

        let first = views.first().expect("views non-empty");
        let occlusion_view = first.occlusion_view_id();
        let host_camera = first.host_camera;
        let mut post_ctx = PostSubmitContext {
            device,
            backend,
            occlusion_view,
            host_camera,
        };
        for pass in self.passes.iter_mut() {
            if pass.phase() == PassPhase::FrameGlobal {
                pass.post_submit(&mut post_ctx)
                    .map_err(GraphExecuteError::Pass)?;
            }
        }
        Ok(())
    }

    fn resolve_graph_resources_for_view(
        &self,
        device: &wgpu::Device,
        backend: &mut RenderBackend,
        resolved: &ResolvedView<'_>,
    ) -> GraphResolvedResources {
        let mut resources = GraphResolvedResources::with_capacity(
            self.transient_textures.len(),
            self.transient_buffers.len(),
            self.imported_textures.len(),
            self.imported_buffers.len(),
        );
        self.resolve_transient_textures(
            device,
            backend,
            resolved.viewport_px,
            resolved.surface_format,
            resolved.sample_count,
            resolved.multiview_stereo,
            &mut resources,
        );
        self.resolve_transient_buffers(device, backend, resolved.viewport_px, &mut resources);
        self.resolve_imported_textures(resolved, &mut resources);
        self.resolve_imported_buffers(backend, resolved, &mut resources);
        resources
    }

    #[allow(clippy::too_many_arguments)]
    fn resolve_transient_textures(
        &self,
        device: &wgpu::Device,
        backend: &mut RenderBackend,
        viewport_px: (u32, u32),
        surface_format: wgpu::TextureFormat,
        sample_count: u32,
        multiview_stereo: bool,
        resources: &mut GraphResolvedResources,
    ) {
        let mut physical_slots: HashMap<usize, ResolvedGraphTexture> = HashMap::new();
        for (idx, compiled) in self.transient_textures.iter().enumerate() {
            if compiled.lifetime.is_none() || compiled.physical_slot == usize::MAX {
                continue;
            }
            let resolved = physical_slots
                .entry(compiled.physical_slot)
                .or_insert_with(|| {
                    let array_layers = compiled.desc.array_layers.resolve(multiview_stereo);
                    let key = TextureKey {
                        format: compiled.desc.format.resolve(surface_format),
                        extent: helpers::resolve_transient_extent(
                            compiled.desc.extent,
                            viewport_px,
                            array_layers,
                        ),
                        mip_levels: compiled.desc.mip_levels,
                        sample_count: compiled.desc.sample_count.resolve(sample_count),
                        dimension: compiled.desc.dimension,
                        array_layers,
                        usage_bits: compiled.usage.bits() as u64,
                    };
                    let lease = backend.transient_pool_mut().acquire_texture_resource(
                        device,
                        key,
                        compiled.desc.label,
                        compiled.usage,
                    );
                    let layer_views = helpers::create_transient_layer_views(&lease.texture, key);
                    ResolvedGraphTexture {
                        pool_id: lease.pool_id,
                        physical_slot: compiled.physical_slot,
                        texture: lease.texture,
                        view: lease.view,
                        layer_views,
                    }
                })
                .clone();
            resources.set_transient_texture(TextureHandle(idx as u32), resolved);
        }
    }

    fn resolve_transient_buffers(
        &self,
        device: &wgpu::Device,
        backend: &mut RenderBackend,
        viewport_px: (u32, u32),
        resources: &mut GraphResolvedResources,
    ) {
        let mut physical_slots: HashMap<usize, ResolvedGraphBuffer> = HashMap::new();
        for (idx, compiled) in self.transient_buffers.iter().enumerate() {
            if compiled.lifetime.is_none() || compiled.physical_slot == usize::MAX {
                continue;
            }
            let resolved = physical_slots
                .entry(compiled.physical_slot)
                .or_insert_with(|| {
                    let key = BufferKey {
                        size_policy: compiled.desc.size_policy,
                        usage_bits: compiled.usage.bits() as u64,
                    };
                    let size = helpers::resolve_buffer_size(compiled.desc.size_policy, viewport_px);
                    let lease = backend.transient_pool_mut().acquire_buffer_resource(
                        device,
                        key,
                        compiled.desc.label,
                        compiled.usage,
                        size,
                    );
                    ResolvedGraphBuffer {
                        pool_id: lease.pool_id,
                        physical_slot: compiled.physical_slot,
                        buffer: lease.buffer,
                        size: lease.size,
                    }
                })
                .clone();
            resources
                .set_transient_buffer(super::super::resources::BufferHandle(idx as u32), resolved);
        }
    }

    fn resolve_imported_textures(
        &self,
        resolved: &ResolvedView<'_>,
        resources: &mut GraphResolvedResources,
    ) {
        for (idx, import) in self.imported_textures.iter().enumerate() {
            let view = match &import.source {
                ImportSource::FrameTarget(FrameTargetRole::ColorAttachment) => {
                    resolved.backbuffer.cloned()
                }
                ImportSource::FrameTarget(FrameTargetRole::DepthAttachment) => {
                    Some(resolved.depth_view.clone())
                }
                ImportSource::External | ImportSource::PingPong(_) => None,
            };
            if let Some(view) = view {
                resources.set_imported_texture(
                    ImportedTextureHandle(idx as u32),
                    ResolvedImportedTexture { view },
                );
            }
        }
    }

    fn resolve_imported_buffers(
        &self,
        backend: &RenderBackend,
        resolved: &ResolvedView<'_>,
        resources: &mut GraphResolvedResources,
    ) {
        let frame_gpu = backend.frame_resources.frame_gpu();
        let cluster_refs = frame_gpu.and_then(|fgpu| {
            fgpu.cluster_cache.get_buffers(
                resolved.viewport_px,
                crate::backend::CLUSTER_COUNT_Z,
                resolved.multiview_stereo,
            )
        });
        for (idx, import) in self.imported_buffers.iter().enumerate() {
            let buffer = match &import.source {
                BufferImportSource::BackendFrameResource(BackendFrameBufferKind::Lights) => {
                    frame_gpu.map(|fgpu| fgpu.lights_buffer.clone())
                }
                BufferImportSource::BackendFrameResource(BackendFrameBufferKind::FrameUniforms) => {
                    frame_gpu.map(|fgpu| fgpu.frame_uniform.clone())
                }
                BufferImportSource::BackendFrameResource(
                    BackendFrameBufferKind::ClusterLightCounts,
                ) => cluster_refs
                    .as_ref()
                    .map(|refs| refs.cluster_light_counts.clone()),
                BufferImportSource::BackendFrameResource(
                    BackendFrameBufferKind::ClusterLightIndices,
                ) => cluster_refs
                    .as_ref()
                    .map(|refs| refs.cluster_light_indices.clone()),
                BufferImportSource::BackendFrameResource(BackendFrameBufferKind::PerDrawSlab) => {
                    backend
                        .frame_resources
                        .per_draw()
                        .map(|per_draw| per_draw.per_draw_storage.clone())
                }
                BufferImportSource::External | BufferImportSource::PingPong(_) => None,
            };
            if let Some(buffer) = buffer {
                resources.set_imported_buffer(
                    ImportedBufferHandle(idx as u32),
                    ResolvedImportedBuffer { buffer },
                );
            }
        }
    }

    fn resolve_view_from_target<'a>(
        target: &'a FrameViewTarget<'a>,
        gpu: &'a mut GpuContext,
        backbuffer_view_holder: &'a Option<wgpu::TextureView>,
    ) -> Result<ResolvedView<'a>, GraphExecuteError> {
        match target {
            FrameViewTarget::Swapchain => {
                let surface_format = gpu.config_format();
                let viewport_px = gpu.surface_extent_px();
                let bb = backbuffer_view_holder
                    .as_ref()
                    .map(|v| v as &wgpu::TextureView);
                let Some(bb_ref) = bb else {
                    return Err(GraphExecuteError::MissingSwapchainView);
                };
                let sample_count = gpu.swapchain_msaa_effective().max(1);
                let (depth_tex, depth_view) = gpu
                    .ensure_depth_target()
                    .map_err(|_| GraphExecuteError::DepthTarget)?;

                Ok(ResolvedView {
                    depth_texture: depth_tex,
                    depth_view,
                    backbuffer: Some(bb_ref),
                    surface_format,
                    viewport_px,
                    multiview_stereo: false,
                    offscreen_write_render_texture_asset_id: None,
                    occlusion_view: OcclusionViewId::Main,
                    sample_count,
                })
            }
            FrameViewTarget::ExternalMultiview(ext) => {
                let sample_count = gpu.swapchain_msaa_effective_stereo().max(1);
                Ok(ResolvedView {
                    depth_texture: ext.depth_texture,
                    depth_view: ext.depth_view,
                    backbuffer: Some(ext.color_view),
                    surface_format: ext.surface_format,
                    viewport_px: ext.extent_px,
                    multiview_stereo: true,
                    offscreen_write_render_texture_asset_id: None,
                    occlusion_view: OcclusionViewId::Main,
                    sample_count,
                })
            }
            FrameViewTarget::OffscreenRt(ext) => Ok(ResolvedView {
                depth_texture: ext.depth_texture,
                depth_view: ext.depth_view,
                backbuffer: Some(ext.color_view),
                surface_format: ext.color_format,
                viewport_px: ext.extent_px,
                multiview_stereo: false,
                offscreen_write_render_texture_asset_id: Some(ext.render_texture_asset_id),
                occlusion_view: OcclusionViewId::OffscreenRenderTexture(
                    ext.render_texture_asset_id,
                ),
                sample_count: 1,
            }),
        }
    }
}
