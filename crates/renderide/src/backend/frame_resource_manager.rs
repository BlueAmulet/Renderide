//! Per-frame GPU bind groups, per-view light staging, shared cluster buffers, and per-view
//! per-draw instance resources.
//!
//! [`FrameResourceManager`] owns the fallback `@group(0)` frame resources
//! ([`FrameGpuResources`]), the empty `@group(1)` fallback ([`EmptyMaterialBindGroup`]),
//! per-view frame/light bind resources ([`PerViewFrameState`]), a `@group(2)` per-draw instance
//! storage slab per render view ([`PerDrawResources`]), and the CPU-side packed light buffers
//! used by [`crate::passes::ClusteredLightPass`] and the forward pass.
//!
//! Cluster buffers are shared through [`FrameGpuResources`] and grow before graph recording so
//! every planned viewport has enough dynamic index storage for its current light pack. Per-view
//! state is keyed by [`ViewId`] and created lazily on first use; retired explicitly when a
//! secondary RT camera is destroyed.
//!
//! Per-draw resources follow the same ownership model: one grow-on-demand slab per
//! [`ViewId`], created lazily so no view can exhaust another view's per-draw capacity.

mod cluster_layout;
mod per_view_state;
mod view_desc;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(test)]
use glam::Mat4;
use hashbrown::HashSet;
use parking_lot::Mutex;

use crate::backend::cluster_gpu::ClusterBufferRefs;
use crate::camera::ViewId;
use crate::gpu::GpuLimits;
use crate::gpu::frame_globals::{FrameGpuUniforms, SkyboxSpecularUniformParams};
use crate::render_graph::execution_backend::{GraphClusterBufferRefs, GraphFrameResources};
use crate::render_graph::frame_params::PreRecordViewResourceLayout;
use crate::render_graph::frame_upload_batch::GraphUploadSink;

use super::frame_gpu::{
    EmptyMaterialBindGroup, FrameGpuResources, PerViewSceneSnapshots,
    ReflectionProbeSpecularResources,
};
use super::frame_gpu_bindings::{FrameGpuBindings, FrameGpuBindingsError};
use super::light_gpu::{
    GpuLight, MAX_LIGHTS, gpu_light_from_resolved, order_lights_for_clustered_shading_in_place,
};
use super::per_draw_resources::PerDrawResources;
use super::per_view_resource_map::PerViewResourceMap;
use crate::mesh_deform::{PaddedPerDrawUniforms, SkinCacheKey};
use crate::scene::{
    ResolvedLight, SceneCoordinator, light_contributes, light_has_negative_contribution,
};

use cluster_layout::{
    cluster_index_capacity_for_layout, make_cluster_params_buffer, per_view_snapshot_sync_params,
    unique_cluster_pre_record_layouts,
};
use per_view_state::PreparedViewLights;

pub(crate) use view_desc::FrameLightViewDesc;
pub use per_view_state::{PerViewFrameState, PerViewPerDrawScratch};

/// Per-frame GPU state: shared frame/light/cluster resources, per-view bind groups,
/// per-view per-draw storage slabs, and the CPU-side packed light buffer.
pub struct FrameResourceManager {
    /// Shared `@group(0)` frame globals (lights, fallback snapshots, bind group layout).
    pub(crate) frame_gpu: Option<FrameGpuResources>,
    /// Placeholder `@group(1)` for materials without per-material bindings.
    pub(crate) empty_material: Option<EmptyMaterialBindGroup>,
    /// Per-view frame uniform buffer and `@group(0)` bind group.
    ///
    /// Created lazily on first use per [`ViewId`]; retired when a secondary RT camera
    /// is destroyed via [`Self::retire_per_view_frame`].
    per_view_frame: PerViewResourceMap<PerViewFrameState>,
    /// One grow-on-demand per-draw slab per stable render-view identity.
    ///
    /// Created lazily; keyed by [`ViewId`] so secondary RT cameras never compete
    /// with the main view (or each other) for buffer space.
    per_view_draw: PerViewResourceMap<Mutex<PerDrawResources>>,
    /// Shared `@group(2)` bind group layout, reflected once at attach time.
    per_draw_bind_group_layout: Option<Arc<wgpu::BindGroupLayout>>,
    /// GPU limits stored at attach time for lazy per-view slab/cluster creation.
    limits: Option<Arc<GpuLimits>>,
    /// Last packed lights for the first prepared view, retained for diagnostics and fallback callers.
    light_scratch: Vec<GpuLight>,
    /// Per-view packed light sets keyed by render view identity.
    per_view_lights: PerViewResourceMap<PreparedViewLights>,
    /// Whether any packed light set subtracts in at least one signed-radiance channel.
    signed_scene_color_required: bool,
    /// Reused each frame to flatten all spaces' [`crate::scene::ResolvedLight`] before ordering and GPU pack.
    resolved_flatten_scratch: Vec<ResolvedLight>,
    /// Reused each view to collect active render spaces that should contribute lights.
    light_space_ids_scratch: Vec<crate::scene::RenderSpaceId>,
    /// When true, [`crate::passes::MeshDeformPass`] already dispatched this tick.
    ///
    /// In VR, the HMD graph runs mesh deform first; secondary cameras skip it via this flag.
    /// Reset with [`Self::reset_light_prep_for_tick`].
    mesh_deform_dispatched_this_tick: AtomicBool,
    /// Optional visible deform filter derived from prefetched per-view draw lists.
    visible_mesh_deform_keys: Mutex<Option<HashSet<SkinCacheKey>>>,
    /// Reused per-view scratch for per-draw VP/pack before [`crate::mesh_deform::write_per_draw_uniform_slab`].
    ///
    /// Each view owns its own mutex-wrapped slot so rayon workers never alias the same scratch.
    per_view_per_draw_scratch: PerViewResourceMap<Mutex<PerViewPerDrawScratch>>,
    /// One-shot guard for the [`MAX_LIGHTS`] overflow warning so a content scene with too many
    /// lights does not spam logs every frame.
    lights_overflow_warned: bool,
    /// One-shot guard for the signed scene-color activation log.
    signed_scene_color_required_logged: bool,
}

impl Default for FrameResourceManager {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameResourceManager {
    /// Creates an empty manager with no GPU resources.
    pub fn new() -> Self {
        Self {
            frame_gpu: None,
            empty_material: None,
            per_view_frame: PerViewResourceMap::new(),
            per_view_draw: PerViewResourceMap::new(),
            per_draw_bind_group_layout: None,
            limits: None,
            light_scratch: Vec::new(),
            per_view_lights: PerViewResourceMap::new(),
            signed_scene_color_required: false,
            resolved_flatten_scratch: Vec::new(),
            light_space_ids_scratch: Vec::new(),
            mesh_deform_dispatched_this_tick: AtomicBool::new(false),
            visible_mesh_deform_keys: Mutex::new(None),
            per_view_per_draw_scratch: PerViewResourceMap::new(),
            lights_overflow_warned: false,
            signed_scene_color_required_logged: false,
        }
    }

    /// Allocates GPU resources for this manager. Called from [`super::RenderBackend::attach`].
    ///
    /// On success, `@group(0)` / `@group(1)` / `@group(2)` layout are present.
    /// `queue` initializes fallback sampled textures used by group-0 bindings.
    /// Per-view per-draw slabs and per-view frame bind resources are created lazily on first use.
    /// On error, frame bind fields remain unset (no partial attach).
    pub fn attach(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        limits: Arc<GpuLimits>,
    ) -> Result<(), FrameGpuBindingsError> {
        let binds = FrameGpuBindings::try_new(device, queue, Arc::clone(&limits))?;
        self.frame_gpu = Some(binds.frame_gpu);
        self.empty_material = Some(binds.empty_material);
        self.per_draw_bind_group_layout = Some(binds.per_draw_bind_group_layout);
        self.limits = Some(limits);
        Ok(())
    }

    /// Clears per-tick frame-resource flags. Call once per winit frame from
    /// [`crate::runtime::RendererRuntime::tick_frame_wall_clock_begin`].
    ///
    /// The flag store uses [`Ordering::Release`] so a worker that observes the cleared state on
    /// the next tick is guaranteed to see the prior tick's GPU writes that produced the work.
    pub fn reset_light_prep_for_tick(&self) {
        self.mesh_deform_dispatched_this_tick
            .store(false, Ordering::Release);
        *self.visible_mesh_deform_keys.lock() = None;
    }

    /// Whether [`crate::passes::MeshDeformPass`] already dispatched this tick.
    ///
    /// Acquire-load pairs with the [`Ordering::Release`] store in
    /// [`Self::set_mesh_deform_dispatched_this_tick`] so a multi-view worker that sees `true` is
    /// guaranteed to see the prior dispatch's encoder/queue writes.
    pub fn mesh_deform_dispatched_this_tick(&self) -> bool {
        self.mesh_deform_dispatched_this_tick
            .load(Ordering::Acquire)
    }

    /// Marks mesh deform as dispatched for this tick.
    pub fn set_mesh_deform_dispatched_this_tick(&self) {
        self.mesh_deform_dispatched_this_tick
            .store(true, Ordering::Release);
    }

    /// Replaces the optional visible deform filter for this graph frame.
    pub fn set_visible_mesh_deform_keys(&mut self, keys: HashSet<SkinCacheKey>) {
        *self.visible_mesh_deform_keys.get_mut() = Some(keys);
    }

    /// Clones the current visible deform filter for lock-free worker iteration.
    pub fn visible_mesh_deform_keys_snapshot(&self) -> Option<HashSet<SkinCacheKey>> {
        self.visible_mesh_deform_keys.lock().clone()
    }

    /// Returns `true` when draw collection proved there is no visible deform work this frame.
    pub fn visible_mesh_deform_filter_is_empty(&self) -> bool {
        self.visible_mesh_deform_keys
            .lock()
            .as_ref()
            .is_some_and(HashSet::is_empty)
    }

    /// Packed GPU lights from the last [`Self::prepare_lights_from_scene`] call.
    pub fn frame_lights(&self) -> &[GpuLight] {
        &self.light_scratch
    }

    /// Packed GPU lights for `view_id`, falling back to the last default frame pack.
    pub fn frame_lights_for_view(&self, view_id: ViewId) -> &[GpuLight] {
        self.per_view_lights
            .get(view_id)
            .map_or(self.light_scratch.as_slice(), |lights| {
                lights.lights.as_slice()
            })
    }

    /// Returns true when the current packed light set needs signed scene-color storage.
    pub fn signed_scene_color_required(&self) -> bool {
        self.signed_scene_color_required
    }

    /// Light count for the specified view's frame uniforms and shaders.
    pub fn frame_light_count_for_view_u32(&self, view_id: ViewId) -> u32 {
        self.frame_lights_for_view(view_id).len().min(MAX_LIGHTS) as u32
    }

    /// Shared `@group(0)` frame globals (camera + lights), after attach.
    pub fn frame_gpu(&self) -> Option<&FrameGpuResources> {
        self.frame_gpu.as_ref()
    }

    /// Mutable shared frame globals (cluster resize, uniform upload).
    pub fn frame_gpu_mut(&mut self) -> Option<&mut FrameGpuResources> {
        self.frame_gpu.as_mut()
    }

    /// Empty `@group(1)` bind group for shaders without per-material bindings.
    pub fn empty_material(&self) -> Option<&EmptyMaterialBindGroup> {
        self.empty_material.as_ref()
    }

    /// Returns the per-view frame state for `view_id`, creating it lazily if it does not exist.
    ///
    /// Grows the shared cluster buffers (on [`FrameGpuResources`]) to cover this view's
    /// layout in `layout` when needed and rebuilds the `@group(0)` bind group whenever the
    /// shared cluster buffers, reflection-probe resources, or this view's snapshots change.
    ///
    /// Returns `None` when the manager has not been attached (no GPU resources available) or
    /// when cluster buffers cannot be allocated for the given viewport.
    pub fn per_view_frame_or_create(
        &mut self,
        view_id: ViewId,
        device: &wgpu::Device,
        layout: PreRecordViewResourceLayout,
    ) -> Option<&mut PerViewFrameState> {
        profiling::scope!("render::ensure_per_view_frame");
        let limits = Arc::clone(self.limits.as_ref()?);
        let viewport = (layout.width, layout.height);
        let stereo = layout.stereo;
        let index_capacity_words = cluster_index_capacity_for_layout(
            layout,
            self.frame_light_count_for_view_u32(view_id),
        )?;
        let snapshot_sync = per_view_snapshot_sync_params(layout);

        let per_view_frame = &mut self.per_view_frame;
        let frame_gpu_opt = &mut self.frame_gpu;
        let fgpu = frame_gpu_opt.as_mut()?;
        // Grow the shared cluster buffers to cover this view if needed; `sync_cluster_viewport`
        // is grow-only so repeated calls from different views consolidate to the max envelope.
        fgpu.sync_cluster_viewport(device, viewport, stereo, index_capacity_words)?;
        let cluster_ver = fgpu.cluster_cache.version;
        let skybox_specular_version = fgpu.skybox_specular_version();

        if !per_view_frame.contains_key(view_id) {
            let frame_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("per_view_frame_uniform"),
                size: size_of::<FrameGpuUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            crate::profiling::note_resource_churn!(Buffer, "backend::per_view_frame_uniform");
            let lights_buffer =
                FrameGpuResources::create_lights_storage_buffer(device, "per_view_lights_storage");
            crate::profiling::note_resource_churn!(Buffer, "backend::per_view_lights_storage");
            let cluster_params_buffer = make_cluster_params_buffer(device, stereo);
            let mut scene_snapshots =
                PerViewSceneSnapshots::new(device, layout.depth_format, layout.color_format);
            scene_snapshots.sync(device, limits.as_ref(), snapshot_sync);
            let refs = fgpu.cluster_cache.current_refs()?;
            let frame_bind_group = fgpu.build_per_view_bind_group(
                device,
                &frame_uniform_buffer,
                &lights_buffer,
                refs,
                scene_snapshots.views(),
            );
            logger::debug!("per-view frame state: allocating for view {view_id:?}");
            let state = PerViewFrameState {
                frame_uniform_buffer,
                lights_buffer,
                frame_bind_group,
                cluster_params_buffer,
                scene_snapshots,
                last_cluster_version: cluster_ver,
                last_skybox_specular_version: skybox_specular_version,
                last_stereo: stereo,
            };
            let _ = per_view_frame.get_or_insert_with(view_id, || state);
        }

        let entry = per_view_frame.get_mut(view_id)?;

        // Resize per-view params buffer on mono->stereo transition (grow-only for consistency).
        if stereo && !entry.last_stereo {
            entry.cluster_params_buffer = make_cluster_params_buffer(device, true);
            entry.last_stereo = true;
        }

        let snapshots_changed = entry
            .scene_snapshots
            .sync(device, limits.as_ref(), snapshot_sync);
        let needs_rebuild = cluster_ver != entry.last_cluster_version
            || skybox_specular_version != entry.last_skybox_specular_version
            || snapshots_changed;

        if needs_rebuild {
            let refs = fgpu.cluster_cache.current_refs()?;
            let new_bg = fgpu.build_per_view_bind_group(
                device,
                &entry.frame_uniform_buffer,
                &entry.lights_buffer,
                refs,
                entry.scene_snapshots.views(),
            );
            entry.frame_bind_group = new_bg;
            entry.last_cluster_version = cluster_ver;
            entry.last_skybox_specular_version = skybox_specular_version;
        }

        per_view_frame.get_mut(view_id)
    }

    /// Uniform parameters for the disabled direct skybox specular slot.
    pub fn skybox_specular_uniform_params(&self) -> SkyboxSpecularUniformParams {
        self.frame_gpu.as_ref().map_or_else(
            SkyboxSpecularUniformParams::disabled,
            FrameGpuResources::skybox_specular_uniform_params,
        )
    }

    /// Synchronizes the frame-global reflection-probe specular resources.
    pub fn sync_reflection_probe_specular_resources(
        &mut self,
        device: &wgpu::Device,
        resources: Option<ReflectionProbeSpecularResources>,
    ) -> bool {
        self.frame_gpu
            .as_mut()
            .is_some_and(|fgpu| fgpu.sync_reflection_probe_specular_resources(device, resources))
    }

    /// Refs to the shared cluster buffers (see [`ClusterBufferCache`]). All views share these.
    pub fn shared_cluster_buffer_refs(&self) -> Option<ClusterBufferRefs<'_>> {
        self.frame_gpu.as_ref()?.cluster_cache.current_refs()
    }

    /// Current [`ClusterBufferCache::version`] on the shared cache. Used for bind-group
    /// invalidation caches that key on cluster-buffer reallocations.
    pub fn shared_cluster_version(&self) -> u64 {
        self.frame_gpu
            .as_ref()
            .map_or(0, |fgpu| fgpu.cluster_cache.version)
    }

    /// Returns the per-view frame state for `view_id`, or `None` if not yet created.
    pub fn per_view_frame(&self, view_id: ViewId) -> Option<&PerViewFrameState> {
        self.per_view_frame.get(view_id)
    }

    /// Frees per-view frame bind resources for a view that is no longer active.
    ///
    /// Call alongside [`Self::retire_per_view_per_draw`] when a secondary RT camera is destroyed.
    /// Has no effect if the view was never allocated.
    pub fn retire_per_view_frame(&mut self, view_id: ViewId) {
        if self.per_view_frame.retire(view_id) {
            logger::debug!("per-view frame state: retired for view {view_id:?}");
        }
    }

    /// Returns the per-draw slab for the given view, creating it if it does not yet exist.
    ///
    /// Returns `None` when the manager has not been attached (no device limits / layout available).
    pub fn per_view_per_draw_or_create(
        &mut self,
        view_id: ViewId,
        device: &wgpu::Device,
    ) -> Option<&Mutex<PerDrawResources>> {
        profiling::scope!("render::ensure_per_view_per_draw");
        let layout = self.per_draw_bind_group_layout.clone()?;
        let limits = self.limits.clone()?;
        let _ = self.per_view_per_draw_scratch_or_create(view_id);
        Some(self.per_view_draw.get_or_insert_with(view_id, || {
            logger::debug!("per-draw slab: allocating new slab for view {view_id:?}");
            Mutex::new(PerDrawResources::new_with_layout(device, layout, limits))
        }))
    }

    /// Returns the per-draw slab for the given view, or `None` if it has not been created yet.
    pub fn per_view_per_draw(&self, view_id: ViewId) -> Option<&Mutex<PerDrawResources>> {
        self.per_view_draw.get(view_id)
    }

    /// Frees the per-draw slab for a view that is no longer active (e.g. render-texture camera destroyed).
    ///
    /// Has no effect if the view was never allocated.
    pub fn retire_per_view_per_draw(&mut self, view_id: ViewId) {
        if self.per_view_draw.retire(view_id) {
            logger::debug!("per-draw slab: retired slab for view {view_id:?}");
        }
    }

    /// Returns the per-view scratch slot used for per-draw uniform packing, creating it on first use.
    ///
    /// Keyed per [`ViewId`] so parallel per-view recording cannot alias the same scratch
    /// across rayon workers.
    pub fn per_view_per_draw_scratch_or_create(
        &mut self,
        view_id: ViewId,
    ) -> &Mutex<PerViewPerDrawScratch> {
        profiling::scope!("render::ensure_per_view_per_draw_scratch");
        self.per_view_per_draw_scratch
            .get_or_insert_with(view_id, || {
                logger::debug!("per-draw scratch: allocating for view {view_id:?}");
                Mutex::new(PerViewPerDrawScratch::default())
            })
    }

    /// Returns the per-view scratch slot, or `None` if it has not been created yet.
    pub fn per_view_per_draw_scratch(
        &self,
        view_id: ViewId,
    ) -> Option<&Mutex<PerViewPerDrawScratch>> {
        self.per_view_per_draw_scratch.get(view_id)
    }

    /// Frees the per-view scratch buffers for a view that is no longer active.
    ///
    /// Call alongside [`Self::retire_per_view_per_draw`] and [`Self::retire_per_view_frame`] when a
    /// secondary RT camera is destroyed. Has no effect if the view was never allocated.
    pub fn retire_per_view_per_draw_scratch(&mut self, view_id: ViewId) {
        if self.per_view_per_draw_scratch.retire(view_id) {
            logger::debug!("per-draw slab scratch: retired for view {view_id:?}");
        }
    }

    /// Retires all view-scoped frame resources for `view_id`.
    pub fn retire_view(&mut self, view_id: ViewId) {
        self.retire_per_view_frame(view_id);
        self.retire_per_view_per_draw(view_id);
        self.retire_per_view_per_draw_scratch(view_id);
        let _ = self.per_view_lights.retire(view_id);
    }

    /// Fills the default main-view light scratch buffer from active render spaces.
    ///
    /// This compatibility entry point is used by unit tests and callers that do not have explicit
    /// view planning information. Normal graph rendering should call [`Self::prepare_lights_for_views`]
    /// so secondary cameras get render-context-aware light packs.
    #[cfg(test)]
    pub fn prepare_lights_from_scene(&mut self, scene: &SceneCoordinator) {
        self.prepare_lights_for_views(
            scene,
            [FrameLightViewDesc {
                view_id: ViewId::Main,
                render_context: scene.active_main_render_context(),
                render_space_filter: None,
                head_output_transform: Mat4::IDENTITY,
            }],
        );
    }

    /// Fills per-view light scratch buffers from [`SceneCoordinator`].
    ///
    /// Inactive spaces are skipped so lights from a previously focused world do not persist into
    /// the next frame's shading. Views with a render-space filter only receive lights from that
    /// space. Non-contributing lights are filtered via [`light_contributes`] before clustered
    /// ordering, and each view's transforms are resolved with the same render context and
    /// head-output transform used by draw collection.
    pub(crate) fn prepare_lights_for_views<I>(&mut self, scene: &SceneCoordinator, views: I)
    where
        I: IntoIterator<Item = FrameLightViewDesc>,
    {
        profiling::scope!("render::prepare_lights_for_views");
        self.light_scratch.clear();
        self.signed_scene_color_required = false;
        let mut wrote_fallback = false;
        for desc in views {
            self.prepare_lights_for_view(scene, desc);
            self.signed_scene_color_required |= self
                .per_view_lights
                .get(desc.view_id)
                .is_some_and(|lights| lights.signed_scene_color_required);
            if !wrote_fallback {
                let fallback_lights = self.frame_lights_for_view(desc.view_id).to_vec();
                self.light_scratch.clear();
                self.light_scratch.extend(fallback_lights);
                wrote_fallback = true;
            }
        }
        if self.signed_scene_color_required && !self.signed_scene_color_required_logged {
            logger::info!(
                "negative direct lights active: signed scene-color HDR will be used while negative lights are packed"
            );
            self.signed_scene_color_required_logged = true;
        }
    }

    fn prepare_lights_for_view(&mut self, scene: &SceneCoordinator, desc: FrameLightViewDesc) {
        profiling::scope!("render::prepare_lights_for_view");
        self.resolved_flatten_scratch.clear();
        self.collect_light_space_ids(scene, desc.render_space_filter);
        self.resolve_lights_for_space_ids(scene, desc);
        {
            profiling::scope!("render::prepare_lights::filter_contributors");
            self.resolved_flatten_scratch.retain(light_contributes);
        }
        order_lights_for_clustered_shading_in_place(&mut self.resolved_flatten_scratch);
        let resolved_len = self.resolved_flatten_scratch.len();
        if resolved_len > MAX_LIGHTS && !self.lights_overflow_warned {
            logger::warn!(
                "scene contains {resolved_len} contributing lights but the engine only uploads \
                 the first {MAX_LIGHTS} (MAX_LIGHTS); the remainder will be ignored for shading. \
                 This warning is only logged once per renderer instance."
            );
            self.lights_overflow_warned = true;
        }
        let kept = resolved_len.min(MAX_LIGHTS);
        let signed_scene_color_required = self
            .resolved_flatten_scratch
            .iter()
            .take(kept)
            .any(light_has_negative_contribution);
        let entry = self
            .per_view_lights
            .get_or_insert_with(desc.view_id, PreparedViewLights::default);
        entry.lights.clear();
        entry.lights.reserve(kept);
        entry.lights.extend(
            self.resolved_flatten_scratch
                .iter()
                .take(kept)
                .map(gpu_light_from_resolved),
        );
        entry.signed_scene_color_required = signed_scene_color_required;
        logger::trace!(
            "prepared lights for view {:?}: lights={} render_context={:?} render_space_filter={:?}",
            desc.view_id,
            entry.lights.len(),
            desc.render_context,
            desc.render_space_filter
        );
    }

    fn collect_light_space_ids(
        &mut self,
        scene: &SceneCoordinator,
        render_space_filter: Option<crate::scene::RenderSpaceId>,
    ) {
        profiling::scope!("render::prepare_lights::collect_active_spaces");
        self.light_space_ids_scratch.clear();
        if let Some(id) = render_space_filter {
            if scene.space(id).is_some_and(|space| space.is_active()) {
                self.light_space_ids_scratch.push(id);
            }
            return;
        }
        self.light_space_ids_scratch.extend(
            scene
                .render_space_ids()
                .filter(|id| scene.space(*id).is_some_and(|space| space.is_active())),
        );
    }

    fn resolve_lights_for_space_ids(&mut self, scene: &SceneCoordinator, desc: FrameLightViewDesc) {
        match self.light_space_ids_scratch.len() {
            0 => {}
            1 => {
                profiling::scope!("render::prepare_lights::resolve_single_space");
                scene.resolve_lights_for_render_context_into(
                    self.light_space_ids_scratch[0],
                    desc.render_context,
                    desc.head_output_transform,
                    &mut self.resolved_flatten_scratch,
                );
            }
            _ => {
                profiling::scope!("render::prepare_lights::resolve_parallel");
                use rayon::prelude::*;
                let per_space: Vec<Vec<ResolvedLight>> = self
                    .light_space_ids_scratch
                    .par_iter()
                    .map(|&id| {
                        let mut local = Vec::new();
                        scene.resolve_lights_for_render_context_into(
                            id,
                            desc.render_context,
                            desc.head_output_transform,
                            &mut local,
                        );
                        local
                    })
                    .collect();
                let total: usize = per_space.iter().map(Vec::len).sum();
                {
                    profiling::scope!("render::prepare_lights::flatten_parallel");
                    self.resolved_flatten_scratch.reserve(total);
                    for chunk in per_space {
                        self.resolved_flatten_scratch.extend(chunk);
                    }
                }
            }
        }
    }

    /// Pre-synchronizes shared cluster buffers for every unique view layout before per-view
    /// recording starts and uploads each view's packed lights buffer.
    pub fn pre_record_sync_for_views(
        &mut self,
        device: &wgpu::Device,
        uploads: GraphUploadSink<'_>,
        view_layouts: &[PreRecordViewResourceLayout],
    ) {
        profiling::scope!("render::pre_record_sync_for_views");
        let cluster_layouts = unique_cluster_pre_record_layouts(view_layouts, |view_id| {
            self.frame_light_count_for_view_u32(view_id)
        });
        for layout in cluster_layouts {
            profiling::scope!("render::pre_record_sync_for_views::cluster_viewport");
            let Some(fgpu) = self.frame_gpu_mut() else {
                return;
            };
            if fgpu
                .sync_cluster_viewport(
                    device,
                    (layout.width, layout.height),
                    layout.stereo,
                    layout.index_capacity_words,
                )
                .is_none()
            {
                logger::warn!(
                    "pre-record cluster sync failed for viewport {}x{} stereo={} index_capacity={}",
                    layout.width,
                    layout.height,
                    layout.stereo,
                    layout.index_capacity_words
                );
            }
        }
        {
            profiling::scope!("render::pre_record_sync_for_views::write_lights");
            for layout in view_layouts {
                let Some(state) = self.per_view_frame(layout.view_id) else {
                    continue;
                };
                FrameGpuResources::write_lights_buffer_to(
                    uploads,
                    &state.lights_buffer,
                    self.frame_lights_for_view(layout.view_id),
                );
            }
        }
    }

    /// Copies the main depth attachment into this view's scene-depth snapshot.
    ///
    /// The snapshot must already have been provisioned by [`Self::per_view_frame_or_create`].
    pub fn copy_scene_depth_snapshot_for_view(
        &self,
        view_id: ViewId,
        encoder: &mut wgpu::CommandEncoder,
        source_depth: &wgpu::Texture,
        viewport: (u32, u32),
        multiview: bool,
    ) {
        let Some(state) = self.per_view_frame.get(view_id) else {
            return;
        };
        state
            .scene_snapshots
            .encode_depth_copy(encoder, source_depth, viewport, multiview);
    }

    /// Copies the main color attachment into this view's scene-color snapshot.
    ///
    /// The snapshot must already have been provisioned by [`Self::per_view_frame_or_create`].
    pub fn copy_scene_color_snapshot_for_view(
        &self,
        view_id: ViewId,
        encoder: &mut wgpu::CommandEncoder,
        source_color: &wgpu::Texture,
        viewport: (u32, u32),
        multiview: bool,
    ) {
        let Some(state) = self.per_view_frame.get(view_id) else {
            return;
        };
        state
            .scene_snapshots
            .encode_color_copy(encoder, source_color, viewport, multiview);
    }
}

impl GraphFrameResources for FrameResourceManager {
    fn has_frame_gpu(&self) -> bool {
        self.frame_gpu().is_some()
    }

    fn frame_lights(&self, view_id: ViewId) -> &[GpuLight] {
        self.frame_lights_for_view(view_id)
    }

    fn frame_light_count_u32(&self, view_id: ViewId) -> u32 {
        self.frame_light_count_for_view_u32(view_id)
    }

    fn lights_buffer(&self, view_id: ViewId) -> Option<wgpu::Buffer> {
        self.per_view_frame(view_id)
            .map(|state| state.lights_buffer.clone())
    }

    fn frame_uniform_buffer(&self) -> Option<wgpu::Buffer> {
        self.frame_gpu().map(|fgpu| fgpu.frame_uniform.clone())
    }

    fn shared_cluster_buffer_refs(&self) -> Option<GraphClusterBufferRefs> {
        self.shared_cluster_buffer_refs()
            .map(|refs| GraphClusterBufferRefs {
                cluster_light_counts: refs.cluster_light_counts.clone(),
                cluster_light_indices: refs.cluster_light_indices.clone(),
            })
    }

    fn shared_cluster_version(&self) -> u64 {
        self.shared_cluster_version()
    }

    fn per_view_cluster_params_buffer(&self, view_id: ViewId) -> Option<wgpu::Buffer> {
        self.per_view_frame(view_id)
            .map(|state| state.cluster_params_buffer.clone())
    }

    fn per_view_frame_bind_group_and_buffer(
        &self,
        view_id: ViewId,
    ) -> Option<(Arc<wgpu::BindGroup>, wgpu::Buffer)> {
        self.per_view_frame(view_id).map(|state| {
            (
                Arc::clone(&state.frame_bind_group),
                state.frame_uniform_buffer.clone(),
            )
        })
    }

    fn ensure_per_view_per_draw_capacity(
        &self,
        device: &wgpu::Device,
        view_id: ViewId,
        draw_count: usize,
    ) -> Option<wgpu::Buffer> {
        let per_draw_slot = self.per_view_per_draw(view_id)?;
        let mut per_draw = per_draw_slot.lock();
        per_draw.ensure_draw_slot_capacity(device, draw_count);
        Some(per_draw.per_draw_storage.clone())
    }

    fn with_per_view_per_draw_scratch(
        &self,
        view_id: ViewId,
        f: &mut dyn FnMut(&mut Vec<PaddedPerDrawUniforms>, &mut Vec<u8>),
    ) -> bool {
        let Some(scratch_slot) = self.per_view_per_draw_scratch(view_id) else {
            return false;
        };
        let mut scratch_guard = scratch_slot.lock();
        let scratch = &mut *scratch_guard;
        let uniforms = &mut scratch.uniforms;
        let slab_bytes = &mut scratch.slab_bytes;
        f(uniforms, slab_bytes);
        drop(scratch_guard);
        true
    }

    fn per_view_per_draw_storage(&self, view_id: ViewId) -> Option<wgpu::Buffer> {
        self.per_view_per_draw(view_id)
            .map(|per_draw| per_draw.lock().per_draw_storage.clone())
    }

    fn per_view_per_draw_bind_group(&self, view_id: ViewId) -> Option<Arc<wgpu::BindGroup>> {
        self.per_view_per_draw(view_id)
            .map(|per_draw| Arc::clone(&per_draw.lock().bind_group))
    }

    fn empty_material_bind_group(&self) -> Option<Arc<wgpu::BindGroup>> {
        self.empty_material()
            .map(|empty| Arc::clone(&empty.bind_group))
    }

    fn copy_scene_depth_snapshot_for_view(
        &self,
        view_id: ViewId,
        encoder: &mut wgpu::CommandEncoder,
        source_depth: &wgpu::Texture,
        viewport: (u32, u32),
        multiview: bool,
    ) {
        self.copy_scene_depth_snapshot_for_view(
            view_id,
            encoder,
            source_depth,
            viewport,
            multiview,
        );
    }

    fn copy_scene_color_snapshot_for_view(
        &self,
        view_id: ViewId,
        encoder: &mut wgpu::CommandEncoder,
        source_color: &wgpu::Texture,
        viewport: (u32, u32),
        multiview: bool,
    ) {
        self.copy_scene_color_snapshot_for_view(
            view_id,
            encoder,
            source_color,
            viewport,
            multiview,
        );
    }

    fn skybox_specular_uniform_params(&self) -> SkyboxSpecularUniformParams {
        self.skybox_specular_uniform_params()
    }

    fn visible_mesh_deform_filter_is_empty(&self) -> bool {
        self.visible_mesh_deform_filter_is_empty()
    }

    fn mesh_deform_dispatched_this_tick(&self) -> bool {
        self.mesh_deform_dispatched_this_tick()
    }

    fn set_mesh_deform_dispatched_this_tick(&self) {
        self.set_mesh_deform_dispatched_this_tick();
    }

    fn visible_mesh_deform_keys_snapshot(&self) -> Option<HashSet<SkinCacheKey>> {
        self.visible_mesh_deform_keys_snapshot()
    }

    fn ensure_per_view_frame_resources(
        &mut self,
        view_id: ViewId,
        device: &wgpu::Device,
        layout: PreRecordViewResourceLayout,
    ) -> bool {
        self.per_view_frame_or_create(view_id, device, layout)
            .is_some()
    }

    fn ensure_per_view_per_draw_resources(
        &mut self,
        view_id: ViewId,
        device: &wgpu::Device,
    ) -> bool {
        self.per_view_per_draw_or_create(view_id, device).is_some()
    }

    fn ensure_per_view_per_draw_scratch(&mut self, view_id: ViewId) {
        let _ = self.per_view_per_draw_scratch_or_create(view_id);
    }

    fn pre_record_sync_for_views(
        &mut self,
        device: &wgpu::Device,
        uploads: GraphUploadSink<'_>,
        view_layouts: &[PreRecordViewResourceLayout],
    ) {
        self.pre_record_sync_for_views(device, uploads, view_layouts);
    }
}

#[cfg(test)]
mod tests;
