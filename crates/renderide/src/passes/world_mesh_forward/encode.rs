//! Encode indexed draws and material bind groups for graph-managed world-mesh forward passes.
//!
//! Drives one raster subpass at a time via [`draw_subset`], walking pre-built
//! [`crate::world_mesh::DrawGroup`]s and issuing one `draw_indexed` per group with
//! pipeline / `@group(1)` / per-draw slab binds skipped when unchanged. Vertex / index buffer
//! binding lives in [`vertex_binding`].

mod vertex_binding;

use crate::gpu::GpuLimits;
use crate::materials::MaterialPipelineSet;
use crate::mesh_deform::PER_DRAW_UNIFORM_STRIDE;
use crate::passes::WorldMeshForwardEncodeRefs;
use crate::world_mesh::{DrawGroup, WorldMeshDrawItem, depth_prepass_group_eligible};

use super::MaterialBatchPacket;
use super::depth_prepass::{
    WorldMeshForwardDepthPrepassPipelineCache, WorldMeshForwardDepthPrepassPipelineKey,
};
use super::normal_pass::{WorldMeshForwardNormalPipelineCache, WorldMeshForwardNormalPipelineKey};

use vertex_binding::{
    LastMeshBindState, draw_mesh_submesh_depth_instanced, draw_mesh_submesh_instanced,
    draw_mesh_submesh_normals_instanced, gpu_refs_for_encode, streams_for_item,
};

/// Pre-grouped draws, bind groups, and precomputed-batch table for one mesh-forward raster subpass.
///
/// Pipelines and `@group(1)` bind groups are pre-resolved by backend world-mesh frame planning,
/// so this struct carries no material-system references and makes no
/// LRU cache lookups during recording.
pub(crate) struct ForwardDrawBatch<'a, 'b, 'c, 'd> {
    /// Active render pass.
    pub rpass: &'a mut wgpu::RenderPass<'b>,
    /// Pre-built [`DrawGroup`]s for this subpass (opaque or intersect), in ascending
    /// `representative_draw_idx` order so the `precomputed` cursor stays monotonic.
    pub groups: &'c [DrawGroup],
    /// Full sorted world mesh draw list for the view (read by representative index).
    pub draws: &'c [WorldMeshDrawItem],
    /// Pre-resolved pipelines and bind groups; one entry per unique batch-key run in `draws`.
    pub precomputed: &'c [MaterialBatchPacket],
    /// Mesh pool and skin cache for vertex/index binding.
    pub encode: &'a mut WorldMeshForwardEncodeRefs<'d>,
    /// Device limits snapshot (storage-offset alignment for `@group(2)`).
    pub gpu_limits: &'a GpuLimits,
    /// Frame globals at `@group(0)`.
    pub frame_bg: &'a wgpu::BindGroup,
    /// Fallback material bind group when a batch has no resolved `@group(1)`.
    pub empty_bg: &'a wgpu::BindGroup,
    /// Per-draw storage slab at `@group(2)` (dynamic offset; see [`Self::supports_base_instance`]).
    pub per_draw_bind_group: &'a wgpu::BindGroup,
    /// Whether `draw_indexed` may use non-zero `first_instance` / base instance. When
    /// false, every group carries `instance_range.len() == 1` and the per-draw slab is
    /// addressed via dynamic offset instead.
    pub supports_base_instance: bool,
    /// Overlay view-projection used to project per-draw `_Rect` corners into screen space for
    /// the GPU scissor optimisation. Same value as
    /// [`super::PreparedWorldMeshForwardFrame::overlay_view_proj`].
    pub overlay_view_proj: glam::Mat4,
    /// Active viewport in pixels for the GPU scissor optimisation.
    pub viewport_px: (u32, u32),
}

/// Pre-grouped draws and normal-prepass state for one mesh-forward raster subpass.
pub(crate) struct NormalDrawBatch<'a, 'b, 'c, 'd> {
    /// Active render pass.
    pub rpass: &'a mut wgpu::RenderPass<'b>,
    /// Pre-built regular draw groups in ascending representative order.
    pub groups: &'c [DrawGroup],
    /// Full sorted world mesh draw list for the view.
    pub draws: &'c [WorldMeshDrawItem],
    /// Mesh pool and skin cache for vertex/index binding.
    pub encode: &'a mut WorldMeshForwardEncodeRefs<'d>,
    /// Device limits snapshot for dynamic storage-buffer offsets.
    pub gpu_limits: &'a GpuLimits,
    /// Per-draw storage slab bound at `@group(0)` for the normal prepass.
    pub per_draw_bind_group: &'a wgpu::BindGroup,
    /// Whether `draw_indexed` may use non-zero `first_instance`.
    pub supports_base_instance: bool,
    /// Pipeline state resolved for the active world-mesh view.
    pub pipeline: &'a super::WorldMeshForwardPipelineState,
    /// GPU device used for lazy normal-prepass pipeline creation.
    pub device: &'a wgpu::Device,
    /// Shared normal-prepass pipeline cache.
    pub normal_pipelines: &'a WorldMeshForwardNormalPipelineCache,
}

/// Pre-grouped draws and pipeline state for the generic opaque depth prepass.
pub(crate) struct DepthPrepassDrawBatch<'a, 'b, 'c, 'd> {
    /// Active render pass.
    pub rpass: &'a mut wgpu::RenderPass<'b>,
    /// Pre-built regular draw groups in ascending representative order.
    pub groups: &'c [DrawGroup],
    /// Slab layout used to resolve every draw member in each group.
    pub slab_layout: &'c [usize],
    /// Full sorted world mesh draw list for the view.
    pub draws: &'c [WorldMeshDrawItem],
    /// Mesh pool and skin cache for vertex/index binding.
    pub encode: &'a mut WorldMeshForwardEncodeRefs<'d>,
    /// Device limits snapshot for dynamic storage-buffer offsets.
    pub gpu_limits: &'a GpuLimits,
    /// Per-draw storage slab bound at `@group(0)` for the depth prepass.
    pub per_draw_bind_group: &'a wgpu::BindGroup,
    /// Whether `draw_indexed` may use non-zero `first_instance`.
    pub supports_base_instance: bool,
    /// Pipeline state resolved for the active world-mesh view.
    pub pipeline: &'a super::WorldMeshForwardPipelineState,
    /// GPU device used for lazy depth-prepass pipeline creation.
    pub device: &'a wgpu::Device,
    /// Shared depth-prepass pipeline cache.
    pub depth_pipelines: &'a WorldMeshForwardDepthPrepassPipelineCache,
}

/// Per-draw slab bind request for one draw group.
struct PerDrawSlabBind<'a> {
    /// Pipeline bind-group index used by the active shader.
    bind_group_index: u32,
    /// Per-draw storage bind group.
    bind_group: &'a wgpu::BindGroup,
    /// Device limits snapshot.
    gpu_limits: &'a GpuLimits,
    /// First row in slab coordinates.
    slab_first_instance: usize,
    /// Number of instances in the draw group.
    instance_count: u32,
    /// Whether instance indices directly address slab rows.
    supports_base_instance: bool,
}

/// Records one raster subpass by walking pre-built [`DrawGroup`]s.
///
/// Each group is one `draw_indexed` covering a contiguous slab range of identical instances.
/// The `precomputed` cursor advances on each group's `representative_draw_idx`, which is
/// monotonically increasing across the group list -- O(1) amortised. Pipelines and `@group(1)`
/// bind groups are bound directly from the table; no cache lookups occur during recording.
pub(crate) fn draw_subset(batch: ForwardDrawBatch<'_, '_, '_, '_>) {
    profiling::scope!("world_mesh::draw_subset");
    let ForwardDrawBatch {
        rpass,
        groups,
        draws,
        precomputed,
        encode,
        gpu_limits,
        frame_bg,
        empty_bg,
        per_draw_bind_group,
        supports_base_instance,
        overlay_view_proj,
        viewport_px,
    } = batch;
    let full_viewport: (u32, u32, u32, u32) = (0, 0, viewport_px.0, viewport_px.1);
    let mut last_scissor: Option<(u32, u32, u32, u32)> = None;

    let (subpass_batch_count, subpass_input_draws) = {
        profiling::scope!("world_mesh::draw_subset::summarize_groups");
        let subpass_batch_count = groups.len();
        let subpass_input_draws: usize = groups
            .iter()
            .map(|g| (g.instance_range.end - g.instance_range.start) as usize)
            .sum();
        (subpass_batch_count, subpass_input_draws)
    };

    let mut last_mesh = LastMeshBindState::new();
    let mut last_per_draw_dyn_offset: Option<u32> = None;
    let mut last_stencil_ref: Option<u32> = None;
    // Track which precomputed batch is currently bound to avoid redundant set_bind_group(1).
    let mut bound_batch_cursor: Option<usize> = None;
    // Track the last pipeline pointer to skip redundant set_pipeline across groups that share
    // the same pipeline (common when one precomputed batch covers many groups, or when
    // adjacent batches resolve to the same multi-pass pipeline set).
    let mut last_pipeline: Option<*const wgpu::RenderPipeline> = None;

    {
        profiling::scope!("world_mesh::draw_subset::bind_frame_group");
        rpass.set_bind_group(0, frame_bg, &[]);
    }

    {
        profiling::scope!("world_mesh::draw_subset::group_loop");
        for group in groups {
            let representative = group.representative_draw_idx;
            let batch_cursor = group.material_packet_idx;
            let Some(pc) = precomputed.get(batch_cursor) else {
                continue;
            };
            debug_assert!(
                representative >= pc.first_draw_idx && representative <= pc.last_draw_idx,
                "precomputed batch [{}, {}] should cover representative draw index {}",
                pc.first_draw_idx,
                pc.last_draw_idx,
                representative,
            );
            debug_assert_eq!(
                pc.pipeline_key.shader_asset_id, draws[representative].batch_key.shader_asset_id,
                "material packet pipeline key must match the representative draw"
            );

            let Some(pipelines) = pc.pipelines.as_ref() else {
                continue; // pipeline unavailable for this batch -- skip draws
            };

            // Bind @group(1) once per unique batch; skip when the cursor hasn't advanced.
            if bound_batch_cursor != Some(batch_cursor) {
                let material_bg = pc.bind_group.as_deref().unwrap_or(empty_bg);
                if let Some(offset) = pc.material_uniform_dynamic_offset {
                    rpass.set_bind_group(1, material_bg, &[offset]);
                } else {
                    rpass.set_bind_group(1, material_bg, &[]);
                }
                bound_batch_cursor = Some(batch_cursor);
            }

            let slab_first_instance = group.instance_range.start as usize;
            let instance_count = group.instance_range.end - group.instance_range.start;
            bind_per_draw_slab_if_changed(
                rpass,
                PerDrawSlabBind {
                    bind_group_index: 2,
                    bind_group: per_draw_bind_group,
                    gpu_limits,
                    slab_first_instance,
                    instance_count,
                    supports_base_instance,
                },
                &mut last_per_draw_dyn_offset,
            );

            let stencil_ref = draws[representative]
                .batch_key
                .render_state
                .stencil_reference();
            if last_stencil_ref != Some(stencil_ref) {
                rpass.set_stencil_reference(stencil_ref);
                last_stencil_ref = Some(stencil_ref);
            }

            // Per-draw scissor for UI rect-mask overlay items: shrinks the rasterised area to
            // the projected `_Rect`, so fragment shaders never run on pixels the
            // `rect_clip.wgsl` predicate would `discard` anyway. Falls back to the full viewport
            // for everything else (the CPU rect-cull above already rejected fully off-screen
            // overlay draws; this handles the partially-on-screen case and all non-UI draws).
            let item = &draws[representative];
            let scissor = match (item.ui_rect_clip_local, item.rigid_world_matrix) {
                (Some(rect), Some(model)) => {
                    project_rect_to_scissor(overlay_view_proj * model, rect, viewport_px)
                        .unwrap_or(full_viewport)
                }
                _ => full_viewport,
            };
            if last_scissor != Some(scissor) {
                rpass.set_scissor_rect(scissor.0, scissor.1, scissor.2, scissor.3);
                last_scissor = Some(scissor);
            }

            let inst_range = instance_range_for_draw_group(group, supports_base_instance);

            issue_material_pipeline_passes(
                rpass,
                encode,
                &draws[representative],
                ActivePipelineSelection { pipelines },
                &inst_range,
                &mut last_mesh,
                &mut last_pipeline,
            );
        }
    }

    // Reset the scissor to the full viewport so the next subpass (skybox, transparent world, ...)
    // starts with a clean state and doesn't inherit a clipped region from the last UI draw.
    if last_scissor.is_some() && last_scissor != Some(full_viewport) {
        rpass.set_scissor_rect(
            full_viewport.0,
            full_viewport.1,
            full_viewport.2,
            full_viewport.3,
        );
    }

    {
        profiling::scope!("world_mesh::draw_subset::plot_subpass");
        crate::profiling::plot_world_mesh_subpass(subpass_batch_count, subpass_input_draws);
    }
}

/// Records the GTAO normal prepass draw subset.
pub(crate) fn draw_normals_subset(batch: NormalDrawBatch<'_, '_, '_, '_>) {
    profiling::scope!("world_mesh::draw_normals_subset");
    let NormalDrawBatch {
        rpass,
        groups,
        draws,
        encode,
        gpu_limits,
        per_draw_bind_group,
        supports_base_instance,
        pipeline,
        device,
        normal_pipelines,
    } = batch;

    let mut last_mesh = LastMeshBindState::new();
    let mut last_per_draw_dyn_offset: Option<u32> = None;
    let mut last_pipeline: Option<*const wgpu::RenderPipeline> = None;

    for group in groups {
        let representative = group.representative_draw_idx;
        let Some(item) = draws.get(representative) else {
            continue;
        };
        let Some(key) = WorldMeshForwardNormalPipelineKey::for_draw(
            pipeline,
            item.batch_key.front_face,
            item.batch_key.primitive_topology,
        ) else {
            continue;
        };

        let slab_first_instance = group.instance_range.start as usize;
        let instance_count = group.instance_range.end - group.instance_range.start;
        bind_per_draw_slab_if_changed(
            rpass,
            PerDrawSlabBind {
                bind_group_index: 0,
                bind_group: per_draw_bind_group,
                gpu_limits,
                slab_first_instance,
                instance_count,
                supports_base_instance,
            },
            &mut last_per_draw_dyn_offset,
        );

        let pipeline = normal_pipelines.pipeline(device, key);
        let pipeline_id: *const wgpu::RenderPipeline = pipeline.as_ref();
        if last_pipeline != Some(pipeline_id) {
            rpass.set_pipeline(pipeline.as_ref());
            last_pipeline = Some(pipeline_id);
        }

        let inst_range = instance_range_for_draw_group(group, supports_base_instance);
        let gpu_refs = gpu_refs_for_encode(encode);
        draw_mesh_submesh_normals_instanced(rpass, item, gpu_refs, inst_range, &mut last_mesh);
    }
}

/// Records the safe opaque depth prepass draw subset.
pub(crate) fn draw_depth_prepass_subset(batch: DepthPrepassDrawBatch<'_, '_, '_, '_>) {
    profiling::scope!("world_mesh::draw_depth_prepass_subset");
    let DepthPrepassDrawBatch {
        rpass,
        groups,
        slab_layout,
        draws,
        encode,
        gpu_limits,
        per_draw_bind_group,
        supports_base_instance,
        pipeline,
        device,
        depth_pipelines,
    } = batch;

    let mut last_mesh = LastMeshBindState::new();
    let mut last_per_draw_dyn_offset: Option<u32> = None;
    let mut last_pipeline: Option<*const wgpu::RenderPipeline> = None;

    for group in groups {
        let representative = group.representative_draw_idx;
        let Some(item) = draws.get(representative) else {
            continue;
        };
        if !depth_prepass_group_eligible(draws, slab_layout, group, pipeline.shader_perm) {
            continue;
        }
        let Some(key) = WorldMeshForwardDepthPrepassPipelineKey::for_draw(item, pipeline) else {
            continue;
        };

        let slab_first_instance = group.instance_range.start as usize;
        let instance_count = group.instance_range.end - group.instance_range.start;
        bind_per_draw_slab_if_changed(
            rpass,
            PerDrawSlabBind {
                bind_group_index: 0,
                bind_group: per_draw_bind_group,
                gpu_limits,
                slab_first_instance,
                instance_count,
                supports_base_instance,
            },
            &mut last_per_draw_dyn_offset,
        );

        let pipeline = depth_pipelines.pipeline(device, key);
        let pipeline_id: *const wgpu::RenderPipeline = pipeline.as_ref();
        if last_pipeline != Some(pipeline_id) {
            rpass.set_pipeline(pipeline.as_ref());
            last_pipeline = Some(pipeline_id);
        }

        let inst_range = instance_range_for_draw_group(group, supports_base_instance);
        let gpu_refs = gpu_refs_for_encode(encode);
        draw_mesh_submesh_depth_instanced(rpass, item, gpu_refs, inst_range, &mut last_mesh);
    }
}

/// Updates a per-draw storage dynamic offset and rebinds the slab when the row offset changes.
///
/// `slab_first_instance` is the slab-coordinate start of the current group's
/// `instance_range`. On base-instance-capable devices the dynamic offset is always zero
/// (rows are addressed via `first_instance`), so the rebind occurs once at most. On
/// downlevel paths each group carries `instance_count == 1` and the slab row is selected
/// via the dynamic offset.
fn bind_per_draw_slab_if_changed(
    rpass: &mut wgpu::RenderPass<'_>,
    bind: PerDrawSlabBind<'_>,
    last_per_draw_dyn_offset: &mut Option<u32>,
) {
    let PerDrawSlabBind {
        bind_group_index,
        bind_group,
        gpu_limits,
        slab_first_instance,
        instance_count,
        supports_base_instance,
    } = bind;
    let storage_align = gpu_limits.min_storage_buffer_offset_alignment();
    let per_draw_dyn_offset = if supports_base_instance {
        // Base-instance path: all rows accessed via `first_instance`; dynamic offset is
        // always zero for the entire pass so the bind is skipped after the first draw.
        0u32
    } else {
        // Downlevel: `first_instance` is always zero; select the draw row via dynamic offset.
        debug_assert_eq!(instance_count, 1);
        let raw = (slab_first_instance * PER_DRAW_UNIFORM_STRIDE) as u32;
        debug_assert_eq!(
            raw % storage_align,
            0,
            "per-draw offset must match min_storage_buffer_offset_alignment"
        );
        raw
    };
    if *last_per_draw_dyn_offset != Some(per_draw_dyn_offset) {
        rpass.set_bind_group(bind_group_index, bind_group, &[per_draw_dyn_offset]);
        *last_per_draw_dyn_offset = Some(per_draw_dyn_offset);
    }
}

/// Per-batch pipeline selection for [`issue_material_pipeline_passes`].
struct ActivePipelineSelection<'a> {
    /// Per-material pipeline objects in pass order.
    pipelines: &'a MaterialPipelineSet,
}

/// Walks the pipeline set for `item` and issues one [`draw_mesh_submesh_instanced`] per pipeline.
///
/// `last_pipeline` is updated and consulted across batches so that adjacent draws sharing a
/// pipeline (the typical case within a precomputed batch) skip the redundant `set_pipeline`.
fn issue_material_pipeline_passes(
    rpass: &mut wgpu::RenderPass<'_>,
    encode: &WorldMeshForwardEncodeRefs<'_>,
    item: &WorldMeshDrawItem,
    pipeline_sel: ActivePipelineSelection<'_>,
    inst_range: &std::ops::Range<u32>,
    last_mesh: &mut LastMeshBindState,
    last_pipeline: &mut Option<*const wgpu::RenderPipeline>,
) {
    let gpu_refs = gpu_refs_for_encode(encode);
    let streams = streams_for_item(item);
    for pipeline in pipeline_sel.pipelines.iter() {
        let pipeline_id: *const wgpu::RenderPipeline = pipeline;
        if *last_pipeline != Some(pipeline_id) {
            rpass.set_pipeline(pipeline);
            *last_pipeline = Some(pipeline_id);
        }
        draw_mesh_submesh_instanced(
            rpass,
            item,
            gpu_refs,
            streams,
            inst_range.clone(),
            last_mesh,
        );
    }
}

/// Resolves the `instance_range` argument to `draw_indexed` for one [`DrawGroup`].
///
/// On base-instance-capable devices, the group's slab range is passed verbatim -- the GPU
/// `instance_index` walks `instance_range.start..instance_range.end`, addressing the
/// per-draw slab directly. On downlevel devices, every group has `instance_range.len() == 1`
/// (forced by `build_plan`'s `supports_base_instance = false` gate), and the slab
/// row is reached via the dynamic offset, so the draw range collapses to `0..1`.
fn instance_range_for_draw_group(
    group: &DrawGroup,
    supports_base_instance: bool,
) -> std::ops::Range<u32> {
    if supports_base_instance {
        group.instance_range.clone()
    } else {
        debug_assert_eq!(
            group.instance_range.end - group.instance_range.start,
            1,
            "downlevel groups must be singletons"
        );
        0..1
    }
}

/// Projects the four corners of an object-local UI rect through `mvp` into NDC, builds the
/// pixel-space AABB clamped to `viewport_px`, and returns it as a `(x, y, w, h)` scissor.
///
/// Returns `None` when:
/// - all four corners have non-positive `w` (rect entirely behind / on the near plane), or
/// - the rect projects to a degenerate (zero-width or zero-height) screen region.
///
/// `viewport_px` is the active viewport in pixels (`width`, `height`). The scissor is clamped to
/// stay inside the viewport for the partially-off-screen case; fully-off-screen rejection is
/// already handled by the CPU rect-cull in
/// [`crate::world_mesh::culling::overlay_rect_clip_visible`].
pub(crate) fn project_rect_to_scissor(
    mvp: glam::Mat4,
    rect: glam::Vec4,
    viewport_px: (u32, u32),
) -> Option<(u32, u32, u32, u32)> {
    let corners = [
        glam::Vec4::new(rect.x, rect.y, 0.0, 1.0),
        glam::Vec4::new(rect.z, rect.y, 0.0, 1.0),
        glam::Vec4::new(rect.z, rect.w, 0.0, 1.0),
        glam::Vec4::new(rect.x, rect.w, 0.0, 1.0),
    ];
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    let mut any_in_front = false;
    for c in corners {
        let clip = mvp * c;
        if clip.w <= 0.0 {
            continue;
        }
        any_in_front = true;
        let ndc_x = (clip.x / clip.w).clamp(-1.0, 1.0);
        let ndc_y = (clip.y / clip.w).clamp(-1.0, 1.0);
        if ndc_x < min_x {
            min_x = ndc_x;
        }
        if ndc_x > max_x {
            max_x = ndc_x;
        }
        if ndc_y < min_y {
            min_y = ndc_y;
        }
        if ndc_y > max_y {
            max_y = ndc_y;
        }
    }
    if !any_in_front {
        return None;
    }
    let (vw, vh) = (viewport_px.0 as f32, viewport_px.1 as f32);
    // Clip space y is +up; pixel space y is +down. Flip y when mapping to pixels.
    let px_min_x = ((min_x * 0.5 + 0.5) * vw).floor().clamp(0.0, vw);
    let px_max_x = ((max_x * 0.5 + 0.5) * vw).ceil().clamp(0.0, vw);
    let px_min_y = (((-max_y) * 0.5 + 0.5) * vh).floor().clamp(0.0, vh);
    let px_max_y = (((-min_y) * 0.5 + 0.5) * vh).ceil().clamp(0.0, vh);
    let x = px_min_x as u32;
    let y = px_min_y as u32;
    let w = (px_max_x - px_min_x) as u32;
    let h = (px_max_y - px_min_y) as u32;
    if w == 0 || h == 0 {
        return None;
    }
    Some((x, y, w, h))
}

#[cfg(test)]
mod tests {
    use super::{instance_range_for_draw_group, project_rect_to_scissor};
    use crate::world_mesh::DrawGroup;
    use glam::{Mat4, Vec4};

    /// Symmetric ortho mapping NDC [-1, 1] in xy to overlay-space [-1, 1].
    fn ortho() -> Mat4 {
        Mat4::orthographic_rh(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
    }

    #[test]
    fn project_rect_inside_viewport_clamps_to_pixel_aabb() {
        let mvp = ortho();
        let rect = Vec4::new(-0.5, -0.5, 0.5, 0.5);
        let scissor = project_rect_to_scissor(mvp, rect, (200, 100)).expect("scissor");
        // x in NDC [-0.5, 0.5] -> pixel [50, 150] -> width 100.
        assert_eq!(scissor.0, 50);
        assert_eq!(scissor.2, 100);
        // y is flipped: NDC y in [-0.5, 0.5] -> pixel y in [25, 75] -> height 50.
        assert_eq!(scissor.1, 25);
        assert_eq!(scissor.3, 50);
    }

    #[test]
    fn project_rect_partially_offscreen_clamps_to_viewport_edges() {
        let mvp = ortho();
        // Rect spans -2..0 in x; the -2 corner clamps to NDC -1 -> pixel 0.
        let rect = Vec4::new(-2.0, -0.5, 0.0, 0.5);
        let scissor = project_rect_to_scissor(mvp, rect, (200, 100)).expect("scissor");
        assert_eq!(scissor.0, 0);
        // NDC max x = 0.0 -> pixel 100.
        assert_eq!(scissor.0 + scissor.2, 100);
    }

    #[test]
    fn project_rect_fully_behind_camera_returns_none() {
        // Build a perspective-style matrix that yields negative w for any input. A diagonal with
        // w = -1 forces every corner to land behind the camera.
        let mvp = Mat4::from_diagonal(Vec4::new(1.0, 1.0, 1.0, -1.0));
        let rect = Vec4::new(-0.5, -0.5, 0.5, 0.5);
        assert!(project_rect_to_scissor(mvp, rect, (200, 100)).is_none());
    }

    #[test]
    fn no_base_instance_draws_from_zero() {
        let group = DrawGroup {
            representative_draw_idx: 17,
            instance_range: 17..18,
            material_packet_idx: 0,
        };
        assert_eq!(instance_range_for_draw_group(&group, false), 0..1);
    }

    #[test]
    fn base_instance_uses_slab_range() {
        let group = DrawGroup {
            representative_draw_idx: 17,
            instance_range: 17..20,
            material_packet_idx: 0,
        };
        assert_eq!(instance_range_for_draw_group(&group, true), 17..20);
    }
}
