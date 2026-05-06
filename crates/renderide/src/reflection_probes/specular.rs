//! Specular IBL reflection-probe baking, binding, and CPU-side selection.

use std::sync::Arc;

use glam::{Vec3, Vec3A, Vec4};
use hashbrown::{HashMap, HashSet};

use crate::assets::AssetTransferQueue;
use crate::backend::frame_gpu::{
    GpuReflectionProbeMetadata, REFLECTION_PROBE_ATLAS_FORMAT,
    REFLECTION_PROBE_METADATA_BOX_PROJECTION, REFLECTION_PROBE_METADATA_SH2_VALID,
    ReflectionProbeSpecularResources,
};
use crate::gpu::{GpuContext, GpuLimits};
use crate::materials::MaterialSystem;
use crate::scene::{
    ReflectionProbeEntry, RenderSpaceId, SceneCoordinator, reflection_probe_skybox_only,
    reflection_probe_use_box_projection,
};
use crate::shared::{ReflectionProbeClear, ReflectionProbeState, ReflectionProbeType, RenderSH2};
use crate::skybox::ibl_cache::{
    SkyboxIblCache, SkyboxIblKey, build_key, clamp_face_size, mip_extent, mip_levels_for_edge,
};
use crate::skybox::specular::{
    CubemapIblSource, SkyboxIblSource, resolve_skybox_material_ibl_source, solid_color_ibl_source,
};
use crate::world_mesh::culling::world_aabb_from_local_bounds;

/// Default destination face size for reflection-probe IBL bakes.
const DEFAULT_REFLECTION_PROBE_FACE_SIZE: u32 = 256;
/// First atlas slot is reserved as a non-sampled black fallback.
const FIRST_PROBE_ATLAS_SLOT: u16 = 1;
/// Maximum number of probes in one BVH leaf.
const BVH_LEAF_SIZE: usize = 8;
/// Minimum object volume used when normalizing intersection weights.
const MIN_OBJECT_VOLUME: f32 = 1e-12;

/// Per-draw reflection-probe selection stored in the per-draw slab.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ReflectionProbeDrawSelection {
    /// First selected reflection-probe atlas index.
    pub first_atlas_index: u16,
    /// Second selected reflection-probe atlas index.
    pub second_atlas_index: u16,
    /// Blend weight for [`Self::second_atlas_index`].
    pub second_weight: f32,
    /// Number of selected probes, clamped to two.
    pub hit_count: u8,
}

impl ReflectionProbeDrawSelection {
    /// Builds a single-probe selection.
    #[must_use]
    pub fn one(first_atlas_index: u16) -> Self {
        Self {
            first_atlas_index,
            second_atlas_index: 0,
            second_weight: 0.0,
            hit_count: 1,
        }
    }

    /// Builds a two-probe selection.
    #[must_use]
    pub fn two(first_atlas_index: u16, second_atlas_index: u16, second_weight: f32) -> Self {
        Self {
            first_atlas_index,
            second_atlas_index,
            second_weight: second_weight.clamp(0.0, 1.0),
            hit_count: 2,
        }
    }
}

/// CPU-side selector snapshot used during world-mesh draw collection.
#[derive(Default)]
pub struct ReflectionProbeFrameSelection {
    spaces: HashMap<RenderSpaceId, ReflectionProbeSpatialIndex>,
    skybox_fallback_slots: HashMap<RenderSpaceId, u16>,
}

impl ReflectionProbeFrameSelection {
    /// Selects up to two probes for one object AABB.
    #[must_use]
    pub fn select(
        &self,
        space_id: RenderSpaceId,
        object_aabb: (Vec3, Vec3),
    ) -> ReflectionProbeDrawSelection {
        if let Some(selection) = self
            .spaces
            .get(&space_id)
            .map(|index| index.select(object_aabb))
            && selection.hit_count > 0
        {
            return selection;
        }
        self.fallback(space_id)
    }

    /// Returns the render-space skybox fallback selection, if its specular IBL is ready.
    #[must_use]
    pub fn fallback(&self, space_id: RenderSpaceId) -> ReflectionProbeDrawSelection {
        self.skybox_fallback_slots
            .get(&space_id)
            .copied()
            .filter(|slot| *slot != 0)
            .map_or_else(ReflectionProbeDrawSelection::default, |slot| {
                ReflectionProbeDrawSelection {
                    first_atlas_index: slot,
                    second_atlas_index: 0,
                    second_weight: 0.0,
                    hit_count: 0,
                }
            })
    }

    fn rebuild<I, J>(&mut self, probes: I, skybox_fallback_slots: J)
    where
        I: IntoIterator<Item = ReadyProbe>,
        J: IntoIterator<Item = (RenderSpaceId, u16)>,
    {
        self.rebuild_spatial(
            probes
                .into_iter()
                .map(|probe| (probe.identity.space_id, probe.spatial)),
            skybox_fallback_slots,
        );
    }

    fn rebuild_spatial<I, J>(&mut self, probes: I, skybox_fallback_slots: J)
    where
        I: IntoIterator<Item = (RenderSpaceId, SpatialProbe)>,
        J: IntoIterator<Item = (RenderSpaceId, u16)>,
    {
        self.spaces.clear();
        self.skybox_fallback_slots.clear();
        self.skybox_fallback_slots.extend(
            skybox_fallback_slots
                .into_iter()
                .filter(|(_, slot)| *slot != 0),
        );
        let mut by_space: HashMap<RenderSpaceId, Vec<SpatialProbe>> = HashMap::new();
        for (space_id, probe) in probes {
            by_space.entry(space_id).or_default().push(probe);
        }
        for (space_id, probes) in by_space {
            self.spaces
                .insert(space_id, ReflectionProbeSpatialIndex::build(probes));
        }
    }
}

/// Specular reflection-probe bake/cache/selection system.
pub struct ReflectionProbeSpecularSystem {
    ibl_cache: SkyboxIblCache,
    atlas: Option<ReflectionProbeAtlas>,
    resources: Option<ReflectionProbeSpecularResources>,
    selection: ReflectionProbeFrameSelection,
    version: u64,
}

impl Default for ReflectionProbeSpecularSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl ReflectionProbeSpecularSystem {
    /// Creates an empty reflection-probe specular system.
    #[must_use]
    pub fn new() -> Self {
        Self {
            ibl_cache: SkyboxIblCache::new(),
            atlas: None,
            resources: None,
            selection: ReflectionProbeFrameSelection::default(),
            version: 1,
        }
    }

    /// Advances GPU bakes, updates the atlas, and rebuilds the CPU selection index.
    pub fn maintain(
        &mut self,
        gpu: &mut GpuContext,
        scene: &SceneCoordinator,
        materials: &MaterialSystem,
        assets: &AssetTransferQueue,
        render_context: crate::shared::RenderingContext,
        sh2_system: &mut super::ReflectionProbeSh2System,
    ) {
        profiling::scope!("reflection_probes::specular::maintain");
        self.ibl_cache.maintain_completed_jobs(gpu.device());
        let face_size = clamp_face_size(DEFAULT_REFLECTION_PROBE_FACE_SIZE, gpu.limits());
        let mut active_keys = HashSet::new();
        let mut ready = Vec::new();
        let mut skybox_fallbacks = Vec::new();

        for space_id in scene.render_space_ids() {
            let Some(space) = scene.space(space_id) else {
                continue;
            };
            if !space.is_active {
                continue;
            }
            if let Some(source) = resolve_space_skybox_fallback_source(
                space.skybox_material_asset_id,
                materials,
                assets,
            ) {
                let key = build_key(&source, face_size);
                active_keys.insert(key.clone());
                self.ibl_cache.ensure_source(gpu, key.clone(), source);
                if let Some(cube) = self.ibl_cache.completed_cube(&key) {
                    skybox_fallbacks.push(ReadySkyboxFallback {
                        space_id,
                        key,
                        texture: cube.texture.clone(),
                        mip_levels: cube.mip_levels,
                    });
                }
            }
            for probe in &space.reflection_probes {
                let identity = ProbeIdentity {
                    space_id,
                    renderable_index: probe.renderable_index,
                };
                let Some(source) =
                    resolve_probe_source(space.skybox_material_asset_id, probe, materials, assets)
                else {
                    continue;
                };
                let key = build_key(&source, face_size);
                active_keys.insert(key.clone());
                let sh2 = sh2_system.ensure_ibl_source(space_id.0, &source);
                self.ibl_cache.ensure_source(gpu, key.clone(), source);
                let Some(cube) = self.ibl_cache.completed_cube(&key) else {
                    continue;
                };
                let Some(sh2) = sh2 else {
                    continue;
                };
                let Some(spatial) =
                    spatial_probe_for_state(scene, space_id, probe, render_context, 0)
                else {
                    continue;
                };
                let mut metadata = metadata_for_spatial(&spatial, probe.state, &sh2);
                metadata.params[1] = cube.mip_levels.saturating_sub(1) as f32;
                ready.push(ReadyProbe {
                    identity,
                    key,
                    texture: cube.texture.clone(),
                    mip_levels: cube.mip_levels,
                    metadata,
                    spatial,
                });
            }
        }
        self.ibl_cache.prune_completed_except(&active_keys);
        ready.sort_unstable_by_key(|probe| {
            (probe.identity.space_id.0, probe.identity.renderable_index)
        });
        skybox_fallbacks.sort_unstable_by_key(|fallback| fallback.space_id.0);
        self.sync_atlas_and_selection(gpu, face_size, ready, skybox_fallbacks);
    }

    /// Current frame-global GPU resources, if allocated.
    #[must_use]
    pub fn resources(&self) -> Option<ReflectionProbeSpecularResources> {
        self.resources.clone()
    }

    /// CPU selection snapshot used by draw collection.
    #[must_use]
    pub fn selection(&self) -> &ReflectionProbeFrameSelection {
        &self.selection
    }

    fn sync_atlas_and_selection(
        &mut self,
        gpu: &GpuContext,
        face_size: u32,
        mut ready: Vec<ReadyProbe>,
        mut skybox_fallbacks: Vec<ReadySkyboxFallback>,
    ) {
        let max_slots = max_atlas_slots(gpu.limits());
        if max_slots <= 1 {
            self.selection.rebuild(Vec::new(), Vec::new());
            return;
        }
        let usable_slots = usize::from(max_slots.saturating_sub(FIRST_PROBE_ATLAS_SLOT));
        if ready.len() > usable_slots {
            logger::warn!(
                "reflection probes: {} ready probes exceed atlas capacity {}; truncating",
                ready.len(),
                usable_slots
            );
            ready.truncate(usable_slots);
        }
        let fallback_slots = usable_slots.saturating_sub(ready.len());
        if skybox_fallbacks.len() > fallback_slots {
            logger::warn!(
                "reflection probes: {} ready skybox fallbacks exceed remaining atlas capacity {}; truncating",
                skybox_fallbacks.len(),
                fallback_slots
            );
            skybox_fallbacks.truncate(fallback_slots);
        }
        let used_slots = ready.len() + skybox_fallbacks.len();
        let required_slots = (used_slots + usize::from(FIRST_PROBE_ATLAS_SLOT)).max(1);
        self.ensure_atlas(gpu.device(), face_size, required_slots as u16);

        let Some(atlas) = self.atlas.as_mut() else {
            self.selection.rebuild(Vec::new(), Vec::new());
            return;
        };
        let mip_levels = atlas.mip_levels;
        let mut metadata = vec![GpuReflectionProbeMetadata::default(); atlas.capacity as usize];
        let mut copy_jobs = Vec::new();
        let mut selectable = Vec::with_capacity(ready.len());
        let mut next_slot = FIRST_PROBE_ATLAS_SLOT;
        for (i, mut probe) in ready.into_iter().enumerate() {
            let slot = FIRST_PROBE_ATLAS_SLOT + i as u16;
            next_slot = slot + 1;
            if atlas.keys[slot as usize].as_ref() != Some(&probe.key) {
                atlas.keys[slot as usize] = Some(probe.key.clone());
                copy_jobs.push(AtlasCopyJob {
                    slot,
                    texture: probe.texture.clone(),
                    mip_levels: probe.mip_levels.min(mip_levels),
                });
            }
            probe.spatial.atlas_index = slot;
            metadata[slot as usize] = probe.metadata;
            selectable.push(probe);
        }
        let mut skybox_fallback_slots = Vec::with_capacity(skybox_fallbacks.len());
        for fallback in skybox_fallbacks {
            let slot = next_slot;
            next_slot = next_slot.saturating_add(1);
            if atlas.keys[slot as usize].as_ref() != Some(&fallback.key) {
                atlas.keys[slot as usize] = Some(fallback.key.clone());
                copy_jobs.push(AtlasCopyJob {
                    slot,
                    texture: fallback.texture.clone(),
                    mip_levels: fallback.mip_levels.min(mip_levels),
                });
            }
            metadata[slot as usize] = skybox_fallback_metadata(fallback.mip_levels);
            skybox_fallback_slots.push((fallback.space_id, slot));
        }
        self.write_metadata(gpu.queue(), &metadata);
        self.encode_atlas_copies(gpu, face_size, mip_levels, copy_jobs);
        self.selection.rebuild(selectable, skybox_fallback_slots);
    }

    fn ensure_atlas(&mut self, device: &wgpu::Device, face_size: u32, required_slots: u16) {
        let needs_new = self
            .atlas
            .as_ref()
            .is_none_or(|atlas| atlas.face_size != face_size || atlas.capacity < required_slots);
        if !needs_new {
            return;
        }
        let capacity = required_slots.max(2);
        let mip_levels = mip_levels_for_edge(face_size);
        let texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("reflection_probe_specular_atlas"),
            size: wgpu::Extent3d {
                width: face_size,
                height: face_size,
                depth_or_array_layers: u32::from(capacity) * 6,
            },
            mip_level_count: mip_levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: REFLECTION_PROBE_ATLAS_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        }));
        let view = Arc::new(texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("reflection_probe_specular_atlas_view"),
            format: Some(REFLECTION_PROBE_ATLAS_FORMAT),
            dimension: Some(wgpu::TextureViewDimension::CubeArray),
            usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(mip_levels),
            base_array_layer: 0,
            array_layer_count: Some(u32::from(capacity) * 6),
        }));
        let sampler = Arc::new(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("reflection_probe_specular_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            lod_min_clamp: 0.0,
            lod_max_clamp: mip_levels.saturating_sub(1) as f32,
            ..Default::default()
        }));
        let metadata_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("reflection_probe_specular_metadata"),
            size: (usize::from(capacity) * size_of::<GpuReflectionProbeMetadata>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.version = self.version.wrapping_add(1).max(1);
        self.resources = Some(ReflectionProbeSpecularResources {
            cube_array_view: view,
            sampler,
            metadata_buffer,
            version: self.version,
        });
        self.atlas = Some(ReflectionProbeAtlas {
            texture,
            face_size,
            mip_levels,
            capacity,
            keys: vec![None; usize::from(capacity)],
        });
    }

    fn write_metadata(&self, queue: &wgpu::Queue, metadata: &[GpuReflectionProbeMetadata]) {
        let Some(resources) = &self.resources else {
            return;
        };
        queue.write_buffer(
            resources.metadata_buffer.as_ref(),
            0,
            bytemuck::cast_slice(metadata),
        );
    }

    fn encode_atlas_copies(
        &self,
        gpu: &GpuContext,
        face_size: u32,
        atlas_mips: u32,
        copy_jobs: Vec<AtlasCopyJob>,
    ) {
        if copy_jobs.is_empty() {
            return;
        }
        let Some(atlas) = &self.atlas else {
            return;
        };
        let mut encoder = gpu
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("reflection_probe_atlas_copy"),
            });
        for job in copy_jobs {
            let mips = job.mip_levels.min(atlas_mips);
            for mip in 0..mips {
                let extent = mip_extent(face_size, mip);
                encoder.copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: job.texture.as_ref(),
                        mip_level: mip,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture: atlas.texture.as_ref(),
                        mip_level: mip,
                        origin: wgpu::Origin3d {
                            x: 0,
                            y: 0,
                            z: u32::from(job.slot) * 6,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::Extent3d {
                        width: extent,
                        height: extent,
                        depth_or_array_layers: 6,
                    },
                );
            }
        }
        gpu.submit_frame_batch(vec![encoder.finish()], None, None);
    }
}

struct ReflectionProbeAtlas {
    texture: Arc<wgpu::Texture>,
    face_size: u32,
    mip_levels: u32,
    capacity: u16,
    keys: Vec<Option<SkyboxIblKey>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
struct ProbeIdentity {
    space_id: RenderSpaceId,
    renderable_index: i32,
}

struct ReadyProbe {
    identity: ProbeIdentity,
    key: SkyboxIblKey,
    texture: Arc<wgpu::Texture>,
    mip_levels: u32,
    metadata: GpuReflectionProbeMetadata,
    spatial: SpatialProbe,
}

struct ReadySkyboxFallback {
    space_id: RenderSpaceId,
    key: SkyboxIblKey,
    texture: Arc<wgpu::Texture>,
    mip_levels: u32,
}

struct AtlasCopyJob {
    slot: u16,
    texture: Arc<wgpu::Texture>,
    mip_levels: u32,
}

fn max_atlas_slots(limits: &GpuLimits) -> u16 {
    (limits.max_texture_array_layers() / 6)
        .min(u32::from(u16::MAX))
        .max(1) as u16
}

fn resolve_probe_source(
    skybox_material_asset_id: i32,
    probe: &ReflectionProbeEntry,
    materials: &MaterialSystem,
    assets: &AssetTransferQueue,
) -> Option<SkyboxIblSource> {
    let state = probe.state;
    if state.intensity <= 0.0 {
        return None;
    }
    if state.clear_flags == ReflectionProbeClear::Color {
        let color = state.background_color;
        return Some(solid_color_ibl_source(
            color_probe_identity(probe.renderable_index, color),
            color.to_array(),
        ));
    }
    if state.r#type == ReflectionProbeType::Baked {
        return resolve_baked_probe_source(state, assets);
    }
    if reflection_probe_skybox_only(state.flags) && skybox_material_asset_id >= 0 {
        return resolve_skybox_material_ibl_source(skybox_material_asset_id, materials, assets);
    }
    None
}

fn resolve_space_skybox_fallback_source(
    skybox_material_asset_id: i32,
    materials: &MaterialSystem,
    assets: &AssetTransferQueue,
) -> Option<SkyboxIblSource> {
    if skybox_material_asset_id < 0 {
        return None;
    }
    resolve_skybox_material_ibl_source(skybox_material_asset_id, materials, assets)
}

fn resolve_baked_probe_source(
    state: ReflectionProbeState,
    assets: &AssetTransferQueue,
) -> Option<SkyboxIblSource> {
    if state.cubemap_asset_id < 0 {
        return None;
    }
    let cubemap = assets.cubemap_pool().get(state.cubemap_asset_id)?;
    if cubemap.mip_levels_resident == 0 {
        return None;
    }
    Some(SkyboxIblSource::Cubemap(CubemapIblSource {
        asset_id: state.cubemap_asset_id,
        face_size: cubemap.size,
        mip_levels_resident: cubemap.mip_levels_resident,
        content_generation: cubemap.content_generation,
        storage_v_inverted: cubemap.storage_v_inverted,
        view: cubemap.view.clone(),
    }))
}

fn color_probe_identity(renderable_index: i32, color: Vec4) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for bits in [
        renderable_index as u32,
        color.x.to_bits(),
        color.y.to_bits(),
        color.z.to_bits(),
        color.w.to_bits(),
    ] {
        hash ^= u64::from(bits);
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}

fn spatial_probe_for_state(
    scene: &SceneCoordinator,
    space_id: RenderSpaceId,
    probe: &ReflectionProbeEntry,
    render_context: crate::shared::RenderingContext,
    atlas_index: u16,
) -> Option<SpatialProbe> {
    if probe.transform_id < 0 {
        return None;
    }
    let box_size = probe.state.box_size.abs();
    if box_size.cmplt(Vec3::splat(1e-6)).any() {
        return None;
    }
    let world =
        scene.world_matrix_for_context(space_id, probe.transform_id as usize, render_context)?;
    let bounds = crate::shared::RenderBoundingBox {
        center: Vec3::ZERO,
        extents: box_size * 0.5,
    };
    let (min, max) = world_aabb_from_local_bounds(&bounds, world)?;
    if !aabb_valid(min, max) {
        return None;
    }
    Some(SpatialProbe {
        renderable_index: probe.renderable_index,
        atlas_index,
        importance: probe.state.importance,
        aabb_min: Vec3A::from(min),
        aabb_max: Vec3A::from(max),
        center: Vec3A::from(world.transform_point3(Vec3::ZERO)),
        volume: aabb_volume(min, max),
    })
}

fn metadata_for_spatial(
    spatial: &SpatialProbe,
    state: ReflectionProbeState,
    sh2: &RenderSH2,
) -> GpuReflectionProbeMetadata {
    let flags = if reflection_probe_use_box_projection(state.flags) {
        REFLECTION_PROBE_METADATA_BOX_PROJECTION
    } else {
        0
    };
    GpuReflectionProbeMetadata {
        box_min: [
            spatial.aabb_min.x,
            spatial.aabb_min.y,
            spatial.aabb_min.z,
            0.0,
        ],
        box_max: [
            spatial.aabb_max.x,
            spatial.aabb_max.y,
            spatial.aabb_max.z,
            0.0,
        ],
        position: [spatial.center.x, spatial.center.y, spatial.center.z, 0.0],
        params: [
            state.intensity.max(0.0),
            0.0,
            flags as f32,
            REFLECTION_PROBE_METADATA_SH2_VALID,
        ],
        sh2: pack_render_sh2_raw(sh2),
    }
}

fn skybox_fallback_metadata(mip_levels: u32) -> GpuReflectionProbeMetadata {
    GpuReflectionProbeMetadata {
        params: [1.0, mip_levels.saturating_sub(1) as f32, 0.0, 0.0],
        ..GpuReflectionProbeMetadata::default()
    }
}

fn pack_render_sh2_raw(sh: &RenderSH2) -> [[f32; 4]; 9] {
    [
        [sh.sh0.x, sh.sh0.y, sh.sh0.z, 0.0],
        [sh.sh1.x, sh.sh1.y, sh.sh1.z, 0.0],
        [sh.sh2.x, sh.sh2.y, sh.sh2.z, 0.0],
        [sh.sh3.x, sh.sh3.y, sh.sh3.z, 0.0],
        [sh.sh4.x, sh.sh4.y, sh.sh4.z, 0.0],
        [sh.sh5.x, sh.sh5.y, sh.sh5.z, 0.0],
        [sh.sh6.x, sh.sh6.y, sh.sh6.z, 0.0],
        [sh.sh7.x, sh.sh7.y, sh.sh7.z, 0.0],
        [sh.sh8.x, sh.sh8.y, sh.sh8.z, 0.0],
    ]
}

fn aabb_valid(min: Vec3, max: Vec3) -> bool {
    min.is_finite() && max.is_finite() && (max - min).cmpgt(Vec3::ZERO).all()
}

fn aabb_volume(min: Vec3, max: Vec3) -> f32 {
    let d = (max - min).max(Vec3::ZERO);
    d.x * d.y * d.z
}

#[derive(Clone, Debug)]
struct SpatialProbe {
    renderable_index: i32,
    atlas_index: u16,
    importance: i32,
    aabb_min: Vec3A,
    aabb_max: Vec3A,
    center: Vec3A,
    volume: f32,
}

/// A BVH over reflection-probe AABBs for one render space.
#[derive(Default)]
pub struct ReflectionProbeSpatialIndex {
    probes: Vec<SpatialProbe>,
    order: Vec<usize>,
    nodes: Vec<BvhNode>,
    root: Option<usize>,
}

impl ReflectionProbeSpatialIndex {
    fn build(probes: Vec<SpatialProbe>) -> Self {
        let mut out = Self {
            order: (0..probes.len()).collect(),
            probes,
            nodes: Vec::new(),
            root: None,
        };
        if !out.probes.is_empty() {
            let mut order = std::mem::take(&mut out.order);
            let end = order.len();
            out.root = Some(out.build_node(&mut order, 0, end));
            out.order = order;
        }
        out
    }

    /// Selects up to two probes for one object AABB.
    #[must_use]
    pub fn select(&self, object_aabb: (Vec3, Vec3)) -> ReflectionProbeDrawSelection {
        let object_min = Vec3A::from(object_aabb.0);
        let object_max = Vec3A::from(object_aabb.1);
        if self.root.is_none() || !aabb_valid(object_aabb.0, object_aabb.1) {
            return ReflectionProbeDrawSelection::default();
        }
        let mut best_priority = i32::MIN;
        let mut top: [Option<ProbeScore>; 2] = [None, None];
        let mut stack = Vec::with_capacity(64);
        stack.push(self.root.unwrap_or(0));
        while let Some(node_index) = stack.pop() {
            let node = self.nodes[node_index];
            if !aabb_intersects(node.aabb_min, node.aabb_max, object_min, object_max) {
                continue;
            }
            if node.count > 0 {
                for &probe_index in &self.order[node.start..node.start + node.count] {
                    let probe = &self.probes[probe_index];
                    if !aabb_intersects(probe.aabb_min, probe.aabb_max, object_min, object_max) {
                        continue;
                    }
                    let intersection = intersection_volume_vec3a(
                        probe.aabb_min,
                        probe.aabb_max,
                        object_min,
                        object_max,
                    );
                    if intersection <= 0.0 {
                        continue;
                    }
                    if probe.importance > best_priority {
                        best_priority = probe.importance;
                        top = [None, None];
                    }
                    if probe.importance != best_priority {
                        continue;
                    }
                    insert_probe_score(
                        &mut top,
                        ProbeScore {
                            atlas_index: probe.atlas_index,
                            intersection,
                            probe_volume: probe.volume,
                            center_distance_sq: (probe.center
                                - object_center(object_min, object_max))
                            .length_squared(),
                            renderable_index: probe.renderable_index,
                        },
                    );
                }
            } else {
                stack.push(node.left);
                stack.push(node.right);
            }
        }
        selection_from_scores(top)
    }

    fn build_node(&mut self, order: &mut [usize], start: usize, end: usize) -> usize {
        let (aabb_min, aabb_max) = bounds_for_order(&self.probes, &order[start..end]);
        let index = self.nodes.len();
        self.nodes.push(BvhNode {
            aabb_min,
            aabb_max,
            start,
            count: 0,
            left: 0,
            right: 0,
        });
        let count = end - start;
        if count <= BVH_LEAF_SIZE {
            self.nodes[index].count = count;
            return index;
        }
        let axis = largest_axis(aabb_max - aabb_min);
        order[start..end].sort_unstable_by(|&a, &b| {
            let ac = axis_value(self.probes[a].center, axis);
            let bc = axis_value(self.probes[b].center, axis);
            ac.total_cmp(&bc).then_with(|| {
                self.probes[a]
                    .renderable_index
                    .cmp(&self.probes[b].renderable_index)
            })
        });
        let mid = start + count / 2;
        let left = self.build_node(order, start, mid);
        let right = self.build_node(order, mid, end);
        self.nodes[index].left = left;
        self.nodes[index].right = right;
        index
    }
}

#[derive(Clone, Copy)]
struct BvhNode {
    aabb_min: Vec3A,
    aabb_max: Vec3A,
    start: usize,
    count: usize,
    left: usize,
    right: usize,
}

#[derive(Clone, Copy, Debug)]
struct ProbeScore {
    atlas_index: u16,
    intersection: f32,
    probe_volume: f32,
    center_distance_sq: f32,
    renderable_index: i32,
}

fn insert_probe_score(top: &mut [Option<ProbeScore>; 2], score: ProbeScore) {
    if top[0].is_none_or(|best| score_better(score, best)) {
        top[1] = top[0];
        top[0] = Some(score);
    } else if top[1].is_none_or(|second| score_better(score, second)) {
        top[1] = Some(score);
    }
}

fn score_better(a: ProbeScore, b: ProbeScore) -> bool {
    a.intersection
        .total_cmp(&b.intersection)
        .reverse()
        .then_with(|| a.probe_volume.total_cmp(&b.probe_volume))
        .then_with(|| a.center_distance_sq.total_cmp(&b.center_distance_sq))
        .then_with(|| a.renderable_index.cmp(&b.renderable_index))
        .is_lt()
}

fn selection_from_scores(top: [Option<ProbeScore>; 2]) -> ReflectionProbeDrawSelection {
    let Some(first) = top[0] else {
        return ReflectionProbeDrawSelection::default();
    };
    let Some(second) = top[1] else {
        return ReflectionProbeDrawSelection::one(first.atlas_index);
    };
    let denom = first.intersection + second.intersection;
    if denom <= MIN_OBJECT_VOLUME {
        return ReflectionProbeDrawSelection::one(first.atlas_index);
    }
    ReflectionProbeDrawSelection::two(
        first.atlas_index,
        second.atlas_index,
        second.intersection / denom,
    )
}

fn bounds_for_order(probes: &[SpatialProbe], order: &[usize]) -> (Vec3A, Vec3A) {
    let mut min = Vec3A::splat(f32::INFINITY);
    let mut max = Vec3A::splat(f32::NEG_INFINITY);
    for &index in order {
        min = min.min(probes[index].aabb_min);
        max = max.max(probes[index].aabb_max);
    }
    (min, max)
}

fn aabb_intersects(a_min: Vec3A, a_max: Vec3A, b_min: Vec3A, b_max: Vec3A) -> bool {
    a_min.cmple(b_max).all() && a_max.cmpge(b_min).all()
}

fn intersection_volume_vec3a(a_min: Vec3A, a_max: Vec3A, b_min: Vec3A, b_max: Vec3A) -> f32 {
    let d = (a_max.min(b_max) - a_min.max(b_min)).max(Vec3A::ZERO);
    d.x * d.y * d.z
}

fn object_center(min: Vec3A, max: Vec3A) -> Vec3A {
    (min + max) * 0.5
}

fn largest_axis(v: Vec3A) -> usize {
    if v.x >= v.y && v.x >= v.z {
        0
    } else if v.y >= v.z {
        1
    } else {
        2
    }
}

fn axis_value(v: Vec3A, axis: usize) -> f32 {
    match axis {
        0 => v.x,
        1 => v.y,
        _ => v.z,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::assets::AssetTransferQueue;

    fn probe(index: i32, atlas: u16, importance: i32, min: Vec3, max: Vec3) -> SpatialProbe {
        SpatialProbe {
            renderable_index: index,
            atlas_index: atlas,
            importance,
            aabb_min: Vec3A::from(min),
            aabb_max: Vec3A::from(max),
            center: Vec3A::from((min + max) * 0.5),
            volume: aabb_volume(min, max),
        }
    }

    #[test]
    fn higher_priority_overrides_lower_priority() {
        let index = ReflectionProbeSpatialIndex::build(vec![
            probe(0, 1, 0, Vec3::splat(-100.0), Vec3::splat(100.0)),
            probe(1, 2, 1, Vec3::splat(-1.0), Vec3::splat(1.0)),
        ]);

        let selection = index.select((Vec3::splat(-0.25), Vec3::splat(0.25)));

        assert_eq!(selection, ReflectionProbeDrawSelection::one(2));
    }

    #[test]
    fn missing_baked_cubemap_is_not_a_probe_source() {
        let assets = AssetTransferQueue::new();
        let state = ReflectionProbeState {
            intensity: 1.0,
            cubemap_asset_id: 42,
            r#type: ReflectionProbeType::Baked,
            ..ReflectionProbeState::default()
        };

        assert!(resolve_baked_probe_source(state, &assets).is_none());
    }

    #[test]
    fn frame_selection_uses_skybox_fallback_when_no_probe_hits() {
        let mut selection = ReflectionProbeFrameSelection::default();
        let space_id = RenderSpaceId(7);
        selection.rebuild_spatial(Vec::new(), [(space_id, 9)]);

        let draw = selection.select(space_id, (Vec3::splat(-1.0), Vec3::splat(1.0)));

        assert_eq!(
            draw,
            ReflectionProbeDrawSelection {
                first_atlas_index: 9,
                second_atlas_index: 0,
                second_weight: 0.0,
                hit_count: 0,
            }
        );
    }

    #[test]
    fn frame_selection_prefers_probe_hit_over_skybox_fallback() {
        let mut selection = ReflectionProbeFrameSelection::default();
        let space_id = RenderSpaceId(7);
        selection.rebuild_spatial(
            [(
                space_id,
                probe(0, 3, 1, Vec3::splat(-1.0), Vec3::splat(1.0)),
            )],
            [(space_id, 9)],
        );

        let draw = selection.select(space_id, (Vec3::splat(-0.5), Vec3::splat(0.5)));

        assert_eq!(draw, ReflectionProbeDrawSelection::one(3));
    }

    #[test]
    fn same_priority_selects_two_by_intersection_volume() {
        let index = ReflectionProbeSpatialIndex::build(vec![
            probe(
                0,
                1,
                1,
                Vec3::new(-1.0, -1.0, -1.0),
                Vec3::new(1.0, 1.0, 1.0),
            ),
            probe(
                1,
                2,
                1,
                Vec3::new(0.0, -1.0, -1.0),
                Vec3::new(2.0, 1.0, 1.0),
            ),
            probe(
                2,
                3,
                1,
                Vec3::new(0.75, -1.0, -1.0),
                Vec3::new(2.0, 1.0, 1.0),
            ),
        ]);

        let selection = index.select((Vec3::new(-0.5, -0.5, -0.5), Vec3::new(1.5, 0.5, 0.5)));

        assert_eq!(selection.hit_count, 2);
        assert_eq!(selection.first_atlas_index, 1);
        assert_eq!(selection.second_atlas_index, 2);
        assert!(selection.second_weight > 0.0 && selection.second_weight < 1.0);
    }

    #[test]
    fn contained_same_priority_probes_still_blend() {
        let index = ReflectionProbeSpatialIndex::build(vec![
            probe(0, 1, 1, Vec3::splat(-10.0), Vec3::splat(10.0)),
            probe(1, 2, 1, Vec3::splat(-1.0), Vec3::splat(1.0)),
        ]);

        let selection = index.select((Vec3::splat(-0.5), Vec3::splat(0.5)));

        assert_eq!(selection.hit_count, 2);
        assert_eq!(selection.first_atlas_index, 2);
        assert_eq!(selection.second_atlas_index, 1);
        assert!(selection.second_weight > 0.0 && selection.second_weight < 1.0);
    }

    #[test]
    fn bvh_matches_bruteforce_candidates() {
        let probes: Vec<_> = (0..32)
            .map(|i| {
                let x = i as f32 * 0.5;
                probe(
                    i,
                    (i + 1) as u16,
                    1,
                    Vec3::new(x, -1.0, -1.0),
                    Vec3::new(x + 1.0, 1.0, 1.0),
                )
            })
            .collect();
        let index = ReflectionProbeSpatialIndex::build(probes.clone());
        let object = (Vec3::new(4.2, -0.25, -0.25), Vec3::new(6.1, 0.25, 0.25));
        let selection = index.select(object);

        let mut brute: Vec<_> = probes
            .iter()
            .filter_map(|probe| {
                let v = intersection_volume_vec3a(
                    probe.aabb_min,
                    probe.aabb_max,
                    Vec3A::from(object.0),
                    Vec3A::from(object.1),
                );
                (v > 0.0).then_some((probe.atlas_index, v))
            })
            .collect();
        brute.sort_by(|a, b| b.1.total_cmp(&a.1));

        assert_eq!(selection.first_atlas_index, brute[0].0);
        assert_eq!(selection.second_atlas_index, brute[1].0);
    }
}
