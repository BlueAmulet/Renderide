use std::sync::Arc;

use hashbrown::HashSet;

use crate::backend::AssetTransferQueue;
use crate::backend::frame_gpu::{
    GpuReflectionProbeMetadata, REFLECTION_PROBE_ATLAS_FORMAT, ReflectionProbeSpecularResources,
};
use crate::gpu::GpuContext;
use crate::materials::MaterialSystem;
use crate::scene::{RenderSpaceId, SceneCoordinator};
use crate::shared::RenderSH2;
use crate::skybox::ibl_cache::{
    SkyboxIblCache, SkyboxIblKey, build_key, clamp_face_size, mip_extent, mip_levels_for_edge,
};
use crate::{profiling, reflection_probes::ReflectionProbeSh2System};

use super::atlas::{AtlasCopyJob, ReflectionProbeAtlas, max_atlas_slots};
use super::captures::{
    RuntimeReflectionProbeCapture, RuntimeReflectionProbeCaptureKey,
    RuntimeReflectionProbeCaptureStore,
};
use super::selection::{ReflectionProbeFrameSelection, SpatialProbe};
use super::source::{
    metadata_for_spatial, resolve_probe_source, resolve_space_skybox_fallback_source,
    skybox_fallback_metadata, spatial_probe_for_state,
};

/// Default destination face size for reflection-probe IBL bakes.
const DEFAULT_REFLECTION_PROBE_FACE_SIZE: u32 = 256;
/// First atlas slot is reserved as a non-sampled black fallback.
const FIRST_PROBE_ATLAS_SLOT: u16 = 1;

/// Specular reflection-probe bake/cache/selection system.
pub struct ReflectionProbeSpecularSystem {
    ibl_cache: SkyboxIblCache,
    atlas: Option<ReflectionProbeAtlas>,
    resources: Option<ReflectionProbeSpecularResources>,
    selection: ReflectionProbeFrameSelection,
    captures: RuntimeReflectionProbeCaptureStore,
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
            captures: RuntimeReflectionProbeCaptureStore::default(),
            version: 1,
        }
    }

    /// Registers a completed runtime cubemap capture for an OnChanges reflection probe.
    pub(crate) fn register_runtime_capture(&mut self, capture: RuntimeReflectionProbeCapture) {
        self.captures.insert(capture);
    }

    /// Runtime OnChanges capture store used by SH2 task resolution.
    #[must_use]
    pub(crate) fn capture_store(&self) -> &RuntimeReflectionProbeCaptureStore {
        &self.captures
    }

    /// Advances GPU bakes, updates the atlas, and rebuilds the CPU selection index.
    pub fn maintain(
        &mut self,
        gpu: &mut GpuContext,
        scene: &SceneCoordinator,
        materials: &MaterialSystem,
        assets: &AssetTransferQueue,
        render_context: crate::shared::RenderingContext,
        sh2_system: &mut ReflectionProbeSh2System,
    ) {
        profiling::scope!("reflection_probes::specular::maintain");
        self.ibl_cache.maintain_completed_jobs(gpu.device());
        let face_size = clamp_face_size(DEFAULT_REFLECTION_PROBE_FACE_SIZE, gpu.limits());
        let mut active_keys = HashSet::new();
        let mut active_capture_keys = HashSet::new();
        let mut ready = Vec::new();
        let mut skybox_fallbacks = Vec::new();

        for space_id in scene.render_space_ids() {
            let Some(space) = scene.space(space_id) else {
                continue;
            };
            if !space.is_active() {
                continue;
            }
            if let Some(source) = resolve_space_skybox_fallback_source(
                space.skybox_material_asset_id(),
                materials,
                assets,
            ) {
                let key = build_key(&source, face_size);
                active_keys.insert(key.clone());
                let sh2 = sh2_system.ensure_ibl_source(space_id.0, &source);
                self.ibl_cache.ensure_source(gpu, key.clone(), source);
                if let Some(cube) = self.ibl_cache.completed_cube(&key) {
                    skybox_fallbacks.push(ReadySkyboxFallback {
                        space_id,
                        key,
                        texture: cube.texture.clone(),
                        mip_levels: cube.mip_levels,
                        sh2,
                    });
                }
            }
            for probe in space.reflection_probes() {
                let identity = ProbeIdentity {
                    space_id,
                    renderable_index: probe.renderable_index,
                };
                if probe.state.r#type == crate::shared::ReflectionProbeType::OnChanges {
                    active_capture_keys.insert(RuntimeReflectionProbeCaptureKey {
                        space_id,
                        renderable_index: probe.renderable_index,
                    });
                }
                let Some(source) = resolve_probe_source(
                    space_id,
                    space.skybox_material_asset_id(),
                    probe,
                    materials,
                    assets,
                    &self.captures,
                ) else {
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
        self.captures.retain_active(&active_capture_keys);
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
            self.selection.rebuild_spatial(Vec::new(), Vec::new());
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
            self.selection.rebuild_spatial(Vec::new(), Vec::new());
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
            selectable.push((probe.identity.space_id, probe.spatial));
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
            metadata[slot as usize] =
                skybox_fallback_metadata(fallback.mip_levels, fallback.sh2.as_ref());
            skybox_fallback_slots.push((fallback.space_id, slot));
        }
        self.write_metadata(gpu.queue(), &metadata);
        self.encode_atlas_copies(gpu, face_size, mip_levels, copy_jobs);
        self.selection
            .rebuild_spatial(selectable, skybox_fallback_slots);
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
        crate::profiling::note_resource_churn!(
            TextureView,
            "reflection_probes::specular_atlas_view"
        );
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
        crate::profiling::note_resource_churn!(
            Buffer,
            "reflection_probes::specular_metadata_buffer"
        );
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
    sh2: Option<RenderSH2>,
}
