//! Unified IBL bake cache for specular reflection sources.
//!
//! Owns one in-flight bake job tracker, three lazily-built mip-0 producer pipelines (analytic
//! procedural / gradient skies, host cubemaps, and Projection360 equirect Texture2Ds), one
//! source-pyramid downsample pipeline, and one GGX convolve pipeline. For each new active
//! reflection source the cache:
//!
//! 1. Allocates a source Rgba16Float cubemap and a filtered output cubemap with full mip chains.
//! 2. Records a mip-0 producer compute pass that converts the source into the source cube's mip 0.
//! 3. Copies source mip 0 into filtered output mip 0 for mirror-smooth reflections.
//! 4. Records downsample passes that build the source radiance mip pyramid.
//! 5. Records one GGX convolve compute pass per filtered mip in `1..N`, sampling the full source
//!    pyramid with solid-angle source-mip selection.
//! 6. Submits the encoder through [`GpuSubmitJobTracker`] and parks the cube in `pending` until
//!    the submit-completion callback promotes it to `completed`.
//!
//! The completed prefiltered cube is reused by reflection probes so every source type reaches
//! shader sampling through a single GGX-prefiltered cube.

use std::sync::Arc;

use hashbrown::HashMap;
use thiserror::Error;

use crate::backend::gpu_jobs::{GpuJobResources, GpuSubmitJobTracker, SubmittedGpuJob};
use crate::gpu::{GpuContext, GpuLimits};
use crate::profiling::GpuProfilerHandle;
use crate::skybox::specular::{SkyboxIblSource, solid_color_params};

mod encode;
mod key;
mod pipeline;
mod resources;

use encode::{
    AnalyticEncodeContext, ConvolveEncodeContext, CubeEncodeContext, DownsampleEncodeContext,
    EquirectEncodeContext, encode_analytic_mip0, encode_convolve_mips, encode_cube_mip0,
    encode_downsample_mips, encode_equirect_mip0,
};
use key::source_max_lod;
pub(crate) use key::{SkyboxIblKey, build_key, mip_extent, mip_levels_for_edge};
#[cfg(test)]
use key::{convolve_sample_count, hash_float4};
use pipeline::{
    ComputePipeline, analytic_layout_entries, downsample_layout_entries, ensure_pipeline,
    mip0_input_layout_entries,
};
use resources::{
    PendingBake, PendingBakeResources, PrefilteredCube, create_full_cube_sample_view,
    create_ibl_cube,
};

/// Maximum concurrent in-flight bakes; matches the analytic-only ceiling we used previously.
const MAX_IN_FLIGHT_IBL_BAKES: usize = 2;
/// Tick budget after which a missing submit-completion callback is treated as lost.
const MAX_PENDING_IBL_BAKE_AGE_FRAMES: u32 = 120;
/// Clamps the configured cube face size against the device texture limit.
pub(crate) fn clamp_face_size(face_size: u32, limits: &GpuLimits) -> u32 {
    face_size.min(limits.max_texture_dimension_2d()).max(1)
}

fn copy_cube_mip0(
    encoder: &mut wgpu::CommandEncoder,
    source: &wgpu::Texture,
    destination: &wgpu::Texture,
    face_size: u32,
) {
    encoder.copy_texture_to_texture(
        wgpu::TexelCopyTextureInfo {
            texture: source,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyTextureInfo {
            texture: destination,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::Extent3d {
            width: face_size,
            height: face_size,
            depth_or_array_layers: 6,
        },
    );
}

/// Errors returned while preparing an IBL bake.
#[derive(Debug, Error)]
enum SkyboxIblBakeError {
    /// Embedded WGSL source was not available at compose time.
    #[error("embedded shader {0} not found")]
    MissingShader(&'static str),
}

/// Errors returned while encoding GGX convolve mips for an existing cubemap.
#[derive(Debug, Error)]
pub(crate) enum SkyboxIblConvolveError {
    /// Embedded WGSL source was not available at compose time.
    #[error("embedded shader {0} not found")]
    MissingShader(&'static str),
}

/// Resources produced while encoding convolve passes and retained until submit completion.
pub(crate) struct SkyboxIblConvolveResources {
    _resources: PendingBakeResources,
    _source_sample_view: Arc<wgpu::TextureView>,
    _sampler: Arc<wgpu::Sampler>,
}

/// Minimal GGX convolver for caller-owned cubemap textures.
#[derive(Default)]
pub(crate) struct SkyboxIblConvolver {
    downsample_pipeline: Option<ComputePipeline>,
    convolve_pipeline: Option<ComputePipeline>,
    input_sampler: Option<Arc<wgpu::Sampler>>,
}

impl SkyboxIblConvolver {
    /// Creates an empty convolver with lazily-built GPU resources.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Encodes GGX convolve passes for mips `1..mip_levels` of `texture`.
    pub(crate) fn encode_existing_cube_mips(
        &mut self,
        gpu: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        texture: &wgpu::Texture,
        face_size: u32,
        mip_levels: u32,
        profiler: Option<&GpuProfilerHandle>,
    ) -> Result<SkyboxIblConvolveResources, SkyboxIblConvolveError> {
        profiling::scope!("skybox_ibl::encode_existing_cube_mips");
        let sampler = self
            .input_sampler
            .get_or_insert_with(|| Arc::new(create_ibl_input_sampler(gpu.device())))
            .clone();
        let pipeline = ensure_pipeline(
            &mut self.convolve_pipeline,
            gpu.device(),
            "skybox_ibl_convolve_params",
            &mip0_input_layout_entries(wgpu::TextureViewDimension::Cube),
        )
        .map_err(|_err| SkyboxIblConvolveError::MissingShader("skybox_ibl_convolve_params"))?;
        let downsample_pipeline = ensure_pipeline(
            &mut self.downsample_pipeline,
            gpu.device(),
            "skybox_ibl_downsample",
            &downsample_layout_entries(),
        )
        .map_err(|_err| SkyboxIblConvolveError::MissingShader("skybox_ibl_downsample"))?;
        let source_cube = create_ibl_cube(
            gpu.device(),
            "skybox_ibl_existing_source_cube",
            face_size,
            mip_levels,
        );
        let source_sample_view = Arc::new(create_full_cube_sample_view(
            source_cube.texture.as_ref(),
            mip_levels,
        ));
        let mut resources = PendingBakeResources::default();
        resources.textures.push(source_cube.texture.clone());
        resources.source_sample_view = Some(source_sample_view.clone());
        copy_cube_mip0(encoder, texture, source_cube.texture.as_ref(), face_size);
        encode_downsample_mips(
            DownsampleEncodeContext {
                device: gpu.device(),
                encoder,
                pipeline: downsample_pipeline,
                texture: source_cube.texture.as_ref(),
                face_size,
                mip_levels,
                profiler,
            },
            &mut resources,
        );
        encode_convolve_mips(
            ConvolveEncodeContext {
                device: gpu.device(),
                encoder,
                pipeline,
                texture,
                src_view: source_sample_view.as_ref(),
                sampler: sampler.as_ref(),
                face_size,
                mip_levels,
                src_max_lod: source_max_lod(mip_levels),
                profiler,
            },
            &mut resources,
        );
        Ok(SkyboxIblConvolveResources {
            _resources: resources,
            _source_sample_view: source_sample_view,
            _sampler: sampler,
        })
    }
}

fn create_ibl_input_sampler(device: &wgpu::Device) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("skybox_ibl_existing_cube_sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        ..Default::default()
    })
}

struct SourceMip0EncodeContext<'a> {
    gpu: &'a GpuContext,
    encoder: &'a mut wgpu::CommandEncoder,
    texture: &'a wgpu::Texture,
    face_size: u32,
    sampler: &'a wgpu::Sampler,
    profiler: Option<&'a GpuProfilerHandle>,
}

/// Owns IBL bakes for prefiltered specular reflection cubemaps.
pub(crate) struct SkyboxIblCache {
    /// Submit-completion tracker for in-flight bakes.
    jobs: GpuSubmitJobTracker<SkyboxIblKey>,
    /// In-flight prefiltered cubes retained until their submit callback fires.
    pending: HashMap<SkyboxIblKey, PendingBake>,
    /// Completed prefiltered cubes for the active skybox key.
    completed: HashMap<SkyboxIblKey, PrefilteredCube>,
    /// Lazily-built analytic mip-0 pipeline (re-uses the existing `skybox_bake_params` shader).
    analytic_pipeline: Option<ComputePipeline>,
    /// Lazily-built cube mip-0 pipeline.
    cube_pipeline: Option<ComputePipeline>,
    /// Lazily-built equirect mip-0 pipeline.
    equirect_pipeline: Option<ComputePipeline>,
    /// Lazily-built source-pyramid downsample pipeline.
    downsample_pipeline: Option<ComputePipeline>,
    /// Lazily-built GGX convolve pipeline (cube -> cube via solid-angle source mip selection).
    convolve_pipeline: Option<ComputePipeline>,
    /// Cached input sampler used by all producers and the convolve pass.
    input_sampler: Option<Arc<wgpu::Sampler>>,
}

impl Default for SkyboxIblCache {
    fn default() -> Self {
        Self::new()
    }
}

impl SkyboxIblCache {
    /// Creates an empty IBL cache.
    pub(crate) fn new() -> Self {
        Self {
            jobs: GpuSubmitJobTracker::new(MAX_PENDING_IBL_BAKE_AGE_FRAMES),
            pending: HashMap::new(),
            completed: HashMap::new(),
            analytic_pipeline: None,
            cube_pipeline: None,
            equirect_pipeline: None,
            downsample_pipeline: None,
            convolve_pipeline: None,
            input_sampler: None,
        }
    }

    /// Drains submit-completed bakes.
    pub(crate) fn maintain_completed_jobs(&mut self, device: &wgpu::Device) {
        let _ = device.poll(wgpu::PollType::Poll);
        self.drain_completed_jobs();
    }

    /// Removes completed cubes whose keys are not retained by the caller.
    pub(crate) fn prune_completed_except(&mut self, retain: &hashbrown::HashSet<SkyboxIblKey>) {
        self.completed.retain(|key, _| retain.contains(key));
    }

    /// Ensures one arbitrary IBL source is scheduled for baking.
    pub(crate) fn ensure_source(
        &mut self,
        gpu: &mut GpuContext,
        key: SkyboxIblKey,
        source: SkyboxIblSource,
    ) {
        if self.completed.contains_key(&key)
            || self.pending.contains_key(&key)
            || self.jobs.contains_key(&key)
            || self.jobs.len() >= MAX_IN_FLIGHT_IBL_BAKES
        {
            return;
        }
        if let Err(e) = self.schedule_bake(gpu, key, source) {
            logger::warn!("skybox_ibl: bake failed: {e}");
        }
    }

    /// Returns a completed prefiltered cube by key.
    pub(crate) fn completed_cube(&self, key: &SkyboxIblKey) -> Option<&PrefilteredCube> {
        self.completed.get(key)
    }

    /// Promotes submit-completed bakes into the completed cache.
    fn drain_completed_jobs(&mut self) {
        let outcomes = self.jobs.maintain();
        for key in outcomes.completed {
            if let Some(pending) = self.pending.remove(&key) {
                self.completed.insert(key, pending.cube);
            }
        }
        for key in outcomes.failed {
            self.pending.remove(&key);
            logger::warn!("skybox_ibl: bake expired before submit completion (key {key:?})");
        }
    }

    /// Encodes one IBL bake (mip-0 producer + per-mip GGX convolves) and submits it.
    fn schedule_bake(
        &mut self,
        gpu: &mut GpuContext,
        key: SkyboxIblKey,
        source: SkyboxIblSource,
    ) -> Result<(), SkyboxIblBakeError> {
        profiling::scope!("skybox_ibl::schedule_bake");
        let mut profiler = gpu.take_gpu_profiler();
        let result = self.schedule_bake_with_profiler(gpu, key, source, profiler.as_mut());
        gpu.restore_gpu_profiler(profiler);
        result
    }

    fn schedule_bake_with_profiler(
        &mut self,
        gpu: &GpuContext,
        key: SkyboxIblKey,
        source: SkyboxIblSource,
        mut profiler: Option<&mut GpuProfilerHandle>,
    ) -> Result<(), SkyboxIblBakeError> {
        self.ensure_pipelines(gpu.device())?;
        let input_sampler = self.ensure_input_sampler(gpu.device()).clone();
        let face_size = key.face_size();
        let mip_levels = mip_levels_for_edge(face_size);
        let source_cube = create_ibl_cube(
            gpu.device(),
            "skybox_ibl_source_cube",
            face_size,
            mip_levels,
        );
        let filtered_cube = create_ibl_cube(
            gpu.device(),
            "skybox_ibl_filtered_cube",
            face_size,
            mip_levels,
        );
        let mut resources = PendingBakeResources::default();
        let source_sample_view = Arc::new(create_full_cube_sample_view(
            &source_cube.texture,
            mip_levels,
        ));
        resources.textures.push(source_cube.texture.clone());
        resources.source_sample_view = Some(source_sample_view.clone());
        let mut encoder = gpu
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("skybox_ibl bake encoder"),
            });
        self.encode_source_mip0(
            SourceMip0EncodeContext {
                gpu,
                encoder: &mut encoder,
                texture: source_cube.texture.as_ref(),
                face_size,
                sampler: input_sampler.as_ref(),
                profiler: profiler.as_deref(),
            },
            source,
            &mut resources,
        )?;
        copy_cube_mip0(
            &mut encoder,
            source_cube.texture.as_ref(),
            filtered_cube.texture.as_ref(),
            face_size,
        );
        let downsample_pipeline = self.downsample_pipeline()?;
        encode_downsample_mips(
            DownsampleEncodeContext {
                device: gpu.device(),
                encoder: &mut encoder,
                pipeline: downsample_pipeline,
                texture: source_cube.texture.as_ref(),
                face_size,
                mip_levels,
                profiler: profiler.as_deref(),
            },
            &mut resources,
        );
        let convolve_pipeline = self.convolve_pipeline()?;
        encode_convolve_mips(
            ConvolveEncodeContext {
                device: gpu.device(),
                encoder: &mut encoder,
                pipeline: convolve_pipeline,
                texture: filtered_cube.texture.as_ref(),
                src_view: source_sample_view.as_ref(),
                sampler: input_sampler.as_ref(),
                face_size,
                mip_levels,
                src_max_lod: source_max_lod(mip_levels),
                profiler: profiler.as_deref(),
            },
            &mut resources,
        );
        if let Some(profiler) = profiler.as_mut() {
            profiling::scope!("skybox_ibl::resolve_profiler_queries");
            profiler.resolve_queries(&mut encoder);
        }
        let pending = PendingBake {
            cube: PrefilteredCube {
                texture: filtered_cube.texture,
                mip_levels,
            },
            _resources: resources,
        };
        self.submit_pending_bake(gpu, key, encoder, pending);
        Ok(())
    }

    fn encode_source_mip0(
        &self,
        ctx: SourceMip0EncodeContext<'_>,
        source: SkyboxIblSource,
        resources: &mut PendingBakeResources,
    ) -> Result<(), SkyboxIblBakeError> {
        match source {
            SkyboxIblSource::Analytic(src) => {
                let pipeline = self.analytic_pipeline()?;
                encode_analytic_mip0(
                    AnalyticEncodeContext {
                        device: ctx.gpu.device(),
                        encoder: ctx.encoder,
                        pipeline,
                        texture: ctx.texture,
                        face_size: ctx.face_size,
                        params: &src.params,
                        profiler: ctx.profiler,
                    },
                    resources,
                );
            }
            SkyboxIblSource::Cubemap(src) => {
                let pipeline = self.cube_pipeline()?;
                encode_cube_mip0(
                    CubeEncodeContext {
                        device: ctx.gpu.device(),
                        encoder: ctx.encoder,
                        pipeline,
                        texture: ctx.texture,
                        face_size: ctx.face_size,
                        src,
                        sampler: ctx.sampler,
                        profiler: ctx.profiler,
                    },
                    resources,
                );
            }
            SkyboxIblSource::Equirect(src) => {
                let pipeline = self.equirect_pipeline()?;
                encode_equirect_mip0(
                    EquirectEncodeContext {
                        device: ctx.gpu.device(),
                        encoder: ctx.encoder,
                        pipeline,
                        texture: ctx.texture,
                        face_size: ctx.face_size,
                        src,
                        sampler: ctx.sampler,
                        profiler: ctx.profiler,
                    },
                    resources,
                );
            }
            SkyboxIblSource::SolidColor(src) => {
                let params = solid_color_params(src.color);
                let pipeline = self.analytic_pipeline()?;
                encode_analytic_mip0(
                    AnalyticEncodeContext {
                        device: ctx.gpu.device(),
                        encoder: ctx.encoder,
                        pipeline,
                        texture: ctx.texture,
                        face_size: ctx.face_size,
                        params: &params,
                        profiler: ctx.profiler,
                    },
                    resources,
                );
            }
        }
        Ok(())
    }

    /// Ensures every compute pipeline used by IBL bakes is resident.
    fn ensure_pipelines(&mut self, device: &wgpu::Device) -> Result<(), SkyboxIblBakeError> {
        profiling::scope!("skybox_ibl::ensure_pipelines");
        let _ = ensure_pipeline(
            &mut self.analytic_pipeline,
            device,
            "skybox_bake_params",
            &analytic_layout_entries(),
        )?;
        let _ = ensure_pipeline(
            &mut self.cube_pipeline,
            device,
            "skybox_mip0_cube_params",
            &mip0_input_layout_entries(wgpu::TextureViewDimension::Cube),
        )?;
        let _ = ensure_pipeline(
            &mut self.equirect_pipeline,
            device,
            "skybox_mip0_equirect_params",
            &mip0_input_layout_entries(wgpu::TextureViewDimension::D2),
        )?;
        let _ = ensure_pipeline(
            &mut self.downsample_pipeline,
            device,
            "skybox_ibl_downsample",
            &downsample_layout_entries(),
        )?;
        let _ = ensure_pipeline(
            &mut self.convolve_pipeline,
            device,
            "skybox_ibl_convolve_params",
            &mip0_input_layout_entries(wgpu::TextureViewDimension::Cube),
        )?;
        Ok(())
    }

    fn analytic_pipeline(&self) -> Result<&ComputePipeline, SkyboxIblBakeError> {
        self.analytic_pipeline
            .as_ref()
            .ok_or(SkyboxIblBakeError::MissingShader("skybox_bake_params"))
    }

    fn cube_pipeline(&self) -> Result<&ComputePipeline, SkyboxIblBakeError> {
        self.cube_pipeline
            .as_ref()
            .ok_or(SkyboxIblBakeError::MissingShader("skybox_mip0_cube_params"))
    }

    fn equirect_pipeline(&self) -> Result<&ComputePipeline, SkyboxIblBakeError> {
        self.equirect_pipeline
            .as_ref()
            .ok_or(SkyboxIblBakeError::MissingShader(
                "skybox_mip0_equirect_params",
            ))
    }

    fn convolve_pipeline(&self) -> Result<&ComputePipeline, SkyboxIblBakeError> {
        self.convolve_pipeline
            .as_ref()
            .ok_or(SkyboxIblBakeError::MissingShader(
                "skybox_ibl_convolve_params",
            ))
    }

    fn downsample_pipeline(&self) -> Result<&ComputePipeline, SkyboxIblBakeError> {
        self.downsample_pipeline
            .as_ref()
            .ok_or(SkyboxIblBakeError::MissingShader("skybox_ibl_downsample"))
    }

    /// Returns a cached linear/clamp sampler used for all source/destination cube reads.
    fn ensure_input_sampler(&mut self, device: &wgpu::Device) -> &Arc<wgpu::Sampler> {
        self.input_sampler.get_or_insert_with(|| {
            Arc::new(device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("skybox_ibl_input_sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Linear,
                ..Default::default()
            }))
        })
    }

    /// Tracks and submits an encoded bake, retaining transient resources until completion.
    fn submit_pending_bake(
        &mut self,
        gpu: &GpuContext,
        key: SkyboxIblKey,
        encoder: wgpu::CommandEncoder,
        pending: PendingBake,
    ) {
        profiling::scope!("skybox_ibl::submit_bake");
        let tx = self.jobs.submit_done_sender();
        let callback_key = key.clone();
        self.jobs.insert(
            key.clone(),
            SubmittedGpuJob {
                resources: GpuJobResources::new(),
            },
        );
        self.pending.insert(key, pending);
        let command_buffer = {
            profiling::scope!("CommandEncoder::finish::skybox_ibl");
            encoder.finish()
        };
        gpu.submit_frame_batch_with_callbacks(
            vec![command_buffer],
            None,
            None,
            vec![Box::new(move || {
                let _ = tx.send(callback_key);
            })],
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip: applying the runtime parabolic LOD then the inverse returns the input.
    #[test]
    fn roughness_lod_round_trip() {
        for i in 0..=20u32 {
            let r = i as f32 / 20.0;
            let lod = r * (2.0 - r);
            let r_back = 1.0 - (1.0 - lod).max(0.0).sqrt();
            assert!((r - r_back).abs() < 1e-6, "r={r} r_back={r_back}");
        }
    }

    /// Mip count includes mip 0 through the one-texel mip.
    #[test]
    fn mip_levels_for_edge_includes_tail_mip() {
        assert_eq!(mip_levels_for_edge(1), 1);
        assert_eq!(mip_levels_for_edge(2), 2);
        assert_eq!(mip_levels_for_edge(128), 8);
        assert_eq!(mip_levels_for_edge(256), 9);
    }

    /// Source-LOD clamping exposes every generated source mip to filtered importance sampling.
    #[test]
    fn source_max_lod_tracks_last_generated_mip() {
        assert_eq!(source_max_lod(0), 0.0);
        assert_eq!(source_max_lod(1), 0.0);
        assert_eq!(source_max_lod(8), 7.0);
    }

    /// Per-mip sample count clamps to the documented base/cap envelope.
    #[test]
    fn convolve_sample_count_envelope() {
        assert_eq!(convolve_sample_count(0), 1);
        assert_eq!(convolve_sample_count(1), 64);
        assert_eq!(convolve_sample_count(2), 128);
        assert_eq!(convolve_sample_count(3), 256);
        assert_eq!(convolve_sample_count(4), 512);
        assert_eq!(convolve_sample_count(5), 1024);
        assert_eq!(convolve_sample_count(8), 1024);
    }

    /// Analytic key invariants: identity bits change the source hash.
    #[test]
    fn analytic_key_hash_changes_with_identity_fields() {
        let a = SkyboxIblKey::Analytic {
            material_asset_id: 1,
            material_generation: 2,
            route_hash: 3,
            face_size: 256,
        };
        let b = SkyboxIblKey::Analytic {
            material_asset_id: 1,
            material_generation: 2,
            route_hash: 3,
            face_size: 128,
        };
        let c = SkyboxIblKey::Analytic {
            material_asset_id: 1,
            material_generation: 9,
            route_hash: 3,
            face_size: 256,
        };
        assert_ne!(a.source_hash(), b.source_hash());
        assert_ne!(a.source_hash(), c.source_hash());
    }

    /// Cubemap key invariants: residency growth and face size resize both invalidate.
    #[test]
    fn cubemap_key_invalidates_on_residency_or_face_change() {
        let a = SkyboxIblKey::Cubemap {
            asset_id: 7,
            mip_levels_resident: 1,
            content_generation: 1,
            storage_v_inverted: false,
            face_size: 256,
        };
        let b = SkyboxIblKey::Cubemap {
            asset_id: 7,
            mip_levels_resident: 4,
            content_generation: 1,
            storage_v_inverted: false,
            face_size: 256,
        };
        let c = SkyboxIblKey::Cubemap {
            asset_id: 7,
            mip_levels_resident: 1,
            content_generation: 1,
            storage_v_inverted: false,
            face_size: 128,
        };
        assert_ne!(a, b);
        assert_ne!(a, c);
        let d = SkyboxIblKey::Cubemap {
            asset_id: 7,
            mip_levels_resident: 1,
            content_generation: 2,
            storage_v_inverted: false,
            face_size: 256,
        };
        assert_ne!(a, d);
    }

    /// Equirect key invariants: FOV / ST hash inputs invalidate the bake.
    #[test]
    fn equirect_key_invalidates_on_param_changes() {
        let base = SkyboxIblKey::Equirect {
            asset_id: 9,
            mip_levels_resident: 3,
            content_generation: 1,
            storage_v_inverted: false,
            fov_hash: hash_float4(&[1.0, 1.0, 0.0, 0.0]),
            st_hash: hash_float4(&[1.0, 1.0, 0.0, 0.0]),
            face_size: 256,
        };
        let altered_fov = SkyboxIblKey::Equirect {
            asset_id: 9,
            mip_levels_resident: 3,
            content_generation: 1,
            storage_v_inverted: false,
            fov_hash: hash_float4(&[2.0, 1.0, 0.0, 0.0]),
            st_hash: hash_float4(&[1.0, 1.0, 0.0, 0.0]),
            face_size: 256,
        };
        let altered_st = SkyboxIblKey::Equirect {
            asset_id: 9,
            mip_levels_resident: 3,
            content_generation: 1,
            storage_v_inverted: false,
            fov_hash: hash_float4(&[1.0, 1.0, 0.0, 0.0]),
            st_hash: hash_float4(&[2.0, 1.0, 0.0, 0.0]),
            face_size: 256,
        };
        assert_ne!(base, altered_fov);
        assert_ne!(base, altered_st);
        let altered_content = SkyboxIblKey::Equirect {
            asset_id: 9,
            mip_levels_resident: 3,
            content_generation: 2,
            storage_v_inverted: false,
            fov_hash: hash_float4(&[1.0, 1.0, 0.0, 0.0]),
            st_hash: hash_float4(&[1.0, 1.0, 0.0, 0.0]),
            face_size: 256,
        };
        assert_ne!(base, altered_content);
    }
}
