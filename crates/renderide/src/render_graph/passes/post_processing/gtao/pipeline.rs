//! Cached pipelines, bind-group layouts, samplers, and per-frame uniform buffer for the GTAO
//! sub-graph (main → denoise[0..N] → apply).
//!
//! The GTAO pipeline cache is keyed by [`GtaoStage`] + output format + multiview-stereo, mirroring
//! [`super::super::bloom::pipeline::BloomPipelineCache`] so all four sub-passes share one cache
//! entry path. Each stage owns its own bind-group layout because the bindings differ:
//!
//! * **Main** consumes the scene depth + the shared `FrameGlobals` UBO + the GTAO params UBO;
//!   writes the AO term (MRT 0) and packed edges (MRT 1).
//! * **Denoise** consumes the previous AO term + the packed edges + the GTAO params UBO;
//!   writes the next AO term. Two pipelines share the same bind-group layout (intermediate vs
//!   final entry points).
//! * **Apply** consumes the chain HDR scene color + a linear sampler + the (denoised) AO term +
//!   the GTAO params UBO; writes the post-process chain output.
//!
//! One process-wide [`GtaoParamsGpu`] uniform buffer is shared across every stage and rewritten
//! once per frame from the live [`crate::config::GtaoSettings`]. Bind groups are cached by their
//! source-resource identities; the transient pool naturally orphans stale entries when textures
//! are recycled.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use bytemuck::{Pod, Zeroable};
use parking_lot::Mutex;

use crate::embedded_shaders::embedded_target_wgsl;

/// Embedded-shader stems for each stage's mono / multiview variants.
mod stems {
    pub(super) const MAIN_MONO: &str = "gtao_main_default";
    pub(super) const MAIN_MULTIVIEW: &str = "gtao_main_multiview";
    pub(super) const DENOISE_MONO: &str = "gtao_denoise_default";
    pub(super) const DENOISE_MULTIVIEW: &str = "gtao_denoise_multiview";
    pub(super) const APPLY_MONO: &str = "gtao_apply_default";
    pub(super) const APPLY_MULTIVIEW: &str = "gtao_apply_multiview";
}

/// Identifies one stage of the GTAO sub-graph for the pipeline cache key. Variants map 1:1 to a
/// shader source file (or, for the two denoise variants, to a fragment entry point inside a
/// shared denoise shader).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum GtaoStage {
    /// AO + edges generator.
    Main,
    /// Denoise pass that compounds with subsequent denoise passes (centre weight `beta / 5`).
    DenoiseIntermediate,
    /// Last denoise pass (centre weight `beta`, output scaled by `OCCLUSION_TERM_SCALE`).
    DenoiseFinal,
    /// Modulates HDR scene color by the (optionally denoised) AO term.
    Apply,
}

impl GtaoStage {
    /// Returns the embedded-shader stem for the active multiview state.
    fn stem(self, multiview_stereo: bool) -> &'static str {
        match (self, multiview_stereo) {
            (Self::Main, false) => stems::MAIN_MONO,
            (Self::Main, true) => stems::MAIN_MULTIVIEW,
            (Self::DenoiseIntermediate | Self::DenoiseFinal, false) => stems::DENOISE_MONO,
            (Self::DenoiseIntermediate | Self::DenoiseFinal, true) => stems::DENOISE_MULTIVIEW,
            (Self::Apply, false) => stems::APPLY_MONO,
            (Self::Apply, true) => stems::APPLY_MULTIVIEW,
        }
    }

    /// Fragment-shader entry point for this stage. Main and apply each carry one entry point;
    /// the denoise shader exposes two so a single source file can serve both intermediate and
    /// final passes without duplicating the kernel body.
    fn entry_point(self) -> &'static str {
        match self {
            Self::Main | Self::Apply => "fs_main",
            Self::DenoiseIntermediate => "fs_denoise_intermediate",
            Self::DenoiseFinal => "fs_denoise_final",
        }
    }

    /// Stable label for diagnostics and pipeline / bind-group naming.
    fn label(self) -> &'static str {
        match self {
            Self::Main => "gtao-main",
            Self::DenoiseIntermediate => "gtao-denoise-intermediate",
            Self::DenoiseFinal => "gtao-denoise-final",
            Self::Apply => "gtao-apply",
        }
    }
}

/// CPU mirror of the WGSL `GtaoParams` uniform (32 bytes, 16-byte aligned).
///
/// Rewritten every record from the live [`crate::config::GtaoSettings`]. Shared across all GTAO
/// stages — denoise reads only `denoise_blur_beta`, apply reads only `intensity` and
/// `albedo_multibounce`, and main reads the rest. One UBO keeps frame-uniform bandwidth low.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(super) struct GtaoParamsGpu {
    /// World-space search radius (meters).
    pub radius_world: f32,
    /// Cap on the horizon search in pixels.
    pub max_pixel_radius: f32,
    /// AO strength exponent applied by the apply pass to the (possibly denoised) visibility.
    pub intensity: f32,
    /// Horizon steps per side in the main pass.
    pub step_count: u32,
    /// Distance-falloff range as a fraction of `radius_world`; matches XeGTAO's `FalloffRange`.
    pub falloff_range: f32,
    /// Gray-albedo proxy for the multi-bounce fit (paper Eq. 10), consumed by the apply pass.
    pub albedo_multibounce: f32,
    /// XeGTAO `DenoiseBlurBeta`: bilateral kernel centre-pixel weight. Intermediate denoise
    /// passes use `beta / 5`; the final pass uses the full value.
    pub denoise_blur_beta: f32,
    /// Padding to 32-byte alignment (matches WGSL `align_pad_tail: f32`).
    pub align_pad_tail: f32,
}

impl GtaoParamsGpu {
    /// Builds the GPU mirror from live settings, applying the same lower bounds the shaders
    /// already enforce so debug HUD slider edits never produce NaNs (e.g. `step_count > 0`).
    pub(super) fn from_settings(settings: &crate::config::GtaoSettings) -> Self {
        Self {
            radius_world: settings.radius_meters.max(0.0),
            max_pixel_radius: settings.max_pixel_radius.max(1.0),
            intensity: settings.intensity.max(0.0),
            step_count: settings.step_count.max(1),
            falloff_range: settings.falloff_range.clamp(0.05, 1.0),
            albedo_multibounce: settings.albedo_multibounce.clamp(0.0, 0.99),
            denoise_blur_beta: settings.denoise_blur_beta.max(1e-4),
            align_pad_tail: 0.0,
        }
    }
}

/// Pipeline cache key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct PipelineKey {
    stage: GtaoStage,
    output_format: wgpu::TextureFormat,
    multiview_stereo: bool,
}

/// Bind-group cache key for the **main** stage. Captures the only resources that vary across
/// frames — depth and the per-view frame-uniforms buffer — plus the multiview shape (which
/// changes the bind-group layout).
#[derive(Clone, Eq, Hash, PartialEq)]
struct MainBindGroupKey {
    scene_depth_texture: wgpu::Texture,
    frame_uniforms: wgpu::Buffer,
    multiview_stereo: bool,
}

/// Bind-group cache key for the **denoise** stages. Both intermediate and final pipelines share
/// the same bind-group layout, so a single cache covers them.
#[derive(Clone, Eq, Hash, PartialEq)]
struct DenoiseBindGroupKey {
    ao_in: wgpu::Texture,
    edges: wgpu::Texture,
    multiview_stereo: bool,
}

/// Bind-group cache key for the **apply** stage.
#[derive(Clone, Eq, Hash, PartialEq)]
struct ApplyBindGroupKey {
    scene_color: wgpu::Texture,
    ao_term: wgpu::Texture,
    multiview_stereo: bool,
}

/// Process-wide GPU state shared by every GTAO pass instance.
pub(super) struct GtaoPipelineCache {
    /// Linear-clamp sampler used by the apply pass.
    sampler: OnceLock<wgpu::Sampler>,
    /// Process-wide GTAO params UBO (rewritten every record by the main pass).
    params_buffer: OnceLock<wgpu::Buffer>,
    /// Bind-group layouts keyed by `(stage_kind, multiview_stereo)` — three kinds (main /
    /// denoise / apply) × two multiview shapes = up to six layouts.
    bgl_main_mono: OnceLock<wgpu::BindGroupLayout>,
    bgl_main_multiview: OnceLock<wgpu::BindGroupLayout>,
    bgl_denoise: OnceLock<wgpu::BindGroupLayout>,
    bgl_apply: OnceLock<wgpu::BindGroupLayout>,
    /// Compiled WGSL modules, one per stage / multiview pair. Cached so subsequent pipeline
    /// requests for the same stage avoid re-parsing the embedded WGSL.
    shader_modules: Mutex<HashMap<(GtaoStage, bool), wgpu::ShaderModule>>,
    /// Pipelines keyed by `(stage, output_format, multiview_stereo)`.
    pipelines: Mutex<HashMap<PipelineKey, Arc<wgpu::RenderPipeline>>>,
    /// Bind groups for the main stage.
    main_bind_groups: Mutex<HashMap<MainBindGroupKey, wgpu::BindGroup>>,
    /// Bind groups for both denoise stages.
    denoise_bind_groups: Mutex<HashMap<DenoiseBindGroupKey, wgpu::BindGroup>>,
    /// Bind groups for the apply stage.
    apply_bind_groups: Mutex<HashMap<ApplyBindGroupKey, wgpu::BindGroup>>,
}

impl Default for GtaoPipelineCache {
    fn default() -> Self {
        Self {
            sampler: OnceLock::new(),
            params_buffer: OnceLock::new(),
            bgl_main_mono: OnceLock::new(),
            bgl_main_multiview: OnceLock::new(),
            bgl_denoise: OnceLock::new(),
            bgl_apply: OnceLock::new(),
            shader_modules: Mutex::new(HashMap::new()),
            pipelines: Mutex::new(HashMap::new()),
            main_bind_groups: Mutex::new(HashMap::new()),
            denoise_bind_groups: Mutex::new(HashMap::new()),
            apply_bind_groups: Mutex::new(HashMap::new()),
        }
    }
}

impl GtaoPipelineCache {
    /// Linear-clamp sampler used to read the HDR scene color and the AO term in the apply pass.
    pub(super) fn sampler(&self, device: &wgpu::Device) -> &wgpu::Sampler {
        self.sampler.get_or_init(|| {
            device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("gtao"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                ..Default::default()
            })
        })
    }

    /// Process-wide `GtaoParams` uniform buffer. Created on first access.
    pub(super) fn params_buffer(&self, device: &wgpu::Device) -> &wgpu::Buffer {
        self.params_buffer.get_or_init(|| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gtao-params"),
                size: std::mem::size_of::<GtaoParamsGpu>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        })
    }

    /// Bind-group layout for the main pass (depth + frame UBO + GTAO params UBO).
    pub(super) fn bind_group_layout_main(
        &self,
        device: &wgpu::Device,
        multiview_stereo: bool,
    ) -> &wgpu::BindGroupLayout {
        let slot = if multiview_stereo {
            &self.bgl_main_multiview
        } else {
            &self.bgl_main_mono
        };
        slot.get_or_init(|| {
            let depth_view_dim = if multiview_stereo {
                wgpu::TextureViewDimension::D2Array
            } else {
                wgpu::TextureViewDimension::D2
            };
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(if multiview_stereo {
                    "gtao-main-multiview"
                } else {
                    "gtao-main-mono"
                }),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: depth_view_dim,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            })
        })
    }

    /// Bind-group layout for the denoise stages (AO term + edges + GTAO params UBO).
    pub(super) fn bind_group_layout_denoise(
        &self,
        device: &wgpu::Device,
    ) -> &wgpu::BindGroupLayout {
        self.bgl_denoise.get_or_init(|| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gtao-denoise"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            })
        })
    }

    /// Bind-group layout for the apply pass (HDR scene + sampler + AO term + GTAO params UBO).
    pub(super) fn bind_group_layout_apply(&self, device: &wgpu::Device) -> &wgpu::BindGroupLayout {
        self.bgl_apply.get_or_init(|| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gtao-apply"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            })
        })
    }

    /// Returns the cached shader module for `stage`, building it on first request.
    fn shader_module(
        &self,
        device: &wgpu::Device,
        stage: GtaoStage,
        multiview_stereo: bool,
    ) -> wgpu::ShaderModule {
        let mut guard = self.shader_modules.lock();
        if let Some(module) = guard.get(&(stage, multiview_stereo)) {
            return module.clone();
        }
        let stem = stage.stem(multiview_stereo);
        #[expect(
            clippy::expect_used,
            reason = "embedded shader is required; absence is a build script regression"
        )]
        let source = embedded_target_wgsl(stem)
            .expect("gtao: embedded shader missing (build script regression)");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(stem),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        guard.insert((stage, multiview_stereo), module.clone());
        module
    }

    /// Returns the per-stage bind-group layout (delegates to one of the three layout helpers).
    fn bind_group_layout_for_stage(
        &self,
        device: &wgpu::Device,
        stage: GtaoStage,
        multiview_stereo: bool,
    ) -> &wgpu::BindGroupLayout {
        match stage {
            GtaoStage::Main => self.bind_group_layout_main(device, multiview_stereo),
            GtaoStage::DenoiseIntermediate | GtaoStage::DenoiseFinal => {
                self.bind_group_layout_denoise(device)
            }
            GtaoStage::Apply => self.bind_group_layout_apply(device),
        }
    }

    /// Returns or builds a render pipeline for the given stage and target format.
    pub(super) fn pipeline(
        &self,
        device: &wgpu::Device,
        stage: GtaoStage,
        output_format: wgpu::TextureFormat,
        multiview_stereo: bool,
    ) -> Arc<wgpu::RenderPipeline> {
        let key = PipelineKey {
            stage,
            output_format,
            multiview_stereo,
        };
        {
            let guard = self.pipelines.lock();
            if let Some(p) = guard.get(&key) {
                return Arc::clone(p);
            }
        }
        logger::debug!(
            "gtao: building pipeline (stage = {:?}, dst format = {:?}, multiview = {})",
            stage,
            output_format,
            multiview_stereo
        );
        let shader = self.shader_module(device, stage, multiview_stereo);
        let layout = self.bind_group_layout_for_stage(device, stage, multiview_stereo);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(stage.label()),
            bind_group_layouts: &[Some(layout)],
            immediate_size: 0,
        });
        let targets = pipeline_targets(stage, output_format);
        let pipeline = Arc::new(
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(stage.label()),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(stage.entry_point()),
                    compilation_options: Default::default(),
                    targets: &targets,
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: Default::default(),
                multiview_mask: multiview_stereo
                    .then(|| std::num::NonZeroU32::new(3))
                    .flatten(),
                cache: None,
            }),
        );
        let mut guard = self.pipelines.lock();
        if let Some(existing) = guard.get(&key) {
            return Arc::clone(existing);
        }
        guard.insert(key, Arc::clone(&pipeline));
        pipeline
    }

    /// Builds (or fetches) the **main** pass bind group.
    pub(super) fn bind_group_main(
        &self,
        device: &wgpu::Device,
        multiview_stereo: bool,
        scene_depth_texture: &wgpu::Texture,
        frame_uniforms: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let key = MainBindGroupKey {
            scene_depth_texture: scene_depth_texture.clone(),
            frame_uniforms: frame_uniforms.clone(),
            multiview_stereo,
        };
        {
            let guard = self.main_bind_groups.lock();
            if let Some(bg) = guard.get(&key) {
                return bg.clone();
            }
        }
        let (depth_dim, depth_layer_count) = if multiview_stereo {
            (wgpu::TextureViewDimension::D2Array, Some(2))
        } else {
            (wgpu::TextureViewDimension::D2, Some(1))
        };
        let scene_depth_view = scene_depth_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("gtao_main_scene_depth"),
            aspect: wgpu::TextureAspect::DepthOnly,
            dimension: Some(depth_dim),
            array_layer_count: depth_layer_count,
            ..Default::default()
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gtao-main"),
            layout: self.bind_group_layout_main(device, multiview_stereo),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&scene_depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: frame_uniforms.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.params_buffer(device).as_entire_binding(),
                },
            ],
        });
        let mut guard = self.main_bind_groups.lock();
        if let Some(existing) = guard.get(&key) {
            return existing.clone();
        }
        if guard.len() >= MAX_CACHED_BIND_GROUPS {
            guard.clear();
        }
        guard.insert(key, bg.clone());
        bg
    }

    /// Builds (or fetches) a **denoise** bind group.
    pub(super) fn bind_group_denoise(
        &self,
        device: &wgpu::Device,
        multiview_stereo: bool,
        ao_in: &wgpu::Texture,
        edges: &wgpu::Texture,
    ) -> wgpu::BindGroup {
        let key = DenoiseBindGroupKey {
            ao_in: ao_in.clone(),
            edges: edges.clone(),
            multiview_stereo,
        };
        {
            let guard = self.denoise_bind_groups.lock();
            if let Some(bg) = guard.get(&key) {
                return bg.clone();
            }
        }
        let array_layer_count = layer_count(ao_in, multiview_stereo);
        let ao_view = ao_in.create_view(&wgpu::TextureViewDescriptor {
            label: Some("gtao_denoise_ao_in"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            array_layer_count: Some(array_layer_count),
            ..Default::default()
        });
        let edges_layers = layer_count(edges, multiview_stereo);
        let edges_view = edges.create_view(&wgpu::TextureViewDescriptor {
            label: Some("gtao_denoise_edges"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            array_layer_count: Some(edges_layers),
            ..Default::default()
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gtao-denoise"),
            layout: self.bind_group_layout_denoise(device),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&ao_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&edges_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.params_buffer(device).as_entire_binding(),
                },
            ],
        });
        let mut guard = self.denoise_bind_groups.lock();
        if let Some(existing) = guard.get(&key) {
            return existing.clone();
        }
        if guard.len() >= MAX_CACHED_BIND_GROUPS {
            guard.clear();
        }
        guard.insert(key, bg.clone());
        bg
    }

    /// Builds (or fetches) the **apply** bind group.
    pub(super) fn bind_group_apply(
        &self,
        device: &wgpu::Device,
        multiview_stereo: bool,
        scene_color: &wgpu::Texture,
        ao_term: &wgpu::Texture,
    ) -> wgpu::BindGroup {
        let key = ApplyBindGroupKey {
            scene_color: scene_color.clone(),
            ao_term: ao_term.clone(),
            multiview_stereo,
        };
        {
            let guard = self.apply_bind_groups.lock();
            if let Some(bg) = guard.get(&key) {
                return bg.clone();
            }
        }
        let scene_layers = layer_count(scene_color, multiview_stereo);
        let scene_view = scene_color.create_view(&wgpu::TextureViewDescriptor {
            label: Some("gtao_apply_scene_color"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            array_layer_count: Some(scene_layers),
            ..Default::default()
        });
        let ao_layers = layer_count(ao_term, multiview_stereo);
        let ao_view = ao_term.create_view(&wgpu::TextureViewDescriptor {
            label: Some("gtao_apply_ao_term"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            array_layer_count: Some(ao_layers),
            ..Default::default()
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gtao-apply"),
            layout: self.bind_group_layout_apply(device),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&scene_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(self.sampler(device)),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&ao_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.params_buffer(device).as_entire_binding(),
                },
            ],
        });
        let mut guard = self.apply_bind_groups.lock();
        if let Some(existing) = guard.get(&key) {
            return existing.clone();
        }
        if guard.len() >= MAX_CACHED_BIND_GROUPS {
            guard.clear();
        }
        guard.insert(key, bg.clone());
        bg
    }
}

/// Builds the color-target list for `stage`. Main writes two attachments (AO + edges); the rest
/// write a single attachment.
fn pipeline_targets(
    stage: GtaoStage,
    output_format: wgpu::TextureFormat,
) -> Vec<Option<wgpu::ColorTargetState>> {
    match stage {
        GtaoStage::Main => vec![
            Some(wgpu::ColorTargetState {
                format: output_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }),
            Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::R8Unorm,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }),
        ],
        _ => vec![Some(wgpu::ColorTargetState {
            format: output_format,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        })],
    }
}

/// Resolves the array-layer count to bind for `texture` given the multiview shape.
fn layer_count(texture: &wgpu::Texture, multiview_stereo: bool) -> u32 {
    let layers = texture.size().depth_or_array_layers.max(1);
    if multiview_stereo {
        2.min(layers)
    } else {
        1
    }
}

/// Upper bound on cached GTAO bind groups before the cache is flushed.
///
/// Expected occupancy is one entry per active view (desktop / HMD / each secondary RT camera),
/// times one per bind-group flavour. Cap protects against unbounded growth when views cycle.
const MAX_CACHED_BIND_GROUPS: usize = 32;
