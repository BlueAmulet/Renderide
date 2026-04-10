//! `@group(1)` bind groups for embedded raster materials (WGSL targets shipped with the renderer).
//!
//! Layouts and uniform packing come from [`crate::materials::reflect_raster_material_wgsl`] (naga).
//! WGSL identifiers in `@group(1)` match Unity [`MaterialPropertyBlock`](https://docs.unity3d.com/ScriptReference/MaterialPropertyBlock.html)
//! names; [`crate::assets::material::PropertyIdRegistry`] resolves them to batch property ids.
//!
//! **UI text (`_TextMode`, `_RectClip`):** When a reflected uniform field is named `_TextMode` or `_RectClip`,
//! packing uses explicit `set_float` when present; otherwise keyword-style floats (`MSDF`, `RASTER`, `SDF`,
//! `RECTCLIP`, case variants) are interpreted the same way as legacy FrooxEngine/Unity keyword bindings—without
//! hard-coding a particular shader stem in the draw pass.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::assets::material::{
    MaterialPropertyLookupIds, MaterialPropertyStore, MaterialPropertyValue, PropertyIdRegistry,
};
use crate::assets::texture::texture2d_asset_id_from_packed;
use crate::embedded_shaders;
use crate::materials::{
    reflect_raster_material_wgsl, ReflectedRasterLayout, ReflectedUniformField,
    ReflectedUniformScalarKind,
};
use crate::resources::{Texture2dSamplerState, TexturePool};

/// GPU resources shared by embedded material bind groups (layouts, default texture, sampler).
pub struct EmbeddedMaterialBindResources {
    device: Arc<wgpu::Device>,
    white_texture: Arc<wgpu::Texture>,
    white_texture_view: Arc<wgpu::TextureView>,
    default_sampler: Arc<wgpu::Sampler>,
    property_registry: Arc<PropertyIdRegistry>,
    shared_keyword_ids: Arc<EmbeddedSharedKeywordIds>,
    stem_cache: RefCell<HashMap<String, Arc<StemMaterialLayout>>>,
    bind_cache: RefCell<HashMap<MaterialBindCacheKey, Arc<wgpu::BindGroup>>>,
    uniform_cache: RefCell<HashMap<MaterialUniformCacheKey, CachedUniformEntry>>,
}

/// Cached reflection for one composed stem.
struct StemMaterialLayout {
    bind_group_layout: wgpu::BindGroupLayout,
    reflected: ReflectedRasterLayout,
    ids: Arc<StemEmbeddedPropertyIds>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
struct MaterialUniformCacheKey {
    stem_hash: u64,
    material_asset_id: i32,
    property_block_slot0: Option<i32>,
    texture_2d_asset_id: i32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
struct MaterialBindCacheKey {
    stem_hash: u64,
    material_asset_id: i32,
    property_block_slot0: Option<i32>,
    texture_bind_signature: u64,
}

/// Cached GPU uniform buffer and last [`crate::assets::material::MaterialPropertyStore::mutation_generation`] uploaded to it.
struct CachedUniformEntry {
    buffer: Arc<wgpu::Buffer>,
    last_written_generation: u64,
}

/// Pre-interned property ids for static keyword names (`MSDF`, `_Flags`, texture probe lists), shared by all embedded stems.
struct EmbeddedSharedKeywordIds {
    text_mode: i32,
    rect_clip: i32,
    msdf: i32,
    sdf: i32,
    raster: i32,
    msdf_lower: i32,
    sdf_lower: i32,
    raster_lower: i32,
    rectclip: i32,
    rectclip_lower: i32,
    flags: i32,
    offset_texture: i32,
    offset_texture_alt: i32,
    mask_texture_mul: i32,
    mask_texture_mul_alt: i32,
    mask_texture_clip: i32,
    mask_texture_clip_alt: i32,
    mul_rgb_by_alpha: i32,
    mul_rgb_by_alpha_alt: i32,
    mul_alpha_intensity: i32,
    mul_alpha_intensity_alt: i32,
    lerp_tex: i32,
    main_tex: i32,
    main_tex1: i32,
    emission_map: i32,
    emission_map1: i32,
    normal_map: i32,
    normal_map1: i32,
    bump_map: i32,
    specular_map: i32,
    specular_map1: i32,
    spec_gloss_map: i32,
    metallic_map: i32,
    metallic_map1: i32,
    metallic_gloss_map: i32,
    occlusion: i32,
    occlusion1: i32,
    occlusion_map: i32,
}

impl EmbeddedSharedKeywordIds {
    fn new(registry: &PropertyIdRegistry) -> Self {
        Self {
            text_mode: registry.intern("_TextMode"),
            rect_clip: registry.intern("_RectClip"),
            msdf: registry.intern("MSDF"),
            sdf: registry.intern("SDF"),
            raster: registry.intern("RASTER"),
            msdf_lower: registry.intern("msdf"),
            sdf_lower: registry.intern("sdf"),
            raster_lower: registry.intern("raster"),
            rectclip: registry.intern("RECTCLIP"),
            rectclip_lower: registry.intern("rectclip"),
            flags: registry.intern("_Flags"),
            offset_texture: registry.intern("_OFFSET_TEXTURE"),
            offset_texture_alt: registry.intern("_OffsetTexture"),
            mask_texture_mul: registry.intern("_MASK_TEXTURE_MUL"),
            mask_texture_mul_alt: registry.intern("_MaskTextureMul"),
            mask_texture_clip: registry.intern("_MASK_TEXTURE_CLIP"),
            mask_texture_clip_alt: registry.intern("_MaskTextureClip"),
            mul_rgb_by_alpha: registry.intern("_MUL_RGB_BY_ALPHA"),
            mul_rgb_by_alpha_alt: registry.intern("_MulRgbByAlpha"),
            mul_alpha_intensity: registry.intern("_MUL_ALPHA_INTENSITY"),
            mul_alpha_intensity_alt: registry.intern("_MulAlphaIntensity"),
            lerp_tex: registry.intern("_LerpTex"),
            main_tex: registry.intern("_MainTex"),
            main_tex1: registry.intern("_MainTex1"),
            emission_map: registry.intern("_EmissionMap"),
            emission_map1: registry.intern("_EmissionMap1"),
            normal_map: registry.intern("_NormalMap"),
            normal_map1: registry.intern("_NormalMap1"),
            bump_map: registry.intern("_BumpMap"),
            specular_map: registry.intern("_SpecularMap"),
            specular_map1: registry.intern("_SpecularMap1"),
            spec_gloss_map: registry.intern("_SpecGlossMap"),
            metallic_map: registry.intern("_MetallicMap"),
            metallic_map1: registry.intern("_MetallicMap1"),
            metallic_gloss_map: registry.intern("_MetallicGlossMap"),
            occlusion: registry.intern("_Occlusion"),
            occlusion1: registry.intern("_Occlusion1"),
            occlusion_map: registry.intern("_OcclusionMap"),
        }
    }
}

/// Per-stem stable property ids from WGSL reflection (uniform members and `@group(1)` texture globals), built once when the stem layout loads.
struct StemEmbeddedPropertyIds {
    shared: Arc<EmbeddedSharedKeywordIds>,
    uniform_field_ids: HashMap<String, i32>,
    texture_binding_to_property_id: HashMap<u32, i32>,
    keyword_field_probe_ids: HashMap<String, [i32; 3]>,
}

impl StemEmbeddedPropertyIds {
    fn build(
        shared: Arc<EmbeddedSharedKeywordIds>,
        registry: &PropertyIdRegistry,
        reflected: &ReflectedRasterLayout,
    ) -> Self {
        let mut uniform_field_ids = HashMap::new();
        let mut keyword_field_probe_ids = HashMap::new();
        if let Some(u) = reflected.material_uniform.as_ref() {
            for field_name in u.fields.keys() {
                let pid = registry.intern(field_name);
                uniform_field_ids.insert(field_name.clone(), pid);
                let stripped = field_name.strip_prefix('_').unwrap_or(field_name);
                let lowercase = stripped.to_ascii_lowercase();
                let pid_strip = registry.intern(stripped);
                let pid_lower = registry.intern(lowercase.as_str());
                keyword_field_probe_ids.insert(field_name.clone(), [pid, pid_strip, pid_lower]);
            }
        }

        let mut texture_binding_to_property_id = HashMap::new();
        for entry in &reflected.material_entries {
            if matches!(entry.ty, wgpu::BindingType::Texture { .. }) {
                if let Some(name) = reflected.material_group1_names.get(&entry.binding) {
                    texture_binding_to_property_id
                        .insert(entry.binding, registry.intern(name.as_str()));
                }
            }
        }

        Self {
            shared,
            uniform_field_ids,
            texture_binding_to_property_id,
            keyword_field_probe_ids,
        }
    }
}

#[cfg(test)]
impl StemEmbeddedPropertyIds {
    /// Shared keyword ids only (no per-stem uniform/texture reflection); for unit tests.
    pub fn minimal_for_tests(registry: &PropertyIdRegistry) -> Self {
        Self {
            shared: Arc::new(EmbeddedSharedKeywordIds::new(registry)),
            uniform_field_ids: HashMap::new(),
            texture_binding_to_property_id: HashMap::new(),
            keyword_field_probe_ids: HashMap::new(),
        }
    }
}

fn stem_hash(stem: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    stem.hash(&mut h);
    h.finish()
}

impl EmbeddedMaterialBindResources {
    /// Builds layouts and placeholder texture.
    pub fn new(
        device: Arc<wgpu::Device>,
        property_registry: Arc<PropertyIdRegistry>,
    ) -> Result<Self, String> {
        let white_texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("embedded_default_white"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        }));
        let white_texture_view =
            Arc::new(white_texture.create_view(&wgpu::TextureViewDescriptor::default()));

        let default_sampler = Arc::new(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("embedded_default_sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        }));

        let shared_keyword_ids =
            Arc::new(EmbeddedSharedKeywordIds::new(property_registry.as_ref()));

        Ok(Self {
            device,
            white_texture,
            white_texture_view,
            default_sampler,
            property_registry,
            shared_keyword_ids,
            stem_cache: RefCell::new(HashMap::new()),
            bind_cache: RefCell::new(HashMap::new()),
            uniform_cache: RefCell::new(HashMap::new()),
        })
    }

    /// Uploads white texel into the placeholder texture (call once after creation with queue).
    pub fn write_default_white(&self, queue: &wgpu::Queue) {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: self.white_texture.as_ref(),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[255u8, 255, 255, 255],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Returns or builds a `@group(1)` bind group for the composed embedded `stem` (e.g. `unlit_default`).
    pub fn embedded_material_bind_group(
        &self,
        stem: &str,
        queue: &wgpu::Queue,
        store: &MaterialPropertyStore,
        texture_pool: &TexturePool,
        lookup: MaterialPropertyLookupIds,
    ) -> Result<Arc<wgpu::BindGroup>, String> {
        let layout = self.stem_layout(stem)?;
        let sh = stem_hash(stem);

        let texture_2d_asset_id =
            primary_texture_2d_asset_id(&layout.reflected, layout.ids.as_ref(), store, lookup);
        let texture_bind_signature = texture_bind_signature(
            &layout.reflected,
            layout.ids.as_ref(),
            store,
            lookup,
            texture_pool,
            texture_2d_asset_id,
        );

        let uniform_key = MaterialUniformCacheKey {
            stem_hash: sh,
            material_asset_id: lookup.material_asset_id,
            property_block_slot0: lookup.mesh_property_block_slot0,
            texture_2d_asset_id,
        };
        let bind_key = MaterialBindCacheKey {
            stem_hash: sh,
            material_asset_id: lookup.material_asset_id,
            property_block_slot0: lookup.mesh_property_block_slot0,
            texture_bind_signature,
        };

        let mutation_gen = store.mutation_generation(lookup);

        let uniform_buf = {
            let mut uniform_cache = self.uniform_cache.borrow_mut();
            if let Some(entry) = uniform_cache.get(&uniform_key) {
                if entry.last_written_generation == mutation_gen {
                    entry.buffer.clone()
                } else {
                    let uniform_bytes = self
                        .build_uniform_bytes(
                            &layout.reflected,
                            layout.ids.as_ref(),
                            store,
                            lookup,
                            texture_2d_asset_id,
                        )
                        .ok_or_else(|| {
                            format!(
                                "stem {stem}: uniform block missing (shader has no material uniform)"
                            )
                        })?;
                    let e = uniform_cache.get_mut(&uniform_key).expect("entry exists");
                    queue.write_buffer(e.buffer.as_ref(), 0, &uniform_bytes);
                    e.last_written_generation = mutation_gen;
                    e.buffer.clone()
                }
            } else {
                let uniform_bytes = self
                    .build_uniform_bytes(
                        &layout.reflected,
                        layout.ids.as_ref(),
                        store,
                        lookup,
                        texture_2d_asset_id,
                    )
                    .ok_or_else(|| {
                        format!(
                            "stem {stem}: uniform block missing (shader has no material uniform)"
                        )
                    })?;
                let buf = Arc::new(self.device.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some("embedded_material_uniform"),
                        contents: &uniform_bytes,
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    },
                ));
                uniform_cache.insert(
                    uniform_key,
                    CachedUniformEntry {
                        buffer: buf.clone(),
                        last_written_generation: mutation_gen,
                    },
                );
                buf
            }
        };

        let mut cache = self.bind_cache.borrow_mut();
        if let Some(bg) = cache.get(&bind_key) {
            return Ok(bg.clone());
        }

        let mut keepalive_views: Vec<Arc<wgpu::TextureView>> = Vec::new();
        let mut keepalive_samplers: Vec<Arc<wgpu::Sampler>> = Vec::new();
        for entry in &layout.reflected.material_entries {
            let b = entry.binding;
            match entry.ty {
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    ..
                } => {}
                wgpu::BindingType::Texture { .. } => {
                    let host_name = layout
                        .reflected
                        .material_group1_names
                        .get(&b)
                        .map(String::as_str)
                        .ok_or_else(|| {
                            format!("reflection: no WGSL name for texture @binding({b})")
                        })?;
                    let tex_pid = layout
                        .ids
                        .texture_binding_to_property_id
                        .get(&b)
                        .copied()
                        .ok_or_else(|| {
                            format!("reflection: missing property id for texture @binding({b})")
                        })?;
                    let tex_view = self
                        .resolve_texture_view_for_host(
                            host_name,
                            tex_pid,
                            texture_2d_asset_id,
                            texture_pool,
                            store,
                            lookup,
                        )
                        .unwrap_or_else(|| self.white_texture_view.clone());
                    keepalive_views.push(tex_view);
                }
                wgpu::BindingType::Sampler(_) => {
                    let tex_binding = sampler_pairs_texture_binding(b);
                    let host_name = layout
                        .reflected
                        .material_group1_names
                        .get(&tex_binding)
                        .map(String::as_str)
                        .ok_or_else(|| {
                            format!("reflection: no texture global for sampler @binding({b})")
                        })?;
                    let tex_pid = layout
                        .ids
                        .texture_binding_to_property_id
                        .get(&tex_binding)
                        .copied()
                        .ok_or_else(|| {
                            format!(
                                "reflection: missing property id for texture @binding({tex_binding})"
                            )
                        })?;
                    let sampler = self.resolve_sampler_for_host(
                        host_name,
                        tex_pid,
                        texture_2d_asset_id,
                        texture_pool,
                        store,
                        lookup,
                    );
                    keepalive_samplers.push(sampler);
                }
                _ => {
                    return Err(format!("unsupported binding type for @binding({b})"));
                }
            }
        }

        let mut view_i = 0usize;
        let mut samp_i = 0usize;
        let mut entries: Vec<wgpu::BindGroupEntry<'_>> =
            Vec::with_capacity(layout.reflected.material_entries.len());
        for entry in &layout.reflected.material_entries {
            let b = entry.binding;
            match entry.ty {
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    ..
                } => {
                    entries.push(wgpu::BindGroupEntry {
                        binding: b,
                        resource: uniform_buf.as_entire_binding(),
                    });
                }
                wgpu::BindingType::Texture { .. } => {
                    let tv = keepalive_views
                        .get(view_i)
                        .ok_or_else(|| format!("internal: texture view index {view_i}"))?;
                    view_i += 1;
                    entries.push(wgpu::BindGroupEntry {
                        binding: b,
                        resource: wgpu::BindingResource::TextureView(tv.as_ref()),
                    });
                }
                wgpu::BindingType::Sampler(_) => {
                    let s = keepalive_samplers
                        .get(samp_i)
                        .ok_or_else(|| format!("internal: sampler index {samp_i}"))?;
                    samp_i += 1;
                    entries.push(wgpu::BindGroupEntry {
                        binding: b,
                        resource: wgpu::BindingResource::Sampler(s.as_ref()),
                    });
                }
                _ => {
                    return Err(format!("unsupported binding type for @binding({b})"));
                }
            }
        }

        let bind_group = Arc::new(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("embedded_material_bind"),
            layout: &layout.bind_group_layout,
            entries: &entries,
        }));
        cache.insert(bind_key, bind_group.clone());
        Ok(bind_group)
    }

    fn stem_layout(&self, stem: &str) -> Result<Arc<StemMaterialLayout>, String> {
        let mut cache = self.stem_cache.borrow_mut();
        if let Some(s) = cache.get(stem) {
            return Ok(s.clone());
        }

        let wgsl = embedded_shaders::embedded_target_wgsl(stem)
            .ok_or_else(|| format!("embedded WGSL missing for stem {stem}"))?;
        let reflected =
            reflect_raster_material_wgsl(wgsl).map_err(|e| format!("reflect {stem}: {e}"))?;

        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("embedded_raster_material"),
                    entries: &reflected.material_entries,
                });

        let ids = Arc::new(StemEmbeddedPropertyIds::build(
            Arc::clone(&self.shared_keyword_ids),
            self.property_registry.as_ref(),
            &reflected,
        ));

        let layout = Arc::new(StemMaterialLayout {
            bind_group_layout,
            reflected,
            ids,
        });
        cache.insert(stem.to_string(), layout.clone());
        Ok(layout)
    }

    fn build_uniform_bytes(
        &self,
        reflected: &ReflectedRasterLayout,
        ids: &StemEmbeddedPropertyIds,
        store: &MaterialPropertyStore,
        lookup: MaterialPropertyLookupIds,
        texture_2d_for_key: i32,
    ) -> Option<Vec<u8>> {
        let u = reflected.material_uniform.as_ref()?;
        let mut buf = vec![0u8; u.total_size as usize];

        // `_Cutoff` is tracked when present so `flags` packing can mirror Unity alpha-test heuristics for any
        // material whose WGSL declares these identifiers (reflection-driven, not per-shader names in Rust).
        let mut cutoff = 0.5f32;
        for (field_name, field) in &u.fields {
            let pid = *ids.uniform_field_ids.get(field_name)?;
            match field.kind {
                ReflectedUniformScalarKind::Vec4 => {
                    let default = default_vec4_for_field(field_name);
                    let mut v = default;
                    if let Some(MaterialPropertyValue::Float4(c)) = store.get_merged(lookup, pid) {
                        v = *c;
                    }
                    write_f32x4_at(&mut buf, field, &v);
                }
                ReflectedUniformScalarKind::F32 => {
                    let v = if field_name == "_TextMode" {
                        packed_text_mode_f32(store, lookup, ids.shared.as_ref())
                    } else if field_name == "_RectClip" {
                        packed_rect_clip_f32(store, lookup, ids.shared.as_ref())
                    } else if let Some(MaterialPropertyValue::Float(f)) =
                        store.get_merged(lookup, pid)
                    {
                        *f
                    } else {
                        default_f32_for_field(field_name, store, lookup, ids)
                    };
                    if field_name == "_Cutoff" {
                        cutoff = v;
                    }
                    write_f32_at(&mut buf, field, v);
                }
                ReflectedUniformScalarKind::U32 => {
                    if field_name == "flags" {
                        let flags = pack_flags_u32(
                            field_name,
                            store,
                            lookup,
                            ids.shared.as_ref(),
                            texture_2d_for_key,
                            cutoff,
                        );
                        write_u32_at(&mut buf, field, flags);
                    }
                }
                ReflectedUniformScalarKind::Unsupported => {}
            }
        }

        Some(buf)
    }

    fn resolve_texture_view_for_host(
        &self,
        host_name: &str,
        texture_property_id: i32,
        primary_texture_2d: i32,
        texture_pool: &TexturePool,
        store: &MaterialPropertyStore,
        lookup: MaterialPropertyLookupIds,
    ) -> Option<Arc<wgpu::TextureView>> {
        let id = resolved_texture_asset_id_for_host(
            host_name,
            texture_property_id,
            primary_texture_2d,
            store,
            lookup,
        );
        self.resolve_texture_view(texture_pool, id)
    }

    fn resolve_sampler_for_host(
        &self,
        host_name: &str,
        texture_property_id: i32,
        primary_texture_2d: i32,
        texture_pool: &TexturePool,
        store: &MaterialPropertyStore,
        lookup: MaterialPropertyLookupIds,
    ) -> Arc<wgpu::Sampler> {
        let tid = resolved_texture_asset_id_for_host(
            host_name,
            texture_property_id,
            primary_texture_2d,
            store,
            lookup,
        );
        self.resolve_sampler(texture_pool, tid)
    }

    fn resolve_texture_view(
        &self,
        texture_pool: &TexturePool,
        texture_asset_id: i32,
    ) -> Option<Arc<wgpu::TextureView>> {
        if texture_asset_id < 0 {
            return None;
        }
        texture_pool
            .get_texture(texture_asset_id)
            .filter(|t| t.mip_levels_resident > 0)
            .map(|t| t.view.clone())
    }

    fn resolve_sampler(
        &self,
        texture_pool: &TexturePool,
        texture_asset_id: i32,
    ) -> Arc<wgpu::Sampler> {
        if texture_asset_id < 0 {
            return self.default_sampler.clone();
        }
        let Some(tex) = texture_pool.get_texture(texture_asset_id) else {
            return self.default_sampler.clone();
        };
        Arc::new(sampler_from_state(&self.device, &tex.sampler))
    }
}

fn sampler_pairs_texture_binding(sampler_binding: u32) -> u32 {
    sampler_binding.saturating_sub(1)
}

/// True when the host material has a `set_float` for `property_id` with value ≥ 0.5 (Unity shader keyword pattern).
fn keyword_float_enabled_by_pid(
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    property_id: i32,
) -> bool {
    matches!(
        store.get_merged(lookup, property_id),
        Some(MaterialPropertyValue::Float(f)) if *f >= 0.5
    )
}

fn keyword_float_enabled_any_pids(
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    pids: &[i32; 3],
) -> bool {
    pids.iter()
        .any(|&pid| keyword_float_enabled_by_pid(store, lookup, pid))
}

fn default_vec4_for_field(field_name: &str) -> [f32; 4] {
    if field_name.ends_with("_ST") {
        return [1.0, 1.0, 0.0, 0.0];
    }
    match field_name {
        "_EmissionColor" | "_EmissionColor1" | "_IntersectEmissionColor" | "_OutsideColor" => {
            [0.0, 0.0, 0.0, 0.0]
        }
        "_SpecularColor" | "_SpecularColor1" => [1.0, 1.0, 1.0, 0.5],
        _ => [1.0, 1.0, 1.0, 1.0],
    }
}

fn is_keyword_like_field(field_name: &str) -> bool {
    let stripped = field_name.strip_prefix('_').unwrap_or(field_name);
    !stripped.is_empty()
        && stripped
            .chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
}

fn texture_property_asset_id_by_pid(
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    property_id: i32,
) -> i32 {
    match store.get_merged(lookup, property_id) {
        Some(MaterialPropertyValue::Texture(packed)) => {
            texture2d_asset_id_from_packed(*packed).unwrap_or(-1)
        }
        _ => -1,
    }
}

fn texture_property_present_pids(
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    pids: &[i32],
) -> bool {
    pids.iter()
        .any(|&pid| texture_property_asset_id_by_pid(store, lookup, pid) >= 0)
}

fn inferred_keyword_float_f32(
    field_name: &str,
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    ids: &StemEmbeddedPropertyIds,
) -> Option<f32> {
    if let Some(probes) = ids.keyword_field_probe_ids.get(field_name) {
        if keyword_float_enabled_any_pids(store, lookup, probes) {
            return Some(1.0);
        }
    }

    let kw = ids.shared.as_ref();
    let inferred = match field_name {
        "_LERPTEX" => texture_property_present_pids(store, lookup, &[kw.lerp_tex]),
        "_ALBEDOTEX" => texture_property_present_pids(store, lookup, &[kw.main_tex, kw.main_tex1]),
        "_EMISSIONTEX" => {
            texture_property_present_pids(store, lookup, &[kw.emission_map, kw.emission_map1])
        }
        "_NORMALMAP" => texture_property_present_pids(
            store,
            lookup,
            &[kw.normal_map, kw.normal_map1, kw.bump_map],
        ),
        "_SPECULARMAP" => texture_property_present_pids(
            store,
            lookup,
            &[kw.specular_map, kw.specular_map1, kw.spec_gloss_map],
        ),
        "_METALLICMAP" => texture_property_present_pids(
            store,
            lookup,
            &[kw.metallic_map, kw.metallic_map1, kw.metallic_gloss_map],
        ),
        "_OCCLUSION" => texture_property_present_pids(
            store,
            lookup,
            &[kw.occlusion, kw.occlusion1, kw.occlusion_map],
        ),
        _ if is_keyword_like_field(field_name) => false,
        _ => return None,
    };
    Some(if inferred { 1.0 } else { 0.0 })
}

fn default_f32_for_field(
    field_name: &str,
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    ids: &StemEmbeddedPropertyIds,
) -> f32 {
    if let Some(v) = inferred_keyword_float_f32(field_name, store, lookup, ids) {
        return v;
    }
    match field_name {
        "_Lerp" | "_Metallic" | "_Metallic1" | "_UVSec" | "_Mode" | "_OffsetFactor"
        | "_OffsetUnits" => 0.0,
        "_NormalScale"
        | "_NormalScale1"
        | "_BumpScale"
        | "_GlossMapScale"
        | "_OcclusionStrength"
        | "_DetailNormalMapScale"
        | "_Exposure"
        | "_Gamma"
        | "_ZWrite" => 1.0,
        "_SrcBlend" => 1.0,
        "_DstBlend" => 0.0,
        "_Cull" => 2.0,
        "_Cutoff" | "_AlphaClip" | "_Glossiness" | "_Glossiness1" => 0.5,
        _ => 0.5,
    }
}

/// Packs `UI_TextUnlit`-style `_TextMode`: explicit `0`/`1`/`2`, else keyword floats `MSDF` / `SDF` / `RASTER`, else `0` (MSDF default).
fn packed_text_mode_f32(
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    kw: &EmbeddedSharedKeywordIds,
) -> f32 {
    if let Some(MaterialPropertyValue::Float(f)) = store.get_merged(lookup, kw.text_mode) {
        return *f;
    }
    if keyword_float_enabled_by_pid(store, lookup, kw.msdf) {
        return 0.0;
    }
    if keyword_float_enabled_by_pid(store, lookup, kw.sdf) {
        return 2.0;
    }
    if keyword_float_enabled_by_pid(store, lookup, kw.raster) {
        return 1.0;
    }
    if keyword_float_enabled_by_pid(store, lookup, kw.msdf_lower) {
        return 0.0;
    }
    if keyword_float_enabled_by_pid(store, lookup, kw.sdf_lower) {
        return 2.0;
    }
    if keyword_float_enabled_by_pid(store, lookup, kw.raster_lower) {
        return 1.0;
    }
    0.0
}

/// Packs `_RectClip`: explicit value, else `RECTCLIP` / `rectclip` keyword floats, else `0`.
fn packed_rect_clip_f32(
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    kw: &EmbeddedSharedKeywordIds,
) -> f32 {
    if let Some(MaterialPropertyValue::Float(f)) = store.get_merged(lookup, kw.rect_clip) {
        return *f;
    }
    if keyword_float_enabled_by_pid(store, lookup, kw.rectclip) {
        return 1.0;
    }
    if keyword_float_enabled_by_pid(store, lookup, kw.rectclip_lower) {
        return 1.0;
    }
    0.0
}

fn primary_texture_2d_asset_id(
    reflected: &ReflectedRasterLayout,
    ids: &StemEmbeddedPropertyIds,
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
) -> i32 {
    for entry in &reflected.material_entries {
        if matches!(entry.ty, wgpu::BindingType::Texture { .. }) {
            let Some(pid) = ids.texture_binding_to_property_id.get(&entry.binding) else {
                continue;
            };
            if let Some(MaterialPropertyValue::Texture(packed)) = store.get_merged(lookup, *pid) {
                return texture2d_asset_id_from_packed(*packed).unwrap_or(-1);
            }
        }
    }
    -1
}

fn should_fallback_to_primary_texture(host_name: &str) -> bool {
    matches!(host_name, "_MainTex" | "_Tex" | "_TEXTURE")
}

fn resolved_texture_asset_id_for_host(
    host_name: &str,
    texture_property_id: i32,
    primary_texture_2d: i32,
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
) -> i32 {
    let tid = texture_property_asset_id_by_pid(store, lookup, texture_property_id);
    if tid >= 0 {
        return tid;
    }
    if should_fallback_to_primary_texture(host_name) {
        return primary_texture_2d;
    }
    -1
}

fn texture_bind_signature(
    reflected: &ReflectedRasterLayout,
    ids: &StemEmbeddedPropertyIds,
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    texture_pool: &TexturePool,
    primary_texture_2d: i32,
) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut h = DefaultHasher::new();
    for entry in &reflected.material_entries {
        if !matches!(entry.ty, wgpu::BindingType::Texture { .. }) {
            continue;
        }
        let Some(name) = reflected.material_group1_names.get(&entry.binding) else {
            continue;
        };
        let Some(&texture_pid) = ids.texture_binding_to_property_id.get(&entry.binding) else {
            continue;
        };
        let texture_asset_id = resolved_texture_asset_id_for_host(
            name.as_str(),
            texture_pid,
            primary_texture_2d,
            store,
            lookup,
        );
        entry.binding.hash(&mut h);
        name.hash(&mut h);
        texture_asset_id.hash(&mut h);
        texture_pool
            .get_texture(texture_asset_id)
            .is_some_and(|t| t.mip_levels_resident > 0)
            .hash(&mut h);
    }
    h.finish()
}

/// Packs `flags` from `_Flags` when present; otherwise derives bits from texture presence,
/// `_Cutoff`, and Unity `#pragma multi_compile` keyword floats.
///
/// ## Bit layout (matches `unlit.wgsl` and `ui_unlit.wgsl` documentation)
/// | Bit | Mask  | Unity keyword / heuristic |
/// |-----|-------|---------------------------|
/// | 0   | 0x01  | Texture present (`_TEXTURE` / `_Tex` or `_MainTex` set) |
/// | 1   | 0x02  | Alpha clip (`_ALPHATEST` / `_Cutoff` in (0, 1)) |
/// | 2   | 0x04  | Offset texture (`_OFFSET_TEXTURE`) |
/// | 3   | 0x08  | Mask multiply alpha (`_MASK_TEXTURE_MUL`) |
/// | 4   | 0x10  | Mask clip (`_MASK_TEXTURE_CLIP`) |
/// | 5   | 0x20  | Premultiply RGB by alpha (`_MUL_RGB_BY_ALPHA`) |
/// | 6   | 0x40  | Additive alpha-by-luminance (`_MUL_ALPHA_INTENSITY`) |
fn pack_flags_u32(
    field_name: &str,
    store: &MaterialPropertyStore,
    lookup: MaterialPropertyLookupIds,
    kw: &EmbeddedSharedKeywordIds,
    texture_2d_for_key: i32,
    cutoff: f32,
) -> u32 {
    if field_name != "flags" {
        return 0;
    }
    // Explicit `_Flags` property from host overrides all heuristics.
    if let Some(v) = store.get_merged(lookup, kw.flags) {
        match v {
            MaterialPropertyValue::Float(f) => return (*f).max(0.0) as u32,
            MaterialPropertyValue::Float4(a) => return a[0].max(0.0) as u32,
            _ => {}
        }
    }

    let mut flags = 0u32;

    // Bit 0 — texture present.
    if texture_2d_for_key >= 0 {
        flags |= 0x01;
    }
    // Bit 1 — alpha clip (cutoff between 0 and 1 exclusive).
    if cutoff > 0.0 && cutoff < 1.0 {
        flags |= 0x02;
    }
    // Bits 2–6 — Unity #pragma multi_compile keyword floats (>= 0.5 = enabled).
    // The host sets these as Float material properties matching the keyword name.
    if keyword_float_enabled_by_pid(store, lookup, kw.offset_texture)
        || keyword_float_enabled_by_pid(store, lookup, kw.offset_texture_alt)
    {
        flags |= 0x04;
    }
    if keyword_float_enabled_by_pid(store, lookup, kw.mask_texture_mul)
        || keyword_float_enabled_by_pid(store, lookup, kw.mask_texture_mul_alt)
    {
        flags |= 0x08;
    }
    if keyword_float_enabled_by_pid(store, lookup, kw.mask_texture_clip)
        || keyword_float_enabled_by_pid(store, lookup, kw.mask_texture_clip_alt)
    {
        flags |= 0x10;
    }
    if keyword_float_enabled_by_pid(store, lookup, kw.mul_rgb_by_alpha)
        || keyword_float_enabled_by_pid(store, lookup, kw.mul_rgb_by_alpha_alt)
    {
        flags |= 0x20;
    }
    if keyword_float_enabled_by_pid(store, lookup, kw.mul_alpha_intensity)
        || keyword_float_enabled_by_pid(store, lookup, kw.mul_alpha_intensity_alt)
    {
        flags |= 0x40;
    }

    flags
}

fn write_f32_at(buf: &mut [u8], field: &ReflectedUniformField, v: f32) {
    let off = field.offset as usize;
    if off + 4 <= buf.len() && field.size >= 4 {
        buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
    }
}

fn write_f32x4_at(buf: &mut [u8], field: &ReflectedUniformField, v: &[f32; 4]) {
    let off = field.offset as usize;
    if off + 16 <= buf.len() && field.size >= 16 {
        for (i, c) in v.iter().enumerate() {
            let o = off + i * 4;
            buf[o..o + 4].copy_from_slice(&c.to_le_bytes());
        }
    }
}

fn write_u32_at(buf: &mut [u8], field: &ReflectedUniformField, v: u32) {
    let off = field.offset as usize;
    if off + 4 <= buf.len() && field.size >= 4 {
        buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
    }
}

fn sampler_from_state(device: &wgpu::Device, state: &Texture2dSamplerState) -> wgpu::Sampler {
    let address_mode_u = match state.wrap_u {
        crate::shared::TextureWrapMode::repeat => wgpu::AddressMode::Repeat,
        crate::shared::TextureWrapMode::clamp => wgpu::AddressMode::ClampToEdge,
        crate::shared::TextureWrapMode::mirror => wgpu::AddressMode::MirrorRepeat,
        crate::shared::TextureWrapMode::mirror_once => wgpu::AddressMode::ClampToEdge,
    };
    let address_mode_v = match state.wrap_v {
        crate::shared::TextureWrapMode::repeat => wgpu::AddressMode::Repeat,
        crate::shared::TextureWrapMode::clamp => wgpu::AddressMode::ClampToEdge,
        crate::shared::TextureWrapMode::mirror => wgpu::AddressMode::MirrorRepeat,
        crate::shared::TextureWrapMode::mirror_once => wgpu::AddressMode::ClampToEdge,
    };
    let (mag, min, mipmap) = match state.filter_mode {
        crate::shared::TextureFilterMode::point => (
            wgpu::FilterMode::Nearest,
            wgpu::FilterMode::Nearest,
            wgpu::MipmapFilterMode::Nearest,
        ),
        crate::shared::TextureFilterMode::bilinear => (
            wgpu::FilterMode::Linear,
            wgpu::FilterMode::Linear,
            wgpu::MipmapFilterMode::Linear,
        ),
        crate::shared::TextureFilterMode::trilinear => (
            wgpu::FilterMode::Linear,
            wgpu::FilterMode::Linear,
            wgpu::MipmapFilterMode::Linear,
        ),
        crate::shared::TextureFilterMode::anisotropic => (
            wgpu::FilterMode::Linear,
            wgpu::FilterMode::Linear,
            wgpu::MipmapFilterMode::Linear,
        ),
    };
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("embedded_texture_sampler"),
        address_mode_u,
        address_mode_v,
        address_mode_w: address_mode_u,
        mag_filter: mag,
        min_filter: min,
        mipmap_filter: mipmap,
        ..Default::default()
    })
}

#[cfg(test)]
mod text_uniform_packing_tests {
    use super::*;
    use crate::assets::material::{MaterialPropertyLookupIds, MaterialPropertyStore};

    fn lookup(material_id: i32) -> MaterialPropertyLookupIds {
        MaterialPropertyLookupIds {
            material_asset_id: material_id,
            mesh_property_block_slot0: None,
        }
    }

    #[test]
    fn packed_text_mode_defaults_to_msdf_when_empty() {
        let store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let kw = EmbeddedSharedKeywordIds::new(&reg);
        assert_eq!(
            packed_text_mode_f32(&store, lookup(1), &kw),
            0.0,
            "FrooxEngine default GlyphRenderMethod is MSDF"
        );
    }

    #[test]
    fn packed_text_mode_explicit_overrides_keywords() {
        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let kw = EmbeddedSharedKeywordIds::new(&reg);
        let pid_tm = reg.intern("_TextMode");
        let pid_msdf = reg.intern("MSDF");
        store.set_material(1, pid_msdf, MaterialPropertyValue::Float(1.0));
        store.set_material(1, pid_tm, MaterialPropertyValue::Float(1.0));
        assert_eq!(packed_text_mode_f32(&store, lookup(1), &kw), 1.0);
    }

    #[test]
    fn packed_text_mode_raster_from_keyword() {
        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let kw = EmbeddedSharedKeywordIds::new(&reg);
        let pid = reg.intern("RASTER");
        store.set_material(2, pid, MaterialPropertyValue::Float(1.0));
        assert_eq!(packed_text_mode_f32(&store, lookup(2), &kw), 1.0);
    }

    #[test]
    fn packed_rect_clip_from_rectclip_keyword() {
        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let kw = EmbeddedSharedKeywordIds::new(&reg);
        let pid = reg.intern("RECTCLIP");
        store.set_material(3, pid, MaterialPropertyValue::Float(1.0));
        assert_eq!(packed_rect_clip_f32(&store, lookup(3), &kw), 1.0);
    }

    #[test]
    fn inferred_pbs_keyword_enables_from_texture_presence() {
        let mut store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
        let pid = reg.intern("_SpecularMap");
        store.set_material(4, pid, MaterialPropertyValue::Texture(123));
        assert_eq!(
            inferred_keyword_float_f32("_SPECULARMAP", &store, lookup(4), &ids),
            Some(1.0)
        );
        assert_eq!(
            inferred_keyword_float_f32("_ALBEDOTEX", &store, lookup(4), &ids),
            Some(0.0)
        );
    }

    #[test]
    fn default_pbs_uniforms_match_unity_style_defaults() {
        let store = MaterialPropertyStore::new();
        let reg = PropertyIdRegistry::new();
        let ids = StemEmbeddedPropertyIds::minimal_for_tests(&reg);
        assert_eq!(default_f32_for_field("_Lerp", &store, lookup(5), &ids), 0.0);
        assert_eq!(
            default_f32_for_field("_NormalScale", &store, lookup(5), &ids),
            1.0
        );
        assert_eq!(default_f32_for_field("_Cull", &store, lookup(5), &ids), 2.0);
        assert_eq!(
            default_vec4_for_field("_EmissionColor"),
            [0.0, 0.0, 0.0, 0.0]
        );
        assert_eq!(
            default_vec4_for_field("_SpecularColor"),
            [1.0, 1.0, 1.0, 0.5]
        );
    }

    #[test]
    fn only_main_texture_bindings_fallback_to_primary_texture() {
        assert!(should_fallback_to_primary_texture("_MainTex"));
        assert!(!should_fallback_to_primary_texture("_MainTex1"));
        assert!(!should_fallback_to_primary_texture("_SpecularMap"));
    }
}
