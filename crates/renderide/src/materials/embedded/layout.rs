//! Stem-level reflection cache for embedded raster materials: composed WGSL, [`wgpu::BindGroupLayout`],
//! and interned property ids per [`crate::materials::ReflectedRasterLayout`].
//!
//! Per-frame uniform bytes and [`wgpu::BindGroup`] instances are built in [`crate::materials::embedded::material_bind`].

use hashbrown::HashMap;
use std::sync::Arc;

use crate::embedded_shaders;
use crate::materials::host_data::PropertyIdRegistry;
use crate::materials::{ReflectedRasterLayout, reflect_raster_material_wgsl};

use super::uniform_pack::MaterialUniformValueSpaces;

/// Cached reflection and layout for one composed shader stem.
pub(crate) struct StemMaterialLayout {
    pub(crate) bind_group_layout: wgpu::BindGroupLayout,
    pub(crate) reflected: ReflectedRasterLayout,
    pub(crate) ids: Arc<StemEmbeddedPropertyIds>,
    pub(crate) uniform_value_spaces: MaterialUniformValueSpaces,
}

/// Per-stem stable property ids from WGSL reflection (uniform members and `@group(1)` texture globals), built once when the stem layout loads.
pub(crate) struct StemEmbeddedPropertyIds {
    pub(crate) uniform_field_ids: HashMap<String, i32>,
    pub(crate) texture_binding_property_ids: HashMap<u32, Arc<[i32]>>,
}

/// Returns alternate host property names for a canonical texture binding name.
///
/// Only the `_Tex` <-> `_MainTex` cross-alias is live (`UnlitMaterial` uses `_Tex`; PBS/Toon
/// materials use `_MainTex`). A host-side audit confirmed that the no-underscore forms `Texture`,
/// `MaskTexture`, `OffsetTexture` are never declared as `MaterialProperty` and thus never sent;
/// they were removed.
fn texture_property_aliases(name: &str) -> &'static [&'static str] {
    match name {
        "_Tex" => &["_MainTex"],
        "_MainTex" => &["_Tex"],
        _ => &[],
    }
}

pub(crate) use crate::materials::wgsl_reflect::identifier_names::unescape_property_name as shader_writer_unescaped_property_name;

impl StemEmbeddedPropertyIds {
    pub(crate) fn build(registry: &PropertyIdRegistry, reflected: &ReflectedRasterLayout) -> Self {
        let mut uniform_field_ids = HashMap::new();
        if let Some(u) = reflected.material_uniform.as_ref() {
            for field_name in u.fields.keys() {
                let host_field_name = shader_writer_unescaped_property_name(field_name);
                let pid = registry.intern(host_field_name);
                uniform_field_ids.insert(field_name.clone(), pid);
            }
        }

        let mut texture_binding_property_ids = HashMap::new();
        for entry in &reflected.material_entries {
            if matches!(entry.ty, wgpu::BindingType::Texture { .. })
                && let Some(name) = reflected.material_group1_names.get(&entry.binding)
            {
                let host_name = shader_writer_unescaped_property_name(name.as_str());
                let pid = registry.intern(host_name);

                let mut pids = Vec::with_capacity(1 + texture_property_aliases(host_name).len());
                pids.push(pid);
                for alias in texture_property_aliases(host_name) {
                    let alias_pid = registry.intern(alias);
                    if !pids.contains(&alias_pid) {
                        pids.push(alias_pid);
                    }
                }
                texture_binding_property_ids.insert(entry.binding, Arc::from(pids));
            }
        }

        Self {
            uniform_field_ids,
            texture_binding_property_ids,
        }
    }
}

/// Stable hash for stem strings (uniform/bind cache keys).
pub(crate) fn stem_hash(stem: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    stem.hash(&mut h);
    h.finish()
}

/// Reflects embedded WGSL for `stem`, builds the `@group(1)` layout, and interns property ids.
pub(crate) fn build_stem_material_layout(
    device: &wgpu::Device,
    stem: &str,
    property_registry: &PropertyIdRegistry,
) -> Result<Arc<StemMaterialLayout>, String> {
    let wgsl = embedded_shaders::embedded_target_wgsl(stem)
        .ok_or_else(|| format!("embedded WGSL missing for stem {stem}"))?;
    let reflected =
        reflect_raster_material_wgsl(wgsl).map_err(|e| format!("reflect {stem}: {e}"))?;

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("embedded_raster_material"),
        entries: &reflected.material_entries,
    });

    let ids = Arc::new(StemEmbeddedPropertyIds::build(
        property_registry,
        &reflected,
    ));
    let uniform_value_spaces = MaterialUniformValueSpaces::for_stem(stem, &reflected);

    Ok(Arc::new(StemMaterialLayout {
        bind_group_layout,
        reflected,
        ids,
        uniform_value_spaces,
    }))
}

#[cfg(test)]
mod tests {
    use super::{StemEmbeddedPropertyIds, shader_writer_unescaped_property_name};
    use crate::materials::host_data::PropertyIdRegistry;
    use crate::materials::reflect_raster_material_wgsl;

    #[test]
    fn xiexe_module_textures_resolve_to_unmangled_property_ids() {
        let wgsl = crate::embedded_shaders::embedded_target_wgsl("xstoon2.0_default")
            .expect("xiexe target WGSL");
        let reflected = reflect_raster_material_wgsl(wgsl).expect("xiexe WGSL reflection");
        let registry = PropertyIdRegistry::new();

        let ids = StemEmbeddedPropertyIds::build(&registry, &reflected);

        assert_eq!(
            ids.texture_binding_property_ids.get(&1).map(|p| &**p),
            Some([registry.intern("_MainTex"), registry.intern("_Tex"),].as_slice())
        );
    }

    #[test]
    fn xiexe_outline_emissive_typo_alias_is_preserved() {
        let wgsl = crate::embedded_shaders::embedded_target_wgsl("xstoon2.0_default")
            .expect("xiexe target WGSL");
        let reflected = reflect_raster_material_wgsl(wgsl).expect("xiexe WGSL reflection");
        let uniform = reflected
            .material_uniform
            .as_ref()
            .expect("xiexe material uniform block");
        assert!(
            uniform.fields.contains_key("_OutlineEmissiveues"),
            "the deliberate `_OutlineEmissiveues` Unity-property typo must remain in the reflected uniform block; XSToon2.0.shader sets this name and removing it would break outline-mode lookup"
        );
    }

    #[test]
    fn xiexe_baked_cubemap_binding_is_reflected() {
        let wgsl = crate::embedded_shaders::embedded_target_wgsl("xstoon2.0_default")
            .expect("xiexe target WGSL");
        let reflected = reflect_raster_material_wgsl(wgsl).expect("xiexe WGSL reflection");
        let unmangled: Vec<String> = reflected
            .material_group1_names
            .values()
            .map(|n| shader_writer_unescaped_property_name(n).to_string())
            .collect();
        assert!(
            unmangled.iter().any(|n| n == "_BakedCubemap"),
            "missing `_BakedCubemap` cube binding in xstoon2 layout: {unmangled:?}"
        );
        assert!(
            unmangled.iter().any(|n| n == "_BakedCubemap_sampler"),
            "missing `_BakedCubemap_sampler` in xstoon2 layout: {unmangled:?}"
        );
    }
}
