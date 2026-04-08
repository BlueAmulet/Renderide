//! Maps host shader asset ids from `set_shader` to renderer [`super::MaterialFamilyId`].
//!
//! Populated from [`crate::assets::shader::resolve_shader_upload`] when the host sends
//! [`crate::shared::ShaderUpload`]. Unknown ids use [`MaterialRouter::fallback`].

use std::collections::HashMap;

use super::MaterialFamilyId;

/// Host shader route: material family plus optional Unity-style name for the debug HUD.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShaderRouteEntry {
    /// Pipeline family for this host shader asset id.
    pub family: MaterialFamilyId,
    /// Logical shader label when known (ShaderLab name, WGSL banner, or upload field).
    pub display_name: Option<String>,
}

/// Shader asset id → route; unknown ids use [`Self::fallback`].
#[derive(Debug)]
pub struct MaterialRouter {
    routes: HashMap<i32, ShaderRouteEntry>,
    /// Optional composed WGSL stem (`shaders/target/<stem>.wgsl`) when an embedded `{key}_default` target exists.
    shader_stem: HashMap<i32, String>,
    /// Default when `routes` has no entry.
    pub fallback: MaterialFamilyId,
}

impl MaterialRouter {
    /// Builds a router with only a fallback family.
    pub fn new(fallback: MaterialFamilyId) -> Self {
        Self {
            routes: HashMap::new(),
            shader_stem: HashMap::new(),
            fallback,
        }
    }

    /// Inserts or replaces a host shader route (family and optional HUD label).
    pub fn set_shader_route(
        &mut self,
        shader_asset_id: i32,
        family: MaterialFamilyId,
        display_name: Option<String>,
    ) {
        self.routes.insert(
            shader_asset_id,
            ShaderRouteEntry {
                family,
                display_name,
            },
        );
    }

    /// Inserts a host shader → family mapping with no HUD display name.
    pub fn set_shader_family(&mut self, shader_asset_id: i32, family: MaterialFamilyId) {
        self.set_shader_route(shader_asset_id, family, None);
    }

    /// Resolves the family for a host shader asset id.
    pub fn family_for_shader_asset(&self, shader_asset_id: i32) -> MaterialFamilyId {
        self.routes
            .get(&shader_asset_id)
            .map(|e| e.family)
            .unwrap_or(self.fallback)
    }

    /// Records a target WGSL stem for `shader_asset_id` (from manifest Unity name resolution).
    pub fn set_shader_stem(&mut self, shader_asset_id: i32, stem: String) {
        self.shader_stem.insert(shader_asset_id, stem);
    }

    /// Clears [`Self::stem_for_shader_asset`] for `shader_asset_id`.
    pub fn remove_shader_stem(&mut self, shader_asset_id: i32) {
        self.shader_stem.remove(&shader_asset_id);
    }

    /// Composed material stem when the host shader name matched the embedded manifest.
    pub fn stem_for_shader_asset(&self, shader_asset_id: i32) -> Option<&str> {
        self.shader_stem.get(&shader_asset_id).map(String::as_str)
    }

    /// Drops a host shader id mapping after [`crate::shared::ShaderUnload`].
    pub fn remove_shader_family(&mut self, shader_asset_id: i32) {
        self.routes.remove(&shader_asset_id);
        self.shader_stem.remove(&shader_asset_id);
    }

    /// Returns the mapped family when the host id was registered via [`Self::set_shader_route`].
    pub fn get_shader_family(&self, shader_asset_id: i32) -> Option<MaterialFamilyId> {
        self.routes.get(&shader_asset_id).map(|e| e.family)
    }

    /// Host shader asset ids, families, and optional display names, sorted by id (for debug HUD).
    pub fn routes_sorted_for_hud(&self) -> Vec<(i32, MaterialFamilyId, Option<String>)> {
        let mut v: Vec<_> = self
            .routes
            .iter()
            .map(|(&k, e)| (k, e.family, e.display_name.clone()))
            .collect();
        v.sort_by_key(|(k, _, _)| *k);
        v
    }
}

#[cfg(test)]
mod tests {
    use super::MaterialRouter;
    use crate::materials::{MaterialFamilyId, SOLID_COLOR_FAMILY_ID};

    #[test]
    fn remove_shader_family_clears_entry() {
        let mut r = MaterialRouter::new(MaterialFamilyId(99));
        r.set_shader_family(7, SOLID_COLOR_FAMILY_ID);
        assert_eq!(r.get_shader_family(7), Some(SOLID_COLOR_FAMILY_ID));
        r.remove_shader_family(7);
        assert_eq!(r.get_shader_family(7), None);
        assert_eq!(r.family_for_shader_asset(7), MaterialFamilyId(99));
    }

    #[test]
    fn remove_shader_family_clears_stem() {
        let mut r = MaterialRouter::new(MaterialFamilyId(99));
        r.set_shader_route(1, SOLID_COLOR_FAMILY_ID, Some("x".to_string()));
        r.set_shader_stem(1, "debug_world_normals_default".to_string());
        assert_eq!(
            r.stem_for_shader_asset(1),
            Some("debug_world_normals_default")
        );
        r.remove_shader_family(1);
        assert_eq!(r.stem_for_shader_asset(1), None);
    }

    #[test]
    fn set_shader_route_stores_display_name_for_hud() {
        let mut r = MaterialRouter::new(MaterialFamilyId(99));
        r.set_shader_route(
            3,
            SOLID_COLOR_FAMILY_ID,
            Some("Custom/ExampleShader".to_string()),
        );
        assert_eq!(
            r.routes_sorted_for_hud(),
            vec![(
                3,
                SOLID_COLOR_FAMILY_ID,
                Some("Custom/ExampleShader".to_string())
            )]
        );
        assert_eq!(r.get_shader_family(3), Some(SOLID_COLOR_FAMILY_ID));
    }

    #[test]
    fn routes_sorted_for_hud_sorted_by_id() {
        let mut r = MaterialRouter::new(MaterialFamilyId(99));
        r.set_shader_route(10, SOLID_COLOR_FAMILY_ID, None);
        r.set_shader_route(2, SOLID_COLOR_FAMILY_ID, Some("a".to_string()));
        assert_eq!(
            r.routes_sorted_for_hud()
                .into_iter()
                .map(|(id, _, _)| id)
                .collect::<Vec<_>>(),
            vec![2, 10]
        );
    }
}
