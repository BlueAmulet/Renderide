//! Host shader asset → raster material family for mesh draws.

use super::router::MaterialRouter;
use super::MaterialFamilyId;

/// Resolves the material family used for **mesh rasterization** for a host shader asset id.
///
/// Uses [`MaterialRouter::family_for_shader_asset`], populated when the host sends
/// [`crate::shared::ShaderUpload`] (see [`crate::assets::shader::resolve_shader_upload`]).
pub fn resolve_raster_family(shader_asset_id: i32, router: &MaterialRouter) -> MaterialFamilyId {
    router.family_for_shader_asset(shader_asset_id)
}

#[cfg(test)]
mod tests {
    use super::resolve_raster_family;
    use crate::materials::{MaterialFamilyId, MaterialRouter};

    /// [`resolve_raster_family`] only forwards [`MaterialRouter::family_for_shader_asset`]; tests need two
    /// distinct ids, not any particular builtin pipeline family.
    const FALLBACK: MaterialFamilyId = MaterialFamilyId(0xE0_00_01);
    const REGISTERED_ROUTE: MaterialFamilyId = MaterialFamilyId(0xE0_00_02);

    #[test]
    fn unknown_shader_uses_router_fallback() {
        let r = MaterialRouter::new(FALLBACK);
        assert_eq!(resolve_raster_family(999, &r), FALLBACK);
    }

    #[test]
    fn registered_shader_uses_route_family() {
        let mut r = MaterialRouter::new(FALLBACK);
        r.set_shader_family(7, REGISTERED_ROUTE);
        assert_eq!(resolve_raster_family(7, &r), REGISTERED_ROUTE);
    }
}
