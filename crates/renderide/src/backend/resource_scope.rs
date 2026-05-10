//! Render-space-scoped ownership tracking for GPU-backed assets.

use hashbrown::{HashMap, HashSet};

use crate::assets::texture::HostTextureAssetKind;
use crate::materials::MaterialSystem;
use crate::scene::{RenderSpaceId, RenderSpaceView, SceneApplyReport, SceneCoordinator};

/// One GPU-relevant asset reference discovered from a render space.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(crate) enum ScopedAssetKey {
    /// Mesh asset in the resident mesh pool.
    Mesh(i32),
    /// Texture2D asset in the resident texture pool.
    Texture2D(i32),
    /// Texture3D asset in the resident texture pool.
    Texture3D(i32),
    /// Cubemap asset in the resident cubemap pool.
    Cubemap(i32),
    /// Host render texture asset.
    RenderTexture(i32),
    /// Host video texture asset.
    VideoTexture(i32),
    /// Host desktop texture asset.
    DesktopTexture(i32),
    /// Point render buffer placeholder asset.
    PointRenderBuffer(i32),
    /// Trail render buffer placeholder asset.
    TrailRenderBuffer(i32),
    /// Gaussian splat placeholder asset.
    GaussianSplat(i32),
    /// Host material asset.
    Material(i32),
    /// Host material property block asset.
    MaterialPropertyBlock(i32),
}

/// Asset ids referenced by one render space, grouped by backend asset family.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct RenderSpaceAssetSet {
    /// Resident mesh asset ids.
    pub(crate) meshes: HashSet<i32>,
    /// Resident Texture2D asset ids.
    pub(crate) texture_2d: HashSet<i32>,
    /// Resident Texture3D asset ids.
    pub(crate) texture_3d: HashSet<i32>,
    /// Resident cubemap asset ids.
    pub(crate) cubemaps: HashSet<i32>,
    /// Host render texture asset ids.
    pub(crate) render_textures: HashSet<i32>,
    /// Host video texture asset ids.
    pub(crate) video_textures: HashSet<i32>,
    /// Host desktop texture asset ids.
    pub(crate) desktop_textures: HashSet<i32>,
    /// Point render buffer asset ids.
    pub(crate) point_render_buffers: HashSet<i32>,
    /// Trail render buffer asset ids.
    pub(crate) trail_render_buffers: HashSet<i32>,
    /// Gaussian splat asset ids.
    pub(crate) gaussian_splats: HashSet<i32>,
    /// Material asset ids.
    pub(crate) materials: HashSet<i32>,
    /// Material property block asset ids.
    pub(crate) property_blocks: HashSet<i32>,
}

impl RenderSpaceAssetSet {
    /// Returns whether this set contains no asset references.
    pub(crate) fn is_empty(&self) -> bool {
        self.meshes.is_empty()
            && self.texture_2d.is_empty()
            && self.texture_3d.is_empty()
            && self.cubemaps.is_empty()
            && self.render_textures.is_empty()
            && self.video_textures.is_empty()
            && self.desktop_textures.is_empty()
            && self.point_render_buffers.is_empty()
            && self.trail_render_buffers.is_empty()
            && self.gaussian_splats.is_empty()
            && self.materials.is_empty()
            && self.property_blocks.is_empty()
    }

    /// Number of unique asset ids across all families.
    pub(crate) fn total_len(&self) -> usize {
        self.meshes.len()
            + self.texture_2d.len()
            + self.texture_3d.len()
            + self.cubemaps.len()
            + self.render_textures.len()
            + self.video_textures.len()
            + self.desktop_textures.len()
            + self.point_render_buffers.len()
            + self.trail_render_buffers.len()
            + self.gaussian_splats.len()
            + self.materials.len()
            + self.property_blocks.len()
    }

    /// Adds a key to the matching family set.
    pub(crate) fn insert_key(&mut self, key: ScopedAssetKey) -> bool {
        match key {
            ScopedAssetKey::Mesh(id) => self.insert_mesh(id),
            ScopedAssetKey::Texture2D(id) => self.insert_texture_2d(id),
            ScopedAssetKey::Texture3D(id) => self.insert_texture_3d(id),
            ScopedAssetKey::Cubemap(id) => self.insert_cubemap(id),
            ScopedAssetKey::RenderTexture(id) => self.insert_render_texture(id),
            ScopedAssetKey::VideoTexture(id) => self.insert_video_texture(id),
            ScopedAssetKey::DesktopTexture(id) => self.insert_desktop_texture(id),
            ScopedAssetKey::PointRenderBuffer(id) => self.insert_point_render_buffer(id),
            ScopedAssetKey::TrailRenderBuffer(id) => self.insert_trail_render_buffer(id),
            ScopedAssetKey::GaussianSplat(id) => self.insert_gaussian_splat(id),
            ScopedAssetKey::Material(id) => self.insert_material(id),
            ScopedAssetKey::MaterialPropertyBlock(id) => self.insert_property_block(id),
        }
    }

    /// Merges another set into this set.
    pub(crate) fn extend(&mut self, other: RenderSpaceAssetSet) {
        self.meshes.extend(other.meshes);
        self.texture_2d.extend(other.texture_2d);
        self.texture_3d.extend(other.texture_3d);
        self.cubemaps.extend(other.cubemaps);
        self.render_textures.extend(other.render_textures);
        self.video_textures.extend(other.video_textures);
        self.desktop_textures.extend(other.desktop_textures);
        self.point_render_buffers.extend(other.point_render_buffers);
        self.trail_render_buffers.extend(other.trail_render_buffers);
        self.gaussian_splats.extend(other.gaussian_splats);
        self.materials.extend(other.materials);
        self.property_blocks.extend(other.property_blocks);
    }

    /// Visits every key in this set.
    pub(crate) fn for_each_key(&self, mut visit: impl FnMut(ScopedAssetKey)) {
        for &id in &self.meshes {
            visit(ScopedAssetKey::Mesh(id));
        }
        for &id in &self.texture_2d {
            visit(ScopedAssetKey::Texture2D(id));
        }
        for &id in &self.texture_3d {
            visit(ScopedAssetKey::Texture3D(id));
        }
        for &id in &self.cubemaps {
            visit(ScopedAssetKey::Cubemap(id));
        }
        for &id in &self.render_textures {
            visit(ScopedAssetKey::RenderTexture(id));
        }
        for &id in &self.video_textures {
            visit(ScopedAssetKey::VideoTexture(id));
        }
        for &id in &self.desktop_textures {
            visit(ScopedAssetKey::DesktopTexture(id));
        }
        for &id in &self.point_render_buffers {
            visit(ScopedAssetKey::PointRenderBuffer(id));
        }
        for &id in &self.trail_render_buffers {
            visit(ScopedAssetKey::TrailRenderBuffer(id));
        }
        for &id in &self.gaussian_splats {
            visit(ScopedAssetKey::GaussianSplat(id));
        }
        for &id in &self.materials {
            visit(ScopedAssetKey::Material(id));
        }
        for &id in &self.property_blocks {
            visit(ScopedAssetKey::MaterialPropertyBlock(id));
        }
    }

    /// Adds a mesh id when it is a valid non-negative host asset id.
    pub(crate) fn insert_mesh(&mut self, id: i32) -> bool {
        insert_valid_id(&mut self.meshes, id)
    }

    /// Adds a Texture2D id when it is a valid non-negative host asset id.
    pub(crate) fn insert_texture_2d(&mut self, id: i32) -> bool {
        insert_valid_id(&mut self.texture_2d, id)
    }

    /// Adds a Texture3D id when it is a valid non-negative host asset id.
    pub(crate) fn insert_texture_3d(&mut self, id: i32) -> bool {
        insert_valid_id(&mut self.texture_3d, id)
    }

    /// Adds a cubemap id when it is a valid non-negative host asset id.
    pub(crate) fn insert_cubemap(&mut self, id: i32) -> bool {
        insert_valid_id(&mut self.cubemaps, id)
    }

    /// Adds a render texture id when it is a valid non-negative host asset id.
    pub(crate) fn insert_render_texture(&mut self, id: i32) -> bool {
        insert_valid_id(&mut self.render_textures, id)
    }

    /// Adds a video texture id when it is a valid non-negative host asset id.
    pub(crate) fn insert_video_texture(&mut self, id: i32) -> bool {
        insert_valid_id(&mut self.video_textures, id)
    }

    /// Adds a desktop texture id when it is a valid non-negative host asset id.
    pub(crate) fn insert_desktop_texture(&mut self, id: i32) -> bool {
        insert_valid_id(&mut self.desktop_textures, id)
    }

    /// Adds a point render buffer id when it is a valid non-negative host asset id.
    pub(crate) fn insert_point_render_buffer(&mut self, id: i32) -> bool {
        insert_valid_id(&mut self.point_render_buffers, id)
    }

    /// Adds a trail render buffer id when it is a valid non-negative host asset id.
    pub(crate) fn insert_trail_render_buffer(&mut self, id: i32) -> bool {
        insert_valid_id(&mut self.trail_render_buffers, id)
    }

    /// Adds a Gaussian splat id when it is a valid non-negative host asset id.
    pub(crate) fn insert_gaussian_splat(&mut self, id: i32) -> bool {
        insert_valid_id(&mut self.gaussian_splats, id)
    }

    /// Adds a material id when it is a valid non-negative host asset id.
    pub(crate) fn insert_material(&mut self, id: i32) -> bool {
        insert_valid_id(&mut self.materials, id)
    }

    /// Adds a property block id when it is a valid non-negative host asset id.
    pub(crate) fn insert_property_block(&mut self, id: i32) -> bool {
        insert_valid_id(&mut self.property_blocks, id)
    }

    /// Adds a packed host texture reference to the matching texture family.
    pub(crate) fn insert_texture_reference(&mut self, id: i32, kind: HostTextureAssetKind) -> bool {
        match kind {
            HostTextureAssetKind::Texture2D => self.insert_texture_2d(id),
            HostTextureAssetKind::Texture3D => self.insert_texture_3d(id),
            HostTextureAssetKind::Cubemap => self.insert_cubemap(id),
            HostTextureAssetKind::RenderTexture => self.insert_render_texture(id),
            HostTextureAssetKind::VideoTexture => self.insert_video_texture(id),
            HostTextureAssetKind::Desktop => self.insert_desktop_texture(id),
        }
    }
}

/// Assets released because one or more render spaces closed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct ReleasedRenderSpaceResources {
    /// Render spaces removed by the host frame submit.
    pub(crate) removed_spaces: Vec<RenderSpaceId>,
    /// Assets that no live render space still references.
    pub(crate) assets: RenderSpaceAssetSet,
}

impl ReleasedRenderSpaceResources {
    /// Returns whether this report has no render-space closes.
    pub(crate) fn is_empty(&self) -> bool {
        self.removed_spaces.is_empty() && self.assets.is_empty()
    }
}

/// Maintains render-space ownership and reverse asset ownership.
#[derive(Default)]
pub(crate) struct RenderSpaceResourceScopes {
    space_assets: HashMap<RenderSpaceId, RenderSpaceAssetSet>,
    asset_owners: HashMap<ScopedAssetKey, HashSet<RenderSpaceId>>,
    material_dependencies_dirty: bool,
}

impl RenderSpaceResourceScopes {
    /// Creates an empty ownership tracker.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Marks material texture dependencies stale after a material/property update.
    pub(crate) fn note_material_dependencies_dirty(&mut self) {
        self.material_dependencies_dirty = true;
    }

    /// Applies one scene mutation report and returns resources released by closed render spaces.
    pub(crate) fn apply_scene_report(
        &mut self,
        report: &SceneApplyReport,
        scene: &SceneCoordinator,
        materials: &MaterialSystem,
    ) -> ReleasedRenderSpaceResources {
        profiling::scope!("backend::resource_scopes::apply_scene_report");
        let refresh_all = self.material_dependencies_dirty;
        for &space_id in &report.submitted_spaces {
            if refresh_all
                || report.changed_spaces.contains(&space_id)
                || !self.space_assets.contains_key(&space_id)
            {
                let assets = collect_render_space_assets(scene, materials, space_id);
                self.replace_space_assets(space_id, assets);
            }
        }
        self.material_dependencies_dirty = false;

        let mut released = ReleasedRenderSpaceResources::default();
        for &space_id in &report.removed_spaces {
            released.removed_spaces.push(space_id);
            released.assets.extend(self.remove_space_assets(space_id));
        }

        self.insert_current_material_texture_references_for_release(
            materials,
            &mut released.assets,
        );
        released
    }

    fn replace_space_assets(&mut self, space_id: RenderSpaceId, assets: RenderSpaceAssetSet) {
        let _ = self.remove_space_assets(space_id);
        assets.for_each_key(|key| {
            self.asset_owners.entry(key).or_default().insert(space_id);
        });
        self.space_assets.insert(space_id, assets);
    }

    fn remove_space_assets(&mut self, space_id: RenderSpaceId) -> RenderSpaceAssetSet {
        let mut released = RenderSpaceAssetSet::default();
        let Some(old_assets) = self.space_assets.remove(&space_id) else {
            return released;
        };
        old_assets.for_each_key(|key| {
            let Some(owners) = self.asset_owners.get_mut(&key) else {
                return;
            };
            owners.remove(&space_id);
            if owners.is_empty() {
                self.asset_owners.remove(&key);
                released.insert_key(key);
            }
        });
        released
    }

    fn insert_current_material_texture_references_for_release(
        &self,
        materials: &MaterialSystem,
        released: &mut RenderSpaceAssetSet,
    ) {
        if released.materials.is_empty() && released.property_blocks.is_empty() {
            return;
        }
        let mut texture_references = Vec::new();
        materials.for_each_texture_reference(
            &released.materials,
            &released.property_blocks,
            |id, kind| texture_references.push((id, kind)),
        );
        for (id, kind) in texture_references {
            let key = texture_asset_key(id, kind);
            if !self.asset_owners.contains_key(&key) {
                released.insert_texture_reference(id, kind);
            }
        }
    }
}

fn collect_render_space_assets(
    scene: &SceneCoordinator,
    materials: &MaterialSystem,
    space_id: RenderSpaceId,
) -> RenderSpaceAssetSet {
    let mut assets = RenderSpaceAssetSet::default();
    let Some(space) = scene.space(space_id) else {
        return assets;
    };

    collect_direct_space_assets(space, scene, &mut assets);
    let mut texture_references = Vec::new();
    materials.for_each_texture_reference(&assets.materials, &assets.property_blocks, |id, kind| {
        texture_references.push((id, kind));
    });
    for (id, kind) in texture_references {
        assets.insert_texture_reference(id, kind);
    }
    assets
}

fn collect_direct_space_assets(
    space: RenderSpaceView<'_>,
    scene: &SceneCoordinator,
    assets: &mut RenderSpaceAssetSet,
) {
    assets.insert_material(space.skybox_material_asset_id());

    for renderer in space.static_mesh_renderers() {
        collect_mesh_renderer_assets(
            renderer.mesh_asset_id,
            renderer.primary_material_asset_id,
            renderer.primary_property_block_id,
            &renderer.material_slots,
            assets,
        );
    }

    for renderer in space.skinned_mesh_renderers() {
        let base = &renderer.base;
        collect_mesh_renderer_assets(
            base.mesh_asset_id,
            base.primary_material_asset_id,
            base.primary_property_block_id,
            &base.material_slots,
            assets,
        );
    }

    for camera in space.cameras() {
        assets.insert_render_texture(camera.state.render_texture_asset_id);
    }

    for probe in space.reflection_probes() {
        assets.insert_cubemap(probe.state.cubemap_asset_id);
    }

    for id in scene.light_cookie_texture_asset_ids(space) {
        assets.insert_texture_2d(id);
    }
}

fn collect_mesh_renderer_assets(
    mesh_asset_id: i32,
    primary_material_asset_id: Option<i32>,
    primary_property_block_id: Option<i32>,
    material_slots: &[crate::scene::MeshMaterialSlot],
    assets: &mut RenderSpaceAssetSet,
) {
    assets.insert_mesh(mesh_asset_id);
    if let Some(id) = primary_material_asset_id {
        assets.insert_material(id);
    }
    if let Some(id) = primary_property_block_id {
        assets.insert_property_block(id);
    }
    for slot in material_slots {
        assets.insert_material(slot.material_asset_id);
        if let Some(id) = slot.property_block_id {
            assets.insert_property_block(id);
        }
    }
}

fn texture_asset_key(id: i32, kind: HostTextureAssetKind) -> ScopedAssetKey {
    match kind {
        HostTextureAssetKind::Texture2D => ScopedAssetKey::Texture2D(id),
        HostTextureAssetKind::Texture3D => ScopedAssetKey::Texture3D(id),
        HostTextureAssetKind::Cubemap => ScopedAssetKey::Cubemap(id),
        HostTextureAssetKind::RenderTexture => ScopedAssetKey::RenderTexture(id),
        HostTextureAssetKind::VideoTexture => ScopedAssetKey::VideoTexture(id),
        HostTextureAssetKind::Desktop => ScopedAssetKey::DesktopTexture(id),
    }
}

fn insert_valid_id(set: &mut HashSet<i32>, id: i32) -> bool {
    id >= 0 && set.insert(id)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set_with_mesh(mesh_id: i32) -> RenderSpaceAssetSet {
        let mut set = RenderSpaceAssetSet::default();
        set.insert_mesh(mesh_id);
        set
    }

    #[test]
    fn shared_assets_are_released_only_after_last_owner_closes() {
        let mut scopes = RenderSpaceResourceScopes::new();
        scopes.replace_space_assets(RenderSpaceId(1), set_with_mesh(10));
        scopes.replace_space_assets(RenderSpaceId(2), set_with_mesh(10));

        assert!(scopes.remove_space_assets(RenderSpaceId(1)).is_empty());

        let released = scopes.remove_space_assets(RenderSpaceId(2));
        assert!(released.meshes.contains(&10));
    }

    #[test]
    fn replacing_space_assets_updates_reverse_owners() {
        let mut scopes = RenderSpaceResourceScopes::new();
        scopes.replace_space_assets(RenderSpaceId(1), set_with_mesh(10));
        scopes.replace_space_assets(RenderSpaceId(1), set_with_mesh(20));

        assert!(!scopes.asset_owners.contains_key(&ScopedAssetKey::Mesh(10)));
        assert!(
            scopes
                .asset_owners
                .get(&ScopedAssetKey::Mesh(20))
                .is_some_and(|owners| owners.contains(&RenderSpaceId(1)))
        );
    }

    #[test]
    fn invalid_negative_ids_are_ignored() {
        let mut set = RenderSpaceAssetSet::default();

        assert!(!set.insert_mesh(-1));
        assert!(!set.insert_material(-1));
        assert!(set.is_empty());
    }
}
