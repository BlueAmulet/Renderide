//! Owns all [`RenderSpaceState`](super::render_space::RenderSpaceState) instances and applies per-frame host data.

use std::collections::{HashMap, HashSet};

use glam::Mat4;

use crate::ipc::SharedMemoryAccessor;
use crate::shared::FrameSubmitData;

use super::error::SceneError;
use super::ids::RenderSpaceId;
use super::math::multiply_root;
use super::render_space::RenderSpaceState;
use super::world::{compute_world_matrices_for_space, ensure_cache_shapes, WorldTransformCache};

/// Scene registry: one entry per host render space, Unity `RenderingManager` dictionary semantics.
pub struct SceneCoordinator {
    spaces: HashMap<RenderSpaceId, RenderSpaceState>,
    world_caches: HashMap<RenderSpaceId, WorldTransformCache>,
    world_dirty: HashSet<RenderSpaceId>,
}

impl Default for SceneCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

impl SceneCoordinator {
    /// Empty registry.
    pub fn new() -> Self {
        Self {
            spaces: HashMap::new(),
            world_caches: HashMap::new(),
            world_dirty: HashSet::new(),
        }
    }

    /// Read-only access for debugging / future systems.
    pub fn space(&self, id: RenderSpaceId) -> Option<&RenderSpaceState> {
        self.spaces.get(&id)
    }

    /// Cached space-local world matrix (`world * root` via [`Self::world_matrix_with_root`]).
    pub fn world_matrix_local(&self, id: RenderSpaceId, transform_index: usize) -> Option<Mat4> {
        self.world_caches
            .get(&id)?
            .world_matrices
            .get(transform_index)
            .copied()
    }

    /// Absolute world matrix including render-space root TRS.
    pub fn world_matrix_with_root(
        &self,
        id: RenderSpaceId,
        transform_index: usize,
    ) -> Option<Mat4> {
        let space = self.spaces.get(&id)?;
        let local = self.world_matrix_local(id, transform_index)?;
        Some(multiply_root(local, &space.root_transform))
    }

    /// Recomputes cached world matrices for every dirty space (no-op if caches clean).
    pub fn flush_world_caches(&mut self) -> Result<(), SceneError> {
        let dirty: Vec<RenderSpaceId> = self.world_dirty.iter().copied().collect();
        for id in dirty {
            let Some(space) = self.spaces.get(&id) else {
                self.world_caches.remove(&id);
                self.world_dirty.remove(&id);
                continue;
            };
            let n = space.nodes.len();
            let cache = self.world_caches.entry(id).or_default();
            ensure_cache_shapes(cache, n, false);
            compute_world_matrices_for_space(id.0, &space.nodes, &space.node_parents, cache)?;
            self.world_dirty.remove(&id);
        }
        Ok(())
    }

    /// Applies [`FrameSubmitData`]: transforms, static mesh renderables, skinned mesh renderables.
    ///
    /// Other render-space payloads (lights, cameras, layers, …) are intentionally omitted in this
    /// phase.
    pub fn apply_frame_submit(
        &mut self,
        shm: &mut SharedMemoryAccessor,
        data: &FrameSubmitData,
    ) -> Result<(), SceneError> {
        let mut seen = HashSet::new();

        for update in &data.render_spaces {
            seen.insert(RenderSpaceId(update.id));
            let space = self
                .spaces
                .entry(RenderSpaceId(update.id))
                .or_insert_with(|| RenderSpaceState {
                    id: RenderSpaceId(update.id),
                    ..Default::default()
                });
            space.id = RenderSpaceId(update.id);
            space.apply_update_header(update);

            let cache = self
                .world_caches
                .entry(RenderSpaceId(update.id))
                .or_default();

            let mut transform_removals = Vec::new();
            if let Some(ref tu) = update.transforms_update {
                transform_removals.extend(super::transforms_apply::apply_transforms_update(
                    space,
                    cache,
                    &mut self.world_dirty,
                    RenderSpaceId(update.id),
                    shm,
                    tu,
                    data.frame_index,
                )?);
            }
            if let Some(ref mu) = update.mesh_renderers_update {
                super::mesh_apply::apply_mesh_renderables_update(
                    space,
                    shm,
                    mu,
                    data.frame_index,
                    update.id,
                )?;
            }
            if let Some(ref su) = update.skinned_mesh_renderers_update {
                super::mesh_apply::apply_skinned_mesh_renderables_update(
                    space,
                    shm,
                    su,
                    data.frame_index,
                    update.id,
                    &transform_removals,
                )?;
            }
        }

        let to_remove: Vec<RenderSpaceId> = self
            .spaces
            .keys()
            .copied()
            .filter(|id| !seen.contains(id))
            .collect();
        for id in to_remove {
            self.spaces.remove(&id);
            self.world_caches.remove(&id);
            self.world_dirty.remove(&id);
        }

        Ok(())
    }
}
