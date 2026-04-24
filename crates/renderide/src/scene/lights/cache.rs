//! [`LightCache`]: merges incremental host light updates and resolves to world space.
//!
//! Dense per-space storage mirrors the reference
//! [`RenderableComponentManager`](https://github.com/Yellow-Dog-Man/FrooxEngine) protocol: the host
//! pre-assigns a `RenderableIndex` equal to its own list length and applies swap-remove on
//! removals, and the renderer maintains an identically-ordered list so subsequent state rows can
//! address renderables by index without any renderer→host handshake.

use hashbrown::HashMap;

use glam::{Mat4, Quat, Vec3};

use crate::shared::{LightData, LightState, LightsBufferRendererState};

use super::super::transforms_apply::TransformRemovalEvent;
use super::super::world::fixup_transform_id;
use super::types::{CachedLight, ResolvedLight};

/// Local axis for light propagation before world transform (host forward = **+Z**).
const LOCAL_LIGHT_PROPAGATION: Vec3 = Vec3::new(0.0, 0.0, 1.0);

/// Dense buffer-renderer entry. Position in the per-space [`Vec`] equals the host's
/// `RenderableIndex`; the pointed-to [`LightData`] rows live in [`LightCache::buffers`] keyed by
/// `state.global_unique_id`.
#[derive(Clone, Copy, Debug)]
struct BufferRenderer {
    /// Dense transform index for world-matrix lookup (from the host additions batch).
    transform_id: usize,
    /// Host state (includes `global_unique_id` selecting which [`LightCache::buffers`] payload to fan out).
    state: LightsBufferRendererState,
}

/// CPU-side cache: buffer submissions, per-render-space flattened lights, regular vs buffer paths.
///
/// Populated from [`crate::shared::FrameSubmitData`] light batches and
/// [`crate::shared::LightsBufferRendererSubmission`]. GPU upload uses
/// [`Self::resolve_lights`] after world matrices are current.
#[derive(Clone, Debug)]
pub struct LightCache {
    /// Monotonic change counter; advanced on any mutation.
    version: u64,
    /// Shared [`LightData`] payloads keyed by `global_unique_id`. Referenced by every
    /// [`BufferRenderer`] whose `state.global_unique_id` matches.
    buffers: HashMap<i32, Vec<LightData>>,
    /// Flattened per-space output, rebuilt after each apply from [`Self::regular_lights`] and
    /// [`Self::buffer_renderers`] fanning out [`Self::buffers`].
    spaces: HashMap<i32, Vec<CachedLight>>,
    /// Dense per-space list of regular (Unity `Light`) renderables; vec index == host `RenderableIndex`.
    regular_lights: HashMap<i32, Vec<CachedLight>>,
    /// Dense per-space list of buffer-renderer entries; vec index == host `RenderableIndex`.
    buffer_renderers: HashMap<i32, Vec<BufferRenderer>>,
}

impl LightCache {
    /// Empty cache.
    pub fn new() -> Self {
        Self {
            version: 0,
            buffers: HashMap::new(),
            spaces: HashMap::new(),
            regular_lights: HashMap::new(),
            buffer_renderers: HashMap::new(),
        }
    }

    /// Number of distinct light buffers stored from submissions (diagnostics).
    pub fn buffer_count(&self) -> usize {
        self.buffers.len()
    }

    /// Monotonic generation for renderable light output.
    pub fn version(&self) -> u64 {
        self.version
    }

    fn mark_changed(&mut self) {
        self.version = self.version.wrapping_add(1);
    }

    /// Stores full [`LightData`] rows from a host submission (overwrites prior buffer id) and
    /// rebuilds every render space that has a [`BufferRenderer`] pointing at this `global_unique_id`.
    pub fn store_full(&mut self, lights_buffer_unique_id: i32, light_data: Vec<LightData>) {
        self.buffers.insert(lights_buffer_unique_id, light_data);
        let mut dirty_spaces: Vec<i32> = self
            .buffer_renderers
            .iter()
            .filter_map(|(sid, v)| {
                v.iter()
                    .any(|br| br.state.global_unique_id == lights_buffer_unique_id)
                    .then_some(*sid)
            })
            .collect();
        dirty_spaces.sort_unstable();
        dirty_spaces.dedup();
        for sid in dirty_spaces {
            self.rebuild_space_vec(sid);
        }
        self.mark_changed();
    }

    /// Rebuilds [`Self::spaces`] for one render space from dense regular and buffer-renderer
    /// lists. Removes the old entry first so the rebuild can read the other maps without
    /// aliasing borrows.
    fn rebuild_space_vec(&mut self, space_id: i32) {
        profiling::scope!("lights::rebuild_space_vec");
        let mut out = self.spaces.remove(&space_id).unwrap_or_default();
        out.clear();

        if let Some(regulars) = self.regular_lights.get(&space_id) {
            out.extend(regulars.iter().cloned());
        }

        if let Some(renderers) = self.buffer_renderers.get(&space_id) {
            for br in renderers {
                let Some(buffer_data) = self.buffers.get(&br.state.global_unique_id) else {
                    continue;
                };
                for data in buffer_data {
                    out.push(CachedLight {
                        data: *data,
                        state: br.state,
                        transform_id: br.transform_id,
                    });
                }
            }
        }

        self.spaces.insert(space_id, out);
    }

    /// Applies [`crate::shared::LightsBufferRendererUpdate`]: removals, additions (transform indices), states.
    ///
    /// Processes in the fixed order **removals → additions → states**, matching the reference
    /// `RenderableManager.HandleUpdate`. Removal uses [`Vec::swap_remove`] so the renderer's
    /// dense list stays in lockstep with the host's swap-remove reindexing; additions append
    /// placeholder entries whose transform ids come from the `additions` buffer; state rows
    /// address those entries by index.
    pub fn apply_update(
        &mut self,
        space_id: i32,
        removals: &[i32],
        additions: &[i32],
        states: &[LightsBufferRendererState],
    ) {
        profiling::scope!("lights::apply_update");
        let v = self.buffer_renderers.entry(space_id).or_default();

        for &idx in removals.iter().take_while(|&&i| i >= 0) {
            let idx_usize = idx as usize;
            if idx_usize >= v.len() {
                logger::warn!(
                    "light_cache: buffer-renderer removal index {idx} out of range (space_id={space_id}, len={})",
                    v.len()
                );
                continue;
            }
            v.swap_remove(idx_usize);
        }

        for &t in additions.iter().take_while(|&&t| t >= 0) {
            v.push(BufferRenderer {
                transform_id: t as usize,
                state: LightsBufferRendererState::default(),
            });
        }

        for state in states {
            if state.renderable_index < 0 {
                break;
            }
            let idx_usize = state.renderable_index as usize;
            let Some(slot) = v.get_mut(idx_usize) else {
                logger::warn!(
                    "light_cache: buffer-renderer state index {} out of range (space_id={space_id}, len={})",
                    state.renderable_index,
                    v.len()
                );
                continue;
            };
            slot.state = *state;
        }

        self.rebuild_space_vec(space_id);
        self.mark_changed();
    }

    /// Applies regular [`LightState`] updates (Unity `Light` components).
    ///
    /// Same three-phase pipeline as [`Self::apply_update`]: removals via [`Vec::swap_remove`] to
    /// mirror the host's reindexing, additions append placeholder [`CachedLight`]s carrying the
    /// transform id from the `additions` buffer, and states address entries by index.
    pub fn apply_regular_lights_update(
        &mut self,
        space_id: i32,
        removals: &[i32],
        additions: &[i32],
        states: &[LightState],
    ) {
        profiling::scope!("lights::apply_regular_lights_update");
        let v = self.regular_lights.entry(space_id).or_default();

        for &idx in removals.iter().take_while(|&&i| i >= 0) {
            let idx_usize = idx as usize;
            if idx_usize >= v.len() {
                logger::warn!(
                    "light_cache: regular-light removal index {idx} out of range (space_id={space_id}, len={})",
                    v.len()
                );
                continue;
            }
            v.swap_remove(idx_usize);
        }

        for &t in additions.iter().take_while(|&&t| t >= 0) {
            v.push(CachedLight {
                data: LightData::default(),
                state: LightsBufferRendererState::default(),
                transform_id: t as usize,
            });
        }

        for state in states {
            if state.renderable_index < 0 {
                break;
            }
            let idx_usize = state.renderable_index as usize;
            let Some(slot) = v.get_mut(idx_usize) else {
                logger::warn!(
                    "light_cache: regular-light state index {} out of range (space_id={space_id}, len={})",
                    state.renderable_index,
                    v.len()
                );
                continue;
            };
            slot.data = LightData {
                point: Vec3::ZERO,
                orientation: Quat::IDENTITY,
                color: Vec3::new(state.color.x, state.color.y, state.color.z),
                intensity: state.intensity,
                range: state.range,
                angle: state.spot_angle,
            };
            slot.state = LightsBufferRendererState {
                renderable_index: state.renderable_index,
                global_unique_id: -1,
                shadow_strength: state.shadow_strength,
                shadow_near_plane: state.shadow_near_plane,
                shadow_map_resolution: state.shadow_map_resolution_override,
                shadow_bias: state.shadow_bias,
                shadow_normal_bias: state.shadow_normal_bias,
                cookie_texture_asset_id: state.cookie_texture_asset_id,
                light_type: state.r#type,
                shadow_type: state.shadow_type,
                _padding: [0; 2],
            };
        }

        self.rebuild_space_vec(space_id);
        self.mark_changed();
    }

    /// Rolls each cached light's `transform_id` forward through this frame's
    /// [`TransformRemovalEvent`]s so stored references follow a transform when it was swap-moved
    /// into a freed slot (matches the host's `RenderableIndex` reindexing). Must run *before* the
    /// frame's light add/remove/state apply so any new state rows land on the correct entry.
    ///
    /// Drops entries whose own transform was the one being removed (fixup returns `-1`) with a
    /// warning; a well-formed host stream won't produce that case because the light's own
    /// removal is sent in the same frame as its slot's transform removal, but this keeps the
    /// cache self-consistent if that invariant ever regresses.
    pub fn fixup_for_transform_removals(
        &mut self,
        space_id: i32,
        removals: &[TransformRemovalEvent],
    ) {
        if removals.is_empty() {
            return;
        }
        profiling::scope!("lights::fixup_for_transform_removals");
        // Sentinel marking an entry whose transform was removed outright — dropped during retain.
        const DEAD: usize = usize::MAX;

        let mut dirty = false;

        if let Some(v) = self.regular_lights.get_mut(&space_id) {
            for removal in removals {
                for light in v.iter_mut() {
                    if light.transform_id == DEAD {
                        continue;
                    }
                    let fixed = fixup_transform_id(
                        light.transform_id as i32,
                        removal.removed_index,
                        removal.last_index_before_swap,
                    );
                    if fixed < 0 {
                        light.transform_id = DEAD;
                        dirty = true;
                    } else if (fixed as usize) != light.transform_id {
                        light.transform_id = fixed as usize;
                        dirty = true;
                    }
                }
            }
            let before = v.len();
            v.retain(|l| {
                if l.transform_id == DEAD {
                    logger::warn!(
                        "light_cache: regular light dropped during transform-removal fixup (space_id={space_id})"
                    );
                    false
                } else {
                    true
                }
            });
            if v.len() != before {
                dirty = true;
            }
        }

        if let Some(v) = self.buffer_renderers.get_mut(&space_id) {
            for removal in removals {
                for br in v.iter_mut() {
                    if br.transform_id == DEAD {
                        continue;
                    }
                    let fixed = fixup_transform_id(
                        br.transform_id as i32,
                        removal.removed_index,
                        removal.last_index_before_swap,
                    );
                    if fixed < 0 {
                        br.transform_id = DEAD;
                        dirty = true;
                    } else if (fixed as usize) != br.transform_id {
                        br.transform_id = fixed as usize;
                        dirty = true;
                    }
                }
            }
            let before = v.len();
            v.retain(|br| {
                if br.transform_id == DEAD {
                    logger::warn!(
                        "light_cache: buffer renderer dropped during transform-removal fixup (space_id={space_id})"
                    );
                    false
                } else {
                    true
                }
            });
            if v.len() != before {
                dirty = true;
            }
        }

        if dirty {
            self.rebuild_space_vec(space_id);
            self.mark_changed();
        }
    }

    /// Cached lights for `space_id` after the last apply.
    pub fn get_lights_for_space(&self, space_id: i32) -> Option<&[CachedLight]> {
        self.spaces.get(&space_id).map(|v| v.as_slice())
    }

    /// Drops all light entries tied to a removed render space.
    pub fn remove_space(&mut self, space_id: i32) {
        self.spaces.remove(&space_id);
        self.regular_lights.remove(&space_id);
        self.buffer_renderers.remove(&space_id);
        self.mark_changed();
    }

    /// Resolves cached lights using space-local transform world matrices (caller composes root).
    pub fn resolve_lights(
        &self,
        space_id: i32,
        get_world_matrix: impl Fn(usize) -> Option<Mat4>,
    ) -> Vec<ResolvedLight> {
        let mut out = Vec::new();
        self.resolve_lights_into(space_id, get_world_matrix, &mut out);
        out
    }

    /// Like [`Self::resolve_lights`], but appends into `out` (caller clears when replacing content).
    pub fn resolve_lights_into(
        &self,
        space_id: i32,
        get_world_matrix: impl Fn(usize) -> Option<Mat4>,
        out: &mut Vec<ResolvedLight>,
    ) {
        profiling::scope!("lights::resolve_lights_into");
        let Some(lights) = self.get_lights_for_space(space_id) else {
            return;
        };

        out.reserve(lights.len());
        for cached in lights {
            let world = get_world_matrix(cached.transform_id).unwrap_or(Mat4::IDENTITY);

            let point = cached.data.point;
            let p = Vec3::new(point.x, point.y, point.z);
            let world_pos = world.transform_point3(p);

            let ori = cached.data.orientation;
            let q = ori;
            let world_dir = (world.to_scale_rotation_translation().1 * q) * LOCAL_LIGHT_PROPAGATION;
            let world_dir = if world_dir.length_squared() > 1e-10 {
                world_dir.normalize()
            } else {
                LOCAL_LIGHT_PROPAGATION
            };

            let color = cached.data.color;
            let color = Vec3::new(color.x, color.y, color.z);

            let range = if cached.state.global_unique_id >= 0 {
                let (scale, _, _) = world.to_scale_rotation_translation();
                let uniform_scale = (scale.x + scale.y + scale.z) / 3.0;
                cached.data.range * uniform_scale
            } else {
                cached.data.range
            };

            out.push(ResolvedLight {
                world_position: world_pos,
                world_direction: world_dir,
                color,
                intensity: cached.data.intensity,
                range,
                spot_angle: cached.data.angle,
                light_type: cached.state.light_type,
                global_unique_id: cached.state.global_unique_id,
                shadow_type: cached.state.shadow_type,
                shadow_strength: cached.state.shadow_strength,
                shadow_near_plane: cached.state.shadow_near_plane,
                shadow_bias: cached.state.shadow_bias,
                shadow_normal_bias: cached.state.shadow_normal_bias,
            });
        }
    }

    /// Alias for [`Self::resolve_lights`] kept for callers that distinguish the "with fallback" name.
    ///
    /// Raw buffer submissions are not renderable by themselves; a matching renderer state is required.
    pub fn resolve_lights_with_fallback(
        &self,
        space_id: i32,
        get_world_matrix: impl Fn(usize) -> Option<Mat4>,
    ) -> Vec<ResolvedLight> {
        self.resolve_lights(space_id, get_world_matrix)
    }

    /// Alias for [`Self::resolve_lights_into`] kept for callers that distinguish the "with fallback" name.
    pub fn resolve_lights_with_fallback_into(
        &self,
        space_id: i32,
        get_world_matrix: impl Fn(usize) -> Option<Mat4>,
        out: &mut Vec<ResolvedLight>,
    ) {
        self.resolve_lights_into(space_id, get_world_matrix, out);
    }
}

impl Default for LightCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use glam::{Mat4, Quat, Vec3};

    use crate::shared::{LightData, LightState, LightType, LightsBufferRendererState, ShadowType};

    use super::LightCache;

    fn make_light_data(pos: (f32, f32, f32), color: (f32, f32, f32)) -> LightData {
        LightData {
            point: Vec3::new(pos.0, pos.1, pos.2),
            orientation: Quat::IDENTITY,
            color: Vec3::new(color.0, color.1, color.2),
            intensity: 1.0,
            range: 10.0,
            angle: 45.0,
        }
    }

    fn make_state(
        renderable_index: i32,
        global_unique_id: i32,
        light_type: LightType,
    ) -> LightsBufferRendererState {
        LightsBufferRendererState {
            renderable_index,
            global_unique_id,
            shadow_strength: 0.0,
            shadow_near_plane: 0.0,
            shadow_map_resolution: 0,
            shadow_bias: 0.0,
            shadow_normal_bias: 0.0,
            cookie_texture_asset_id: -1,
            light_type,
            shadow_type: ShadowType::None,
            _padding: [0; 2],
        }
    }

    fn make_regular_state(renderable_index: i32, intensity: f32, range: f32) -> LightState {
        LightState {
            renderable_index,
            intensity,
            range,
            spot_angle: 45.0,
            color: glam::Vec4::new(1.0, 1.0, 1.0, 1.0),
            shadow_strength: 0.0,
            shadow_near_plane: 0.0,
            shadow_map_resolution_override: 0,
            shadow_bias: 0.0,
            shadow_normal_bias: 0.0,
            cookie_texture_asset_id: -1,
            r#type: LightType::Point,
            shadow_type: ShadowType::None,
            _padding: [0; 2],
        }
    }

    #[test]
    fn light_cache_store_full_and_apply_additions() {
        let mut cache = LightCache::new();
        let space_id = 0;
        let light_data = vec![
            make_light_data((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            make_light_data((0.0, 2.0, 0.0), (0.0, 1.0, 0.0)),
        ];
        cache.store_full(100, light_data);

        let additions: Vec<i32> = vec![0];
        let states = vec![make_state(0, 100, LightType::Point)];
        cache.apply_update(space_id, &[], &additions, &states);

        let lights = cache
            .get_lights_for_space(space_id)
            .expect("test setup: space should have lights");
        assert_eq!(lights.len(), 2);
        assert_eq!(lights[0].data.point.x, 1.0);
        assert_eq!(lights[0].state.global_unique_id, 100);
        assert_eq!(lights[1].data.point.y, 2.0);
        assert_eq!(lights[1].state.light_type, LightType::Point);
    }

    #[test]
    fn store_full_refreshes_existing_buffer_contributions() {
        let mut cache = LightCache::new();
        let space_id = 0;
        cache.store_full(100, vec![make_light_data((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))]);
        cache.apply_update(space_id, &[], &[0], &[make_state(0, 100, LightType::Point)]);

        cache.store_full(100, vec![make_light_data((2.0, 0.0, 0.0), (0.0, 1.0, 0.0))]);

        let lights = cache
            .get_lights_for_space(space_id)
            .expect("test setup: space should have lights");
        assert_eq!(lights.len(), 1);
        assert!((lights[0].data.point.x - 2.0).abs() < 1e-5);
        assert!((lights[0].data.color.y - 1.0).abs() < 1e-5);
        assert_eq!(lights[0].state.global_unique_id, 100);
    }

    #[test]
    fn store_full_after_state_creates_buffer_contributions() {
        let mut cache = LightCache::new();
        let space_id = 0;
        cache.apply_update(space_id, &[], &[0], &[make_state(0, 100, LightType::Point)]);
        assert_eq!(
            cache
                .get_lights_for_space(space_id)
                .expect("test setup: space should exist")
                .len(),
            0
        );

        cache.store_full(100, vec![make_light_data((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))]);

        let lights = cache
            .get_lights_for_space(space_id)
            .expect("test setup: space should have lights");
        assert_eq!(lights.len(), 1);
        assert!((lights[0].data.color.y - 1.0).abs() < 1e-5);
        assert_eq!(lights[0].transform_id, 0);
    }

    #[test]
    fn light_cache_version_changes_on_light_mutations() {
        let mut cache = LightCache::new();
        let version0 = cache.version();

        cache.store_full(100, vec![make_light_data((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))]);
        let version1 = cache.version();
        assert_ne!(version0, version1);

        cache.apply_update(0, &[], &[0], &[make_state(0, 100, LightType::Point)]);
        let version2 = cache.version();
        assert_ne!(version1, version2);

        cache.apply_update(0, &[0], &[], &[]);
        assert_ne!(version2, cache.version());
    }

    #[test]
    fn apply_update_replacing_renderable_guid_removes_previous_buffer() {
        let mut cache = LightCache::new();
        let space_id = 0;
        cache.store_full(100, vec![make_light_data((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))]);
        cache.store_full(200, vec![make_light_data((2.0, 0.0, 0.0), (0.0, 1.0, 0.0))]);

        cache.apply_update(space_id, &[], &[0], &[make_state(0, 100, LightType::Point)]);
        cache.apply_update(space_id, &[], &[], &[make_state(0, 200, LightType::Point)]);

        let lights = cache
            .get_lights_for_space(space_id)
            .expect("test setup: space should have lights");
        assert_eq!(lights.len(), 1);
        assert_eq!(lights[0].state.global_unique_id, 200);
        assert!((lights[0].data.color.y - 1.0).abs() < 1e-5);
    }

    #[test]
    fn light_cache_removals() {
        let mut cache = LightCache::new();
        let space_id = 0;
        cache.store_full(100, vec![make_light_data((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))]);
        cache.store_full(101, vec![make_light_data((0.0, 2.0, 0.0), (0.0, 1.0, 0.0))]);
        cache.store_full(102, vec![make_light_data((0.0, 0.0, 3.0), (0.0, 0.0, 1.0))]);

        let additions: Vec<i32> = vec![0, 1, 2];
        let states = vec![
            make_state(0, 100, LightType::Point),
            make_state(1, 101, LightType::Point),
            make_state(2, 102, LightType::Point),
        ];
        cache.apply_update(space_id, &[], &additions, &states);
        assert_eq!(
            cache
                .get_lights_for_space(space_id)
                .expect("test setup: space should have lights")
                .len(),
            3
        );

        cache.apply_update(space_id, &[1], &[], &[]);
        let lights = cache
            .get_lights_for_space(space_id)
            .expect("test setup: space should have lights");
        assert_eq!(lights.len(), 2);
        // Swap-remove of index 1 moves the last entry (guid=102) into slot 1. Space vec is
        // rebuilt from the dense list so output order is [guid=100, guid=102].
        assert_eq!(lights[0].state.global_unique_id, 100);
        assert_eq!(lights[1].state.global_unique_id, 102);
    }

    #[test]
    fn light_cache_resolve_world_space() {
        let mut cache = LightCache::new();
        let space_id = 0;
        cache.store_full(100, vec![make_light_data((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))]);

        let additions: Vec<i32> = vec![0];
        let states = vec![make_state(0, 100, LightType::Point)];
        cache.apply_update(space_id, &[], &additions, &states);

        let world_matrix = Mat4::from_translation(Vec3::new(10.0, 0.0, 0.0));
        let resolved =
            cache.resolve_lights(
                space_id,
                |tid| {
                    if tid == 0 {
                        Some(world_matrix)
                    } else {
                        None
                    }
                },
            );

        assert_eq!(resolved.len(), 1);
        assert!((resolved[0].world_position.x - 11.0).abs() < 1e-5);
        assert!((resolved[0].world_position.y - 0.0).abs() < 1e-5);
        assert!((resolved[0].world_position.z - 0.0).abs() < 1e-5);
    }

    #[test]
    fn resolve_lights_with_fallback_does_not_synthesize_raw_buffers() {
        let mut cache = LightCache::new();
        let space_id = 0;
        let light_data = vec![
            make_light_data((5.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            make_light_data((0.0, 3.0, 0.0), (0.0, 1.0, 0.0)),
        ];
        cache.store_full(space_id, light_data);

        let resolved = cache.resolve_lights_with_fallback(space_id, |_| None);

        assert!(
            resolved.is_empty(),
            "raw LightData buffers require an active LightsBufferRendererState before rendering"
        );
    }

    #[test]
    fn gpu_light_from_resolved_point() {
        let mut cache = LightCache::new();
        let space_id = 0;
        cache.store_full(100, vec![make_light_data((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))]);
        cache.apply_update(space_id, &[], &[0], &[make_state(0, 100, LightType::Point)]);
        let resolved = cache.resolve_lights(space_id, |_| Some(Mat4::IDENTITY));
        assert_eq!(resolved.len(), 1);
        let gpu = crate::backend::GpuLight::from_resolved(&resolved[0]);
        assert_eq!(gpu.light_type, 0);
        assert!((gpu.position[0] - 1.0).abs() < 1e-5);
    }

    /// Regression: removing a middle buffer-renderer slot must swap-remove the last entry into
    /// the freed index so the dense list stays aligned with the host's swap-remove reindexing.
    #[test]
    fn buffer_renderer_swap_remove_middle_preserves_last_entry() {
        let mut cache = LightCache::new();
        let space_id = 0;
        cache.store_full(100, vec![make_light_data((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))]);
        cache.store_full(101, vec![make_light_data((0.0, 2.0, 0.0), (0.0, 1.0, 0.0))]);
        cache.store_full(102, vec![make_light_data((0.0, 0.0, 3.0), (0.0, 0.0, 1.0))]);

        cache.apply_update(
            space_id,
            &[],
            &[10, 11, 12],
            &[
                make_state(0, 100, LightType::Point),
                make_state(1, 101, LightType::Point),
                make_state(2, 102, LightType::Point),
            ],
        );

        cache.apply_update(space_id, &[1], &[], &[]);

        let lights = cache
            .get_lights_for_space(space_id)
            .expect("test setup: space should have lights");
        assert_eq!(lights.len(), 2);
        assert_eq!(lights[0].state.global_unique_id, 100);
        assert_eq!(lights[0].transform_id, 10);
        // Swapped-in entry keeps its original transform id and buffer guid.
        assert_eq!(lights[1].state.global_unique_id, 102);
        assert_eq!(lights[1].transform_id, 12);
    }

    /// Regression: after removing and re-adding, the dense list must have exactly one entry per
    /// live renderable with its provided transform id — no ghost carrying `transform_id = 0`.
    #[test]
    fn buffer_renderer_remove_then_add_has_no_ghost() {
        let mut cache = LightCache::new();
        let space_id = 0;
        cache.store_full(100, vec![make_light_data((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))]);
        cache.store_full(101, vec![make_light_data((0.0, 2.0, 0.0), (0.0, 1.0, 0.0))]);
        cache.apply_update(
            space_id,
            &[],
            &[10, 11],
            &[
                make_state(0, 100, LightType::Point),
                make_state(1, 101, LightType::Point),
            ],
        );

        cache.store_full(102, vec![make_light_data((0.0, 0.0, 3.0), (0.0, 0.0, 1.0))]);
        cache.apply_update(
            space_id,
            &[0],
            &[12],
            &[make_state(1, 102, LightType::Point)],
        );

        let lights = cache
            .get_lights_for_space(space_id)
            .expect("test setup: space should have lights");
        assert_eq!(lights.len(), 2);
        assert_eq!(lights[0].state.global_unique_id, 101);
        assert_eq!(lights[0].transform_id, 11);
        assert_eq!(lights[1].state.global_unique_id, 102);
        assert_eq!(lights[1].transform_id, 12);
    }

    /// Regression: a state-only update after a swap-remove must apply to the swapped entry —
    /// the original `transform_id` must be preserved (no fallback to `transform_id = 0`, which
    /// would render the light at the world origin).
    #[test]
    fn buffer_renderer_state_only_update_after_swap_uses_swapped_transform() {
        let mut cache = LightCache::new();
        let space_id = 0;
        cache.store_full(100, vec![make_light_data((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))]);
        cache.store_full(101, vec![make_light_data((0.0, 2.0, 0.0), (0.0, 1.0, 0.0))]);
        cache.apply_update(
            space_id,
            &[],
            &[10, 11],
            &[
                make_state(0, 100, LightType::Point),
                make_state(1, 101, LightType::Point),
            ],
        );

        cache.apply_update(space_id, &[0], &[], &[]);

        cache.apply_update(
            space_id,
            &[],
            &[],
            &[make_state(0, 101, LightType::Directional)],
        );

        let lights = cache
            .get_lights_for_space(space_id)
            .expect("test setup: space should have lights");
        assert_eq!(lights.len(), 1);
        assert_eq!(lights[0].state.global_unique_id, 101);
        assert_eq!(lights[0].state.light_type, LightType::Directional);
        // Critical: transform id is preserved, not reset to 0 (which would put the light at origin).
        assert_eq!(lights[0].transform_id, 11);
    }

    /// Regression: the same swap-remove contract must hold for regular (Unity `Light`)
    /// renderables — the reported bug was about a Light component added via the component picker
    /// ending up with `renderable_index = -1` / at the world origin after add/remove cycles.
    #[test]
    fn regular_light_swap_remove_middle_preserves_last_entry() {
        let mut cache = LightCache::new();
        let space_id = 0;
        cache.apply_regular_lights_update(
            space_id,
            &[],
            &[10, 11, 12],
            &[
                make_regular_state(0, 1.0, 10.0),
                make_regular_state(1, 2.0, 20.0),
                make_regular_state(2, 3.0, 30.0),
            ],
        );

        cache.apply_regular_lights_update(space_id, &[1], &[], &[]);

        let lights = cache
            .get_lights_for_space(space_id)
            .expect("test setup: space should have lights");
        assert_eq!(lights.len(), 2);
        assert!((lights[0].data.intensity - 1.0).abs() < 1e-5);
        assert_eq!(lights[0].transform_id, 10);
        // Former index 2 swapped into index 1; its transform id and state follow it.
        assert!((lights[1].data.intensity - 3.0).abs() < 1e-5);
        assert_eq!(lights[1].transform_id, 12);
    }

    #[test]
    fn regular_light_remove_then_add_has_no_ghost() {
        let mut cache = LightCache::new();
        let space_id = 0;
        cache.apply_regular_lights_update(
            space_id,
            &[],
            &[10, 11],
            &[
                make_regular_state(0, 1.0, 10.0),
                make_regular_state(1, 2.0, 20.0),
            ],
        );

        cache.apply_regular_lights_update(
            space_id,
            &[0],
            &[12],
            &[make_regular_state(1, 3.0, 30.0)],
        );

        let lights = cache
            .get_lights_for_space(space_id)
            .expect("test setup: space should have lights");
        assert_eq!(lights.len(), 2);
        // Former index 1 (intensity 2.0, transform 11) swapped into index 0.
        assert!((lights[0].data.intensity - 2.0).abs() < 1e-5);
        assert_eq!(lights[0].transform_id, 11);
        // New renderable at index 1 with transform id 12.
        assert!((lights[1].data.intensity - 3.0).abs() < 1e-5);
        assert_eq!(lights[1].transform_id, 12);
    }

    /// Regression: reproduces the "light at origin" symptom — a state update for a swapped
    /// regular light must address the swapped entry and preserve its non-zero transform id.
    #[test]
    fn regular_light_state_only_update_after_swap_uses_swapped_transform() {
        let mut cache = LightCache::new();
        let space_id = 0;
        cache.apply_regular_lights_update(
            space_id,
            &[],
            &[10, 11],
            &[
                make_regular_state(0, 1.0, 10.0),
                make_regular_state(1, 2.0, 20.0),
            ],
        );

        cache.apply_regular_lights_update(space_id, &[0], &[], &[]);

        cache.apply_regular_lights_update(space_id, &[], &[], &[make_regular_state(0, 5.0, 50.0)]);

        let lights = cache
            .get_lights_for_space(space_id)
            .expect("test setup: space should have lights");
        assert_eq!(lights.len(), 1);
        assert!((lights[0].data.intensity - 5.0).abs() < 1e-5);
        // Critical: the swapped entry keeps its original transform id, not the `0` origin fallback.
        assert_eq!(lights[0].transform_id, 11);
    }

    /// Regression: when an unrelated transform elsewhere in the space is swap-removed and the
    /// host moves the last transform into the freed slot, the light's stored `transform_id`
    /// must be rolled forward so it still points at its slot's transform. Without the fixup,
    /// `get_world_matrix` returns `None` and the light falls back to `Mat4::IDENTITY` → visible
    /// at world origin. Reproduces the user-reported bug directly.
    #[test]
    fn regular_light_transform_id_follows_swap_remove() {
        use crate::scene::transforms_apply::TransformRemovalEvent;

        let mut cache = LightCache::new();
        let space_id = 0;
        // Two lights: one at transform 5 (some unrelated slot), one at transform 42 (which
        // happens to be the last transform in the dense list on the host).
        cache.apply_regular_lights_update(
            space_id,
            &[],
            &[5, 42],
            &[
                make_regular_state(0, 1.0, 10.0),
                make_regular_state(1, 2.0, 20.0),
            ],
        );

        // Simulate: transform 10 was removed on the host (unrelated slot lost its last
        // renderable); the last transform (index 42) was swap-moved into slot 10.
        cache.fixup_for_transform_removals(
            space_id,
            &[TransformRemovalEvent {
                removed_index: 10,
                last_index_before_swap: 42,
            }],
        );

        let lights = cache
            .get_lights_for_space(space_id)
            .expect("test setup: space should have lights");
        assert_eq!(lights.len(), 2);
        // Unrelated light keeps transform 5.
        assert_eq!(lights[0].transform_id, 5);
        assert!((lights[0].data.intensity - 1.0).abs() < 1e-5);
        // The light whose transform was at the pre-swap last index (42) now points at the
        // freed slot (10). Without the fixup, this would still read 42 → out of range → origin.
        assert_eq!(lights[1].transform_id, 10);
        assert!((lights[1].data.intensity - 2.0).abs() < 1e-5);
    }

    /// Regression: same swap-remove fixup for buffer-renderer lights.
    #[test]
    fn buffer_renderer_transform_id_follows_swap_remove() {
        use crate::scene::transforms_apply::TransformRemovalEvent;

        let mut cache = LightCache::new();
        let space_id = 0;
        cache.store_full(100, vec![make_light_data((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))]);
        cache.store_full(101, vec![make_light_data((0.0, 2.0, 0.0), (0.0, 1.0, 0.0))]);
        cache.apply_update(
            space_id,
            &[],
            &[5, 42],
            &[
                make_state(0, 100, LightType::Point),
                make_state(1, 101, LightType::Point),
            ],
        );

        cache.fixup_for_transform_removals(
            space_id,
            &[TransformRemovalEvent {
                removed_index: 10,
                last_index_before_swap: 42,
            }],
        );

        let lights = cache
            .get_lights_for_space(space_id)
            .expect("test setup: space should have lights");
        assert_eq!(lights.len(), 2);
        // Unrelated renderer (transform 5, buffer 100) is untouched.
        assert_eq!(lights[0].state.global_unique_id, 100);
        assert_eq!(lights[0].transform_id, 5);
        // Swapped-in renderer follows: transform 42 → 10.
        assert_eq!(lights[1].state.global_unique_id, 101);
        assert_eq!(lights[1].transform_id, 10);
    }

    /// Regression: when a light's OWN transform is removed (its slot lost all renderables),
    /// the fixup drops it defensively. In a real host stream the light would also be in the
    /// same frame's removals array; this guards against that invariant regressing.
    #[test]
    fn regular_light_whose_own_transform_was_removed_is_dropped() {
        use crate::scene::transforms_apply::TransformRemovalEvent;

        let mut cache = LightCache::new();
        let space_id = 0;
        cache.apply_regular_lights_update(
            space_id,
            &[],
            &[5, 42],
            &[
                make_regular_state(0, 1.0, 10.0),
                make_regular_state(1, 2.0, 20.0),
            ],
        );

        // Remove transform 5 (which is the first light's transform). last_index_before_swap = 42
        // → the second light's transform moves into slot 5.
        cache.fixup_for_transform_removals(
            space_id,
            &[TransformRemovalEvent {
                removed_index: 5,
                last_index_before_swap: 42,
            }],
        );

        let lights = cache
            .get_lights_for_space(space_id)
            .expect("test setup: space should have lights");
        assert_eq!(lights.len(), 1);
        assert_eq!(lights[0].transform_id, 5);
        assert!((lights[0].data.intensity - 2.0).abs() < 1e-5);
    }

    /// Regression: an unrelated transform removal (not touching any light) must leave every
    /// cached light's `transform_id` untouched.
    #[test]
    fn light_fixup_ignores_unrelated_removals() {
        use crate::scene::transforms_apply::TransformRemovalEvent;

        let mut cache = LightCache::new();
        let space_id = 0;
        cache.apply_regular_lights_update(
            space_id,
            &[],
            &[10, 11],
            &[
                make_regular_state(0, 1.0, 10.0),
                make_regular_state(1, 2.0, 20.0),
            ],
        );

        cache.fixup_for_transform_removals(
            space_id,
            &[TransformRemovalEvent {
                removed_index: 5,
                last_index_before_swap: 42,
            }],
        );

        let lights = cache
            .get_lights_for_space(space_id)
            .expect("test setup: space should have lights");
        assert_eq!(lights.len(), 2);
        assert_eq!(lights[0].transform_id, 10);
        assert_eq!(lights[1].transform_id, 11);
    }
}
