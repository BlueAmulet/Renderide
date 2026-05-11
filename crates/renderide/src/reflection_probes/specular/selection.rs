use glam::{Vec3, Vec3A};
use hashbrown::HashMap;

use crate::scene::RenderSpaceId;

/// Maximum number of probes in one BVH leaf.
const BVH_LEAF_SIZE: usize = 8;
/// Minimum object volume used when normalizing intersection weights.
const MIN_OBJECT_VOLUME: f32 = 1e-12;
const CONTAINMENT_EPSILON: f32 = 1e-5;

/// Per-draw reflection-probe selection stored in the per-draw slab.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ReflectionProbeDrawSelection {
    /// First selected reflection-probe atlas index.
    pub first_atlas_index: u16,
    /// Second selected reflection-probe atlas index.
    pub second_atlas_index: u16,
    /// Blend weight for [`Self::second_atlas_index`].
    pub second_weight: f32,
    /// Number of selected probes, clamped to two.
    pub hit_count: u8,
}

impl ReflectionProbeDrawSelection {
    /// Builds a single-probe selection.
    #[must_use]
    pub fn one(first_atlas_index: u16) -> Self {
        Self {
            first_atlas_index,
            second_atlas_index: 0,
            second_weight: 0.0,
            hit_count: 1,
        }
    }

    /// Builds a two-probe selection.
    #[must_use]
    pub fn two(first_atlas_index: u16, second_atlas_index: u16, second_weight: f32) -> Self {
        Self {
            first_atlas_index,
            second_atlas_index,
            second_weight: second_weight.clamp(0.0, 1.0),
            hit_count: 2,
        }
    }
}

/// CPU-side selector snapshot used during world-mesh draw collection.
#[derive(Default)]
pub struct ReflectionProbeFrameSelection {
    spaces: HashMap<RenderSpaceId, ReflectionProbeSpatialIndex>,
    skybox_fallback_slots: HashMap<RenderSpaceId, u16>,
}

impl ReflectionProbeFrameSelection {
    /// Selects up to two probes for one object AABB.
    #[must_use]
    pub fn select(
        &self,
        space_id: RenderSpaceId,
        object_aabb: (Vec3, Vec3),
    ) -> ReflectionProbeDrawSelection {
        if let Some(selection) = self
            .spaces
            .get(&space_id)
            .map(|index| index.select(object_aabb))
            && selection.hit_count > 0
        {
            return selection;
        }
        self.fallback(space_id)
    }

    /// Returns the render-space skybox fallback selection, if its specular IBL is ready.
    #[must_use]
    pub fn fallback(&self, space_id: RenderSpaceId) -> ReflectionProbeDrawSelection {
        self.skybox_fallback_slots
            .get(&space_id)
            .copied()
            .filter(|slot| *slot != 0)
            .map_or_else(ReflectionProbeDrawSelection::default, |slot| {
                ReflectionProbeDrawSelection {
                    first_atlas_index: slot,
                    second_atlas_index: 0,
                    second_weight: 0.0,
                    hit_count: 0,
                }
            })
    }

    pub(super) fn rebuild_spatial<I, J>(&mut self, probes: I, skybox_fallback_slots: J)
    where
        I: IntoIterator<Item = (RenderSpaceId, SpatialProbe)>,
        J: IntoIterator<Item = (RenderSpaceId, u16)>,
    {
        self.spaces.clear();
        self.skybox_fallback_slots.clear();
        self.skybox_fallback_slots.extend(
            skybox_fallback_slots
                .into_iter()
                .filter(|(_, slot)| *slot != 0),
        );
        let mut by_space: HashMap<RenderSpaceId, Vec<SpatialProbe>> = HashMap::new();
        for (space_id, probe) in probes {
            by_space.entry(space_id).or_default().push(probe);
        }
        for (space_id, probes) in by_space {
            self.spaces
                .insert(space_id, ReflectionProbeSpatialIndex::build(probes));
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct SpatialProbe {
    pub(super) renderable_index: i32,
    pub(super) atlas_index: u16,
    pub(super) importance: i32,
    pub(super) aabb_min: Vec3A,
    pub(super) aabb_max: Vec3A,
    pub(super) center: Vec3A,
    pub(super) volume: f32,
}

/// A BVH over reflection-probe AABBs for one render space.
#[derive(Default)]
pub struct ReflectionProbeSpatialIndex {
    probes: Vec<SpatialProbe>,
    order: Vec<usize>,
    nodes: Vec<BvhNode>,
    root: Option<usize>,
}

impl ReflectionProbeSpatialIndex {
    pub(super) fn build(probes: Vec<SpatialProbe>) -> Self {
        let mut out = Self {
            order: (0..probes.len()).collect(),
            probes,
            nodes: Vec::new(),
            root: None,
        };
        if !out.probes.is_empty() {
            let mut order = std::mem::take(&mut out.order);
            let end = order.len();
            out.root = Some(out.build_node(&mut order, 0, end));
            out.order = order;
        }
        out
    }

    /// Selects up to two probes for one object AABB.
    #[must_use]
    pub fn select(&self, object_aabb: (Vec3, Vec3)) -> ReflectionProbeDrawSelection {
        let object_min = Vec3A::from(object_aabb.0);
        let object_max = Vec3A::from(object_aabb.1);
        if self.root.is_none() || !aabb_valid(object_aabb.0, object_aabb.1) {
            return ReflectionProbeDrawSelection::default();
        }
        let object_center = object_center(object_min, object_max);
        let object_volume = aabb_volume_vec3a(object_min, object_max);
        let mut best_importance = i32::MIN;
        let mut top: [Option<ProbeScore>; 2] = [None, None];
        let mut stack = Vec::with_capacity(64);
        stack.push(self.root.unwrap_or(0));
        while let Some(node_index) = stack.pop() {
            let node = self.nodes[node_index];
            if !aabb_intersects(node.aabb_min, node.aabb_max, object_min, object_max) {
                continue;
            }
            if node.count > 0 {
                for &probe_index in &self.order[node.start..node.start + node.count] {
                    let probe = &self.probes[probe_index];
                    if !aabb_intersects(probe.aabb_min, probe.aabb_max, object_min, object_max) {
                        continue;
                    }
                    let intersection = intersection_volume_vec3a(
                        probe.aabb_min,
                        probe.aabb_max,
                        object_min,
                        object_max,
                    );
                    if intersection <= 0.0 {
                        continue;
                    }
                    if probe.importance > best_importance {
                        best_importance = probe.importance;
                        top = [None, None];
                    }
                    if probe.importance != best_importance {
                        continue;
                    }
                    insert_probe_score(
                        &mut top,
                        ProbeScore {
                            atlas_index: probe.atlas_index,
                            intersection,
                            probe_volume: probe.volume,
                            center_distance_sq: (probe.center - object_center).length_squared(),
                            renderable_index: probe.renderable_index,
                            aabb_min: probe.aabb_min,
                            aabb_max: probe.aabb_max,
                        },
                    );
                }
            } else {
                stack.push(node.left);
                stack.push(node.right);
            }
        }
        selection_from_scores(top, object_volume)
    }

    fn build_node(&mut self, order: &mut [usize], start: usize, end: usize) -> usize {
        let (aabb_min, aabb_max) = bounds_for_order(&self.probes, &order[start..end]);
        let index = self.nodes.len();
        self.nodes.push(BvhNode {
            aabb_min,
            aabb_max,
            start,
            count: 0,
            left: 0,
            right: 0,
        });
        let count = end - start;
        if count <= BVH_LEAF_SIZE {
            self.nodes[index].count = count;
            return index;
        }
        let axis = largest_axis(aabb_max - aabb_min);
        order[start..end].sort_unstable_by(|&a, &b| {
            let ac = axis_value(self.probes[a].center, axis);
            let bc = axis_value(self.probes[b].center, axis);
            ac.total_cmp(&bc).then_with(|| {
                self.probes[a]
                    .renderable_index
                    .cmp(&self.probes[b].renderable_index)
            })
        });
        let mid = start + count / 2;
        let left = self.build_node(order, start, mid);
        let right = self.build_node(order, mid, end);
        self.nodes[index].left = left;
        self.nodes[index].right = right;
        index
    }
}

#[derive(Clone, Copy)]
struct BvhNode {
    aabb_min: Vec3A,
    aabb_max: Vec3A,
    start: usize,
    count: usize,
    left: usize,
    right: usize,
}

#[derive(Clone, Copy, Debug)]
struct ProbeScore {
    atlas_index: u16,
    intersection: f32,
    probe_volume: f32,
    center_distance_sq: f32,
    renderable_index: i32,
    aabb_min: Vec3A,
    aabb_max: Vec3A,
}

fn insert_probe_score(top: &mut [Option<ProbeScore>; 2], score: ProbeScore) {
    if top[0].is_none_or(|best| score_better(score, best)) {
        top[1] = top[0];
        top[0] = Some(score);
    } else if top[1].is_none_or(|second| score_better(score, second)) {
        top[1] = Some(score);
    }
}

fn score_better(a: ProbeScore, b: ProbeScore) -> bool {
    a.intersection
        .total_cmp(&b.intersection)
        .reverse()
        .then_with(|| a.probe_volume.total_cmp(&b.probe_volume))
        .then_with(|| a.center_distance_sq.total_cmp(&b.center_distance_sq))
        .then_with(|| a.renderable_index.cmp(&b.renderable_index))
        .is_lt()
}

fn selection_from_scores(
    top: [Option<ProbeScore>; 2],
    object_volume: f32,
) -> ReflectionProbeDrawSelection {
    let Some(first) = top[0] else {
        return ReflectionProbeDrawSelection::default();
    };
    let Some(second) = top[1] else {
        return ReflectionProbeDrawSelection::one(first.atlas_index);
    };
    if let Some(selection) = contained_selection(first, second, object_volume) {
        return selection;
    }
    let denom = first.intersection + second.intersection;
    if denom <= MIN_OBJECT_VOLUME {
        return ReflectionProbeDrawSelection::one(first.atlas_index);
    }
    ReflectionProbeDrawSelection::two(
        first.atlas_index,
        second.atlas_index,
        second.intersection / denom,
    )
}

fn contained_selection(
    first: ProbeScore,
    second: ProbeScore,
    object_volume: f32,
) -> Option<ReflectionProbeDrawSelection> {
    if larger_probe_contains(first, second) {
        return Some(inner_outer_selection(second, first, object_volume));
    }
    if larger_probe_contains(second, first) {
        return Some(inner_outer_selection(first, second, object_volume));
    }
    None
}

fn larger_probe_contains(outer: ProbeScore, inner: ProbeScore) -> bool {
    outer.probe_volume > inner.probe_volume
        && aabb_contains(
            outer.aabb_min,
            outer.aabb_max,
            inner.aabb_min,
            inner.aabb_max,
        )
}

fn inner_outer_selection(
    inner: ProbeScore,
    outer: ProbeScore,
    object_volume: f32,
) -> ReflectionProbeDrawSelection {
    if object_volume <= MIN_OBJECT_VOLUME {
        return ReflectionProbeDrawSelection::one(inner.atlas_index);
    }
    let inner_weight = (inner.intersection / object_volume).clamp(0.0, 1.0);
    if inner_weight >= 1.0 - CONTAINMENT_EPSILON {
        return ReflectionProbeDrawSelection::one(inner.atlas_index);
    }
    ReflectionProbeDrawSelection::two(inner.atlas_index, outer.atlas_index, 1.0 - inner_weight)
}

fn bounds_for_order(probes: &[SpatialProbe], order: &[usize]) -> (Vec3A, Vec3A) {
    let mut min = Vec3A::splat(f32::INFINITY);
    let mut max = Vec3A::splat(f32::NEG_INFINITY);
    for &index in order {
        min = min.min(probes[index].aabb_min);
        max = max.max(probes[index].aabb_max);
    }
    (min, max)
}

fn aabb_intersects(a_min: Vec3A, a_max: Vec3A, b_min: Vec3A, b_max: Vec3A) -> bool {
    a_min.cmple(b_max).all() && a_max.cmpge(b_min).all()
}

fn aabb_contains(outer_min: Vec3A, outer_max: Vec3A, inner_min: Vec3A, inner_max: Vec3A) -> bool {
    let epsilon = Vec3A::splat(CONTAINMENT_EPSILON);
    outer_min.cmple(inner_min + epsilon).all() && outer_max.cmpge(inner_max - epsilon).all()
}

pub(super) fn aabb_valid(min: Vec3, max: Vec3) -> bool {
    min.is_finite() && max.is_finite() && (max - min).cmpgt(Vec3::ZERO).all()
}

pub(super) fn aabb_volume(min: Vec3, max: Vec3) -> f32 {
    aabb_volume_vec3a(Vec3A::from(min), Vec3A::from(max))
}

fn aabb_volume_vec3a(min: Vec3A, max: Vec3A) -> f32 {
    let d = (max - min).max(Vec3A::ZERO);
    d.x * d.y * d.z
}

pub(super) fn intersection_volume_vec3a(
    a_min: Vec3A,
    a_max: Vec3A,
    b_min: Vec3A,
    b_max: Vec3A,
) -> f32 {
    let d = (a_max.min(b_max) - a_min.max(b_min)).max(Vec3A::ZERO);
    d.x * d.y * d.z
}

fn object_center(min: Vec3A, max: Vec3A) -> Vec3A {
    (min + max) * 0.5
}

fn largest_axis(v: Vec3A) -> usize {
    if v.x >= v.y && v.x >= v.z {
        0
    } else if v.y >= v.z {
        1
    } else {
        2
    }
}

fn axis_value(v: Vec3A, axis: usize) -> f32 {
    match axis {
        0 => v.x,
        1 => v.y,
        _ => v.z,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probe(index: i32, atlas: u16, importance: i32, min: Vec3, max: Vec3) -> SpatialProbe {
        SpatialProbe {
            renderable_index: index,
            atlas_index: atlas,
            importance,
            aabb_min: Vec3A::from(min),
            aabb_max: Vec3A::from(max),
            center: Vec3A::from((min + max) * 0.5),
            volume: aabb_volume(min, max),
        }
    }

    #[test]
    fn higher_priority_overrides_lower_priority() {
        let index = ReflectionProbeSpatialIndex::build(vec![
            probe(0, 1, 0, Vec3::splat(-100.0), Vec3::splat(100.0)),
            probe(1, 2, 1, Vec3::splat(-1.0), Vec3::splat(1.0)),
        ]);

        let selection = index.select((Vec3::splat(-0.25), Vec3::splat(0.25)));

        assert_eq!(selection, ReflectionProbeDrawSelection::one(2));
    }

    #[test]
    fn frame_selection_uses_skybox_fallback_when_no_probe_hits() {
        let mut selection = ReflectionProbeFrameSelection::default();
        let space_id = RenderSpaceId(7);
        selection.rebuild_spatial(Vec::new(), [(space_id, 9)]);

        let draw = selection.select(space_id, (Vec3::splat(-1.0), Vec3::splat(1.0)));

        assert_eq!(
            draw,
            ReflectionProbeDrawSelection {
                first_atlas_index: 9,
                second_atlas_index: 0,
                second_weight: 0.0,
                hit_count: 0,
            }
        );
    }

    #[test]
    fn frame_selection_prefers_probe_hit_over_skybox_fallback() {
        let mut selection = ReflectionProbeFrameSelection::default();
        let space_id = RenderSpaceId(7);
        selection.rebuild_spatial(
            [(
                space_id,
                probe(0, 3, 1, Vec3::splat(-1.0), Vec3::splat(1.0)),
            )],
            [(space_id, 9)],
        );

        let draw = selection.select(space_id, (Vec3::splat(-0.5), Vec3::splat(0.5)));

        assert_eq!(draw, ReflectionProbeDrawSelection::one(3));
    }

    #[test]
    fn same_importance_selects_two_by_intersection_volume() {
        let index = ReflectionProbeSpatialIndex::build(vec![
            probe(
                0,
                1,
                1,
                Vec3::new(-1.0, -1.0, -1.0),
                Vec3::new(1.0, 1.0, 1.0),
            ),
            probe(
                1,
                2,
                1,
                Vec3::new(0.0, -1.0, -1.0),
                Vec3::new(2.0, 1.0, 1.0),
            ),
            probe(
                2,
                3,
                1,
                Vec3::new(0.75, -1.0, -1.0),
                Vec3::new(2.0, 1.0, 1.0),
            ),
        ]);

        let selection = index.select((Vec3::new(-0.5, -0.5, -0.5), Vec3::new(1.5, 0.5, 0.5)));

        assert_eq!(selection.hit_count, 2);
        assert_eq!(selection.first_atlas_index, 1);
        assert_eq!(selection.second_atlas_index, 2);
        assert!((selection.second_weight - 0.5).abs() < 1e-6);
    }

    #[test]
    fn contained_same_importance_probe_selects_inner_when_object_fully_inside() {
        let index = ReflectionProbeSpatialIndex::build(vec![
            probe(0, 1, 1, Vec3::splat(-10.0), Vec3::splat(10.0)),
            probe(1, 2, 1, Vec3::splat(-1.0), Vec3::splat(1.0)),
        ]);

        let selection = index.select((Vec3::splat(-0.5), Vec3::splat(0.5)));

        assert_eq!(selection, ReflectionProbeDrawSelection::one(2));
    }

    #[test]
    fn contained_same_importance_probe_blends_when_object_partially_leaves_inner() {
        let index = ReflectionProbeSpatialIndex::build(vec![
            probe(0, 1, 1, Vec3::splat(-10.0), Vec3::splat(10.0)),
            probe(1, 2, 1, Vec3::splat(-1.0), Vec3::splat(1.0)),
        ]);

        let selection = index.select((Vec3::new(-0.5, -0.5, -0.5), Vec3::new(1.5, 0.5, 0.5)));

        assert_eq!(selection.hit_count, 2);
        assert_eq!(selection.first_atlas_index, 2);
        assert_eq!(selection.second_atlas_index, 1);
        assert!((selection.second_weight - 0.25).abs() < 1e-6);
    }

    #[test]
    fn identical_same_importance_probe_boxes_use_intersection_blend() {
        let index = ReflectionProbeSpatialIndex::build(vec![
            probe(0, 1, 1, Vec3::splat(-1.0), Vec3::splat(1.0)),
            probe(1, 2, 1, Vec3::splat(-1.0), Vec3::splat(1.0)),
        ]);

        let selection = index.select((Vec3::splat(-0.5), Vec3::splat(0.5)));

        assert_eq!(selection.hit_count, 2);
        assert_eq!(selection.first_atlas_index, 1);
        assert_eq!(selection.second_atlas_index, 2);
        assert!((selection.second_weight - 0.5).abs() < 1e-6);
    }

    #[test]
    fn bvh_matches_bruteforce_candidates() {
        let probes: Vec<_> = (0..32)
            .map(|i| {
                let x = i as f32 * 0.5;
                probe(
                    i,
                    (i + 1) as u16,
                    1,
                    Vec3::new(x, -1.0, -1.0),
                    Vec3::new(x + 1.0, 1.0, 1.0),
                )
            })
            .collect();
        let index = ReflectionProbeSpatialIndex::build(probes.clone());
        let object = (Vec3::new(4.2, -0.25, -0.25), Vec3::new(6.1, 0.25, 0.25));
        let selection = index.select(object);

        let mut brute: Vec<_> = probes
            .iter()
            .filter_map(|probe| {
                let v = intersection_volume_vec3a(
                    probe.aabb_min,
                    probe.aabb_max,
                    Vec3A::from(object.0),
                    Vec3A::from(object.1),
                );
                (v > 0.0).then_some((probe.atlas_index, v))
            })
            .collect();
        brute.sort_by(|a, b| b.1.total_cmp(&a.1));

        assert_eq!(selection.first_atlas_index, brute[0].0);
        assert_eq!(selection.second_atlas_index, brute[1].0);
    }
}
