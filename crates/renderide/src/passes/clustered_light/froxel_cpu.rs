//! CPU light-centric froxel assignment for clustered forward lighting.
//!
//! This path mirrors the clustered-light storage contract (`cluster_light_counts` range rows plus
//! compact `cluster_light_indices`) so dense-light frames can use a light-centric alternative to
//! the O(froxels x lights) GPU scan.

use glam::{Mat4, Vec2, Vec3, Vec4};

use crate::gpu::GpuLight;
use crate::world_mesh::cluster::{
    CLUSTER_COUNT_Z, ClusterFrameParams, TILE_SIZE, sanitize_cluster_clip_planes,
};

/// Light count at which `Auto` mode starts considering CPU froxel assignment.
pub(super) const AUTO_CPU_FROXEL_LIGHT_THRESHOLD: u32 = 128;

/// Point light tag in [`GpuLight::light_type`].
const LIGHT_TYPE_POINT: u32 = 0;
/// Directional light tag in [`GpuLight::light_type`].
const LIGHT_TYPE_DIRECTIONAL: u32 = 1;
/// Spot light tag in [`GpuLight::light_type`].
const LIGHT_TYPE_SPOT: u32 = 2;

/// Cluster-grid layout for one eye.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct FroxelLayout {
    /// Cluster count in screen X.
    pub cluster_count_x: u32,
    /// Cluster count in screen Y.
    pub cluster_count_y: u32,
    /// Cluster count in depth.
    pub cluster_count_z: u32,
    /// Viewport width in physical pixels.
    pub viewport_width: u32,
    /// Viewport height in physical pixels.
    pub viewport_height: u32,
}

impl FroxelLayout {
    /// Builds a layout from the frame's clustered camera params.
    fn from_cluster_params(params: &ClusterFrameParams) -> Self {
        Self {
            cluster_count_x: params.cluster_count_x.max(1),
            cluster_count_y: params.cluster_count_y.max(1),
            cluster_count_z: CLUSTER_COUNT_Z.max(1),
            viewport_width: params.viewport_width.max(1),
            viewport_height: params.viewport_height.max(1),
        }
    }

    /// Number of froxels in this eye.
    fn cluster_count(self) -> Option<usize> {
        let xy = self.cluster_count_x.checked_mul(self.cluster_count_y)?;
        xy.checked_mul(self.cluster_count_z).map(|v| v as usize)
    }
}

/// Per-frame CPU-produced cluster storage matching the existing WGSL buffers.
#[derive(Clone, Debug, Default)]
pub(super) struct CpuClusterAssignments {
    /// Per-froxel `[offset, count]` rows addressing [`Self::indices`].
    pub ranges: Vec<[u32; 2]>,
    /// Compact light indices for every froxel membership.
    pub indices: Vec<u32>,
    /// Assignment diagnostics for profiling and tests.
    pub stats: CpuFroxelStats,
}

/// CPU froxel assignment diagnostics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct CpuFroxelStats {
    /// Number of light/froxel memberships emitted into compact storage.
    pub assigned_memberships: u64,
    /// Number of light/froxel memberships dropped because compact storage could not represent them.
    pub overflowed_memberships: u64,
    /// Number of lights rejected before assignment because their conservative bounds miss the view.
    pub culled_lights: u32,
}

/// Stateless CPU froxel assignment entry point.
pub(super) struct FroxelLightPlanner;

impl FroxelLightPlanner {
    /// Builds dynamic-range cluster assignments for every eye in `eye_params`.
    pub fn build(
        lights: &[GpuLight],
        eye_params: &[ClusterFrameParams],
        clusters_per_eye: u32,
    ) -> Option<CpuClusterAssignments> {
        profiling::scope!("clustered_light::cpu_froxel_build");
        if eye_params.is_empty() {
            return Some(CpuClusterAssignments::default());
        }

        let eye_count = eye_params.len();
        let total_clusters = usize::try_from(clusters_per_eye)
            .ok()?
            .checked_mul(eye_count)?;
        let mut counts = vec![0u32; total_clusters];
        let mut stats = CpuFroxelStats::default();

        for (eye_idx, params) in eye_params.iter().enumerate() {
            let layout = FroxelLayout::from_cluster_params(params);
            let expected_clusters = layout.cluster_count()?;
            if expected_clusters != usize::try_from(clusters_per_eye).ok()? {
                return None;
            }
            let cluster_base = eye_idx.checked_mul(expected_clusters)?;
            let mut emit_count = |cluster_id: usize, _light_idx: u32| {
                let Some(count) = counts.get_mut(cluster_id) else {
                    return;
                };
                *count = count.saturating_add(1);
                stats.assigned_memberships = stats.assigned_memberships.saturating_add(1);
            };
            stats.culled_lights = stats.culled_lights.saturating_add(assign_eye_lights(
                lights,
                *params,
                layout,
                cluster_base,
                &mut emit_count,
            ));
        }

        let (ranges, total_indices) = prefix_counts_to_ranges(&counts)?;
        let mut indices = vec![0u32; total_indices];
        let mut cursors = vec![0u32; total_clusters];

        for (eye_idx, params) in eye_params.iter().enumerate() {
            let layout = FroxelLayout::from_cluster_params(params);
            let expected_clusters = layout.cluster_count()?;
            if expected_clusters != usize::try_from(clusters_per_eye).ok()? {
                return None;
            }
            let cluster_base = eye_idx.checked_mul(expected_clusters)?;
            let mut emit_index = |cluster_id: usize, light_idx: u32| {
                write_membership(cluster_id, light_idx, &ranges, &mut cursors, &mut indices);
            };
            assign_eye_lights(lights, *params, layout, cluster_base, &mut emit_index);
        }

        Some(CpuClusterAssignments {
            ranges,
            indices,
            stats,
        })
    }
}

/// Assigns every light to one eye's froxel grid.
fn assign_eye_lights(
    lights: &[GpuLight],
    params: ClusterFrameParams,
    layout: FroxelLayout,
    cluster_base: usize,
    emit: &mut impl FnMut(usize, u32),
) -> u32 {
    let view = params.world_to_view;
    let view_scale = params.world_to_view_scale_max();
    let mut culled_lights = 0u32;
    for (light_idx, light) in lights.iter().enumerate() {
        if light.light_type == LIGHT_TYPE_DIRECTIONAL {
            assign_directional(light_idx as u32, layout, cluster_base, emit);
            continue;
        }
        let Some(bounds) =
            light_froxel_bounds(light, view, params.proj, view_scale, layout, params)
        else {
            culled_lights = culled_lights.saturating_add(1);
            continue;
        };
        assign_bounded_light(light_idx as u32, bounds, layout, cluster_base, emit);
    }
    culled_lights
}

/// Inclusive froxel bounds touched by a light.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FroxelBounds {
    /// First X froxel.
    x0: u32,
    /// Last X froxel.
    x1: u32,
    /// First Y froxel.
    y0: u32,
    /// Last Y froxel.
    y1: u32,
    /// First Z froxel.
    z0: u32,
    /// Last Z froxel.
    z1: u32,
}

/// Computes conservative froxel bounds for point and spot lights.
fn light_froxel_bounds(
    light: &GpuLight,
    view: Mat4,
    proj: Mat4,
    view_scale: f32,
    layout: FroxelLayout,
    params: ClusterFrameParams,
) -> Option<FroxelBounds> {
    let mut center = transform_point(view, Vec3::from_array(light.position));
    let mut radius = (light.range * view_scale).max(0.0);
    if radius <= 0.0 || !radius.is_finite() {
        return None;
    }

    if light.light_type == LIGHT_TYPE_SPOT {
        let axis = transform_vector(view, Vec3::from_array(light.direction))
            .try_normalize()
            .unwrap_or(Vec3::Z);
        let cos_half = light.spot_cos_half_angle.clamp(0.0, 1.0);
        if cos_half < 0.9999 {
            let sin_sq = (1.0 - cos_half * cos_half).max(0.0);
            let tan_sq = sin_sq / (cos_half * cos_half).max(1e-8);
            center += axis * (radius * 0.5);
            radius *= (0.25 + tan_sq).sqrt();
        }
    } else if light.light_type != LIGHT_TYPE_POINT {
        return None;
    }

    let (near, far) = params.sanitized_clip_planes();
    let raw_nearest_depth = -(center.z + radius);
    let raw_farthest_depth = -(center.z - radius);
    if raw_farthest_depth < near || raw_nearest_depth > far {
        return None;
    }

    let nearest_depth = raw_nearest_depth.clamp(near, far);
    let farthest_depth = raw_farthest_depth.clamp(near, far);
    let z0 = cluster_z_from_depth(nearest_depth, near, far, layout.cluster_count_z);
    let z1 = cluster_z_from_depth(farthest_depth, near, far, layout.cluster_count_z);
    let (x0, x1, y0, y1) = projected_sphere_xy_bounds(center, radius, proj, near, far, layout)?;

    Some(FroxelBounds {
        x0,
        x1,
        y0,
        y1,
        z0: z0.min(z1),
        z1: z0.max(z1),
    })
}

/// Transforms a world-space point by `matrix`.
fn transform_point(matrix: Mat4, point: Vec3) -> Vec3 {
    (matrix * point.extend(1.0)).truncate()
}

/// Transforms a world-space vector by `matrix`.
fn transform_vector(matrix: Mat4, vector: Vec3) -> Vec3 {
    (matrix * vector.extend(0.0)).truncate()
}

/// Maps positive depth to a logarithmic clustered Z slice.
fn cluster_z_from_depth(depth: f32, near_clip: f32, far_clip: f32, cluster_count_z: u32) -> u32 {
    let z_count = cluster_count_z.max(1);
    let (near_safe, far_safe) = sanitize_cluster_clip_planes(near_clip, far_clip);
    let ratio = (far_safe / near_safe).max(1.0 + f32::EPSILON);
    let z = (depth.clamp(near_safe, far_safe) / near_safe).log(ratio) * z_count as f32;
    z.clamp(0.0, z_count.saturating_sub(1) as f32) as u32
}

/// Computes conservative screen-space froxel bounds for a view-space sphere.
fn projected_sphere_xy_bounds(
    center: Vec3,
    radius: f32,
    proj: Mat4,
    near: f32,
    far: f32,
    layout: FroxelLayout,
) -> Option<(u32, u32, u32, u32)> {
    let near_z = (center.z + radius).min(-near).max(-far);
    let far_z = (center.z - radius).min(-near).max(-far);
    let mut ndc_min = Vec2::splat(f32::INFINITY);
    let mut ndc_max = Vec2::splat(f32::NEG_INFINITY);
    for z in [near_z, far_z] {
        for x_sign in [-1.0, 1.0] {
            for y_sign in [-1.0, 1.0] {
                let p = Vec3::new(center.x + radius * x_sign, center.y + radius * y_sign, z);
                let ndc = project_view_point(proj, p)?;
                ndc_min = ndc_min.min(ndc);
                ndc_max = ndc_max.max(ndc);
            }
        }
    }
    let x0 = ndc_x_to_cluster(ndc_min.x, layout);
    let x1 = ndc_x_to_cluster(ndc_max.x, layout);
    let y0 = ndc_y_to_cluster(ndc_max.y, layout);
    let y1 = ndc_y_to_cluster(ndc_min.y, layout);
    Some((x0.min(x1), x0.max(x1), y0.min(y1), y0.max(y1)))
}

/// Projects a view-space point into normalized device coordinates.
fn project_view_point(proj: Mat4, point: Vec3) -> Option<Vec2> {
    let clip = proj * Vec4::new(point.x, point.y, point.z, 1.0);
    if clip.w.abs() <= 1e-8 || !clip.w.is_finite() {
        return None;
    }
    let ndc = clip.truncate() / clip.w;
    (ndc.x.is_finite() && ndc.y.is_finite()).then(|| ndc.truncate())
}

/// Converts NDC X to a froxel coordinate.
fn ndc_x_to_cluster(ndc_x: f32, layout: FroxelLayout) -> u32 {
    let px = ((ndc_x.clamp(-1.0, 1.0) + 1.0) * 0.5 * layout.viewport_width as f32).floor();
    (px as u32 / TILE_SIZE).min(layout.cluster_count_x - 1)
}

/// Converts NDC Y to a froxel coordinate with top-left screen origin.
fn ndc_y_to_cluster(ndc_y: f32, layout: FroxelLayout) -> u32 {
    let py = ((1.0 - ndc_y.clamp(-1.0, 1.0)) * 0.5 * layout.viewport_height as f32).floor();
    (py as u32 / TILE_SIZE).min(layout.cluster_count_y - 1)
}

/// Assigns a directional light to every froxel.
fn assign_directional(
    light_idx: u32,
    layout: FroxelLayout,
    cluster_base: usize,
    emit: &mut impl FnMut(usize, u32),
) {
    let Some(cluster_count) = layout.cluster_count() else {
        return;
    };
    for cluster_local in 0..cluster_count {
        emit(cluster_base + cluster_local, light_idx);
    }
}

/// Assigns a bounded local light to its touched froxel range.
fn assign_bounded_light(
    light_idx: u32,
    bounds: FroxelBounds,
    layout: FroxelLayout,
    cluster_base: usize,
    emit: &mut impl FnMut(usize, u32),
) {
    for z in bounds.z0..=bounds.z1 {
        for y in bounds.y0..=bounds.y1 {
            for x in bounds.x0..=bounds.x1 {
                let local = x + layout.cluster_count_x * (y + layout.cluster_count_y * z);
                emit(cluster_base + local as usize, light_idx);
            }
        }
    }
}

/// Converts per-froxel counts into compact `[offset, count]` rows.
fn prefix_counts_to_ranges(counts: &[u32]) -> Option<(Vec<[u32; 2]>, usize)> {
    let mut ranges = Vec::with_capacity(counts.len());
    let mut offset = 0u64;
    for &count in counts {
        let range_offset = u32::try_from(offset).ok()?;
        ranges.push([range_offset, count]);
        offset = offset.checked_add(u64::from(count))?;
        if offset > u64::from(u32::MAX) {
            return None;
        }
    }
    let total_indices = usize::try_from(offset).ok()?;
    Some((ranges, total_indices))
}

/// Appends one light index to one froxel's compact index range.
fn write_membership(
    cluster_id: usize,
    light_idx: u32,
    ranges: &[[u32; 2]],
    cursors: &mut [u32],
    indices: &mut [u32],
) {
    let Some(range) = ranges.get(cluster_id) else {
        return;
    };
    let Some(cursor) = cursors.get_mut(cluster_id) else {
        return;
    };
    if *cursor >= range[1] {
        return;
    }
    let index_offset = u64::from(range[0]).checked_add(u64::from(*cursor));
    let Some(index) = index_offset.and_then(|offset| usize::try_from(offset).ok()) else {
        return;
    };
    let Some(dst) = indices.get_mut(index) else {
        return;
    };
    *dst = light_idx;
    *cursor += 1;
}

#[cfg(test)]
mod tests {
    use glam::Mat4;

    use super::*;

    /// Builds a compact 2x2x16 test layout.
    fn test_params() -> ClusterFrameParams {
        ClusterFrameParams {
            near_clip: 0.1,
            far_clip: 100.0,
            world_to_view: Mat4::IDENTITY,
            proj: Mat4::IDENTITY,
            cluster_count_x: 2,
            cluster_count_y: 2,
            viewport_width: 64,
            viewport_height: 64,
            projection_flags: 0,
        }
    }

    /// Builds a point light at `position`.
    fn point_light(position: Vec3, range: f32) -> GpuLight {
        GpuLight {
            position: position.to_array(),
            range,
            light_type: LIGHT_TYPE_POINT,
            ..Default::default()
        }
    }

    /// Builds a directional light.
    fn directional_light() -> GpuLight {
        GpuLight {
            light_type: LIGHT_TYPE_DIRECTIONAL,
            ..Default::default()
        }
    }

    /// Returns the compact light-index slice for one cluster.
    fn cluster_indices(assignments: &CpuClusterAssignments, cluster_id: usize) -> &[u32] {
        let [offset, count] = assignments.ranges[cluster_id];
        let start = offset as usize;
        let end = start + count as usize;
        &assignments.indices[start..end]
    }

    #[test]
    fn empty_lights_write_zero_ranges_without_indices() {
        let params = test_params();
        let assignments = FroxelLightPlanner::build(
            &[],
            &[params],
            params.cluster_count_x * params.cluster_count_y * CLUSTER_COUNT_Z,
        )
        .expect("assignments");
        assert_eq!(assignments.ranges.len(), 64);
        assert!(assignments.ranges.iter().all(|range| range[1] == 0));
        assert!(assignments.indices.is_empty());
    }

    #[test]
    fn directional_light_hits_every_froxel() {
        let params = test_params();
        let assignments = FroxelLightPlanner::build(
            &[directional_light()],
            &[params],
            params.cluster_count_x * params.cluster_count_y * CLUSTER_COUNT_Z,
        )
        .expect("assignments");

        assert!(assignments.ranges.iter().all(|range| range[1] == 1));
        assert_eq!(cluster_indices(&assignments, 0), &[0]);
    }

    #[test]
    fn local_light_touches_subset_of_froxels() {
        let params = test_params();
        let assignments = FroxelLightPlanner::build(
            &[point_light(Vec3::new(0.0, 0.0, -5.0), 0.25)],
            &[params],
            params.cluster_count_x * params.cluster_count_y * CLUSTER_COUNT_Z,
        )
        .expect("assignments");

        let touched = assignments
            .ranges
            .iter()
            .filter(|range| range[1] > 0)
            .count();
        assert!(touched > 0);
        assert!(touched < assignments.ranges.len());
    }

    #[test]
    fn compact_indices_store_all_lights() {
        let params = test_params();
        let assignments = FroxelLightPlanner::build(
            &[directional_light(), directional_light()],
            &[params],
            params.cluster_count_x * params.cluster_count_y * CLUSTER_COUNT_Z,
        )
        .expect("assignments");

        assert_eq!(assignments.ranges[0][1], 2);
        assert_eq!(cluster_indices(&assignments, 0), &[0, 1]);
    }

    #[test]
    fn compact_indices_do_not_truncate_dense_clusters() {
        let params = test_params();
        let lights = (0..70).map(|_| directional_light()).collect::<Vec<_>>();
        let assignments = FroxelLightPlanner::build(
            &lights,
            &[params],
            params.cluster_count_x * params.cluster_count_y * CLUSTER_COUNT_Z,
        )
        .expect("assignments");

        assert!(assignments.ranges.iter().all(|range| range[1] == 70));
        assert_eq!(cluster_indices(&assignments, 0).len(), 70);
        assert_eq!(assignments.stats.overflowed_memberships, 0);
    }
}
