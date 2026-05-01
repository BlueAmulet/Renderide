//! Batch keys and draw list ordering for world mesh forward.

use std::cmp::Ordering;

use rayon::slice::ParallelSliceMut;

use crate::materials::render_queue_is_transparent;

use super::item::WorldMeshDrawItem;

/// Compact ordering prefix for the hot draw-sort comparator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct DrawSortKey {
    /// Overlay draws sort after main-world draws.
    pub overlay: bool,
    /// Effective Unity render queue.
    pub render_queue: i32,
    /// Queues after Unity's opaque cutoff use transparent ordering.
    pub transparent_queue: bool,
    /// Coarse front-to-back bucket for opaque draws.
    pub opaque_depth_bucket: u16,
}

impl DrawSortKey {
    /// Builds the sort-key prefix for a draw item.
    fn from_draw(item: &WorldMeshDrawItem) -> Self {
        Self {
            overlay: item.is_overlay,
            render_queue: item.batch_key.render_queue,
            transparent_queue: render_queue_is_transparent(item.batch_key.render_queue),
            opaque_depth_bucket: item.opaque_depth_bucket,
        }
    }
}

/// Maps camera-distance squared into a coarse logarithmic front-to-back bucket.
///
/// Called once per draw at candidate evaluation and the result stored on
/// [`WorldMeshDrawItem::opaque_depth_bucket`]; the comparator then reads the field directly
/// instead of recomputing `sqrt` + `log2` on every pairwise compare.
pub(super) fn opaque_depth_bucket(distance_sq: f32) -> u16 {
    if !distance_sq.is_finite() || distance_sq <= 0.0 {
        return 0;
    }
    let distance = distance_sq.sqrt().max(1e-4);
    ((distance.log2() + 16.0).floor().clamp(0.0, 255.0)) as u16
}

/// Ordering for world mesh draws (Unity render queue, then opaque batching vs transparent distance sort).
///
/// Shared by [`sort_draws`] (parallel) and [`sort_draws_serial`].
///
/// The opaque-bucket tiebreaker drives the comparator's hot path: every pair of draws sharing the
/// `(overlay, render_queue, transparent_queue, opaque_depth_bucket)` prefix used to fall through to a full
/// [`MaterialDrawBatchKey::cmp`] walk (including `RasterPipelineKind` and
/// `MaterialRenderState`). The hash compare on
/// [`super::item::WorldMeshDrawItem::batch_key_hash`] resolves the dominant case in one
/// `u64::cmp`, falling back to the structural compare only on hash collisions so deterministic
/// instance batching survives the (statistically negligible) collision case.
#[inline]
pub(super) fn cmp_world_mesh_draw_items(a: &WorldMeshDrawItem, b: &WorldMeshDrawItem) -> Ordering {
    let a_sort = DrawSortKey::from_draw(a);
    let b_sort = DrawSortKey::from_draw(b);
    a_sort
        .overlay
        .cmp(&b_sort.overlay)
        .then(a_sort.render_queue.cmp(&b_sort.render_queue))
        .then(a_sort.transparent_queue.cmp(&b_sort.transparent_queue))
        .then_with(
            || match (a_sort.transparent_queue, b_sort.transparent_queue) {
                (false, false) => a_sort
                    .opaque_depth_bucket
                    .cmp(&b_sort.opaque_depth_bucket)
                    .then_with(|| a.batch_key_hash.cmp(&b.batch_key_hash))
                    .then_with(|| a.batch_key.cmp(&b.batch_key))
                    .then(b.sorting_order.cmp(&a.sorting_order))
                    .then(a.mesh_asset_id.cmp(&b.mesh_asset_id))
                    .then(a.node_id.cmp(&b.node_id))
                    .then(a.slot_index.cmp(&b.slot_index)),
                (true, true) => a
                    .sorting_order
                    .cmp(&b.sorting_order)
                    .then_with(|| b.camera_distance_sq.total_cmp(&a.camera_distance_sq))
                    .then(a.collect_order.cmp(&b.collect_order)),
                _ => Ordering::Equal,
            },
        )
}

/// Pre-depth-bucket ordering retained for regression tests that need to isolate batch-key order.
#[cfg(test)]
fn cmp_world_mesh_draw_items_without_depth_bucket(
    a: &WorldMeshDrawItem,
    b: &WorldMeshDrawItem,
) -> Ordering {
    a.is_overlay
        .cmp(&b.is_overlay)
        .then(a.batch_key.render_queue.cmp(&b.batch_key.render_queue))
        .then(
            render_queue_is_transparent(a.batch_key.render_queue)
                .cmp(&render_queue_is_transparent(b.batch_key.render_queue)),
        )
        .then_with(|| {
            match (
                render_queue_is_transparent(a.batch_key.render_queue),
                render_queue_is_transparent(b.batch_key.render_queue),
            ) {
                (false, false) => a
                    .batch_key
                    .cmp(&b.batch_key)
                    .then(b.sorting_order.cmp(&a.sorting_order))
                    .then(a.mesh_asset_id.cmp(&b.mesh_asset_id))
                    .then(a.node_id.cmp(&b.node_id))
                    .then(a.slot_index.cmp(&b.slot_index)),
                (true, true) => a
                    .sorting_order
                    .cmp(&b.sorting_order)
                    .then_with(|| b.camera_distance_sq.total_cmp(&a.camera_distance_sq))
                    .then(a.collect_order.cmp(&b.collect_order)),
                _ => Ordering::Equal,
            }
        })
}

/// Sorts opaque draws for batching and alpha UI/text draws in stable canvas order.
pub fn sort_draws(items: &mut [WorldMeshDrawItem]) {
    profiling::scope!("mesh::sort_draws");
    items.par_sort_unstable_by(cmp_world_mesh_draw_items);
}

/// Same ordering as [`sort_draws`] without rayon (for nested parallel batches).
pub(super) fn sort_draws_serial(items: &mut [WorldMeshDrawItem]) {
    profiling::scope!("mesh::sort_draws_serial");
    items.sort_unstable_by(cmp_world_mesh_draw_items);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::materials::{
        UNITY_RENDER_QUEUE_ALPHA_TEST, UNITY_RENDER_QUEUE_OVERLAY, UNITY_RENDER_QUEUE_TRANSPARENT,
    };
    use crate::render_graph::test_fixtures::{DummyDrawItemSpec, dummy_world_mesh_draw_item};
    use crate::world_mesh::materials::compute_batch_key_hash;

    /// Sets `camera_distance_sq` and refreshes the precomputed `opaque_depth_bucket` so test
    /// fixtures match what `evaluate_draw_candidate` would produce in production.
    fn set_camera_distance(item: &mut WorldMeshDrawItem, distance_sq: f32) {
        item.camera_distance_sq = distance_sq;
        item.opaque_depth_bucket = opaque_depth_bucket(distance_sq);
    }

    fn set_render_queue(item: &mut WorldMeshDrawItem, render_queue: i32) {
        item.batch_key.render_queue = render_queue;
        item.batch_key_hash = compute_batch_key_hash(&item.batch_key);
    }

    #[test]
    fn opaque_sort_prefers_nearer_depth_bucket_before_batch_key() {
        let mut near = dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 2,
            property_block: None,
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: 1,
            node_id: 2,
            slot_index: 0,
            collect_order: 0,
            alpha_blended: false,
        });
        set_camera_distance(&mut near, 1.0);
        let mut far = dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 1,
            property_block: None,
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: 1,
            node_id: 1,
            slot_index: 0,
            collect_order: 1,
            alpha_blended: false,
        });
        set_camera_distance(&mut far, 4096.0);

        assert_eq!(
            cmp_world_mesh_draw_items(&near, &far),
            Ordering::Less,
            "near opaque draws should sort before lower material ids when depth buckets differ"
        );
        assert_eq!(
            cmp_world_mesh_draw_items_without_depth_bucket(&near, &far),
            Ordering::Greater,
            "the regression setup must differ from pure batch-key ordering"
        );
    }

    #[test]
    fn transparent_sort_remains_back_to_front() {
        let mut near = dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 1,
            property_block: None,
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: 1,
            node_id: 1,
            slot_index: 0,
            collect_order: 0,
            alpha_blended: true,
        });
        set_camera_distance(&mut near, 1.0);
        let mut far = dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 1,
            property_block: None,
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: 1,
            node_id: 2,
            slot_index: 0,
            collect_order: 1,
            alpha_blended: true,
        });
        set_camera_distance(&mut far, 4096.0);

        assert_eq!(cmp_world_mesh_draw_items(&far, &near), Ordering::Less);
    }

    #[test]
    fn render_queue_orders_before_transparent_distance() {
        let mut near_early_queue = dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 1,
            property_block: None,
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: 1,
            node_id: 1,
            slot_index: 0,
            collect_order: 0,
            alpha_blended: true,
        });
        set_camera_distance(&mut near_early_queue, 1.0);
        set_render_queue(&mut near_early_queue, UNITY_RENDER_QUEUE_TRANSPARENT);

        let mut far_late_queue = dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 1,
            property_block: None,
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: 1,
            node_id: 2,
            slot_index: 0,
            collect_order: 1,
            alpha_blended: true,
        });
        set_camera_distance(&mut far_late_queue, 4096.0);
        set_render_queue(&mut far_late_queue, UNITY_RENDER_QUEUE_TRANSPARENT + 5);

        assert_eq!(
            cmp_world_mesh_draw_items(&near_early_queue, &far_late_queue),
            Ordering::Less,
            "lower transparent render queues must draw before farther later queues"
        );
    }

    #[test]
    fn render_queue_orders_alpha_test_transparent_and_overlay_ranges() {
        let mut transparent = dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 1,
            property_block: None,
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: 1,
            node_id: 1,
            slot_index: 0,
            collect_order: 0,
            alpha_blended: true,
        });
        set_render_queue(&mut transparent, UNITY_RENDER_QUEUE_TRANSPARENT);

        let mut alpha_test = transparent.clone();
        set_render_queue(&mut alpha_test, UNITY_RENDER_QUEUE_ALPHA_TEST);

        let mut late_transparent = transparent.clone();
        set_render_queue(&mut late_transparent, UNITY_RENDER_QUEUE_TRANSPARENT + 5);

        let mut overlay = transparent.clone();
        set_render_queue(&mut overlay, UNITY_RENDER_QUEUE_OVERLAY);

        let mut items = vec![overlay, late_transparent, transparent, alpha_test];
        sort_draws_serial(&mut items);

        let queues: Vec<_> = items
            .iter()
            .map(|item| item.batch_key.render_queue)
            .collect();
        assert_eq!(
            queues,
            vec![
                UNITY_RENDER_QUEUE_ALPHA_TEST,
                UNITY_RENDER_QUEUE_TRANSPARENT,
                UNITY_RENDER_QUEUE_TRANSPARENT + 5,
                UNITY_RENDER_QUEUE_OVERLAY,
            ]
        );
    }
}
