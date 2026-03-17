//! World matrix computation and transform hierarchy utilities.
//!
//! Uses glam for SIMD-optimized matrix operations in the hot path.

use glam::Mat4;

use crate::scene::{math::render_transform_to_matrix, Scene};

use super::error::SceneError;

/// Per-scene cache for world matrices and computed flags.
pub(super) struct SceneCache {
    /// World-space matrices for each transform.
    pub world_matrices: Vec<Mat4>,
    /// Whether each transform's world matrix has been computed this frame.
    pub computed: Vec<bool>,
    /// Cached local (TRS) matrices; avoids redundant render_transform_to_matrix when propagating.
    pub local_matrices: Vec<Mat4>,
    /// Whether each node's local matrix cache is stale (pose changed).
    pub local_dirty: Vec<bool>,
}

/// Fixes transform ID references after swap_remove: removed ID becomes -1,
/// last index (swapped into removed slot) becomes removed ID.
pub(super) fn fixup_transform_id(old: i32, removed_id: i32, last_index: usize) -> i32 {
    if old == removed_id {
        -1
    } else if old == last_index as i32 {
        removed_id
    } else {
        old
    }
}

/// Builds a parent→children index from `node_parents`. Root nodes (parent < 0) have no parent.
fn build_node_children(node_parents: &[i32], n: usize) -> Vec<Vec<usize>> {
    let mut children: Vec<Vec<usize>> = (0..n).map(|_| Vec::new()).collect();
    for (i, &p) in node_parents.iter().take(n).enumerate() {
        if p >= 0 && (p as usize) < n && p != i as i32 {
            children[p as usize].push(i);
        }
    }
    children
}

/// Marks descendants of uncomputed transforms as uncomputed.
/// Uses a parent→children index to traverse down from each uncomputed node (O(n) total)
/// instead of walking up from every node (O(n²) for deep chains).
pub(super) fn mark_descendants_uncomputed(node_parents: &[i32], computed: &mut [bool]) {
    let n = computed.len();
    if n == 0 {
        return;
    }
    let children = build_node_children(node_parents, n);
    let mut stack = Vec::with_capacity(64.min(n));
    for i in 0..n {
        if computed[i] {
            continue;
        }
        stack.clear();
        stack.extend(children[i].iter().copied());
        while let Some(child) = stack.pop() {
            computed[child] = false;
            stack.extend(children[child].iter().copied());
        }
    }
}

/// Returns the local matrix for node `i`, using cache when valid.
#[inline]
fn get_local_matrix(
    nodes: &[crate::shared::RenderTransform],
    local_matrices: &mut [Mat4],
    local_dirty: &mut [bool],
    i: usize,
) -> Mat4 {
    if i < local_dirty.len() && local_dirty[i] {
        let m = render_transform_to_matrix(&nodes[i]);
        local_matrices[i] = m;
        local_dirty[i] = false;
        m
    } else if i < local_matrices.len() {
        local_matrices[i]
    } else {
        render_transform_to_matrix(&nodes[i])
    }
}

/// Incremental world matrix computation: only recomputes nodes with `computed[i] == false`.
/// Walks up from each uncomputed node to find the first computed ancestor, then multiplies down.
/// Uses glam for SIMD-optimized matrix multiply and local matrix cache to avoid redundant TRS conversion.
pub(super) fn compute_world_matrices_incremental(
    scene: &Scene,
    world_matrices: &mut [Mat4],
    computed: &mut [bool],
    local_matrices: &mut [Mat4],
    local_dirty: &mut [bool],
) -> Result<(), SceneError> {
    let n = scene.nodes.len();
    let node_parents = &scene.node_parents;
    let nodes = &scene.nodes;
    let mut stack = Vec::with_capacity(64.min(n));

    for transform_index in (0..n).rev() {
        if computed[transform_index] {
            continue;
        }

        let mut maybe_uppermost_matrix: Option<Mat4> = None;
        let mut id = transform_index;
        let mut steps = 0;
        while id < n && steps < n {
            steps += 1;
            if computed[id] {
                maybe_uppermost_matrix = Some(world_matrices[id]);
                break;
            }
            stack.push(id);
            let p = node_parents.get(id).copied().unwrap_or(-1);
            if p < 0 || (p as usize) >= n || p == id as i32 {
                break;
            }
            id = p as usize;
        }

        let mut parent_matrix = match maybe_uppermost_matrix {
            Some(m) => m,
            None => {
                let top = match stack.pop() {
                    Some(t) => t,
                    None => continue,
                };
                let local = get_local_matrix(nodes, local_matrices, local_dirty, top);
                world_matrices[top] = local;
                computed[top] = true;
                local
            }
        };

        while let Some(child_id) = stack.pop() {
            let local = get_local_matrix(nodes, local_matrices, local_dirty, child_id);
            parent_matrix = parent_matrix * local;
            world_matrices[child_id] = parent_matrix;
            computed[child_id] = true;
        }
    }

    Ok(())
}

/// Full iterative DFS world matrix computation with cycle detection.
/// Used by tests; root-level nodes use identity as parent.
#[cfg(test)]
pub(crate) fn compute_world_matrices_from_scene(scene: &Scene) -> Vec<Mat4> {
    let n = scene.nodes.len();
    if n == 0 {
        return Vec::new();
    }

    let mut world = vec![Mat4::IDENTITY; n];
    let mut visited = vec![false; n];
    let mut in_stack = vec![false; n];

    let mut stack: Vec<usize> = Vec::new();
    for start in 0..n {
        if visited[start] {
            continue;
        }
        stack.push(start);
        in_stack[start] = true;
        while let Some(&i) = stack.last() {
            if visited[i] {
                in_stack[i] = false;
                stack.pop();
                continue;
            }
            let p = scene.node_parents.get(i).copied().unwrap_or(-1);
            let p_usize = if p >= 0 && (p as usize) < n && p != i as i32 {
                p as usize
            } else {
                let local = render_transform_to_matrix(&scene.nodes[i]);
                world[i] = local;
                visited[i] = true;
                in_stack[i] = false;
                stack.pop();
                continue;
            };

            if in_stack[p_usize] {
                logger::trace!(
                    "Cycle detected in scene {} at transform {} (parent {}); treating as root",
                    scene.id,
                    i,
                    p
                );
                let local = render_transform_to_matrix(&scene.nodes[i]);
                world[i] = local;
                visited[i] = true;
                in_stack[i] = false;
                stack.pop();
                continue;
            }

            if !visited[p_usize] {
                stack.push(p_usize);
                in_stack[p_usize] = true;
                continue;
            }

            let local = render_transform_to_matrix(&scene.nodes[i]);
            world[i] = world[p_usize] * local;
            visited[i] = true;
            in_stack[i] = false;
            stack.pop();
        }
    }

    world
}
