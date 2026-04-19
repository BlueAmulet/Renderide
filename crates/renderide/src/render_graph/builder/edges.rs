//! Explicit group edges and resource read/write dependency edges.

use hashbrown::HashMap;
use std::collections::HashSet;

use super::super::error::GraphBuildError;
use super::super::ids::{GroupId, PassId};
use super::super::pass::GroupScope;
use super::super::resources::{BufferResourceHandle, ResourceHandle, TextureResourceHandle};
use super::decl::SetupEntry;
use super::GraphBuilder;

pub(super) fn explicit_edges(
    builder: &GraphBuilder,
    n: usize,
) -> Result<HashSet<(usize, usize)>, GraphBuildError> {
    let mut edges = HashSet::new();
    for &(from, to) in &builder.edges {
        if from >= n || to >= n {
            return Err(GraphBuildError::InvalidEdge {
                from: PassId(from),
                to: PassId(to),
            });
        }
        if from != to {
            edges.insert((from, to));
        }
    }
    Ok(edges)
}

/// Adds linear-size relay edges so every pass in set `a` precedes every pass in set `b`.
fn relay_all_before(a: &[usize], b: &[usize], edges: &mut HashSet<(usize, usize)>) {
    if a.is_empty() || b.is_empty() {
        return;
    }
    let mut b_sorted = b.to_vec();
    b_sorted.sort_unstable();
    let rep_b = b_sorted[0];
    for &ai in a {
        if ai != rep_b {
            edges.insert((ai, rep_b));
        }
    }
    for &bi in b_sorted.iter().skip(1) {
        edges.insert((rep_b, bi));
    }
}

pub(super) fn add_group_edges(
    builder: &GraphBuilder,
    setups: &[SetupEntry],
    edges: &mut HashSet<(usize, usize)>,
) -> Result<(), GraphBuildError> {
    for entry in &builder.groups {
        for &dep in &entry.after {
            if dep.0 >= builder.groups.len() {
                return Err(GraphBuildError::CycleDetected);
            }
        }
    }

    let mut frame_global = Vec::new();
    let mut per_view = Vec::new();
    for (idx, setup) in setups.iter().enumerate() {
        match builder.groups[setup.group.0].scope {
            GroupScope::FrameGlobal => frame_global.push(idx),
            GroupScope::PerView => per_view.push(idx),
        }
    }
    frame_global.sort_unstable();
    per_view.sort_unstable();
    relay_all_before(&frame_global, &per_view, edges);

    for (gb_idx, gb) in builder.groups.iter().enumerate() {
        let gb_id = GroupId(gb_idx);
        let mut passes_b: Vec<usize> = setups
            .iter()
            .enumerate()
            .filter_map(|(i, s)| (s.group == gb_id).then_some(i))
            .collect();
        passes_b.sort_unstable();
        for &ga_id in &gb.after {
            let mut passes_a: Vec<usize> = setups
                .iter()
                .enumerate()
                .filter_map(|(i, s)| (s.group == ga_id).then_some(i))
                .collect();
            passes_a.sort_unstable();
            relay_all_before(&passes_a, &passes_b, edges);
        }
    }
    Ok(())
}

pub(super) fn add_resource_edges(
    builder: &GraphBuilder,
    setups: &[SetupEntry],
    edges: &mut HashSet<(usize, usize)>,
) -> Result<(), GraphBuildError> {
    let mut by_resource: HashMap<ResourceHandle, Vec<(usize, bool, bool)>> = HashMap::new();
    for (pass_idx, setup) in setups.iter().enumerate() {
        for access in &setup.setup.accesses {
            by_resource.entry(access.resource).or_default().push((
                pass_idx,
                access.reads(),
                access.writes(),
            ));
        }
    }

    for (resource, mut accesses) in by_resource {
        accesses.sort_by_key(|(pass_idx, _, _)| *pass_idx);
        accesses.dedup();
        let mut last_writer: Option<usize> = None;
        let mut readers: HashSet<usize> = HashSet::new();
        for (pass_idx, reads, writes) in accesses {
            if reads {
                if let Some(writer) = last_writer {
                    if writer != pass_idx {
                        edges.insert((writer, pass_idx));
                    }
                } else if !resource.is_imported() {
                    return Err(GraphBuildError::MissingDependency {
                        pass: PassId(pass_idx),
                        resource: resource_label(builder, resource),
                    });
                }
                readers.insert(pass_idx);
            }
            if writes {
                if let Some(writer) = last_writer {
                    if writer != pass_idx {
                        edges.insert((writer, pass_idx));
                    }
                }
                for reader in readers.drain() {
                    if reader != pass_idx {
                        edges.insert((reader, pass_idx));
                    }
                }
                last_writer = Some(pass_idx);
            }
        }
    }
    Ok(())
}

fn resource_label(builder: &GraphBuilder, resource: ResourceHandle) -> String {
    match resource {
        ResourceHandle::Texture(TextureResourceHandle::Transient(h)) => builder
            .textures
            .get(h.index())
            .map(|d| d.label.to_string())
            .unwrap_or_else(|| format!("texture#{}", h.index())),
        ResourceHandle::Texture(TextureResourceHandle::Imported(h)) => builder
            .imports_tex
            .get(h.index())
            .map(|d| d.label.to_string())
            .unwrap_or_else(|| format!("imported_texture#{}", h.index())),
        ResourceHandle::Buffer(BufferResourceHandle::Transient(h)) => builder
            .buffers
            .get(h.index())
            .map(|d| d.label.to_string())
            .unwrap_or_else(|| format!("buffer#{}", h.index())),
        ResourceHandle::Buffer(BufferResourceHandle::Imported(h)) => builder
            .imports_buf
            .get(h.index())
            .map(|d| d.label.to_string())
            .unwrap_or_else(|| format!("imported_buffer#{}", h.index())),
    }
}
