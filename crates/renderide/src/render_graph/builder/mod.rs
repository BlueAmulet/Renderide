//! Render graph builder: setup collection, dependency synthesis, culling, and alias planning.

mod decl;
mod edges;
mod lifetime;
mod topo;
mod validate;

#[cfg(test)]
mod tests;

use decl::{GroupEntry, PassEntry, SetupEntry};
use edges::{add_group_edges, add_resource_edges, explicit_edges};
use lifetime::{compile_buffers, compile_textures};
use topo::{retained_ordinals, retained_passes, topo_sort};
use validate::validate_handles;

use super::compiled::{
    ColorAttachmentTemplate, CompileStats, CompiledBufferResource, CompiledGroup, CompiledPassInfo,
    CompiledRenderGraph, CompiledTextureResource, DepthAttachmentTemplate, RenderPassTemplate,
};
use super::error::GraphBuildError;
use super::ids::{GroupId, PassId};
use super::pass::{GroupScope, PassBuilder, PassPhase, PassSetup, RenderPass};
use super::resources::{
    BufferHandle, ImportSource, ImportedBufferDecl, ImportedBufferHandle, ImportedTextureDecl,
    ImportedTextureHandle, ResourceHandle, TextureHandle, TextureResourceHandle,
    TransientBufferDesc, TransientTextureDesc,
};

/// Builder for a typed render graph.
pub struct GraphBuilder {
    textures: Vec<TransientTextureDesc>,
    buffers: Vec<TransientBufferDesc>,
    imports_tex: Vec<ImportedTextureDecl>,
    imports_buf: Vec<ImportedBufferDecl>,
    passes: Vec<PassEntry>,
    edges: Vec<(usize, usize)>,
    groups: Vec<GroupEntry>,
    default_frame_group: GroupId,
    default_per_view_group: GroupId,
}

impl GraphBuilder {
    /// Empty builder with default frame-global and per-view groups.
    pub fn new() -> Self {
        let default_frame_group = GroupId(0);
        let default_per_view_group = GroupId(1);
        Self {
            textures: Vec::new(),
            buffers: Vec::new(),
            imports_tex: Vec::new(),
            imports_buf: Vec::new(),
            passes: Vec::new(),
            edges: Vec::new(),
            groups: vec![
                GroupEntry {
                    name: "frame-global",
                    scope: GroupScope::FrameGlobal,
                    after: Vec::new(),
                },
                GroupEntry {
                    name: "per-view",
                    scope: GroupScope::PerView,
                    after: vec![default_frame_group],
                },
            ],
            default_frame_group,
            default_per_view_group,
        }
    }

    /// Declares a graph-owned transient texture.
    pub fn create_texture(&mut self, desc: TransientTextureDesc) -> TextureHandle {
        let handle = TextureHandle(self.textures.len() as u32);
        self.textures.push(desc);
        handle
    }

    /// Declares a graph-owned transient buffer.
    pub fn create_buffer(&mut self, desc: TransientBufferDesc) -> BufferHandle {
        let handle = BufferHandle(self.buffers.len() as u32);
        self.buffers.push(desc);
        handle
    }

    /// Declares an imported texture.
    pub fn import_texture(&mut self, decl: ImportedTextureDecl) -> ImportedTextureHandle {
        let handle = ImportedTextureHandle(self.imports_tex.len() as u32);
        self.imports_tex.push(decl);
        handle
    }

    /// Declares an imported buffer.
    pub fn import_buffer(&mut self, decl: ImportedBufferDecl) -> ImportedBufferHandle {
        let handle = ImportedBufferHandle(self.imports_buf.len() as u32);
        self.imports_buf.push(decl);
        handle
    }

    /// Creates an explicit scheduling group.
    pub fn group(&mut self, name: &'static str, scope: GroupScope) -> GroupId {
        let id = GroupId(self.groups.len());
        self.groups.push(GroupEntry {
            name,
            scope,
            after: Vec::new(),
        });
        id
    }

    /// Orders `group` after `dependency`.
    pub fn group_after(&mut self, group: GroupId, dependency: GroupId) {
        if let Some(entry) = self.groups.get_mut(group.0) {
            entry.after.push(dependency);
        }
    }

    /// Appends a pass to the default group matching its [`PassPhase`].
    pub fn add_pass(&mut self, pass: Box<dyn RenderPass>) -> PassId {
        let group = match pass.phase() {
            PassPhase::FrameGlobal => self.default_frame_group,
            PassPhase::PerView => self.default_per_view_group,
        };
        self.add_pass_to_group(group, pass)
    }

    /// Appends a pass to a specific group.
    pub fn add_pass_to_group(&mut self, group: GroupId, pass: Box<dyn RenderPass>) -> PassId {
        let id = PassId(self.passes.len());
        self.passes.push(PassEntry { group, pass });
        id
    }

    /// Appends a pass only when `condition` is true.
    pub fn add_pass_if(&mut self, condition: bool, pass: Box<dyn RenderPass>) -> Option<PassId> {
        if condition {
            Some(self.add_pass(pass))
        } else {
            None
        }
    }

    /// Ensures `from` is scheduled before `to`.
    pub fn add_edge(&mut self, from: PassId, to: PassId) {
        self.edges.push((from.0, to.0));
    }

    /// Compiles setup declarations into an immutable graph.
    pub fn build(mut self) -> Result<CompiledRenderGraph, GraphBuildError> {
        let n = self.passes.len();
        if n == 0 {
            return Ok(self.empty_graph());
        }

        let setups = self.collect_setup()?;
        let mut edges = explicit_edges(&self, n)?;
        add_group_edges(&self, &setups, &mut edges)?;
        add_resource_edges(&self, &setups, &mut edges)?;

        let (sorted, topo_levels) = topo_sort(n, &edges)?;
        #[cfg(debug_assertions)]
        {
            let mut pos = vec![0usize; n];
            for (ord, &node) in sorted.iter().enumerate() {
                pos[node] = ord;
            }
            for &(u, v) in &edges {
                debug_assert!(
                    pos[u] < pos[v],
                    "topological order violates edge ({u} -> {v})"
                );
            }
        }
        let keep = retained_passes(n, &edges, &setups);
        let culled_count = n.saturating_sub(keep.len());
        let ordered: Vec<usize> = sorted
            .into_iter()
            .filter(|idx| keep.contains(idx))
            .collect();

        let retained_ord = retained_ordinals(&ordered);
        let (compiled_textures, texture_slots) =
            compile_textures(&self.textures, &setups, &retained_ord);
        let (compiled_buffers, buffer_slots) =
            compile_buffers(&self.buffers, &setups, &retained_ord);
        let pass_info = compile_pass_info(&setups, &ordered);
        let groups = compile_groups(&self.groups, &pass_info);
        let needs_surface_acquire = needs_surface_acquire(&pass_info, &self.imports_tex);

        let mut pass_take: Vec<Option<Box<dyn RenderPass>>> = self
            .passes
            .into_iter()
            .map(|entry| Some(entry.pass))
            .collect();
        let mut ordered_passes = Vec::with_capacity(ordered.len());
        for idx in ordered {
            let Some(pass) = pass_take[idx].take() else {
                return Err(GraphBuildError::PassOwnershipInvariant {
                    message: "pass index taken more than once during build",
                });
            };
            ordered_passes.push(pass);
        }

        let mut frame_global_pass_indices = Vec::new();
        let mut per_view_pass_indices = Vec::new();
        for (i, pass) in ordered_passes.iter().enumerate() {
            match pass.phase() {
                PassPhase::FrameGlobal => frame_global_pass_indices.push(i),
                PassPhase::PerView => per_view_pass_indices.push(i),
            }
        }

        Ok(CompiledRenderGraph {
            passes: ordered_passes,
            needs_surface_acquire,
            compile_stats: CompileStats {
                pass_count: pass_info.len(),
                topo_levels,
                culled_count,
                transient_texture_count: self.textures.len(),
                transient_texture_slots: texture_slots,
                transient_buffer_count: self.buffers.len(),
                transient_buffer_slots: buffer_slots,
                imported_texture_count: self.imports_tex.len(),
                imported_buffer_count: self.imports_buf.len(),
            },
            groups,
            pass_info,
            transient_textures: compiled_textures,
            transient_buffers: compiled_buffers,
            imported_textures: self.imports_tex,
            imported_buffers: self.imports_buf,
            frame_global_pass_indices,
            per_view_pass_indices,
            main_graph_msaa_transient_handles: None,
        })
    }

    fn empty_graph(self) -> CompiledRenderGraph {
        CompiledRenderGraph {
            passes: Vec::new(),
            needs_surface_acquire: false,
            compile_stats: CompileStats {
                transient_texture_count: self.textures.len(),
                transient_buffer_count: self.buffers.len(),
                imported_texture_count: self.imports_tex.len(),
                imported_buffer_count: self.imports_buf.len(),
                ..CompileStats::default()
            },
            groups: Vec::new(),
            pass_info: Vec::new(),
            transient_textures: self
                .textures
                .into_iter()
                .map(|desc| CompiledTextureResource {
                    usage: desc.base_usage,
                    desc,
                    lifetime: None,
                    physical_slot: usize::MAX,
                })
                .collect(),
            transient_buffers: self
                .buffers
                .into_iter()
                .map(|desc| CompiledBufferResource {
                    usage: desc.base_usage,
                    desc,
                    lifetime: None,
                    physical_slot: usize::MAX,
                })
                .collect(),
            imported_textures: self.imports_tex,
            imported_buffers: self.imports_buf,
            frame_global_pass_indices: Vec::new(),
            per_view_pass_indices: Vec::new(),
            main_graph_msaa_transient_handles: None,
        }
    }

    fn collect_setup(&mut self) -> Result<Vec<SetupEntry>, GraphBuildError> {
        let texture_count = self.textures.len();
        let buffer_count = self.buffers.len();
        let imported_texture_count = self.imports_tex.len();
        let imported_buffer_count = self.imports_buf.len();

        let mut setups = Vec::with_capacity(self.passes.len());
        for (idx, entry) in self.passes.iter_mut().enumerate() {
            let id = PassId(idx);
            let name = entry.pass.name().to_string();
            let mut builder = PassBuilder::new(&name);
            entry
                .pass
                .setup(&mut builder)
                .map_err(|source| GraphBuildError::Setup {
                    pass: id,
                    name: name.clone(),
                    source,
                })?;
            let setup = builder.finish().map_err(|source| GraphBuildError::Setup {
                pass: id,
                name: name.clone(),
                source,
            })?;
            validate_handles(
                &setup,
                texture_count,
                buffer_count,
                imported_texture_count,
                imported_buffer_count,
            )
            .map_err(|source| GraphBuildError::Setup {
                pass: id,
                name: name.clone(),
                source,
            })?;
            setups.push(SetupEntry {
                id,
                group: entry.group,
                name,
                setup,
            });
        }
        Ok(setups)
    }
}

impl Default for GraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

fn compile_pass_info(setups: &[SetupEntry], ordered: &[usize]) -> Vec<CompiledPassInfo> {
    ordered
        .iter()
        .copied()
        .map(|idx| {
            let setup = &setups[idx];
            let raster_template = compile_raster_template(&setup.setup);
            CompiledPassInfo {
                id: setup.id,
                name: setup.name.clone(),
                group: setup.group,
                kind: setup.setup.kind,
                accesses: setup.setup.accesses.clone(),
                multiview_mask: setup.setup.multiview_mask,
                raster_template,
            }
        })
        .collect()
}

fn compile_raster_template(setup: &PassSetup) -> Option<RenderPassTemplate> {
    let color_attachments: Vec<ColorAttachmentTemplate> = setup
        .color_attachments
        .iter()
        .map(|color| ColorAttachmentTemplate {
            target: color.target,
            load: color.load,
            store: color.store,
            resolve_to: color.resolve_to,
        })
        .collect();
    let depth_stencil_attachment =
        setup
            .depth_stencil_attachment
            .as_ref()
            .map(|depth| DepthAttachmentTemplate {
                target: depth.target,
                depth: depth.depth,
                stencil: depth.stencil,
            });
    (!color_attachments.is_empty() || depth_stencil_attachment.is_some()).then_some(
        RenderPassTemplate {
            color_attachments,
            depth_stencil_attachment,
            multiview_mask: setup.multiview_mask,
        },
    )
}

fn compile_groups(groups: &[GroupEntry], pass_info: &[CompiledPassInfo]) -> Vec<CompiledGroup> {
    groups
        .iter()
        .enumerate()
        .filter_map(|(idx, group)| {
            let id = GroupId(idx);
            let pass_indices: Vec<usize> = pass_info
                .iter()
                .enumerate()
                .filter_map(|(pass_idx, info)| (info.group == id).then_some(pass_idx))
                .collect();
            (!pass_indices.is_empty()).then_some(CompiledGroup {
                id,
                name: group.name,
                scope: group.scope,
                pass_indices,
            })
        })
        .collect()
}

fn needs_surface_acquire(pass_info: &[CompiledPassInfo], imports: &[ImportedTextureDecl]) -> bool {
    pass_info.iter().any(|pass| {
        pass.accesses.iter().any(|access| {
            if !access.writes() {
                return false;
            }
            let ResourceHandle::Texture(TextureResourceHandle::Imported(handle)) = access.resource
            else {
                return false;
            };
            imports.get(handle.index()).is_some_and(|decl| {
                matches!(
                    decl.source,
                    ImportSource::FrameTarget(super::resources::FrameTargetRole::ColorAttachment)
                )
            })
        })
    })
}
