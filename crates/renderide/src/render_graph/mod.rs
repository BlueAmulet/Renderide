//! Compile-time validated **render graph**: pass ordering, resource flow checks, and a single
//! command-encode path per frame (v1).
//!
//! ## Responsibilities
//!
//! - **[`GraphBuilder`]** — register [`RenderPass`] nodes, optional [`GraphBuilder::add_pass_if`],
//!   edges, then [`GraphBuilder::build`] for a topological order and producer/consumer validation.
//! - **[`CompiledRenderGraph`]** — immutable schedule; [`CompiledRenderGraph::execute`] acquires
//!   the swapchain at most once when any pass writes [`ResourceSlot::Backbuffer`], records all
//!   passes into one encoder, submits, and presents.
//!
//! ## Phase 2 (not implemented here)
//!
//! - Nested subgraphs / phase labels.
//! - Real GPU resource handles and automatic barriers per slot.
//! - Multiple encoders, parallel recording, and async compute queue routing.
//! - Graph reuse across frames with invalidation keys (resolution, MSAA, toggles).

mod builder;
mod compiled;
mod context;
mod error;
mod frame_params;
mod ids;
mod pass;
mod resources;

pub mod passes;

pub use builder::GraphBuilder;
pub use compiled::{CompileStats, CompiledRenderGraph};
pub use context::RenderPassContext;
pub use error::{GraphBuildError, GraphExecuteError, RenderPassError};
pub use frame_params::FrameRenderParams;
pub use ids::PassId;
pub use pass::RenderPass;
pub use resources::{PassResources, ResourceSlot};

/// Builds the default graph: mesh deform compute, then world forward (clear + depth + mesh draw).
pub fn build_default_main_graph() -> Result<CompiledRenderGraph, GraphBuildError> {
    let mut builder = GraphBuilder::new();
    let deform = builder.add_pass(Box::new(passes::MeshDeformPass::new()));
    let forward = builder.add_pass(Box::new(passes::WorldMeshForwardPass::new()));
    builder.add_edge(deform, forward);
    builder.build()
}

#[cfg(test)]
mod default_graph_tests {
    use super::*;

    #[test]
    fn default_main_needs_surface_and_two_passes() {
        let g = build_default_main_graph().expect("default graph");
        assert!(g.needs_surface_acquire());
        assert_eq!(g.pass_count(), 2);
        assert_eq!(g.compile_stats.topo_levels, 2);
    }
}
