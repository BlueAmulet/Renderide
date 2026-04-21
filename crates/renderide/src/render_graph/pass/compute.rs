//! [`ComputePass`] trait for encoder-driven compute and mixed compute/copy work.
//!
//! Unlike [`super::raster::RasterPass`], the graph does not open any GPU pass object for compute
//! passes — the implementor receives the context (which includes the [`wgpu::CommandEncoder`])
//! and dispatches compute workgroups or uses the encoder API directly.

use crate::render_graph::context::{ComputePassCtx, PostSubmitContext};
use crate::render_graph::error::{RenderPassError, SetupError};

use super::builder::PassBuilder;
use super::node::PassPhase;

/// A graph node whose GPU work is encoder-driven compute (compute shaders, pipeline barriers,
/// compute dispatch, or mixed compute/copy operations).
pub trait ComputePass: Send {
    /// Stable name for logging, profiling, and error messages.
    fn name(&self) -> &str;

    /// Declares resource accesses and compute intent.
    ///
    /// The implementor must call `builder.compute()`.
    fn setup(&mut self, builder: &mut PassBuilder<'_>) -> Result<(), SetupError>;

    /// Records GPU compute commands.
    ///
    /// The [`wgpu::CommandEncoder`] is accessible via `ctx.encoder`. The pass opens and closes
    /// compute sub-passes on it directly.
    fn record(&mut self, ctx: &mut ComputePassCtx<'_, '_, '_>) -> Result<(), RenderPassError>;

    /// Scheduling phase. Defaults to [`PassPhase::PerView`].
    fn phase(&self) -> PassPhase {
        PassPhase::PerView
    }

    /// Runs after the encoder containing this pass is submitted.
    ///
    /// Default is a no-op.
    fn post_submit(&mut self, _ctx: &mut PostSubmitContext<'_>) -> Result<(), RenderPassError> {
        Ok(())
    }
}
