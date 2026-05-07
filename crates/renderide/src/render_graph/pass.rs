//! Pass-node trait hierarchy, builder, and setup data.
//!
//! ## Pass kinds
//!
//! The render graph stores `Vec<PassNode>`. Each node wraps one of two typed pass traits:
//!
//! | Kind | Trait | GPU work |
//! |------|-------|----------|
//! | [`PassKind::Raster`] | [`RasterPass`] | Graph opens render pass; pass records draws. |
//! | [`PassKind::Compute`] | [`ComputePass`] | Pass receives raw encoder; dispatches compute. |
//!
//! ## Setup flow
//!
//! During graph build, each pass's [`RasterPass::setup`] / [`ComputePass::setup`] is called with a
//! [`PassBuilder`]. The builder accumulates resource declarations, attachment templates, and the
//! pass kind flag (`raster()` / `compute()`).
//! [`PassBuilder::finish`] validates the combination and emits a [`PassSetup`].

mod attachments;
pub mod builder;
pub mod compute;
pub mod node;
pub mod raster;
pub(crate) mod setup;

pub use builder::PassBuilder;
pub use compute::ComputePass;
#[cfg(test)]
pub use node::PassMergeHint;
pub use node::{GroupScope, PassKind, PassNode, PassPhase};
pub use raster::RasterPass;
pub use setup::PassSetup;
