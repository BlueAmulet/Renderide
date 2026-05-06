//! Pass-node trait hierarchy, builder, and setup data.
//!
//! ## Pass kinds
//!
//! The render graph stores `Vec<PassNode>`. Each node wraps one of three typed pass traits:
//!
//! | Kind | Trait | GPU work |
//! |------|-------|----------|
//! | [`PassKind::Raster`] | [`RasterPass`] | Graph opens render pass; pass records draws. |
//! | [`PassKind::Compute`] | [`ComputePass`] | Pass receives raw encoder; dispatches compute. |
//! | [`PassKind::Callback`] | [`CallbackPass`] | CPU-only; no encoder; uploads / blackboard writes. |
//!
//! ## Setup flow
//!
//! During graph build, each pass's [`RasterPass::setup`] / [`ComputePass::setup`] / etc. is
//! called with a [`PassBuilder`]. The builder accumulates resource declarations, attachment
//! templates, and the pass kind flag (`raster()` / `compute()` / `callback()`).
//! [`PassBuilder::finish`] validates the combination and emits a [`PassSetup`].

mod attachments;
pub mod builder;
pub mod callback;
pub mod compute;
pub mod node;
pub mod raster;
pub(crate) mod setup;

pub use builder::PassBuilder;
pub use callback::CallbackPass;
pub use compute::ComputePass;
#[cfg(test)]
pub use node::PassMergeHint;
pub use node::{GroupScope, PassKind, PassNode, PassPhase};
pub use raster::RasterPass;
pub use setup::PassSetup;
