//! Post-processing stack framework: trait, signature, and graph wiring helpers.
//!
//! Effects are inserted between the world-mesh forward HDR producer
//! ([`crate::passes::WorldMeshForwardOpaquePass`]) and the displayable target blit
//! ([`crate::passes::SceneColorComposePass`]). Each effect registers a subgraph on
//! the builder whose head samples one HDR float texture and whose tail writes another; the
//! [`PostProcessChain`] allocates the ping-pong HDR slots and wires edges between effects. Most
//! effects contribute a single raster pass (GTAO, ACES tonemap); a few (bloom) register a mip
//! ladder terminating in a single composite pass.
//!
//! See [`crate::passes::post_processing`] for concrete effect implementations:
//! [`GtaoPass`](crate::passes::post_processing::GtaoPass),
//! [`BloomEffect`](crate::passes::post_processing::BloomEffect), and
//! [`AcesTonemapPass`](crate::passes::post_processing::AcesTonemapPass).

mod chain;
pub(crate) mod effect;
mod ping_pong;

pub use chain::{ChainOutput, PostProcessChain, PostProcessChainSignature};
pub use effect::{EffectPasses, PostProcessEffect, PostProcessEffectId};
