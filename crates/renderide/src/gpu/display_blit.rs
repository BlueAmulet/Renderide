//! Desktop window blit for the host `BlitToDisplay` renderable.
//!
//! Mirrors `Renderite.Unity.TextureDisplayBlitter`: a host renderable carries a `texture_id`,
//! `display_index`, `background_color`, and per-axis flip flags. For the local user's display, the
//! renderer clears the swapchain to `background_color`, fits the texture into the surface with
//! letterbox bars (object-fit: contain), and draws it through a fullscreen triangle bound to a
//! viewport sized to the fitted rect. The bars stay in the cleared color.
//!
//! ### Mapping vs VR mirror
//!
//! [`crate::gpu::vr_mirror`] uses **cover** (fill) UV mapping for the HMD eye preview. This module
//! uses **fit/letterbox** to match Unity's `TextureDisplayBlitter.Blit()` math.

mod fit;
mod pipelines;
mod resources;
mod surface_blit;

pub use resources::DisplayBlitResources;
pub use surface_blit::DisplayBlitSource;
