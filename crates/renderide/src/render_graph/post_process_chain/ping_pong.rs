//! Ping-pong HDR transient slot helper for post-processing chains.
//!
//! A two-slot rotation lets each effect read the previous effect's output and write into a
//! sibling slot without forcing the chain to allocate `N+1` transient targets. The first effect
//! reads the chain input and writes into [`PingPongHdrSlots::ping`]; subsequent effects swap
//! between [`PingPongHdrSlots::ping`] and [`PingPongHdrSlots::pong`] each step.

use crate::render_graph::builder::GraphBuilder;
use crate::render_graph::resources::{
    TextureHandle, TransientArrayLayers, TransientExtent, TransientSampleCount,
    TransientTextureDesc, TransientTextureFormat,
};

/// The two HDR ping-pong transient texture handles used by a post-processing chain.
#[derive(Clone, Copy, Debug)]
pub(super) struct PingPongHdrSlots {
    /// First-write target slot.
    pub ping: TextureHandle,
    /// Second-write target slot.
    pub pong: TextureHandle,
}

impl PingPongHdrSlots {
    /// Creates the two ping-pong transient HDR scene-color textures.
    pub fn new(builder: &mut GraphBuilder) -> Self {
        Self {
            ping: builder.create_texture(post_process_color_transient_desc(
                "post_processed_color_hdr_a",
            )),
            pong: builder.create_texture(post_process_color_transient_desc(
                "post_processed_color_hdr_b",
            )),
        }
    }
}

/// Walks a sequence of effects through a [`PingPongHdrSlots`] pair, exposing the current effect's
/// `(input, output)` handles and advancing the cursor each time an effect registers.
///
/// The first advance abandons the chain input and starts the rotation between
/// [`PingPongHdrSlots::ping`] and [`PingPongHdrSlots::pong`]. Subsequent advances are pure swaps.
pub(super) struct PingPongCursor {
    slots: PingPongHdrSlots,
    input: TextureHandle,
    output: TextureHandle,
    advanced_once: bool,
}

impl PingPongCursor {
    /// Starts the cursor at `(initial_input, slots.ping)`.
    pub fn start(slots: PingPongHdrSlots, initial_input: TextureHandle) -> Self {
        Self {
            input: initial_input,
            output: slots.ping,
            slots,
            advanced_once: false,
        }
    }

    /// Read source for the current effect.
    pub fn input(&self) -> TextureHandle {
        self.input
    }

    /// Write target for the current effect.
    pub fn output(&self) -> TextureHandle {
        self.output
    }

    /// Moves to the next effect: `input` becomes what the just-completed effect wrote, `output`
    /// becomes the sibling slot.
    pub fn advance(&mut self) {
        if self.advanced_once {
            std::mem::swap(&mut self.input, &mut self.output);
        } else {
            self.input = self.slots.ping;
            self.output = self.slots.pong;
            self.advanced_once = true;
        }
    }

    /// Slot the most recent effect wrote into (i.e. the chain's final output).
    pub fn last_output(&self) -> TextureHandle {
        self.input
    }
}

/// Standard transient texture descriptor used by both ping-pong slots.
fn post_process_color_transient_desc(label: &'static str) -> TransientTextureDesc {
    TransientTextureDesc {
        label,
        format: TransientTextureFormat::SceneColorHdr,
        extent: TransientExtent::Backbuffer,
        mip_levels: 1,
        sample_count: TransientSampleCount::Fixed(1),
        dimension: wgpu::TextureDimension::D2,
        array_layers: TransientArrayLayers::Frame,
        base_usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        alias: true,
    }
}
