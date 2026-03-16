//! Texture command handlers: 2D, 3D, cubemap, render texture, desktop texture.
//!
//! Placeholder; returns Ignored for all texture commands. StubCommandHandler handles them until implemented.

use crate::shared::RendererCommand;

use super::{CommandContext, CommandHandler, CommandResult};

/// Handles texture commands. Placeholder; returns Ignored until texture support is implemented.
pub struct TextureCommandHandler;

impl CommandHandler for TextureCommandHandler {
    fn handle(&mut self, _cmd: RendererCommand, _ctx: &mut CommandContext<'_>) -> CommandResult {
        CommandResult::Ignored
    }
}
