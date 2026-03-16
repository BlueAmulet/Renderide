//! Frame command handlers: frame_submit_data.

use crate::shared::RendererCommand;

use super::{CommandContext, CommandHandler, CommandResult};

/// Handles `frame_submit_data`. Stores data in context for Session to process after dispatch.
pub struct FrameSubmitCommandHandler;

impl CommandHandler for FrameSubmitCommandHandler {
    fn handle(&mut self, cmd: RendererCommand, ctx: &mut CommandContext<'_>) -> CommandResult {
        match cmd {
            RendererCommand::frame_submit_data(data) => {
                ctx.pending_frame_data = Some(data);
                CommandResult::Handled
            }
            _ => CommandResult::Ignored,
        }
    }
}
