//! Material command handlers: material_property_id_request, materials_update_batch, unload_material, etc.
//!
//! Placeholder; returns Ignored for all material commands. StubCommandHandler handles them until implemented.

use crate::shared::RendererCommand;

use super::{CommandContext, CommandHandler, CommandResult};

/// Handles material commands. Placeholder; returns Ignored until material support is implemented.
pub struct MaterialCommandHandler;

impl CommandHandler for MaterialCommandHandler {
    fn handle(&mut self, _cmd: RendererCommand, _ctx: &mut CommandContext<'_>) -> CommandResult {
        CommandResult::Ignored
    }
}
