//! Config command handlers: desktop_config, resolution_config, quality_config, etc.

use crate::config::RenderConfig;
use crate::shared::RendererCommand;

use super::{CommandContext, CommandHandler, CommandResult};

/// Handles `desktop_config`. Updates view state and render config. Post-finalize only.
pub struct ConfigCommandHandler;

impl CommandHandler for ConfigCommandHandler {
    fn handle(&mut self, cmd: RendererCommand, ctx: &mut CommandContext<'_>) -> CommandResult {
        if !*ctx.init_finalized {
            return CommandResult::Ignored;
        }
        match cmd {
            RendererCommand::desktop_config(x) => {
                ctx.view_state.near_clip = 0.01;
                ctx.view_state.far_clip = 1024.0;
                ctx.view_state.desktop_fov = 75.0;
                *ctx.render_config = RenderConfig {
                    near_clip: 0.01,
                    far_clip: 1024.0,
                    desktop_fov: 75.0,
                    vsync: x.v_sync,
                    use_debug_uv: ctx.render_config.use_debug_uv,
                };
                CommandResult::Handled
            }
            _ => CommandResult::Ignored,
        }
    }
}
