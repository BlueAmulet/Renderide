//! Command handler pattern for renderer commands.
//!
//! Replaces the monolithic match in Session with a registry of handlers.
//! New commands can be added by implementing CommandHandler without editing Session.
//!
//! ## Handler order matters
//!
//! - **Init first**: Before `init_received`, only InitCommandHandler accepts commands; others must Ignore.
//! - **Frame before assets**: Within a single poll, frame_submit_data must be processed before asset
//!   uploads so scene updates and render tasks are applied before new meshes/textures are used.
//!   Session::process_commands partitions commands accordingly.
//! - **Stub last**: StubCommandHandler catches all unimplemented variants; add real handlers above.

mod config;
mod frame;
mod init;
mod material;
mod mesh;
mod noop;
mod shared_memory;
mod shader;
mod shutdown;
mod stub;
mod texture;
mod window;

use crate::assets::AssetRegistry;
use crate::config::RenderConfig;
use crate::ipc::receiver::CommandReceiver;
use crate::ipc::shared_memory::SharedMemoryAccessor;
use crate::scene::SceneGraph;
use crate::session::state::ViewState;
use crate::shared::{FrameSubmitData, RendererCommand};

/// Result of handling a command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandResult {
    /// Handler processed the command; dispatch stops.
    Handled,
    /// Handler did not handle; try next handler.
    Ignored,
    /// Fatal error; dispatch stops and Session sets fatal_error.
    FatalError,
}

/// Context passed to command handlers with mutable access to Session state.
pub struct CommandContext<'a> {
    /// Shared memory accessor for reading asset data.
    pub shared_memory: &'a mut Option<SharedMemoryAccessor>,
    /// Asset registry for mesh uploads/unloads.
    pub asset_registry: &'a mut AssetRegistry,
    /// Scene graph for frame updates.
    pub scene_graph: &'a mut SceneGraph,
    /// View state (clip planes, FOV).
    pub view_state: &'a mut ViewState,
    /// Command receiver for sending responses.
    pub receiver: &'a mut CommandReceiver,
    /// Whether renderer_init_data has been received.
    pub init_received: &'a mut bool,
    /// Whether renderer_init_finalize_data has been received.
    pub init_finalized: &'a mut bool,
    /// Whether shutdown was requested.
    pub shutdown: &'a mut bool,
    /// Whether a fatal error occurred.
    pub fatal_error: &'a mut bool,
    /// Whether the last frame was processed (for FrameStartData timing).
    pub last_frame_data_processed: &'a mut bool,
    /// Asset IDs unloaded this frame (drained by Session).
    pub pending_mesh_unloads: &'a mut Vec<i32>,
    /// Render configuration (clip planes, vsync).
    pub render_config: &'a mut RenderConfig,
    /// Whether cursor lock was requested.
    pub lock_cursor: &'a mut bool,
    /// Frame data to process; set by FrameSubmitCommandHandler, drained by Session.
    pub pending_frame_data: Option<FrameSubmitData>,
}

/// Trait for command handlers. Handlers are tried in order until one returns Handled or FatalError.
pub trait CommandHandler {
    /// Handles a command. Returns Handled to stop dispatch, Ignored to try next handler, FatalError to abort.
    fn handle(&mut self, cmd: RendererCommand, ctx: &mut CommandContext<'_>) -> CommandResult;
}

/// Dispatches commands to a list of handlers. Stops on Handled or FatalError.
pub struct CommandDispatcher {
    handlers: Vec<Box<dyn CommandHandler>>,
}

impl CommandDispatcher {
    /// Creates a new dispatcher with the default handler set.
    ///
    /// Handler order: init, init_finalize, shutdown, frame, mesh, shader, texture, material,
    /// config, window, shared_memory, noop, stub.
    pub fn new() -> Self {
        Self {
            handlers: vec![
                Box::new(init::InitCommandHandler),
                Box::new(init::InitFinalizeCommandHandler),
                Box::new(shutdown::ShutdownCommandHandler),
                Box::new(frame::FrameSubmitCommandHandler),
                Box::new(mesh::MeshCommandHandler),
                Box::new(shader::ShaderCommandHandler),
                Box::new(texture::TextureCommandHandler),
                Box::new(material::MaterialCommandHandler),
                Box::new(config::ConfigCommandHandler),
                Box::new(window::WindowCommandHandler),
                Box::new(shared_memory::FreeSharedMemoryCommandHandler),
                Box::new(noop::NoopCommandHandler),
                Box::new(stub::StubCommandHandler),
            ],
        }
    }

    /// Dispatches a command to handlers. Returns when a handler returns Handled or FatalError.
    pub fn dispatch(&mut self, cmd: RendererCommand, ctx: &mut CommandContext<'_>) -> CommandResult {
        for handler in &mut self.handlers {
            let result = handler.handle(cmd.clone(), ctx);
            match result {
                CommandResult::Handled | CommandResult::FatalError => return result,
                CommandResult::Ignored => continue,
            }
        }
        CommandResult::Handled
    }
}

impl Default for CommandDispatcher {
    fn default() -> Self {
        Self::new()
    }
}
