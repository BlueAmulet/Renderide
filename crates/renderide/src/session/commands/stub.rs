//! Stub command handler for unimplemented RendererCommand variants.
//!
//! Must be last in the dispatcher. Exhaustive match ensures new variants cause a compile error.
//! Add real handlers above to implement features.

use crate::shared::RendererCommand;

use super::{CommandContext, CommandHandler, CommandResult};

/// Stub handler for all unimplemented RendererCommand variants. Documents the IPC surface and ensures
/// the dispatcher never silently falls through. Add real handlers above to implement features.
pub struct StubCommandHandler;

impl CommandHandler for StubCommandHandler {
    fn handle(&mut self, cmd: RendererCommand, _ctx: &mut CommandContext<'_>) -> CommandResult {
        match cmd {
            // --- Handled by earlier handlers; should not reach here ---
            RendererCommand::renderer_init_data(_)
            | RendererCommand::renderer_init_finalize_data(_)
            | RendererCommand::renderer_shutdown(_)
            | RendererCommand::renderer_shutdown_request(_)
            | RendererCommand::frame_submit_data(_)
            | RendererCommand::mesh_upload_data(_)
            | RendererCommand::mesh_unload(_)
            | RendererCommand::shader_upload(_)
            | RendererCommand::desktop_config(_)
            | RendererCommand::free_shared_memory_view(_)
            | RendererCommand::keep_alive(_)
            | RendererCommand::renderer_init_progress_update(_)
            | RendererCommand::renderer_engine_ready(_)
            | RendererCommand::renderer_init_result(_)
            | RendererCommand::frame_start_data(_) => {
                unreachable!("command handled by earlier handler")
            }

            // --- Window ---
            RendererCommand::renderer_parent_window(_) => {
                crate::trace!("Unhandled: renderer_parent_window (TODO: implement parent window embedding)");
                CommandResult::Handled
            }
            RendererCommand::set_window_icon(_) => {
                crate::trace!("Unhandled: set_window_icon (TODO: implement window icon)");
                CommandResult::Handled
            }
            RendererCommand::set_window_icon_result(_) => CommandResult::Handled,
            RendererCommand::set_taskbar_progress(_) => {
                crate::trace!("Unhandled: set_taskbar_progress (TODO: implement taskbar progress)");
                CommandResult::Handled
            }

            // --- Config ---
            RendererCommand::post_processing_config(_) => {
                crate::trace!("Unhandled: post_processing_config");
                CommandResult::Handled
            }
            RendererCommand::quality_config(_) => {
                crate::trace!("Unhandled: quality_config");
                CommandResult::Handled
            }
            RendererCommand::resolution_config(_) => {
                crate::trace!("Unhandled: resolution_config");
                CommandResult::Handled
            }
            RendererCommand::render_decoupling_config(_) => {
                crate::trace!("Unhandled: render_decoupling_config");
                CommandResult::Handled
            }
            RendererCommand::gaussian_splat_config(_) => {
                crate::trace!("Unhandled: gaussian_splat_config");
                CommandResult::Handled
            }

            // --- Response / no-op ---
            RendererCommand::mesh_upload_result(_) => CommandResult::Handled,

            // --- Shaders ---
            RendererCommand::shader_unload(_) => {
                crate::trace!("Unhandled: shader_unload");
                CommandResult::Handled
            }
            RendererCommand::shader_upload_result(_) => CommandResult::Handled,

            // --- Materials ---
            RendererCommand::material_property_id_request(_) => {
                crate::trace!("Unhandled: material_property_id_request");
                CommandResult::Handled
            }
            RendererCommand::material_property_id_result(_) => CommandResult::Handled,
            RendererCommand::materials_update_batch(_) => {
                crate::trace!("Unhandled: materials_update_batch");
                CommandResult::Handled
            }
            RendererCommand::materials_update_batch_result(_) => CommandResult::Handled,
            RendererCommand::unload_material(_) => {
                crate::trace!("Unhandled: unload_material");
                CommandResult::Handled
            }
            RendererCommand::unload_material_property_block(_) => {
                crate::trace!("Unhandled: unload_material_property_block");
                CommandResult::Handled
            }

            // --- Textures 2D ---
            RendererCommand::set_texture_2d_format(_) => {
                crate::trace!("Unhandled: set_texture_2d_format");
                CommandResult::Handled
            }
            RendererCommand::set_texture_2d_properties(_) => {
                crate::trace!("Unhandled: set_texture_2d_properties");
                CommandResult::Handled
            }
            RendererCommand::set_texture_2d_data(_) => {
                crate::trace!("Unhandled: set_texture_2d_data");
                CommandResult::Handled
            }
            RendererCommand::set_texture_2d_result(_) => CommandResult::Handled,
            RendererCommand::unload_texture_2d(_) => {
                crate::trace!("Unhandled: unload_texture_2d");
                CommandResult::Handled
            }

            // --- Textures 3D ---
            RendererCommand::set_texture_3d_format(_) => {
                crate::trace!("Unhandled: set_texture_3d_format");
                CommandResult::Handled
            }
            RendererCommand::set_texture_3d_properties(_) => {
                crate::trace!("Unhandled: set_texture_3d_properties");
                CommandResult::Handled
            }
            RendererCommand::set_texture_3d_data(_) => {
                crate::trace!("Unhandled: set_texture_3d_data");
                CommandResult::Handled
            }
            RendererCommand::set_texture_3d_result(_) => CommandResult::Handled,
            RendererCommand::unload_texture_3d(_) => {
                crate::trace!("Unhandled: unload_texture_3d");
                CommandResult::Handled
            }

            // --- Cubemaps ---
            RendererCommand::set_cubemap_format(_) => {
                crate::trace!("Unhandled: set_cubemap_format");
                CommandResult::Handled
            }
            RendererCommand::set_cubemap_properties(_) => {
                crate::trace!("Unhandled: set_cubemap_properties");
                CommandResult::Handled
            }
            RendererCommand::set_cubemap_data(_) => {
                crate::trace!("Unhandled: set_cubemap_data");
                CommandResult::Handled
            }
            RendererCommand::set_cubemap_result(_) => CommandResult::Handled,
            RendererCommand::unload_cubemap(_) => {
                crate::trace!("Unhandled: unload_cubemap");
                CommandResult::Handled
            }

            // --- Render textures ---
            RendererCommand::set_render_texture_format(_) => {
                crate::trace!("Unhandled: set_render_texture_format");
                CommandResult::Handled
            }
            RendererCommand::render_texture_result(_) => CommandResult::Handled,
            RendererCommand::unload_render_texture(_) => {
                crate::trace!("Unhandled: unload_render_texture");
                CommandResult::Handled
            }

            // --- Desktop textures ---
            RendererCommand::set_desktop_texture_properties(_) => {
                crate::trace!("Unhandled: set_desktop_texture_properties");
                CommandResult::Handled
            }
            RendererCommand::desktop_texture_properties_update(_) => {
                crate::trace!("Unhandled: desktop_texture_properties_update");
                CommandResult::Handled
            }
            RendererCommand::unload_desktop_texture(_) => {
                crate::trace!("Unhandled: unload_desktop_texture");
                CommandResult::Handled
            }

            // --- Point / trail buffers ---
            RendererCommand::point_render_buffer_upload(_) => {
                crate::trace!("Unhandled: point_render_buffer_upload");
                CommandResult::Handled
            }
            RendererCommand::point_render_buffer_consumed(_) => CommandResult::Handled,
            RendererCommand::point_render_buffer_unload(_) => {
                crate::trace!("Unhandled: point_render_buffer_unload");
                CommandResult::Handled
            }
            RendererCommand::trail_render_buffer_upload(_) => {
                crate::trace!("Unhandled: trail_render_buffer_upload");
                CommandResult::Handled
            }
            RendererCommand::trail_render_buffer_consumed(_) => CommandResult::Handled,
            RendererCommand::trail_render_buffer_unload(_) => {
                crate::trace!("Unhandled: trail_render_buffer_unload");
                CommandResult::Handled
            }

            // --- Gaussian splats ---
            RendererCommand::gaussian_splat_upload_raw(_) => {
                crate::trace!("Unhandled: gaussian_splat_upload_raw");
                CommandResult::Handled
            }
            RendererCommand::gaussian_splat_upload_encoded(_) => {
                crate::trace!("Unhandled: gaussian_splat_upload_encoded");
                CommandResult::Handled
            }
            RendererCommand::gaussian_splat_result(_) => CommandResult::Handled,
            RendererCommand::unload_gaussian_splat(_) => {
                crate::trace!("Unhandled: unload_gaussian_splat");
                CommandResult::Handled
            }

            // --- Lights buffer ---
            RendererCommand::lights_buffer_renderer_submission(_) => {
                crate::trace!("Unhandled: lights_buffer_renderer_submission");
                CommandResult::Handled
            }
            RendererCommand::lights_buffer_renderer_consumed(_) => CommandResult::Handled,

            // --- Reflection probes ---
            RendererCommand::reflection_probe_render_result(_) => {
                crate::trace!("Unhandled: reflection_probe_render_result");
                CommandResult::Handled
            }

            // --- Video ---
            RendererCommand::video_texture_load(_) => {
                crate::trace!("Unhandled: video_texture_load");
                CommandResult::Handled
            }
            RendererCommand::video_texture_update(_) => {
                crate::trace!("Unhandled: video_texture_update");
                CommandResult::Handled
            }
            RendererCommand::video_texture_ready(_) => CommandResult::Handled,
            RendererCommand::video_texture_changed(_) => {
                crate::trace!("Unhandled: video_texture_changed");
                CommandResult::Handled
            }
            RendererCommand::video_texture_properties(_) => {
                crate::trace!("Unhandled: video_texture_properties");
                CommandResult::Handled
            }
            RendererCommand::video_texture_start_audio_track(_) => {
                crate::trace!("Unhandled: video_texture_start_audio_track");
                CommandResult::Handled
            }
            RendererCommand::unload_video_texture(_) => {
                crate::trace!("Unhandled: unload_video_texture");
                CommandResult::Handled
            }
        }
    }
}
