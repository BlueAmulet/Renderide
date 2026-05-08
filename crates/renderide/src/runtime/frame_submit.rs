//! Host [`crate::shared::FrameSubmitData`] application on [`super::RendererRuntime`].

use std::time::Instant;

use super::camera_render_tasks::zero_camera_render_task_results;
use super::{RendererRuntime, lockstep};
use crate::shared::FrameSubmitData;

impl RendererRuntime {
    /// Applies a host frame submit to lock-step, output state, camera fields, scene caches, and
    /// head-output transform.
    pub(crate) fn apply_frame_submit_data(&mut self, data: FrameSubmitData) {
        let prev_frame_index = self.host_camera.frame_index;
        lockstep::trace_duplicate_frame_index_if_interesting(data.frame_index, prev_frame_index);
        self.process_frame_submit(data);
    }

    fn process_frame_submit(&mut self, data: FrameSubmitData) {
        profiling::scope!("scene::frame_submit");
        {
            profiling::scope!("scene::frame_submit_frontend_bookkeeping");
            self.frontend.note_frame_submit_processed(data.frame_index);
            self.frontend
                .apply_frame_submit_output(data.output_state.clone());
            self.set_last_submit_render_task_count(data.render_tasks.len());
        };

        {
            profiling::scope!("scene::frame_submit_camera_fields");
            crate::camera::apply_frame_submit_fields(&mut self.host_camera, &data);
        };

        let start = Instant::now();
        let mut apply_failed = false;
        let mut rendered_reflection_probes = Vec::new();
        let mut queue_camera_tasks = false;
        let mut failed_camera_tasks = 0u64;
        if let Some(ref mut shm) = self.frontend.shared_memory_mut() {
            {
                profiling::scope!("scene::frame_submit_apply_scene");
                match self.scene.apply_frame_submit(shm, &data) {
                    Ok(report) => self.backend.note_scene_apply_report(&report),
                    Err(e) => {
                        logger::error!("scene apply_frame_submit failed: {e}");
                        apply_failed = true;
                    }
                }
            }
            {
                profiling::scope!("scene::frame_submit_flush_world_caches");
                match self.scene.flush_world_caches() {
                    Ok(report) => self.backend.note_scene_cache_flush_report(&report),
                    Err(e) => {
                        logger::error!("scene flush_world_caches failed: {e}");
                        apply_failed = true;
                    }
                }
            }
            if !apply_failed {
                profiling::scope!("scene::frame_submit_reflection_probes");
                self.backend
                    .answer_reflection_probe_sh2_tasks(shm, &self.scene, &data);
                rendered_reflection_probes =
                    self.scene.take_supported_reflection_probe_render_results();
                queue_camera_tasks = !data.render_tasks.is_empty();
            } else if !data.render_tasks.is_empty() {
                let zero_failed = zero_camera_render_task_results(shm, &data.render_tasks);
                logger::warn!(
                    "zero-filled {} CameraRenderTask readback(s) after failed frame submit apply (zero_fill_failed={zero_failed})",
                    data.render_tasks.len()
                );
                failed_camera_tasks =
                    failed_camera_tasks.saturating_add(data.render_tasks.len() as u64);
            }
        } else if !data.render_tasks.is_empty() {
            logger::warn!(
                "dropping {} CameraRenderTask readback(s): frame submit has no shared memory accessor",
                data.render_tasks.len()
            );
            failed_camera_tasks =
                failed_camera_tasks.saturating_add(data.render_tasks.len() as u64);
        }
        if queue_camera_tasks {
            self.queue_camera_render_tasks(&data.render_tasks);
        }
        if failed_camera_tasks > 0 {
            self.note_camera_readback_results(0, failed_camera_tasks);
        }
        self.frontend
            .enqueue_rendered_reflection_probes(rendered_reflection_probes);
        if apply_failed {
            self.note_frame_submit_apply_failure();
            self.frontend.set_fatal_error(true);
        }
        {
            profiling::scope!("scene::frame_submit_host_camera_derive");
            self.host_camera.head_output_transform =
                crate::camera::head_output_from_active_main_space(&self.scene);
            self.host_camera.eye_world_position =
                crate::camera::eye_world_position_from_active_main_space(&self.scene);
        };

        logger::trace!(
            "frame_submit frame_index={} render_spaces={} render_tasks={} output_state={} debug_log={} near_clip={} far_clip={} desktop_fov_deg={} vr_active={} scene_apply_ms={:.3}",
            data.frame_index,
            data.render_spaces.len(),
            data.render_tasks.len(),
            data.output_state.is_some(),
            data.debug_log,
            self.host_camera.clip.near,
            self.host_camera.clip.far,
            self.host_camera.desktop_fov_degrees,
            self.host_camera.vr_active,
            start.elapsed().as_secs_f64() * 1000.0
        );
    }
}

#[cfg(test)]
mod tests {
    use glam::IVec2;

    use crate::shared::{CameraRenderParameters, CameraRenderTask, FrameSubmitData, TextureFormat};

    use super::RendererRuntime;

    #[test]
    fn successful_frame_submit_queues_camera_render_tasks() {
        let mut runtime = RendererRuntime::new(
            Option::<crate::connection::ConnectionParams>::None,
            std::sync::Arc::new(std::sync::RwLock::new(
                crate::config::RendererSettings::default(),
            )),
            std::path::PathBuf::from("test_config.toml"),
        );
        runtime.test_set_shared_memory("renderide-test-camera-queue");
        let data = FrameSubmitData {
            render_tasks: vec![CameraRenderTask {
                parameters: Some(CameraRenderParameters {
                    resolution: IVec2 { x: 2, y: 2 },
                    texture_format: TextureFormat::RGBA32,
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        };

        runtime.apply_frame_submit_data(data);

        assert_eq!(runtime.pending_camera_render_task_count(), 1);
    }
}
