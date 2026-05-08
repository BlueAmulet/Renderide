//! Persistent display-blit state: per-format pipeline cache and a shared 16-byte UV uniform.
//!
//! Per-frame blit logic lives in the sibling [`super::surface_blit`] module.

use super::pipelines::surface_pipeline;

/// GPU resources for the desktop `BlitToDisplay` pass.
///
/// Shared across frames; the only per-format reconfigure is the surface pipeline when the
/// swapchain format changes (rare, e.g. window-move HDR transition).
#[derive(Debug, Default)]
pub struct DisplayBlitResources {
    uniform_buf: Option<wgpu::Buffer>,
    pipeline: Option<(wgpu::TextureFormat, wgpu::RenderPipeline)>,
}

impl DisplayBlitResources {
    /// Empty resources; the GPU buffer and pipeline are lazily created on first blit.
    pub fn new() -> Self {
        Self::default()
    }

    pub(super) fn uniform_buffer(&self) -> Option<&wgpu::Buffer> {
        self.uniform_buf.as_ref()
    }

    pub(super) fn ensure_uniform(&mut self, device: &wgpu::Device) {
        if self.uniform_buf.is_some() {
            return;
        }
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("display_blit_uv"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        crate::profiling::note_resource_churn!(Buffer, "gpu::display_blit_uniform");
        self.uniform_buf = Some(buf);
    }

    pub(super) fn pipeline_for_format(
        &mut self,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
    ) -> &wgpu::RenderPipeline {
        let entry = self
            .pipeline
            .get_or_insert_with(|| (format, surface_pipeline(device, format)));
        if entry.0 != format {
            *entry = (format, surface_pipeline(device, format));
        }
        &entry.1
    }
}
