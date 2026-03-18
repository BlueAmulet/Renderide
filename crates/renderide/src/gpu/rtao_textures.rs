//! Cached RTAO (Ray-Traced Ambient Occlusion) MRT textures.
//!
//! Reuses textures across frames; recreates only when viewport dimensions change.
//! Avoids per-frame texture allocation which is one of the most expensive GPU operations.

/// Cached RTAO MRT textures and views.
///
/// Stored in [`GpuState`](super::state::GpuState) when RTAO is enabled.
/// Recreated only when viewport (width, height) changes.
pub struct RtaoTextureCache {
    /// Viewport dimensions these textures were created for.
    pub width: u32,
    /// Viewport height.
    pub height: u32,
    /// Color texture (matches surface format). Mesh pass renders to this.
    pub color_texture: wgpu::Texture,
    /// Color texture view.
    pub color_view: wgpu::TextureView,
    /// Position G-buffer (Rgba16Float).
    pub position_texture: wgpu::Texture,
    /// Position texture view.
    pub position_view: wgpu::TextureView,
    /// Normal G-buffer (Rgba16Float).
    pub normal_texture: wgpu::Texture,
    /// Normal texture view.
    pub normal_view: wgpu::TextureView,
    /// Raw AO output from RTAO compute (Rgba8Unorm).
    pub ao_raw_texture: wgpu::Texture,
    /// Raw AO texture view.
    pub ao_raw_view: wgpu::TextureView,
    /// Blurred AO output (Rgba8Unorm).
    pub ao_texture: wgpu::Texture,
    /// AO texture view.
    pub ao_view: wgpu::TextureView,
}

impl RtaoTextureCache {
    /// Creates RTAO textures for the given viewport and color format.
    ///
    /// Call only when cache is missing or dimensions changed.
    pub fn create(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        color_format: wgpu::TextureFormat,
    ) -> Self {
        let color_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("RTAO MRT color texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: color_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let position_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("RTAO MRT position texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let normal_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("RTAO MRT normal texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let ao_raw_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("RTAO AO raw texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let ao_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("RTAO AO texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let color_view = color_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let position_view = position_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let normal_view = normal_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let ao_raw_view = ao_raw_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let ao_view = ao_tex.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            width,
            height,
            color_texture: color_tex,
            color_view,
            position_texture: position_tex,
            position_view,
            normal_texture: normal_tex,
            normal_view,
            ao_raw_texture: ao_raw_tex,
            ao_raw_view,
            ao_texture: ao_tex,
            ao_view,
        }
    }

    /// Returns true if this cache matches the given viewport dimensions.
    pub fn matches_viewport(&self, width: u32, height: u32) -> bool {
        self.width == width && self.height == height
    }
}
