//! 1x1 placeholder textures used as the default binding for unset `@group(1)` texture slots.

use std::sync::Arc;

use super::super::bind_kind::TextureBindKind;

/// Placeholder 1x1 texture for one [`TextureBindKind`].
pub(super) struct PlaceholderTexture {
    /// Underlying device texture.
    pub(super) texture: Arc<wgpu::Texture>,
    /// Default texture view used at binding time.
    pub(super) view: Arc<wgpu::TextureView>,
}

/// Texel color uploaded into a placeholder texture.
#[derive(Clone, Copy)]
pub(super) enum PlaceholderTextureColor {
    /// Opaque white texel.
    White,
    /// Opaque black texel.
    Black,
    /// Opaque flat tangent-space normal `(0.5, 0.5, 1.0)`.
    FlatNormal,
}

impl PlaceholderTextureColor {
    fn rgba(self) -> [u8; 4] {
        match self {
            Self::White => [255, 255, 255, 255],
            Self::Black => [0, 0, 0, 255],
            Self::FlatNormal => [128, 128, 255, 255],
        }
    }

    /// Texture format whose sampler decode preserves [`Self::rgba`] as the intended linear value.
    ///
    /// `White` and `Black` keep the sRGB format the color-slot placeholders shipped with so 0 and 1
    /// round-trip identically. `FlatNormal` must stay linear (`Rgba8Unorm`): a tangent-space normal
    /// component of 0.5 is stored as the byte 128, and the sRGB EOTF would decode that as ~0.216
    /// instead of 0.502.
    fn format(self) -> wgpu::TextureFormat {
        match self {
            Self::White | Self::Black => wgpu::TextureFormat::Rgba8UnormSrgb,
            Self::FlatNormal => wgpu::TextureFormat::Rgba8Unorm,
        }
    }
}

impl TextureBindKind {
    /// Texture descriptor parameters (label, dimension, layer count, view dimension, format).
    fn placeholder_descriptor(self, color: PlaceholderTextureColor) -> PlaceholderDescriptor {
        let format = color.format();
        match (self, color) {
            (TextureBindKind::Tex2D, PlaceholderTextureColor::White) => PlaceholderDescriptor {
                label: "embedded_default_white",
                view_label: None,
                dimension: wgpu::TextureDimension::D2,
                view_dimension: None,
                depth_or_array_layers: 1,
                format,
            },
            (TextureBindKind::Tex2D, PlaceholderTextureColor::Black) => PlaceholderDescriptor {
                label: "embedded_default_black",
                view_label: None,
                dimension: wgpu::TextureDimension::D2,
                view_dimension: None,
                depth_or_array_layers: 1,
                format,
            },
            (TextureBindKind::Tex2D, PlaceholderTextureColor::FlatNormal) => {
                PlaceholderDescriptor {
                    label: "embedded_default_flat_normal",
                    view_label: None,
                    dimension: wgpu::TextureDimension::D2,
                    view_dimension: None,
                    depth_or_array_layers: 1,
                    format,
                }
            }
            (TextureBindKind::Tex3D, PlaceholderTextureColor::White) => PlaceholderDescriptor {
                label: "embedded_default_white_3d",
                view_label: Some("embedded_default_white_3d_view"),
                dimension: wgpu::TextureDimension::D3,
                view_dimension: Some(wgpu::TextureViewDimension::D3),
                depth_or_array_layers: 1,
                format,
            },
            (TextureBindKind::Tex3D, PlaceholderTextureColor::Black) => PlaceholderDescriptor {
                label: "embedded_default_black_3d",
                view_label: Some("embedded_default_black_3d_view"),
                dimension: wgpu::TextureDimension::D3,
                view_dimension: Some(wgpu::TextureViewDimension::D3),
                depth_or_array_layers: 1,
                format,
            },
            (TextureBindKind::Cube, PlaceholderTextureColor::White) => PlaceholderDescriptor {
                label: "embedded_default_white_cube",
                view_label: Some("embedded_default_white_cube_view"),
                dimension: wgpu::TextureDimension::D2,
                view_dimension: Some(wgpu::TextureViewDimension::Cube),
                depth_or_array_layers: 6,
                format,
            },
            (TextureBindKind::Cube, PlaceholderTextureColor::Black) => PlaceholderDescriptor {
                label: "embedded_default_black_cube",
                view_label: Some("embedded_default_black_cube_view"),
                dimension: wgpu::TextureDimension::D2,
                view_dimension: Some(wgpu::TextureViewDimension::Cube),
                depth_or_array_layers: 6,
                format,
            },
            (
                TextureBindKind::Tex3D | TextureBindKind::Cube,
                PlaceholderTextureColor::FlatNormal,
            ) => {
                unreachable!(
                    "FlatNormal placeholder is 2D-only; normal-map fallbacks never request a 3D or cube view"
                )
            }
        }
    }
}

struct PlaceholderDescriptor {
    label: &'static str,
    view_label: Option<&'static str>,
    dimension: wgpu::TextureDimension,
    view_dimension: Option<wgpu::TextureViewDimension>,
    depth_or_array_layers: u32,
    format: wgpu::TextureFormat,
}

fn create_placeholder(
    device: &wgpu::Device,
    kind: TextureBindKind,
    color: PlaceholderTextureColor,
) -> PlaceholderTexture {
    let desc = kind.placeholder_descriptor(color);
    let texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
        label: Some(desc.label),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: desc.depth_or_array_layers,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: desc.dimension,
        format: desc.format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    }));
    let view_descriptor = wgpu::TextureViewDescriptor {
        label: desc.view_label,
        dimension: desc.view_dimension,
        ..Default::default()
    };
    let view = Arc::new(texture.create_view(&view_descriptor));
    crate::profiling::note_resource_churn!(
        TextureView,
        "materials::embedded_placeholder_texture_view"
    );
    PlaceholderTexture { texture, view }
}

/// Allocates a 1x1 white texture and a default view for `kind`.
pub(super) fn create_white(device: &wgpu::Device, kind: TextureBindKind) -> PlaceholderTexture {
    create_placeholder(device, kind, PlaceholderTextureColor::White)
}

/// Allocates a 1x1 black texture and a default view for `kind`.
pub(super) fn create_black(device: &wgpu::Device, kind: TextureBindKind) -> PlaceholderTexture {
    create_placeholder(device, kind, PlaceholderTextureColor::Black)
}

/// Allocates a 1x1 flat tangent-space normal texture (`(0.5, 0.5, 1.0)`) in linear `Rgba8Unorm`.
///
/// Used as the fallback view for normal-map slots so missing/unloaded bump maps render as a flat
/// surface rather than the tilted `(1, 1, 1)` direction a white placeholder decodes to.
pub(super) fn create_flat_normal(
    device: &wgpu::Device,
    kind: TextureBindKind,
) -> PlaceholderTexture {
    create_placeholder(device, kind, PlaceholderTextureColor::FlatNormal)
}

fn upload_placeholder(
    queue: &wgpu::Queue,
    placeholder: &PlaceholderTexture,
    kind: TextureBindKind,
    color: PlaceholderTextureColor,
) {
    let depth_or_array_layers = match kind {
        TextureBindKind::Tex2D | TextureBindKind::Tex3D => 1,
        TextureBindKind::Cube => 6,
    };
    let texel = color.rgba();
    let mut bytes: Vec<u8> = Vec::with_capacity(4 * depth_or_array_layers as usize);
    for _ in 0..depth_or_array_layers {
        bytes.extend_from_slice(&texel);
    }
    let rows_per_image = match kind {
        TextureBindKind::Tex2D => None,
        TextureBindKind::Tex3D | TextureBindKind::Cube => Some(1),
    };
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: placeholder.texture.as_ref(),
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4),
            rows_per_image,
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers,
        },
    );
}

/// Uploads a single white texel into every layer of `white` (1 layer for 2D / 3D, 6 for cubes).
pub(super) fn upload_white(queue: &wgpu::Queue, white: &PlaceholderTexture, kind: TextureBindKind) {
    upload_placeholder(queue, white, kind, PlaceholderTextureColor::White);
}

/// Uploads a single black texel into every layer of `black` (1 layer for 2D / 3D, 6 for cubes).
pub(super) fn upload_black(queue: &wgpu::Queue, black: &PlaceholderTexture, kind: TextureBindKind) {
    upload_placeholder(queue, black, kind, PlaceholderTextureColor::Black);
}

/// Uploads a single flat-normal texel `(128, 128, 255, 255)` into `flat_normal`.
pub(super) fn upload_flat_normal(
    queue: &wgpu::Queue,
    flat_normal: &PlaceholderTexture,
    kind: TextureBindKind,
) {
    upload_placeholder(
        queue,
        flat_normal,
        kind,
        PlaceholderTextureColor::FlatNormal,
    );
}
