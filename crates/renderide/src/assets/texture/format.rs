//! Maps host [`TextureFormat`] + [`ColorProfile`] to [`wgpu::TextureFormat`] when the device reports required compression features.

use crate::shared::{ColorProfile, TextureFormat};

/// Picks a [`wgpu::TextureFormat`] for `host` if this device advertises the needed [`wgpu::Features`].
///
/// Returns [`None`] when the combination is unknown or compression features are missing (caller may decode to `Rgba8UnormSrgb`).
pub fn pick_wgpu_storage_format(
    device: &wgpu::Device,
    host: TextureFormat,
    profile: ColorProfile,
) -> Option<wgpu::TextureFormat> {
    let f = map_host_format(host, profile)?;
    if texture_format_supported(device, f) {
        Some(f)
    } else {
        None
    }
}

/// Maps host format without feature checks (for estimating sizes or documentation).
pub fn map_host_format(host: TextureFormat, profile: ColorProfile) -> Option<wgpu::TextureFormat> {
    use ColorProfile::{SRGBAlpha, SRGB};
    use TextureFormat::*;

    let srgb = matches!(profile, SRGB | SRGBAlpha);

    Some(match host {
        Unknown => return None,
        Alpha8 | R8 => wgpu::TextureFormat::R8Unorm,
        RGB24 | RGB565 | BGR565 => return None, // decode path
        RGBA32 => {
            if srgb {
                wgpu::TextureFormat::Rgba8UnormSrgb
            } else {
                wgpu::TextureFormat::Rgba8Unorm
            }
        }
        ARGB32 | BGRA32 => {
            if srgb {
                wgpu::TextureFormat::Rgba8UnormSrgb
            } else {
                wgpu::TextureFormat::Rgba8Unorm
            }
        }
        RGBAHalf | ARGBHalf => wgpu::TextureFormat::Rgba16Float,
        RHalf => wgpu::TextureFormat::R16Float,
        RGHalf => wgpu::TextureFormat::Rg16Float,
        RGBAFloat | ARGBFloat => wgpu::TextureFormat::Rgba32Float,
        RFloat => wgpu::TextureFormat::R32Float,
        RGFloat => wgpu::TextureFormat::Rg32Float,
        BC1 => {
            if srgb {
                wgpu::TextureFormat::Bc1RgbaUnormSrgb
            } else {
                wgpu::TextureFormat::Bc1RgbaUnorm
            }
        }
        BC2 => {
            if srgb {
                wgpu::TextureFormat::Bc2RgbaUnormSrgb
            } else {
                wgpu::TextureFormat::Bc2RgbaUnorm
            }
        }
        BC3 => {
            if srgb {
                wgpu::TextureFormat::Bc3RgbaUnormSrgb
            } else {
                wgpu::TextureFormat::Bc3RgbaUnorm
            }
        }
        BC4 => wgpu::TextureFormat::Bc4RUnorm,
        BC5 => wgpu::TextureFormat::Bc5RgUnorm,
        BC6H => wgpu::TextureFormat::Bc6hRgbUfloat,
        BC7 => {
            if srgb {
                wgpu::TextureFormat::Bc7RgbaUnormSrgb
            } else {
                wgpu::TextureFormat::Bc7RgbaUnorm
            }
        }
        ETC2RGB => {
            if srgb {
                wgpu::TextureFormat::Etc2Rgb8UnormSrgb
            } else {
                wgpu::TextureFormat::Etc2Rgb8Unorm
            }
        }
        ETC2RGBA1 => {
            if srgb {
                wgpu::TextureFormat::Etc2Rgb8A1UnormSrgb
            } else {
                wgpu::TextureFormat::Etc2Rgb8A1Unorm
            }
        }
        ETC2RGBA8 => {
            if srgb {
                wgpu::TextureFormat::Etc2Rgba8UnormSrgb
            } else {
                wgpu::TextureFormat::Etc2Rgba8Unorm
            }
        }
        ASTC4x4 => astc_wgpu(wgpu::AstcBlock::B4x4, srgb),
        ASTC5x5 => astc_wgpu(wgpu::AstcBlock::B5x5, srgb),
        ASTC6x6 => astc_wgpu(wgpu::AstcBlock::B6x6, srgb),
        ASTC8x8 => astc_wgpu(wgpu::AstcBlock::B8x8, srgb),
        ASTC10x10 => astc_wgpu(wgpu::AstcBlock::B10x10, srgb),
        ASTC12x12 => astc_wgpu(wgpu::AstcBlock::B12x12, srgb),
    })
}

fn astc_wgpu(block: wgpu::AstcBlock, srgb: bool) -> wgpu::TextureFormat {
    let channel = if srgb {
        wgpu::AstcChannel::UnormSrgb
    } else {
        wgpu::AstcChannel::Unorm
    };
    wgpu::TextureFormat::Astc { block, channel }
}

fn texture_format_supported(device: &wgpu::Device, format: wgpu::TextureFormat) -> bool {
    if !format.is_compressed() {
        return true;
    }
    let feats = device.features();
    if format_required_bc(format) && !feats.contains(wgpu::Features::TEXTURE_COMPRESSION_BC) {
        return false;
    }
    if format_required_etc2(format) && !feats.contains(wgpu::Features::TEXTURE_COMPRESSION_ETC2) {
        return false;
    }
    if format_required_astc(format) && !feats.contains(wgpu::Features::TEXTURE_COMPRESSION_ASTC) {
        return false;
    }
    true
}

fn format_required_bc(f: wgpu::TextureFormat) -> bool {
    matches!(
        f,
        wgpu::TextureFormat::Bc1RgbaUnorm
            | wgpu::TextureFormat::Bc1RgbaUnormSrgb
            | wgpu::TextureFormat::Bc2RgbaUnorm
            | wgpu::TextureFormat::Bc2RgbaUnormSrgb
            | wgpu::TextureFormat::Bc3RgbaUnorm
            | wgpu::TextureFormat::Bc3RgbaUnormSrgb
            | wgpu::TextureFormat::Bc4RUnorm
            | wgpu::TextureFormat::Bc4RSnorm
            | wgpu::TextureFormat::Bc5RgUnorm
            | wgpu::TextureFormat::Bc5RgSnorm
            | wgpu::TextureFormat::Bc6hRgbUfloat
            | wgpu::TextureFormat::Bc6hRgbFloat
            | wgpu::TextureFormat::Bc7RgbaUnorm
            | wgpu::TextureFormat::Bc7RgbaUnormSrgb
    )
}

fn format_required_etc2(f: wgpu::TextureFormat) -> bool {
    matches!(
        f,
        wgpu::TextureFormat::Etc2Rgb8Unorm
            | wgpu::TextureFormat::Etc2Rgb8UnormSrgb
            | wgpu::TextureFormat::Etc2Rgb8A1Unorm
            | wgpu::TextureFormat::Etc2Rgb8A1UnormSrgb
            | wgpu::TextureFormat::Etc2Rgba8Unorm
            | wgpu::TextureFormat::Etc2Rgba8UnormSrgb
    )
}

fn format_required_astc(f: wgpu::TextureFormat) -> bool {
    matches!(
        f,
        wgpu::TextureFormat::Astc {
            channel: wgpu::AstcChannel::Unorm | wgpu::AstcChannel::UnormSrgb,
            ..
        }
    )
}

/// Formats we can accept via GPU-native storage or transient RGBA8 decode (advertised to the host).
pub fn supported_host_formats_for_init() -> Vec<TextureFormat> {
    use TextureFormat::*;
    vec![
        Alpha8, R8, RGB24, RGBA32, ARGB32, BGRA32, RGB565, BGR565, RGBAHalf, ARGBHalf, RHalf,
        RGHalf, RGBAFloat, ARGBFloat, RFloat, RGFloat, BC1, BC2, BC3, BC4, BC5, BC6H, BC7, ETC2RGB,
        ETC2RGBA1, ETC2RGBA8, ASTC4x4, ASTC5x5, ASTC6x6, ASTC8x8, ASTC10x10, ASTC12x12,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgba32_linear_maps() {
        assert_eq!(
            map_host_format(TextureFormat::RGBA32, ColorProfile::Linear),
            Some(wgpu::TextureFormat::Rgba8Unorm)
        );
    }
}
