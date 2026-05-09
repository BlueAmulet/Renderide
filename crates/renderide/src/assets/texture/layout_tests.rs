//! Unit tests for [`super::layout`] (mip byte sizing).

use glam::IVec2;

use super::layout::{
    clamp_host_texture_mip_count, legal_texture2d_mip_level_count, legal_texture3d_mip_level_count,
    mip_byte_len, mip_dimensions_at_level, mip_tight_bytes_per_texel, validate_mip_upload_layout,
};
use crate::shared::{SetTexture2DData, TextureFormat};

#[test]
fn validate_mip_layout_accepts_contiguous_payload() {
    let mut d = SetTexture2DData::default();
    d.data.length = 4 * 4 * 4 + 2 * 2 * 4; // rgba32 mip0 + mip1
    d.mip_map_sizes = vec![IVec2::new(4, 4), IVec2::new(2, 2)];
    d.mip_starts = vec![0, 64];
    assert!(validate_mip_upload_layout(TextureFormat::RGBA32, &d).is_ok());
}

#[test]
fn validate_mip_layout_rejects_overflow() {
    let mut d = SetTexture2DData::default();
    d.data.length = 10;
    d.mip_map_sizes = vec![IVec2::new(4, 4)];
    d.mip_starts = vec![0];
    assert!(validate_mip_upload_layout(TextureFormat::RGBA32, &d).is_err());
}

#[test]
fn bc1_mip0_128_bytes_for_32x32() {
    let b = mip_byte_len(TextureFormat::BC1, 32, 32).expect("bc1");
    assert_eq!(b, (32 / 4) * (32 / 4) * 8);
}

#[test]
fn rgba32_mip0_byte_len() {
    assert_eq!(
        mip_byte_len(TextureFormat::RGBA32, 16, 16).unwrap(),
        16 * 16 * 4
    );
}

#[test]
fn rgba_float_matches_rgba32_float_texel_size() {
    assert_eq!(mip_byte_len(TextureFormat::RGBAFloat, 1, 1).unwrap(), 16);
    assert_eq!(mip_tight_bytes_per_texel(16 * 4 * 4, 4, 4), Some(16));
}

#[test]
fn mip_dimensions_at_level_halves_each_step() {
    assert_eq!(mip_dimensions_at_level(114, 200, 0), (114, 200));
    assert_eq!(mip_dimensions_at_level(114, 200, 1), (57, 100));
    assert_eq!(mip_dimensions_at_level(114, 200, 2), (28, 50));
    assert_eq!(mip_dimensions_at_level(1, 1, 5), (1, 1));
}

#[test]
fn legal_texture2d_mips_follow_largest_axis() {
    assert_eq!(legal_texture2d_mip_level_count(1, 1), 1);
    assert_eq!(legal_texture2d_mip_level_count(256, 128), 9);
    assert_eq!(legal_texture2d_mip_level_count(257, 16), 9);
}

#[test]
fn legal_texture3d_mips_follow_largest_axis() {
    assert_eq!(legal_texture3d_mip_level_count(1, 1, 1), 1);
    assert_eq!(legal_texture3d_mip_level_count(64, 32, 16), 7);
    assert_eq!(legal_texture3d_mip_level_count(7, 9, 33), 6);
}

#[test]
fn host_mip_count_clamps_to_allocated_texture_count() {
    assert_eq!(clamp_host_texture_mip_count(-4, 8), 1);
    assert_eq!(clamp_host_texture_mip_count(0, 8), 1);
    assert_eq!(clamp_host_texture_mip_count(12, 9), 9);
    assert_eq!(clamp_host_texture_mip_count(5, 9), 5);
}
