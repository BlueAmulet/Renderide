//! Per-draw uniform packing for mesh forward passes (WebGPU dynamic uniform offset = 256 bytes).

use glam::Mat4;

/// Stride between consecutive draw slots in the uniform slab (`mat4`×3 + WGSL padding).
pub const PER_DRAW_UNIFORM_STRIDE: usize = 256;

/// Initial number of draw slots allocated for [`super::debug_draw::DebugDrawResources`].
pub const INITIAL_PER_DRAW_UNIFORM_SLOTS: usize = 256;

/// GPU layout: left/right view–projection, `model`, then padding to 256 bytes.
///
/// Matches `shaders/debug_world_normals.wgsl` and `debug_world_normals_multiview.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PaddedPerDrawUniforms {
    /// Column-major 4×4 view–projection for the left eye (or single desktop view).
    pub view_proj_left: [f32; 16],
    /// Column-major 4×4 view–projection for the right eye (duplicated for desktop).
    pub view_proj_right: [f32; 16],
    /// Column-major 4×4 model matrix.
    pub model: [f32; 16],
    /// Padding to [`PER_DRAW_UNIFORM_STRIDE`] bytes.
    pub _pad: [f32; 16],
}

impl PaddedPerDrawUniforms {
    /// Single-view path: duplicates `view_proj` into both eye slots.
    pub fn new_single(view_proj: Mat4, model: Mat4) -> Self {
        let vp = view_proj.to_cols_array();
        Self {
            view_proj_left: vp,
            view_proj_right: vp,
            model: model.to_cols_array(),
            _pad: [0.0; 16],
        }
    }

    /// Stereo path: separate per-eye view–projection (multiview or two-pass fallback).
    pub fn new_stereo(view_proj_left: Mat4, view_proj_right: Mat4, model: Mat4) -> Self {
        Self {
            view_proj_left: view_proj_left.to_cols_array(),
            view_proj_right: view_proj_right.to_cols_array(),
            model: model.to_cols_array(),
            _pad: [0.0; 16],
        }
    }
}

/// Writes `count` consecutive [`PaddedPerDrawUniforms`] into `out` (must be `count * 256` bytes).
pub fn write_per_draw_uniform_slab(slots: &[PaddedPerDrawUniforms], out: &mut [u8]) {
    let need = slots.len().saturating_mul(PER_DRAW_UNIFORM_STRIDE);
    assert!(
        out.len() >= need,
        "slab buffer too small: need {need}, have {}",
        out.len()
    );
    for (i, slot) in slots.iter().enumerate() {
        let start = i * PER_DRAW_UNIFORM_STRIDE;
        let bytes: &[u8] = bytemuck::bytes_of(slot);
        out[start..start + bytes.len()].copy_from_slice(bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padded_size_is_256() {
        assert_eq!(
            std::mem::size_of::<PaddedPerDrawUniforms>(),
            PER_DRAW_UNIFORM_STRIDE
        );
    }

    #[test]
    fn slab_roundtrip_bytes() {
        let vp = Mat4::from_translation(glam::Vec3::new(1.0, 2.0, 3.0));
        let m = Mat4::from_scale(glam::Vec3::new(4.0, 5.0, 6.0));
        let slot = PaddedPerDrawUniforms::new_single(vp, m);
        let mut buf = vec![0u8; PER_DRAW_UNIFORM_STRIDE * 2];
        write_per_draw_uniform_slab(
            &[
                slot,
                PaddedPerDrawUniforms::new_single(Mat4::IDENTITY, Mat4::IDENTITY),
            ],
            &mut buf,
        );
        let a: &PaddedPerDrawUniforms = bytemuck::from_bytes(&buf[0..PER_DRAW_UNIFORM_STRIDE]);
        assert_eq!(a.view_proj_left, vp.to_cols_array());
        assert_eq!(a.view_proj_right, vp.to_cols_array());
        assert_eq!(a.model, m.to_cols_array());
    }
}
