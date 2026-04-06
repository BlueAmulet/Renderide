//! Per-draw uniform packing for mesh forward passes (WebGPU dynamic uniform offset = 256 bytes).

use glam::Mat4;

/// Stride between consecutive draw slots in the uniform slab (`mat4`×2 + WGSL padding).
pub const PER_DRAW_UNIFORM_STRIDE: usize = 256;

/// Initial number of draw slots allocated for [`super::debug_draw::DebugDrawResources`].
pub const INITIAL_PER_DRAW_UNIFORM_SLOTS: usize = 256;

/// GPU layout: `view_proj` and `model` column-major (`glam`), then 128 bytes padding to 256.
///
/// Matches `shaders/debug_world_normals.wgsl` `PerDrawUniforms`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PaddedPerDrawUniforms {
    /// Column-major 4×4 view-projection matrix.
    pub view_proj: [f32; 16],
    /// Column-major 4×4 model matrix.
    pub model: [f32; 16],
    /// Padding so each slot is [`PER_DRAW_UNIFORM_STRIDE`] bytes (dynamic uniform alignment).
    pub _pad: [f32; 32],
}

impl PaddedPerDrawUniforms {
    /// Packs `view_proj` and `model` into one 256-byte slot.
    pub fn new(view_proj: Mat4, model: Mat4) -> Self {
        Self {
            view_proj: view_proj.to_cols_array(),
            model: model.to_cols_array(),
            _pad: [0.0; 32],
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
        let slot = PaddedPerDrawUniforms::new(vp, m);
        let mut buf = vec![0u8; PER_DRAW_UNIFORM_STRIDE * 2];
        write_per_draw_uniform_slab(
            &[
                slot,
                PaddedPerDrawUniforms::new(Mat4::IDENTITY, Mat4::IDENTITY),
            ],
            &mut buf,
        );
        let a: &PaddedPerDrawUniforms = bytemuck::from_bytes(&buf[0..PER_DRAW_UNIFORM_STRIDE]);
        assert_eq!(a.view_proj, vp.to_cols_array());
        assert_eq!(a.model, m.to_cols_array());
    }
}
