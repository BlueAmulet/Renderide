//! Shared per-frame bindings (`@group(0)`) for all raster materials.
//! Import with `#import renderide::globals` from `source/materials/*.wgsl`.
//!
//! CPU packing must match [`crate::gpu::frame_globals::FrameGpuUniforms`] and
//! [`crate::backend::light_gpu::GpuLight`].

#define_import_path renderide::globals

struct GpuLight {
    position: vec3<f32>,
    align_pad_vec3_pos: f32,
    direction: vec3<f32>,
    align_pad_vec3_dir: f32,
    color: vec3<f32>,
    intensity: f32,
    range: f32,
    spot_cos_half_angle: f32,
    light_type: u32,
    align_pad_before_shadow: u32,
    shadow_strength: f32,
    shadow_near_plane: f32,
    shadow_bias: f32,
    shadow_normal_bias: f32,
    shadow_type: u32,
    align_pad_vec3_tail: vec3<u32>,
}

struct FrameGlobals {
    camera_world_pos: vec4<f32>,
    light_count: u32,
    align_pad_frame_a: u32,
    align_pad_frame_b: u32,
    align_pad_frame_c: u32,
}

@group(0) @binding(0) var<uniform> frame: FrameGlobals;
@group(0) @binding(1) var<storage, read> lights: array<GpuLight>;
