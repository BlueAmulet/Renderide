//! Shared shader ABI contracts for frame-global bind groups and packed GPU rows.

use std::mem::size_of;
use std::num::NonZeroU64;

use bytemuck::{Pod, Zeroable};

use super::frame_globals::FrameGpuUniforms;

/// Max lights copied into the frame light buffer.
pub const MAX_LIGHTS: usize = 65536;

/// Maximum lights assigned to a single cluster.
pub const MAX_LIGHTS_PER_TILE: u32 = 64;

const _: () = assert!(MAX_LIGHTS_PER_TILE.is_multiple_of(2));

/// Uniform buffer size for clustered light compute `ClusterParams`.
pub const CLUSTER_PARAMS_UNIFORM_SIZE: u64 = 256;

/// GPU-facing light record for a storage buffer upload.
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct GpuLight {
    /// Light position in world space.
    pub position: [f32; 3],
    /// Aligns `position` to 16 bytes for WGSL `vec3` storage rules.
    pub _pad0: f32,
    /// Forward axis for spot/directional lights.
    pub direction: [f32; 3],
    /// Aligns `direction` to 16 bytes.
    pub _pad1: f32,
    /// Linear RGB color.
    pub color: [f32; 3],
    /// Scalar brightness multiplier.
    pub intensity: f32,
    /// Attenuation range in world units.
    pub range: f32,
    /// Cosine of the spot half-angle.
    pub spot_cos_half_angle: f32,
    /// Light type as a `u32`.
    pub light_type: u32,
    /// Padding before shadow parameter block.
    pub _pad_before_shadow_params: u32,
    /// Shadow strength / visibility factor.
    pub shadow_strength: f32,
    /// Shadow projection near plane.
    pub shadow_near_plane: f32,
    /// Depth bias for shadow sampling.
    pub shadow_bias: f32,
    /// Normal offset bias for shadowing.
    pub shadow_normal_bias: f32,
    /// Shadow type as a `u32`.
    pub shadow_type: u32,
    /// Padding so `_pad_trailing` starts at a 16-byte aligned offset.
    pub _pad_align_vec3_trailing: [u8; 4],
    /// Trailing `vec3<u32>`-shaped padding in WGSL.
    pub _pad_trailing: [u32; 3],
    /// Pads struct size to match WGSL struct alignment.
    pub _pad_struct_end: [u8; 12],
}

impl Default for GpuLight {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            _pad0: 0.0,
            direction: [0.0, 0.0, 1.0],
            _pad1: 0.0,
            color: [1.0; 3],
            intensity: 1.0,
            range: 10.0,
            spot_cos_half_angle: 1.0,
            light_type: 0,
            _pad_before_shadow_params: 0,
            shadow_strength: 0.0,
            shadow_near_plane: 0.0,
            shadow_bias: 0.0,
            shadow_normal_bias: 0.0,
            shadow_type: 0,
            _pad_align_vec3_trailing: [0; 4],
            _pad_trailing: [0; 3],
            _pad_struct_end: [0; 12],
        }
    }
}

/// One reflection-probe metadata row consumed by PBS fragment shaders.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct GpuReflectionProbeMetadata {
    /// World-space AABB minimum, padded to a vec4.
    pub box_min: [f32; 4],
    /// World-space AABB maximum, padded to a vec4.
    pub box_max: [f32; 4],
    /// World-space probe position, padded to a vec4.
    pub position: [f32; 4],
    /// `.x` intensity, `.y` max LOD, `.z` flags, `.w` SH2 source kind.
    pub params: [f32; 4],
    /// Probe SH2 coefficients in host order, padded to vec4 rows.
    pub sh2: [[f32; 4]; 9],
}

/// Probe metadata flag for box-projected reflection sampling.
pub const REFLECTION_PROBE_METADATA_BOX_PROJECTION: u32 = 1;
/// Probe metadata parameter value for local reflection-probe SH2 coefficients.
pub const REFLECTION_PROBE_METADATA_SH2_SOURCE_LOCAL: f32 = 1.0;
/// Probe metadata parameter value for skybox-derived SH2 coefficients.
pub const REFLECTION_PROBE_METADATA_SH2_SOURCE_SKYBOX: f32 = 2.0;

/// Texture format used by prefiltered reflection-probe IBL cubemaps.
pub const REFLECTION_PROBE_ATLAS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Returns the `@group(0)` layout entries shared by every material pipeline.
pub fn frame_bind_group_layout_entries() -> Vec<wgpu::BindGroupLayoutEntry> {
    let mut entries = Vec::with_capacity(13);
    append_frame_buffer_layout_entries(&mut entries);
    append_scene_snapshot_layout_entries(&mut entries);
    append_ibl_layout_entries(&mut entries);
    entries
}

/// Layout for `@group(0)`: frame globals, lights, cluster lists, snapshots, and IBL.
pub fn frame_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let entries = frame_bind_group_layout_entries();
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("frame_globals"),
        entries: &entries,
    })
}

/// Layout for an empty `@group(1)` material bind group.
pub fn empty_material_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("empty_material_props_layout"),
        entries: &[],
    })
}

fn append_frame_buffer_layout_entries(entries: &mut Vec<wgpu::BindGroupLayoutEntry>) {
    entries.extend([
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: NonZeroU64::new(size_of::<FrameGpuUniforms>() as u64),
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: NonZeroU64::new(size_of::<GpuLight>() as u64),
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: NonZeroU64::new(4),
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: NonZeroU64::new(4),
            },
            count: None,
        },
    ]);
}

fn append_scene_snapshot_layout_entries(entries: &mut Vec<wgpu::BindGroupLayoutEntry>) {
    entries.extend([
        wgpu::BindGroupLayoutEntry {
            binding: 4,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Depth,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 5,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Depth,
                view_dimension: wgpu::TextureViewDimension::D2Array,
                multisampled: false,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 6,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 7,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2Array,
                multisampled: false,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 8,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        },
    ]);
}

fn append_ibl_layout_entries(entries: &mut Vec<wgpu::BindGroupLayoutEntry>) {
    entries.extend([
        wgpu::BindGroupLayoutEntry {
            binding: 9,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::CubeArray,
                multisampled: false,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 10,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 11,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 12,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: NonZeroU64::new(size_of::<GpuReflectionProbeMetadata>() as u64),
            },
            count: None,
        },
    ]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_light_row_size_matches_wgsl_storage_stride() {
        assert_eq!(size_of::<GpuLight>(), 112);
    }

    #[test]
    fn reflection_probe_metadata_row_size_matches_wgsl_storage_stride() {
        assert_eq!(size_of::<GpuReflectionProbeMetadata>(), 208);
    }
}
