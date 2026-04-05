//! Reusable GPU buffers for per-frame mesh deformation (bone palette, blendshape uniforms).

/// CPU-reserved caps; buffers grow when exceeded.
const INITIAL_MAX_BONES: u32 = 256;
const INITIAL_MAX_BLENDSHAPES: u32 = 256;

/// Scratch storage written each frame before compute dispatches.
pub struct MeshDeformScratch {
    /// Linear blend skinning bone palette (`mat4` column-major, 64 bytes each).
    pub bone_matrices: wgpu::Buffer,
    /// 16-byte uniform: `vertex_count`, `shape_count`, `vertices_per_shape`, padding.
    pub blendshape_params: wgpu::Buffer,
    /// `f32` weight per blendshape (dense prefix).
    pub blendshape_weights: wgpu::Buffer,
    max_bones: u32,
    max_shapes: u32,
}

impl MeshDeformScratch {
    /// Allocates initial scratch buffers on `device`.
    pub fn new(device: &wgpu::Device) -> Self {
        let bone_bytes = (INITIAL_MAX_BONES as u64) * 64;
        let weight_bytes = (INITIAL_MAX_BLENDSHAPES as u64) * 4;
        Self {
            bone_matrices: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mesh_deform_bone_palette"),
                size: bone_bytes.max(64),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            blendshape_params: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mesh_deform_blendshape_params"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            blendshape_weights: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mesh_deform_blendshape_weights"),
                size: weight_bytes.max(16),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            max_bones: INITIAL_MAX_BONES,
            max_shapes: INITIAL_MAX_BLENDSHAPES,
        }
    }

    /// Ensures the bone palette buffer fits at least `need_bones` matrices.
    pub fn ensure_bone_capacity(&mut self, device: &wgpu::Device, need_bones: u32) {
        if need_bones <= self.max_bones {
            return;
        }
        let next = need_bones.next_power_of_two().max(INITIAL_MAX_BONES);
        let bone_bytes = (next as u64) * 64;
        self.bone_matrices = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh_deform_bone_palette"),
            size: bone_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.max_bones = next;
    }

    /// Ensures the blendshape weight buffer fits at least `need_shapes` floats.
    pub fn ensure_shape_weight_capacity(&mut self, device: &wgpu::Device, need_shapes: u32) {
        if need_shapes <= self.max_shapes {
            return;
        }
        let next = need_shapes.next_power_of_two().max(INITIAL_MAX_BLENDSHAPES);
        let weight_bytes = (next as u64) * 4;
        self.blendshape_weights = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh_deform_blendshape_weights"),
            size: weight_bytes.max(16),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.max_shapes = next;
    }
}
