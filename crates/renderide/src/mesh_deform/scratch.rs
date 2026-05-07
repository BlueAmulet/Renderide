//! Reusable GPU buffers for per-frame mesh deformation (bone palette, blendshape uniforms).

use std::sync::Arc;

use hashbrown::HashMap;

/// CPU-reserved caps; buffers grow when exceeded.
const INITIAL_MAX_BONES: u32 = 256;
/// Initial number of 256-byte slots for per-dispatch `SkinDispatchParams`.
const INITIAL_SKIN_DISPATCH_SLOTS: u64 = 16;

/// Bytes per skinning palette matrix (column-major `mat4`).
const BONE_MATRIX_BYTES: u64 = 64;
/// Pads to the per-draw slab stride (matches [`crate::mesh_deform::PER_DRAW_UNIFORM_STRIDE`]).
///
/// The device's `min_storage_buffer_offset_alignment` is verified to be `<= 256` in
/// [`crate::gpu::GpuLimits::try_new`], so this constant satisfies dynamic-offset alignment for
/// every supported adapter. Use 256 here (not the device alignment) because the slab payload
/// stride is a fixed CPU/GPU contract, not a per-device value.
#[inline]
fn align256(n: u64) -> u64 {
    (n + 255) & !255
}

/// Static description of a growable scratch buffer: label, usage, and minimum size floor.
///
/// Centralises the per-buffer recipe so [`MeshDeformScratch::new`] and the [`Self::ensure`]
/// growth path share one buffer descriptor and one log message format.
struct GrowableBuffer {
    label: &'static str,
    usage: wgpu::BufferUsages,
    /// Floor below which the buffer is never sized. Matches the `.max(N)` literals from the
    /// per-call buffer descriptors.
    min_size: u64,
}

impl GrowableBuffer {
    fn create(&self, device: &wgpu::Device, requested: u64) -> wgpu::Buffer {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(self.label),
            size: requested.max(self.min_size),
            usage: self.usage,
            mapped_at_creation: false,
        });
        crate::profiling::note_resource_churn!(Buffer, "mesh_deform::scratch_buffer");
        buffer
    }

    /// Ensures `buf` is at least `need_bytes` long, growing to the next power of two.
    ///
    /// Returns `false` (and logs a warning) when growth would exceed `max_buffer_size`. The buffer
    /// is left unchanged in that case so the caller can fall back to a smaller dispatch.
    fn ensure(
        &self,
        device: &wgpu::Device,
        buf: &mut wgpu::Buffer,
        need_bytes: u64,
        max_buffer_size: u64,
    ) -> bool {
        if need_bytes <= buf.size() {
            return true;
        }
        let next = need_bytes.next_power_of_two().max(self.min_size);
        if next > max_buffer_size {
            logger::warn!(
                "mesh deform scratch: {} would need {} bytes (max_buffer_size={})",
                self.label,
                next,
                max_buffer_size
            );
            return false;
        }
        *buf = self.create(device, next);
        true
    }
}

const BONE_MATRICES: GrowableBuffer = GrowableBuffer {
    label: "mesh_deform_bone_palette",
    usage: wgpu::BufferUsages::STORAGE.union(wgpu::BufferUsages::COPY_DST),
    min_size: BONE_MATRIX_BYTES,
};

const BLENDSHAPE_PARAMS: GrowableBuffer = GrowableBuffer {
    label: "mesh_deform_blendshape_params",
    usage: wgpu::BufferUsages::UNIFORM.union(wgpu::BufferUsages::COPY_DST),
    min_size: 32,
};

const SKIN_DISPATCH: GrowableBuffer = GrowableBuffer {
    label: "mesh_deform_skin_dispatch",
    usage: wgpu::BufferUsages::UNIFORM.union(wgpu::BufferUsages::COPY_DST),
    min_size: 256,
};

const DUMMY_VEC4_READ: GrowableBuffer = GrowableBuffer {
    label: "mesh_deform_dummy_vec4_read",
    usage: wgpu::BufferUsages::STORAGE,
    min_size: 16,
};

const DUMMY_VEC4_WRITE: GrowableBuffer = GrowableBuffer {
    label: "mesh_deform_dummy_vec4_write",
    usage: wgpu::BufferUsages::STORAGE,
    min_size: 16,
};

/// Cache key for a blendshape scatter bind group.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlendshapeBindGroupKey {
    /// Scratch buffer generation used by the params binding.
    pub scratch_generation: u64,
    /// Stable identity of the mesh sparse-delta buffer.
    pub sparse_buffer: u64,
    /// Stable identity of the output stream buffer.
    pub output_buffer: u64,
}

/// Cache key for a skinning bind group.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SkinningBindGroupKey {
    /// Scratch buffer generation used by palette, params, and dummy bindings.
    pub scratch_generation: u64,
    /// Stable identity of the source position buffer.
    pub src_positions: u64,
    /// Stable identity of the bone-index buffer.
    pub bone_indices: u64,
    /// Stable identity of the bone-weight buffer.
    pub bone_weights: u64,
    /// Stable identity of the destination position buffer.
    pub dst_positions: u64,
    /// Stable identity of the source normal buffer.
    pub src_normals: u64,
    /// Stable identity of the destination normal buffer.
    pub dst_normals: u64,
    /// Stable identity of the source tangent buffer or the dummy tangent buffer.
    pub src_tangents: u64,
    /// Stable identity of the destination tangent buffer or the dummy tangent buffer.
    pub dst_tangents: u64,
}

/// Returns a process-local identity for a `wgpu::Buffer` handle.
#[inline]
pub fn buffer_identity(buffer: &wgpu::Buffer) -> u64 {
    let ptr: *const wgpu::Buffer = buffer;
    let ptr = ptr as usize as u64;
    ptr ^ buffer.size().rotate_left(17)
}

/// Scratch storage written each frame before compute dispatches.
pub struct MeshDeformScratch {
    /// Linear blend skinning bone palette (`mat4` column-major, 64 bytes each); subranges use 256-byte-aligned offsets.
    pub bone_matrices: wgpu::Buffer,
    /// Dynamic-uniform slab for sparse blendshape scatter (`mesh_blendshape.wgsl` `Params`).
    pub blendshape_params: wgpu::Buffer,
    /// Slab of `mesh_skinning.wgsl` [`SkinDispatchParams`] at 256-byte-aligned offsets.
    pub skin_dispatch: wgpu::Buffer,
    /// Dummy read-only storage used for optional shader input bindings when an attribute path is disabled.
    pub dummy_vec4_read: wgpu::Buffer,
    /// Dummy writable storage used for optional shader output bindings when an attribute path is disabled.
    pub dummy_vec4_write: wgpu::Buffer,
    /// Reusable byte buffer for one skinning palette before it is copied into the frame upload batch.
    ///
    /// Cleared (length-only, capacity retained) at the start of each skinning record call.
    pub bone_palette_bytes: Vec<u8>,
    /// Reusable byte buffer for packed scatter `Params` per mesh; one entry per dispatch chunk.
    ///
    /// Cleared (length-only, capacity retained) at the start of each blendshape record call.
    pub packed_scatter_params: Vec<u8>,
    /// Reusable workgroup count per scatter dispatch chunk; parallels [`Self::packed_scatter_params`].
    ///
    /// Cleared (length-only, capacity retained) at the start of each blendshape record call.
    pub scatter_dispatch_wgs: Vec<u32>,
    /// Reusable blendshape output channel per scatter dispatch chunk.
    ///
    /// Cleared (length-only, capacity retained) at the start of each blendshape record call.
    pub scatter_dispatch_targets: Vec<u32>,
    max_bones: u32,
    /// [`wgpu::Limits::max_buffer_size`]; growth refuses past this cap.
    max_buffer_size: u64,
    /// Grow operations observed since the previous diagnostics drain.
    frame_grow_count: u64,
    /// Incremented whenever a scratch GPU buffer is replaced.
    resource_generation: u64,
    blendshape_bind_groups: HashMap<BlendshapeBindGroupKey, Arc<wgpu::BindGroup>>,
    skinning_bind_groups: HashMap<SkinningBindGroupKey, Arc<wgpu::BindGroup>>,
}

impl MeshDeformScratch {
    /// Allocates initial scratch buffers on `device`.
    ///
    /// `max_buffer_size` must be [`wgpu::Device::limits`].`max_buffer_size` (see [`crate::gpu::GpuLimits::max_buffer_size`]).
    pub fn new(device: &wgpu::Device, max_buffer_size: u64) -> Self {
        let bone_bytes = u64::from(INITIAL_MAX_BONES) * BONE_MATRIX_BYTES;
        let skin_dispatch_bytes = INITIAL_SKIN_DISPATCH_SLOTS.saturating_mul(256);
        Self {
            bone_matrices: BONE_MATRICES.create(device, bone_bytes),
            blendshape_params: BLENDSHAPE_PARAMS.create(device, BLENDSHAPE_PARAMS.min_size),
            skin_dispatch: SKIN_DISPATCH.create(device, skin_dispatch_bytes),
            dummy_vec4_read: DUMMY_VEC4_READ.create(device, DUMMY_VEC4_READ.min_size),
            dummy_vec4_write: DUMMY_VEC4_WRITE.create(device, DUMMY_VEC4_WRITE.min_size),
            bone_palette_bytes: Vec::new(),
            packed_scatter_params: Vec::new(),
            scatter_dispatch_wgs: Vec::new(),
            scatter_dispatch_targets: Vec::new(),
            max_bones: INITIAL_MAX_BONES,
            max_buffer_size,
            frame_grow_count: 0,
            resource_generation: 0,
            blendshape_bind_groups: HashMap::new(),
            skinning_bind_groups: HashMap::new(),
        }
    }

    fn note_gpu_buffer_grow(&mut self) {
        self.frame_grow_count = self.frame_grow_count.saturating_add(1);
        self.resource_generation = self.resource_generation.wrapping_add(1);
        self.blendshape_bind_groups.clear();
        self.skinning_bind_groups.clear();
    }

    /// Current scratch-resource generation for bind-group cache keys.
    #[inline]
    pub fn resource_generation(&self) -> u64 {
        self.resource_generation
    }

    /// Looks up a cached blendshape scatter bind group.
    #[inline]
    pub fn blendshape_bind_group(
        &self,
        key: BlendshapeBindGroupKey,
    ) -> Option<Arc<wgpu::BindGroup>> {
        self.blendshape_bind_groups.get(&key).cloned()
    }

    /// Inserts a blendshape scatter bind group into the scratch cache.
    #[inline]
    pub fn insert_blendshape_bind_group(
        &mut self,
        key: BlendshapeBindGroupKey,
        bind_group: Arc<wgpu::BindGroup>,
    ) {
        self.blendshape_bind_groups.insert(key, bind_group);
    }

    /// Looks up a cached skinning bind group.
    #[inline]
    pub fn skinning_bind_group(&self, key: SkinningBindGroupKey) -> Option<Arc<wgpu::BindGroup>> {
        self.skinning_bind_groups.get(&key).cloned()
    }

    /// Inserts a skinning bind group into the scratch cache.
    #[inline]
    pub fn insert_skinning_bind_group(
        &mut self,
        key: SkinningBindGroupKey,
        bind_group: Arc<wgpu::BindGroup>,
    ) {
        self.skinning_bind_groups.insert(key, bind_group);
    }

    /// Ensures the bone palette buffer fits at least `need_bones` matrices for a single-mesh dispatch.
    pub fn ensure_bone_capacity(&mut self, device: &wgpu::Device, need_bones: u32) {
        if need_bones <= self.max_bones {
            return;
        }
        let next = need_bones.next_power_of_two().max(INITIAL_MAX_BONES);
        let bone_bytes = u64::from(next) * BONE_MATRIX_BYTES;
        let old_size = self.bone_matrices.size();
        if BONE_MATRICES.ensure(
            device,
            &mut self.bone_matrices,
            bone_bytes,
            self.max_buffer_size,
        ) {
            if self.bone_matrices.size() > old_size {
                self.note_gpu_buffer_grow();
            }
            self.max_bones = next;
        }
    }

    /// Ensures the bone buffer is large enough for byte range `[0, end_exclusive)`.
    pub fn ensure_bone_byte_capacity(&mut self, device: &wgpu::Device, end_exclusive: u64) {
        let old_size = self.bone_matrices.size();
        if BONE_MATRICES.ensure(
            device,
            &mut self.bone_matrices,
            end_exclusive,
            self.max_buffer_size,
        ) && self.bone_matrices.size() > old_size
        {
            self.note_gpu_buffer_grow();
        }
    }

    /// Ensures the blendshape params uniform slab can address byte range `[0, end_exclusive)`.
    pub fn ensure_blendshape_param_byte_capacity(
        &mut self,
        device: &wgpu::Device,
        end_exclusive: u64,
    ) {
        let old_size = self.blendshape_params.size();
        if BLENDSHAPE_PARAMS.ensure(
            device,
            &mut self.blendshape_params,
            end_exclusive,
            self.max_buffer_size,
        ) && self.blendshape_params.size() > old_size
        {
            self.note_gpu_buffer_grow();
        }
    }

    /// Ensures the skin-dispatch uniform slab can address byte range `[0, end_exclusive)`.
    ///
    /// Each skinning dispatch writes a small uniform payload at a 256-byte-aligned cursor; callers advance with
    /// [`advance_slab_cursor`].
    pub fn ensure_skin_dispatch_byte_capacity(
        &mut self,
        device: &wgpu::Device,
        end_exclusive: u64,
    ) {
        let old_size = self.skin_dispatch.size();
        if SKIN_DISPATCH.ensure(
            device,
            &mut self.skin_dispatch,
            end_exclusive,
            self.max_buffer_size,
        ) && self.skin_dispatch.size() > old_size
        {
            self.note_gpu_buffer_grow();
        }
    }

    /// Returns and clears the grow count accumulated since the previous diagnostics drain.
    pub fn take_frame_grow_count(&mut self) -> u64 {
        let count = self.frame_grow_count;
        self.frame_grow_count = 0;
        count
    }
}

/// Returns the next slab cursor after placing `byte_len` bytes at `cursor`, padding to 256-byte
/// boundaries so subsequent storage/uniform bindings meet typical WebGPU offset alignment.
pub fn advance_slab_cursor(cursor: u64, byte_len: u64) -> u64 {
    if byte_len == 0 {
        return cursor;
    }
    cursor + align256(byte_len)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_cursor_advances_by_256_for_small_payloads() {
        assert_eq!(advance_slab_cursor(0, 32), 256);
        assert_eq!(advance_slab_cursor(256, 48), 512);
    }
}
