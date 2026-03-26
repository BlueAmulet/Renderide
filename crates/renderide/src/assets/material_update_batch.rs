//! Parses [`crate::shared::MaterialsUpdateBatch`] using the same layout as FrooxEngine
//! `MaterialUpdateWriter` and Renderite `MaterialUpdateReader`:
//! `MaterialPropertyUpdate` records live in `material_updates` buffers; payload values live in
//! separate `int_buffers`, `float_buffers`, `float4_buffers`, and `matrix_buffers`, consumed in
//! global order across each list.

use bytemuck::{Pod, Zeroable};

use crate::assets::{MaterialPropertyStore, MaterialPropertyValue};
use crate::ipc::shared_memory::SharedMemoryAccessor;
use crate::shared::buffer::SharedMemoryBufferDescriptor;
use crate::shared::{MaterialPropertyUpdate, MaterialPropertyUpdateType, MaterialsUpdateBatch};

/// Copies the bytes for a material batch descriptor (production: shared-memory mmap).
pub trait MaterialBatchBlobLoader {
    /// Returns a copy of the region described by `descriptor`, or `None` on failure / empty.
    fn load_blob(&mut self, descriptor: &SharedMemoryBufferDescriptor) -> Option<Vec<u8>>;
}

impl MaterialBatchBlobLoader for SharedMemoryAccessor {
    fn load_blob(&mut self, descriptor: &SharedMemoryBufferDescriptor) -> Option<Vec<u8>> {
        self.access_copy::<u8>(descriptor)
    }
}

/// Applies all material updates in `batch` into `store` using `loader`.
///
/// Chains every `material_updates` descriptor into one logical update stream (matching Renderite’s
/// reader). Typed side buffers are consumed in order for payloads.
pub fn parse_materials_update_batch_into_store(
    loader: &mut impl MaterialBatchBlobLoader,
    batch: &MaterialsUpdateBatch,
    store: &mut MaterialPropertyStore,
) {
    let mut p = BatchParser {
        loader,
        updates: ChainCursor::new(&batch.material_updates),
        ints: ChainCursor::new(&batch.int_buffers),
        floats: ChainCursor::new(&batch.float_buffers),
        float4s: ChainCursor::new(&batch.float4_buffers),
        matrices: ChainCursor::new(&batch.matrix_buffers),
    };

    let mut current_block: Option<i32> = None;

    loop {
        let Some(update) = p.next_update() else {
            break;
        };
        if update.update_type == MaterialPropertyUpdateType::update_batch_end {
            break;
        }

        let Some(block) = current_block else {
            if update.update_type == MaterialPropertyUpdateType::select_target {
                current_block = Some(update.property_id);
            }
            continue;
        };

        match update.update_type {
            MaterialPropertyUpdateType::select_target => {
                current_block = Some(update.property_id);
            }
            MaterialPropertyUpdateType::set_shader => {
                store.set_shader_asset(block, update.property_id);
            }
            MaterialPropertyUpdateType::set_render_queue
            | MaterialPropertyUpdateType::set_instancing
            | MaterialPropertyUpdateType::set_render_type => {}
            MaterialPropertyUpdateType::set_float => {
                if let Some(v) = p.next_float() {
                    store.set(block, update.property_id, MaterialPropertyValue::Float(v));
                }
            }
            MaterialPropertyUpdateType::set_float4 => {
                if let Some(v) = p.next_float4() {
                    store.set(block, update.property_id, MaterialPropertyValue::Float4(v));
                }
            }
            MaterialPropertyUpdateType::set_float4x4 => {
                let _ = p.next_matrix();
            }
            MaterialPropertyUpdateType::set_texture => {
                if let Some(packed) = p.next_int() {
                    store.set(
                        block,
                        update.property_id,
                        MaterialPropertyValue::Texture(packed),
                    );
                }
            }
            MaterialPropertyUpdateType::set_float_array => {
                let Some(len) = p.next_int() else {
                    continue;
                };
                let len = len.max(0) as usize;
                for _ in 0..len {
                    let _ = p.next_float();
                }
            }
            MaterialPropertyUpdateType::set_float4_array => {
                let Some(len) = p.next_int() else {
                    continue;
                };
                let len = len.max(0) as usize;
                for _ in 0..len {
                    let _ = p.next_float4();
                }
            }
            MaterialPropertyUpdateType::update_batch_end => break,
        }
    }
}

struct ChainCursor<'a> {
    descriptors: &'a [SharedMemoryBufferDescriptor],
    descriptor_index: usize,
    data: Vec<u8>,
    offset: usize,
}

impl<'a> ChainCursor<'a> {
    fn new(descriptors: &'a [SharedMemoryBufferDescriptor]) -> Self {
        Self {
            descriptors,
            descriptor_index: 0,
            data: Vec::new(),
            offset: 0,
        }
    }

    fn advance<L: MaterialBatchBlobLoader + ?Sized>(&mut self, loader: &mut L) -> bool {
        while self.descriptor_index < self.descriptors.len() {
            let desc = &self.descriptors[self.descriptor_index];
            self.descriptor_index += 1;
            if desc.length <= 0 {
                continue;
            }
            if let Some(bytes) = loader.load_blob(desc) {
                self.data = bytes;
                self.offset = 0;
                return !self.data.is_empty();
            }
        }
        self.data.clear();
        self.offset = 0;
        false
    }

    fn ensure_capacity<L: MaterialBatchBlobLoader + ?Sized>(
        &mut self,
        loader: &mut L,
        elem_size: usize,
    ) -> bool {
        loop {
            if self.offset + elem_size <= self.data.len() {
                return true;
            }
            if !self.advance(loader) {
                return false;
            }
        }
    }

    fn next<T: Pod + Zeroable, L: MaterialBatchBlobLoader + ?Sized>(
        &mut self,
        loader: &mut L,
    ) -> Option<T> {
        let elem_size = std::mem::size_of::<T>();
        if elem_size == 0 {
            return Some(T::zeroed());
        }
        if !self.ensure_capacity(loader, elem_size) {
            return None;
        }
        let slice = &self.data[self.offset..self.offset + elem_size];
        let v = bytemuck::pod_read_unaligned(slice);
        self.offset += elem_size;
        Some(v)
    }
}

struct BatchParser<'a, L: MaterialBatchBlobLoader + ?Sized> {
    loader: &'a mut L,
    updates: ChainCursor<'a>,
    ints: ChainCursor<'a>,
    floats: ChainCursor<'a>,
    float4s: ChainCursor<'a>,
    matrices: ChainCursor<'a>,
}

impl<'a, L: MaterialBatchBlobLoader + ?Sized> BatchParser<'a, L> {
    fn next_update(&mut self) -> Option<MaterialPropertyUpdate> {
        self.updates.next(self.loader)
    }

    fn next_int(&mut self) -> Option<i32> {
        self.ints.next(self.loader)
    }

    fn next_float(&mut self) -> Option<f32> {
        self.floats.next(self.loader)
    }

    fn next_float4(&mut self) -> Option<[f32; 4]> {
        self.float4s.next(self.loader)
    }

    fn next_matrix(&mut self) -> Option<[f32; 16]> {
        self.matrices.next(self.loader)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::buffer::SharedMemoryBufferDescriptor;

    /// Test loader: `buffer_id` indexes into `blobs`.
    struct TestLoader {
        blobs: Vec<Vec<u8>>,
    }

    impl MaterialBatchBlobLoader for TestLoader {
        fn load_blob(&mut self, descriptor: &SharedMemoryBufferDescriptor) -> Option<Vec<u8>> {
            let i = descriptor.buffer_id.max(0) as usize;
            self.blobs.get(i).cloned()
        }
    }

    fn desc(blob_idx: i32, bytes: &[u8]) -> SharedMemoryBufferDescriptor {
        SharedMemoryBufferDescriptor {
            buffer_id: blob_idx,
            buffer_capacity: bytes.len() as i32,
            offset: 0,
            length: bytes.len() as i32,
        }
    }

    fn write_update(property_id: i32, ty: MaterialPropertyUpdateType) -> MaterialPropertyUpdate {
        MaterialPropertyUpdate {
            property_id,
            update_type: ty,
            _padding: [0; 3],
        }
    }

    #[test]
    fn select_target_uses_property_id_set_shader_in_property_id() {
        let b0 = bytemuck::bytes_of(&write_update(42, MaterialPropertyUpdateType::select_target))
            .to_vec();
        let b1 =
            bytemuck::bytes_of(&write_update(7, MaterialPropertyUpdateType::set_shader)).to_vec();
        let b2 = bytemuck::bytes_of(&write_update(
            0,
            MaterialPropertyUpdateType::update_batch_end,
        ))
        .to_vec();
        let mut loader = TestLoader {
            blobs: vec![b0.clone(), b1.clone(), b2.clone()],
        };
        let batch = MaterialsUpdateBatch {
            material_updates: vec![desc(0, &b0), desc(1, &b1), desc(2, &b2)],
            ..Default::default()
        };
        let mut store = MaterialPropertyStore::new();
        parse_materials_update_batch_into_store(&mut loader, &batch, &mut store);
        assert_eq!(store.shader_asset_for_block(42), Some(7));
    }

    #[test]
    fn set_texture_reads_packed_from_int_buffer() {
        let stream: Vec<u8> =
            bytemuck::bytes_of(&write_update(99, MaterialPropertyUpdateType::select_target))
                .iter()
                .chain(bytemuck::bytes_of(&write_update(
                    1,
                    MaterialPropertyUpdateType::set_texture,
                )))
                .chain(bytemuck::bytes_of(&write_update(
                    0,
                    MaterialPropertyUpdateType::update_batch_end,
                )))
                .copied()
                .collect();
        let packed: i32 = 0x00AB_CD01;
        let int_bytes = bytemuck::bytes_of(&packed).to_vec();

        let mut loader = TestLoader {
            blobs: vec![stream.clone(), int_bytes.clone()],
        };
        let batch = MaterialsUpdateBatch {
            material_updates: vec![desc(0, &stream)],
            int_buffers: vec![desc(1, &int_bytes)],
            ..Default::default()
        };
        let mut store = MaterialPropertyStore::new();
        parse_materials_update_batch_into_store(&mut loader, &batch, &mut store);
        assert_eq!(
            store.get(99, 1),
            Some(&MaterialPropertyValue::Texture(0x00AB_CD01))
        );
    }

    #[test]
    fn set_float_and_float4_from_typed_buffers() {
        let stream: Vec<u8> =
            bytemuck::bytes_of(&write_update(10, MaterialPropertyUpdateType::select_target))
                .iter()
                .chain(bytemuck::bytes_of(&write_update(
                    2,
                    MaterialPropertyUpdateType::set_float,
                )))
                .chain(bytemuck::bytes_of(&write_update(
                    3,
                    MaterialPropertyUpdateType::set_float4,
                )))
                .chain(bytemuck::bytes_of(&write_update(
                    0,
                    MaterialPropertyUpdateType::update_batch_end,
                )))
                .copied()
                .collect();
        let fv: f32 = 2.5;
        let v4 = [1.0f32, 2.0, 3.0, 4.0];

        let fbytes = bytemuck::bytes_of(&fv).to_vec();
        let v4bytes = bytemuck::cast_slice(&v4).to_vec();
        let mut loader = TestLoader {
            blobs: vec![stream.clone(), fbytes.clone(), v4bytes.clone()],
        };
        let batch = MaterialsUpdateBatch {
            material_updates: vec![desc(0, &stream)],
            float_buffers: vec![desc(1, &fbytes)],
            float4_buffers: vec![desc(2, &v4bytes)],
            ..Default::default()
        };
        let mut store = MaterialPropertyStore::new();
        parse_materials_update_batch_into_store(&mut loader, &batch, &mut store);
        assert_eq!(store.get(10, 2), Some(&MaterialPropertyValue::Float(2.5)));
        assert_eq!(
            store.get(10, 3),
            Some(&MaterialPropertyValue::Float4([1.0, 2.0, 3.0, 4.0]))
        );
    }

    #[test]
    fn chained_material_update_buffers() {
        let b0 = bytemuck::bytes_of(&write_update(5, MaterialPropertyUpdateType::select_target))
            .to_vec();
        let b1 =
            bytemuck::bytes_of(&write_update(9, MaterialPropertyUpdateType::set_shader)).to_vec();
        let mut loader = TestLoader {
            blobs: vec![b0.clone(), b1.clone()],
        };
        let batch = MaterialsUpdateBatch {
            material_updates: vec![desc(0, &b0), desc(1, &b1)],
            ..Default::default()
        };
        let mut store = MaterialPropertyStore::new();
        parse_materials_update_batch_into_store(&mut loader, &batch, &mut store);
        assert_eq!(store.shader_asset_for_block(5), Some(9));
    }
}
