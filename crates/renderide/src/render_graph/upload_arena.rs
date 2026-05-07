//! Fence-backed staging-buffer reuse for render-graph uploads.
//!
//! The render graph records many small buffer writes through
//! [`super::frame_upload_batch::FrameUploadBatch`]. This arena owns a small set of persistent
//! `MAP_WRITE | COPY_SRC` buffers that those writes can reuse across frames. A slot is unmapped
//! before submit, marked reusable only after `Queue::on_submitted_work_done` fires, and remapped
//! from the main thread during the next maintenance pass.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;

const UPLOAD_ARENA_SLOTS: usize = 3;
const DEFAULT_SLOT_BYTES: u64 = 1024 * 1024;
static OVERSIZED_UPLOAD_LOG_COUNTER: AtomicU64 = AtomicU64::new(0);

/// One completed upload slot event delivered from a wgpu callback to the main thread.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum UploadArenaCompletion {
    /// Submitted GPU work that references this slot has completed.
    Submitted { slot: usize, generation: u64 },
    /// A post-submit remap request has completed.
    Remapped {
        slot: usize,
        generation: u64,
        success: bool,
    },
}

/// Lifecycle of one persistent upload arena slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum UploadArenaSlotState {
    /// No buffer has been allocated for this slot yet.
    Empty,
    /// Buffer is mapped and available for a new frame's writes.
    Free,
    /// Buffer is currently being filled on the main thread.
    Writing { generation: u64 },
    /// Buffer is referenced by submitted GPU work.
    InFlight { generation: u64 },
    /// GPU work completed; the buffer is waiting for `map_async` to finish.
    Remapping { generation: u64 },
}

impl UploadArenaSlotState {
    fn can_write(self) -> bool {
        matches!(self, Self::Empty | Self::Free)
    }
}

/// One buffer slot in the persistent upload arena.
struct UploadArenaSlot {
    buffer: Option<wgpu::Buffer>,
    capacity: u64,
    state: UploadArenaSlotState,
}

impl UploadArenaSlot {
    fn empty() -> Self {
        Self {
            buffer: None,
            capacity: 0,
            state: UploadArenaSlotState::Empty,
        }
    }
}

/// Stats captured while acquiring staging storage for one frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct UploadArenaAcquireStats {
    /// Bytes staged through a persistent slot this frame.
    pub(crate) persistent_staging_bytes: u64,
    /// Persistent slot reuse count.
    pub(crate) persistent_slot_reuses: usize,
    /// Persistent slot allocation or growth count.
    pub(crate) persistent_slot_grows: usize,
    /// Bytes staged through a one-frame temporary fallback buffer.
    pub(crate) temporary_staging_bytes: u64,
    /// Temporary fallback count caused by all persistent slots being unavailable.
    pub(crate) temporary_staging_fallbacks: usize,
    /// Staged writes replayed through `Queue::write_buffer` because no staging buffer could fit.
    pub(crate) oversized_queue_fallback_writes: usize,
}

/// Current persistent arena pressure after an upload drain.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct UploadArenaPressure {
    /// Total bytes currently allocated across persistent slots.
    pub(crate) capacity_bytes: u64,
    /// Persistent slots that are mapped and free.
    pub(crate) free_slots: usize,
    /// Persistent slots referenced by submitted GPU work.
    pub(crate) in_flight_slots: usize,
    /// Persistent slots waiting for `map_async` completion.
    pub(crate) remapping_slots: usize,
}

/// Staging storage prepared for one upload drain.
pub(crate) struct PreparedUploadStaging {
    buffer: Option<wgpu::Buffer>,
    source: UploadStagingSource,
    size: u64,
    acquire_stats: UploadArenaAcquireStats,
}

impl PreparedUploadStaging {
    /// Buffer to fill while it is mapped.
    pub(crate) fn buffer(&self) -> Option<&wgpu::Buffer> {
        self.buffer.as_ref()
    }

    /// Stats for the acquisition path that produced this staging storage.
    pub(crate) fn acquire_stats(&self) -> UploadArenaAcquireStats {
        self.acquire_stats
    }

    /// Whether staged writes must be replayed through `Queue::write_buffer`.
    pub(crate) fn requires_queue_fallback(&self) -> bool {
        self.size > 0 && self.buffer.is_none()
    }

    /// Unmaps staging storage and returns the buffer/callback pair required by submit.
    pub(crate) fn finish(self, arena: &mut PersistentUploadArena) -> FinishedUploadStaging {
        match self.source {
            UploadStagingSource::None | UploadStagingSource::QueueFallbackOversized => {
                FinishedUploadStaging {
                    buffer: None,
                    on_submitted_work_done: None,
                }
            }
            UploadStagingSource::Temporary => {
                if let Some(buffer) = self.buffer.as_ref() {
                    buffer.unmap();
                }
                FinishedUploadStaging {
                    buffer: self.buffer,
                    on_submitted_work_done: None,
                }
            }
            UploadStagingSource::Persistent { slot, generation } => {
                let on_submitted_work_done = arena.finish_persistent_write(slot, generation);
                FinishedUploadStaging {
                    buffer: self.buffer,
                    on_submitted_work_done,
                }
            }
        }
    }
}

/// Finished staging storage ready for copy-command recording and submit callbacks.
pub(crate) struct FinishedUploadStaging {
    /// Buffer used as `COPY_SRC` for staged writes.
    pub(crate) buffer: Option<wgpu::Buffer>,
    /// Callback that marks a persistent slot submitted after GPU completion.
    pub(crate) on_submitted_work_done: Option<Box<dyn FnOnce() + Send + 'static>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum UploadStagingSource {
    None,
    Persistent { slot: usize, generation: u64 },
    Temporary,
    QueueFallbackOversized,
}

/// Persistent triple-buffered upload staging arena for render-graph buffer writes.
pub(crate) struct PersistentUploadArena {
    slots: [UploadArenaSlot; UPLOAD_ARENA_SLOTS],
    completion_tx: mpsc::Sender<UploadArenaCompletion>,
    completion_rx: mpsc::Receiver<UploadArenaCompletion>,
    next_generation: u64,
}

impl PersistentUploadArena {
    /// Creates an empty arena. Slots allocate lazily on first use.
    pub(crate) fn new() -> Self {
        let (completion_tx, completion_rx) = mpsc::channel();
        Self {
            slots: std::array::from_fn(|_| UploadArenaSlot::empty()),
            completion_tx,
            completion_rx,
            next_generation: 1,
        }
    }

    /// Drains submitted-work and remap callbacks, then polls once to advance pending remaps.
    pub(crate) fn maintain(&mut self, device: &wgpu::Device) {
        profiling::scope!("frame_upload_arena::maintain");
        self.drain_completions();
        let _ = device.poll(wgpu::PollType::Poll);
        self.drain_completions();
    }

    /// Prepares staging storage for `required` aligned bytes.
    pub(crate) fn prepare_staging_buffer(
        &mut self,
        device: &wgpu::Device,
        max_buffer_size: u64,
        required: u64,
        staged_writes: usize,
    ) -> PreparedUploadStaging {
        profiling::scope!("frame_upload_arena::prepare_staging");
        if required == 0 {
            return PreparedUploadStaging {
                buffer: None,
                source: UploadStagingSource::None,
                size: 0,
                acquire_stats: UploadArenaAcquireStats::default(),
            };
        }
        if required > max_buffer_size {
            log_oversized_upload(required, max_buffer_size);
            return PreparedUploadStaging {
                buffer: None,
                source: UploadStagingSource::QueueFallbackOversized,
                size: required,
                acquire_stats: UploadArenaAcquireStats {
                    oversized_queue_fallback_writes: staged_writes,
                    ..UploadArenaAcquireStats::default()
                },
            };
        }

        if let Some(slot) = self.select_writable_slot(required) {
            return self.prepare_persistent_slot(device, max_buffer_size, required, slot);
        }

        logger::debug!(
            "frame upload arena: no persistent slot available; using temporary staging buffer bytes={required}"
        );
        PreparedUploadStaging {
            buffer: Some(create_temporary_staging_buffer(device, required)),
            source: UploadStagingSource::Temporary,
            size: required,
            acquire_stats: UploadArenaAcquireStats {
                temporary_staging_bytes: required,
                temporary_staging_fallbacks: 1,
                ..UploadArenaAcquireStats::default()
            },
        }
    }

    /// Current pressure/capacity sample for diagnostics.
    pub(crate) fn pressure(&self) -> UploadArenaPressure {
        let mut pressure = UploadArenaPressure::default();
        for slot in &self.slots {
            pressure.capacity_bytes = pressure.capacity_bytes.saturating_add(slot.capacity);
            match slot.state {
                UploadArenaSlotState::Free => {
                    pressure.free_slots = pressure.free_slots.saturating_add(1);
                }
                UploadArenaSlotState::InFlight { .. } => {
                    pressure.in_flight_slots = pressure.in_flight_slots.saturating_add(1);
                }
                UploadArenaSlotState::Remapping { .. } => {
                    pressure.remapping_slots = pressure.remapping_slots.saturating_add(1);
                }
                UploadArenaSlotState::Empty | UploadArenaSlotState::Writing { .. } => {}
            }
        }
        pressure
    }

    fn prepare_persistent_slot(
        &mut self,
        device: &wgpu::Device,
        max_buffer_size: u64,
        required: u64,
        slot: usize,
    ) -> PreparedUploadStaging {
        let generation = self.next_generation;
        self.next_generation = self.next_generation.saturating_add(1).max(1);

        let Some(arena_slot) = self.slots.get_mut(slot) else {
            return PreparedUploadStaging {
                buffer: Some(create_temporary_staging_buffer(device, required)),
                source: UploadStagingSource::Temporary,
                size: required,
                acquire_stats: UploadArenaAcquireStats {
                    temporary_staging_bytes: required,
                    temporary_staging_fallbacks: 1,
                    ..UploadArenaAcquireStats::default()
                },
            };
        };

        let mut acquire_stats = UploadArenaAcquireStats {
            persistent_staging_bytes: required,
            ..UploadArenaAcquireStats::default()
        };
        if arena_slot.buffer.is_none() || arena_slot.capacity < required {
            let Some(capacity) = next_slot_capacity(required, arena_slot.capacity, max_buffer_size)
            else {
                return PreparedUploadStaging {
                    buffer: None,
                    source: UploadStagingSource::QueueFallbackOversized,
                    size: required,
                    acquire_stats: UploadArenaAcquireStats {
                        oversized_queue_fallback_writes: 1,
                        ..UploadArenaAcquireStats::default()
                    },
                };
            };
            arena_slot.buffer = Some(create_persistent_slot_buffer(device, capacity));
            arena_slot.capacity = capacity;
            acquire_stats.persistent_slot_grows = 1;
        } else {
            acquire_stats.persistent_slot_reuses = 1;
        }
        arena_slot.state = UploadArenaSlotState::Writing { generation };
        PreparedUploadStaging {
            buffer: arena_slot.buffer.clone(),
            source: UploadStagingSource::Persistent { slot, generation },
            size: required,
            acquire_stats,
        }
    }

    fn finish_persistent_write(
        &mut self,
        slot: usize,
        generation: u64,
    ) -> Option<Box<dyn FnOnce() + Send + 'static>> {
        let arena_slot = self.slots.get_mut(slot)?;
        if arena_slot.state != (UploadArenaSlotState::Writing { generation }) {
            return None;
        }
        let buffer = arena_slot.buffer.as_ref()?;
        buffer.unmap();
        arena_slot.state = UploadArenaSlotState::InFlight { generation };
        let tx = self.completion_tx.clone();
        Some(Box::new(move || {
            let _ = tx.send(UploadArenaCompletion::Submitted { slot, generation });
        }))
    }

    fn select_writable_slot(&self, required: u64) -> Option<usize> {
        select_writable_slot(&self.slots, required)
    }

    fn drain_completions(&mut self) {
        while let Ok(completion) = self.completion_rx.try_recv() {
            match completion {
                UploadArenaCompletion::Submitted { slot, generation } => {
                    self.start_remap(slot, generation);
                }
                UploadArenaCompletion::Remapped {
                    slot,
                    generation,
                    success,
                } => self.finish_remap(slot, generation, success),
            }
        }
    }

    fn start_remap(&mut self, slot: usize, generation: u64) {
        let Some(arena_slot) = self.slots.get_mut(slot) else {
            return;
        };
        if arena_slot.state != (UploadArenaSlotState::InFlight { generation }) {
            return;
        }
        let Some(buffer) = arena_slot.buffer.clone() else {
            arena_slot.state = UploadArenaSlotState::Empty;
            arena_slot.capacity = 0;
            return;
        };
        arena_slot.state = UploadArenaSlotState::Remapping { generation };
        let tx = self.completion_tx.clone();
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Write, move |result| {
                let _ = tx.send(UploadArenaCompletion::Remapped {
                    slot,
                    generation,
                    success: result.is_ok(),
                });
            });
    }

    fn finish_remap(&mut self, slot: usize, generation: u64, success: bool) {
        let Some(arena_slot) = self.slots.get_mut(slot) else {
            return;
        };
        if arena_slot.state != (UploadArenaSlotState::Remapping { generation }) {
            return;
        }
        if success {
            arena_slot.state = UploadArenaSlotState::Free;
            return;
        }
        logger::warn!("frame upload arena: persistent slot remap failed; dropping slot");
        arena_slot.buffer = None;
        arena_slot.capacity = 0;
        arena_slot.state = UploadArenaSlotState::Empty;
    }
}

impl Default for PersistentUploadArena {
    fn default() -> Self {
        Self::new()
    }
}

fn select_writable_slot(
    slots: &[UploadArenaSlot; UPLOAD_ARENA_SLOTS],
    required: u64,
) -> Option<usize> {
    slots
        .iter()
        .position(|slot| slot.state.can_write() && slot.capacity >= required)
        .or_else(|| {
            slots
                .iter()
                .position(|slot| matches!(slot.state, UploadArenaSlotState::Empty))
        })
        .or_else(|| {
            slots
                .iter()
                .position(|slot| matches!(slot.state, UploadArenaSlotState::Free))
        })
}

fn next_slot_capacity(required: u64, current: u64, max_buffer_size: u64) -> Option<u64> {
    if required == 0 || required > max_buffer_size {
        return None;
    }
    let doubled = current.saturating_mul(2);
    let target = DEFAULT_SLOT_BYTES.max(required).max(doubled);
    let rounded = target
        .checked_next_power_of_two()
        .unwrap_or(max_buffer_size);
    Some(rounded.min(max_buffer_size).max(required))
}

fn create_persistent_slot_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    profiling::scope!("frame_upload_arena::create_persistent_slot");
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("frame_upload_arena_slot"),
        size,
        usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: true,
    });
    crate::profiling::note_resource_churn!(Buffer, "render_graph::frame_upload_arena_slot");
    buffer
}

fn create_temporary_staging_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    profiling::scope!("frame_upload_arena::create_temporary_staging");
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("frame_upload_temporary_staging"),
        size,
        usage: wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: true,
    });
    crate::profiling::note_resource_churn!(Buffer, "render_graph::frame_upload_temporary_staging");
    buffer
}

fn log_oversized_upload(required: u64, max_buffer_size: u64) {
    let count = OVERSIZED_UPLOAD_LOG_COUNTER.fetch_add(1, Ordering::Relaxed) + 1;
    if count <= 5 || count.is_multiple_of(120) {
        logger::warn!(
            "frame upload arena: staging bytes {required} exceed max_buffer_size {max_buffer_size}; falling back to queue writes"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_slots() -> [UploadArenaSlot; UPLOAD_ARENA_SLOTS] {
        std::array::from_fn(|_| UploadArenaSlot::empty())
    }

    #[test]
    fn slot_capacity_grows_to_power_of_two_with_default_floor() {
        assert_eq!(
            next_slot_capacity(64, 0, 8 * 1024 * 1024),
            Some(1024 * 1024)
        );
        assert_eq!(
            next_slot_capacity(2 * 1024 * 1024 + 1, 1024 * 1024, 8 * 1024 * 1024),
            Some(4 * 1024 * 1024)
        );
    }

    #[test]
    fn slot_capacity_clamps_to_device_max_without_underallocating() {
        assert_eq!(next_slot_capacity(900, 0, 1000), Some(1000));
        assert_eq!(next_slot_capacity(1001, 0, 1000), None);
    }

    #[test]
    fn select_writable_slot_prefers_existing_capacity() {
        let mut slots = empty_slots();
        slots[0].state = UploadArenaSlotState::Free;
        slots[0].capacity = 64;
        slots[1].state = UploadArenaSlotState::Free;
        slots[1].capacity = 1024;

        assert_eq!(select_writable_slot(&slots, 512), Some(1));
    }

    #[test]
    fn select_writable_slot_uses_empty_before_growing_small_free_slot() {
        let mut slots = empty_slots();
        slots[0].state = UploadArenaSlotState::Free;
        slots[0].capacity = 64;

        assert_eq!(select_writable_slot(&slots, 512), Some(1));
    }

    #[test]
    fn select_writable_slot_returns_none_when_all_slots_are_busy() {
        let mut slots = empty_slots();
        for (i, slot) in slots.iter_mut().enumerate() {
            slot.state = UploadArenaSlotState::InFlight {
                generation: i as u64 + 1,
            };
        }

        assert_eq!(select_writable_slot(&slots, 512), None);
    }

    #[test]
    fn stale_completion_does_not_free_newer_generation() {
        let mut arena = PersistentUploadArena::new();
        arena.slots[0].state = UploadArenaSlotState::InFlight { generation: 2 };

        arena.start_remap(0, 1);

        assert_eq!(
            arena.slots[0].state,
            UploadArenaSlotState::InFlight { generation: 2 }
        );
    }

    #[test]
    fn successful_remap_returns_slot_to_free_state() {
        let mut arena = PersistentUploadArena::new();
        arena.slots[0].state = UploadArenaSlotState::Remapping { generation: 4 };

        arena.finish_remap(0, 4, true);

        assert_eq!(arena.slots[0].state, UploadArenaSlotState::Free);
    }

    #[test]
    fn failed_remap_drops_slot() {
        let mut arena = PersistentUploadArena::new();
        arena.slots[0].state = UploadArenaSlotState::Remapping { generation: 4 };
        arena.slots[0].capacity = 1024;

        arena.finish_remap(0, 4, false);

        assert_eq!(arena.slots[0].state, UploadArenaSlotState::Empty);
        assert_eq!(arena.slots[0].capacity, 0);
    }

    #[test]
    fn pressure_counts_in_flight_and_remapping_slots() {
        let mut arena = PersistentUploadArena::new();
        arena.slots[0].state = UploadArenaSlotState::Free;
        arena.slots[0].capacity = 64;
        arena.slots[1].state = UploadArenaSlotState::InFlight { generation: 1 };
        arena.slots[1].capacity = 128;
        arena.slots[2].state = UploadArenaSlotState::Remapping { generation: 2 };
        arena.slots[2].capacity = 256;

        let pressure = arena.pressure();

        assert_eq!(pressure.capacity_bytes, 448);
        assert_eq!(pressure.free_slots, 1);
        assert_eq!(pressure.in_flight_slots, 1);
        assert_eq!(pressure.remapping_slots, 1);
    }
}
