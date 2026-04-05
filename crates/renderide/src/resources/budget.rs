//! VRAM accounting and streaming policy hooks (implement eviction later).

/// Running tally of GPU bytes tied to pooled resources.
#[derive(Debug, Default, Clone)]
pub struct VramAccounting {
    total_resident_bytes: u64,
}

impl VramAccounting {
    /// Adds `bytes` when a resource becomes resident.
    pub fn on_resident_added(&mut self, bytes: u64) {
        self.total_resident_bytes = self.total_resident_bytes.saturating_add(bytes);
    }

    /// Subtracts `bytes` when a resource is freed or evicted.
    pub fn on_resident_removed(&mut self, bytes: u64) {
        self.total_resident_bytes = self.total_resident_bytes.saturating_sub(bytes);
    }

    /// Current accounted resident size (approximate; mirrors buffer sizes at upload time).
    pub fn total_resident_bytes(&self) -> u64 {
        self.total_resident_bytes
    }
}

/// Future **LRU / priority / budget clamp** / mipmap residency: suggest IDs to drop under pressure.
///
/// Default implementation is a no-op. Replace with a policy that tracks last frame touched,
/// material importance, or host hints when implementing streaming.
pub trait StreamingPolicy: Send {
    /// Called when a draw or upload touches a mesh (for future LRU).
    fn note_mesh_access(&mut self, _asset_id: i32) {}

    /// Under memory pressure, return mesh asset IDs to evict (highest priority first).
    fn suggest_mesh_evictions(&self, _budget: &VramAccounting) -> Vec<i32> {
        Vec::new()
    }
}

/// No-op policy until streaming is implemented.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopStreamingPolicy;

impl StreamingPolicy for NoopStreamingPolicy {}

/// Extension hook: classify resources for future tiered residency (`Hot`, `Streaming`, ...).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ResidencyTier {
    /// Always try to keep resident (hero assets, bound materials).
    #[default]
    Hot,
    /// May be evicted when over budget (background LODs).
    Streaming,
    /// Not required to stay resident across frames.
    Volatile,
}

/// Metadata for future eviction (not enforced yet).
#[derive(Clone, Debug)]
pub struct MeshResidencyMeta {
    pub tier: ResidencyTier,
}

impl Default for MeshResidencyMeta {
    fn default() -> Self {
        Self {
            tier: ResidencyTier::Hot,
        }
    }
}
