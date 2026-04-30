//! Render-graph lifetime state owned by [`super::RenderBackend`].
//!
//! This keeps graph cache/history/transient ownership together instead of scattering long-lived
//! graph resources across the backend facade.

use crate::camera::ViewId;
use crate::render_graph::{GraphCache, TransientPool};

use super::super::{HistoryRegistry, ViewResourceRegistry};

/// Long-lived render-graph resources retained across frames.
pub(super) struct RenderGraphState {
    /// Cached compiled frame graph keyed by the shared render-graph cache inputs.
    pub(super) frame_graph_cache: GraphCache,
    /// Render-graph transient texture/buffer pool retained across frames.
    transient_pool: TransientPool,
    /// Persistent ping-pong resources used by graph history slots
    /// (`ImportSource::PingPong` / `BufferImportSource::PingPong`).
    history_registry: HistoryRegistry,
    /// Retained logical-view ownership for every backend cache that lives beyond one frame.
    view_resources: ViewResourceRegistry,
}

impl RenderGraphState {
    /// Creates empty graph state before GPU attach.
    pub(super) fn new() -> Self {
        Self {
            frame_graph_cache: GraphCache::default(),
            transient_pool: TransientPool::new(),
            history_registry: HistoryRegistry::new(),
            view_resources: ViewResourceRegistry::new(),
        }
    }

    /// Mutable graph transient pool.
    pub(super) fn transient_pool_mut(&mut self) -> &mut TransientPool {
        &mut self.transient_pool
    }

    /// Mutable history registry.
    pub(super) fn history_registry_mut(&mut self) -> &mut HistoryRegistry {
        &mut self.history_registry
    }

    /// Shared history registry.
    pub(super) fn history_registry(&self) -> &HistoryRegistry {
        &self.history_registry
    }

    /// Mutable transient pool and history registry for graph execution after the cached graph
    /// has been temporarily removed from [`Self::frame_graph_cache`].
    pub(super) fn execution_resources_mut(&mut self) -> (&mut TransientPool, &mut HistoryRegistry) {
        (&mut self.transient_pool, &mut self.history_registry)
    }

    /// Synchronizes active view ownership and releases graph-owned view resources immediately.
    pub(super) fn sync_active_views<I>(&mut self, active_views: I) -> Vec<ViewId>
    where
        I: IntoIterator<Item = ViewId>,
    {
        let retired = self.view_resources.sync_active_views(active_views);
        self.frame_graph_cache.release_view_resources(&retired);
        retired
    }
}
