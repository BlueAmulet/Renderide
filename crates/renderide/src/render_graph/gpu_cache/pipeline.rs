//! Render pipeline cache keyed by pass-specific descriptors.

use std::hash::Hash;
use std::sync::Arc;

use hashbrown::HashMap;
use parking_lot::Mutex;

use crate::concurrency::{KeyedSingleFlight, SingleFlightPermit};

/// Typed cache for `wgpu::RenderPipeline` values.
#[derive(Debug)]
pub(crate) struct RenderPipelineMap<K>
where
    K: Eq + Hash,
{
    /// Shared map storing pipelines behind `Arc` so record paths can clone handles cheaply.
    pipelines: Mutex<HashMap<K, Arc<wgpu::RenderPipeline>>>,
    /// Per-key compile coordination for cache misses.
    compiles: KeyedSingleFlight<K>,
}

impl<K> Default for RenderPipelineMap<K>
where
    K: Eq + Hash,
{
    fn default() -> Self {
        Self {
            pipelines: Mutex::new(HashMap::new()),
            compiles: KeyedSingleFlight::default(),
        }
    }
}

impl<K> RenderPipelineMap<K>
where
    K: Clone + Eq + Hash,
{
    /// Returns a cached render pipeline or builds one outside the map lock.
    pub(crate) fn get_or_create(
        &self,
        key: K,
        build: impl Fn(&K) -> wgpu::RenderPipeline,
    ) -> Arc<wgpu::RenderPipeline> {
        loop {
            if let Some(existing) = self.pipelines.lock().get(&key) {
                return existing.clone();
            }

            let leader = match self.compiles.acquire(key.clone()) {
                SingleFlightPermit::Leader(leader) => leader,
                SingleFlightPermit::Waiter(waiter) => {
                    waiter.wait();
                    continue;
                }
            };

            if let Some(existing) = self.pipelines.lock().get(&key) {
                return existing.clone();
            }

            let pipeline = Arc::new(build(&key));
            self.pipelines.lock().insert(key, pipeline.clone());
            drop(leader);
            return pipeline;
        }
    }
}
