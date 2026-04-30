//! Generic locked GPU object cache used by [`super::pipeline::RenderPipelineMap`] and
//! [`super::bindgroup::BindGroupMap`].

use std::hash::Hash;

use hashbrown::HashMap;
use parking_lot::Mutex;

/// Generic locked cache with double-check insertion and optional clear-on-overflow eviction.
#[derive(Debug)]
pub(super) struct GpuCache<K, V> {
    /// Cached objects keyed by pass-specific descriptors.
    entries: Mutex<HashMap<K, V>>,
    /// Maximum number of entries retained before the map is cleared.
    max_entries: Option<usize>,
}

impl<K, V> Default for GpuCache<K, V> {
    fn default() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            max_entries: None,
        }
    }
}

impl<K, V> GpuCache<K, V> {
    /// Creates an empty unbounded map.
    pub(super) fn new() -> Self {
        Self::default()
    }

    /// Creates an empty map that clears itself before inserting once it reaches `max_entries`.
    pub(super) fn with_max_entries(max_entries: usize) -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            max_entries: Some(max_entries),
        }
    }
}

impl<K, V> GpuCache<K, V>
where
    K: Clone + Eq + Hash,
    V: Clone,
{
    /// Returns a cached value or builds, double-checks, and inserts a new one.
    pub(super) fn get_or_create(&self, key: K, build: impl FnOnce(&K) -> V) -> V {
        {
            let guard = self.entries.lock();
            if let Some(existing) = guard.get(&key) {
                return existing.clone();
            }
        }

        let value = build(&key);
        let mut guard = self.entries.lock();
        if let Some(existing) = guard.get(&key) {
            return existing.clone();
        }
        if self
            .max_entries
            .is_some_and(|max_entries| guard.len() >= max_entries)
        {
            guard.clear();
        }
        guard.insert(key, value.clone());
        value
    }

    /// Clears all cached entries.
    #[cfg(test)]
    pub(super) fn clear(&self) {
        self.entries.lock().clear();
    }

    /// Returns the number of cached entries.
    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.entries.lock().len()
    }
}

#[cfg(test)]
mod tests {
    use super::GpuCache;

    #[test]
    fn cache_separates_keys_and_reuses_values() {
        let cache = GpuCache::<u32, u32>::new();

        assert_eq!(cache.get_or_create(1, |_| 10), 10);
        assert_eq!(cache.get_or_create(1, |_| 20), 10);
        assert_eq!(cache.get_or_create(2, |_| 20), 20);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn bounded_cache_clears_before_overflow_insert() {
        let cache = GpuCache::<u32, u32>::with_max_entries(2);

        assert_eq!(cache.get_or_create(1, |_| 10), 10);
        assert_eq!(cache.get_or_create(2, |_| 20), 20);
        assert_eq!(cache.len(), 2);

        assert_eq!(cache.get_or_create(3, |_| 30), 30);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get_or_create(1, |_| 40), 40);
    }

    #[test]
    fn cache_clear_drops_entries() {
        let cache = GpuCache::<u32, u32>::new();

        cache.get_or_create(1, |_| 10);
        cache.clear();

        assert_eq!(cache.len(), 0);
    }
}
