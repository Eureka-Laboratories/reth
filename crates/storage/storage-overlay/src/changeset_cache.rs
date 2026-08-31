//! Trie changeset caching utilities.
//!
//! This module provides functionality to compute trie changesets for a given block,
//! which represent the old trie node values before the block was processed.
//!
//! It also provides an efficient in-memory cache for these changesets, which is essential for:
//! - **Reorg support**: Quickly access changesets to revert blocks during chain reorganizations
//! - **Memory efficiency**: Explicit eviction releases persisted changesets

use crate::{database_state_frontiers, OverlayManager, OverlayStateProvider};
use alloy_eips::BlockNumHash;
use alloy_primitives::{map::B256Map, BlockNumber, B256};
use parking_lot::RwLock;
use reth_metrics::{
    metrics::{Counter, Gauge},
    Metrics,
};
use reth_primitives_traits::{FastInstant as Instant, NodePrimitives};
use reth_storage_api::{
    BlockNumReader, ChangeSetReader, DBProvider, PruneCheckpointReader, StageCheckpointReader,
    StorageChangeSetReader, StorageSettingsCache,
};
use reth_storage_errors::provider::{ProviderError, ProviderResult};
use reth_trie::trie_cursor::{InMemoryTrieCursorFactory, TrieCursor, TrieCursorFactory};
use reth_trie_common::updates::{StorageTrieUpdatesSorted, TrieUpdatesSorted};
use reth_trie_db::{DatabaseTrieCursorFactory, TrieTableAdapter};
use std::{
    collections::{BTreeMap, HashMap},
    fmt,
    ops::RangeInclusive,
    sync::{Arc, OnceLock},
};
use tracing::{debug, debug_span, warn};

#[cfg(test)]
use reth_trie::{changesets::compute_trie_changesets, HashedPostStateSorted, TrieInputSorted};
#[cfg(test)]
use reth_trie_db::{DatabaseHashedCursorFactory, DatabaseHashedPostState, DatabaseStateRoot};

/// Computes block trie updates using the changeset cache.
///
/// # Algorithm
///
/// For block N:
/// 1. Get cumulative trie reverts from block N+1 to db tip using the cache
/// 2. Create an overlay cursor factory with these reverts (representing trie state after block N)
/// 3. Walk through account trie changesets for block N
/// 4. For each changed path, look up the current value using the overlay cursor
/// 5. Walk through storage trie changesets for block N
/// 6. For each changed path, look up the current value using the overlay cursor
/// 7. Return the collected trie updates
///
/// # Arguments
///
/// * `provider` - Database provider for accessing changesets and block data
/// * `block_number` - Block number to compute trie updates for
///
/// # Returns
///
/// Trie updates representing the state of trie nodes after the block was processed
///
/// # Errors
///
/// Returns error if:
/// - Block number exceeds database tip
/// - Database access fails
/// - Cache retrieval fails
pub(crate) fn compute_block_trie_updates<N, Provider>(
    overlay_manager: &OverlayManager<N>,
    provider: &Provider,
    block_number: BlockNumber,
) -> ProviderResult<TrieUpdatesSorted>
where
    N: NodePrimitives,
    Provider: DBProvider
        + ChangeSetReader
        + StorageChangeSetReader
        + PruneCheckpointReader
        + StageCheckpointReader
        + BlockNumReader
        + StorageSettingsCache,
{
    reth_trie_db::with_adapter!(provider, |A| {
        compute_block_trie_updates_inner::<_, _, A>(overlay_manager, provider, block_number)
    })
}

fn compute_block_trie_updates_inner<N, Provider, A>(
    overlay_manager: &OverlayManager<N>,
    provider: &Provider,
    block_number: BlockNumber,
) -> ProviderResult<TrieUpdatesSorted>
where
    N: NodePrimitives,
    Provider: DBProvider
        + ChangeSetReader
        + StorageChangeSetReader
        + PruneCheckpointReader
        + StageCheckpointReader
        + BlockNumReader
        + StorageSettingsCache,
    A: TrieTableAdapter,
{
    let tx = provider.tx_ref();
    let cache = overlay_manager.changeset_cache();
    let (partial_state_trie, finish) = database_state_frontiers(provider)?;

    // Step 1: Get the trie changesets for the target block from cache
    let changesets = cache.get_or_compute(
        overlay_manager,
        provider,
        block_number,
        partial_state_trie,
        finish,
    )?;

    // Step 2: Get the trie reverts for the state after the target block using the cache
    let reverts = cache.get_or_compute_range(
        overlay_manager,
        provider,
        (block_number + 1)..=finish.number,
        partial_state_trie,
        finish,
    )?;

    // Step 3: Create an InMemoryTrieCursorFactory with the reverts
    // This gives us the trie state as it was after the target block was processed
    let db_cursor_factory = DatabaseTrieCursorFactory::<_, A>::new(tx);
    let cursor_factory = InMemoryTrieCursorFactory::new(db_cursor_factory, &reverts);

    // Step 4: Collect all account trie nodes that changed in the target block
    let account_nodes_ref = changesets.account_nodes_ref();
    let mut account_nodes = Vec::with_capacity(account_nodes_ref.len());
    let mut account_cursor = cursor_factory.account_trie_cursor()?;

    // Iterate over the account nodes from the changesets
    for (nibbles, _old_node) in account_nodes_ref {
        // Look up the current value of this trie node using the overlay cursor
        let node_value = account_cursor.seek_exact(*nibbles)?.map(|(_, node)| node);
        account_nodes.push((*nibbles, node_value));
    }

    // Step 5: Collect all storage trie nodes that changed in the target block
    let mut storage_tries = B256Map::default();

    // Iterate over the storage tries from the changesets
    for (hashed_address, storage_changeset) in changesets.storage_tries_ref() {
        let mut storage_cursor = cursor_factory.storage_trie_cursor(*hashed_address)?;
        let storage_nodes_ref = storage_changeset.storage_nodes_ref();
        let mut storage_nodes = Vec::with_capacity(storage_nodes_ref.len());

        // Iterate over the storage nodes for this account
        for (nibbles, _old_node) in storage_nodes_ref {
            // Look up the current value of this storage trie node
            let node_value = storage_cursor.seek_exact(*nibbles)?.map(|(_, node)| node);
            storage_nodes.push((*nibbles, node_value));
        }

        storage_tries.insert(
            *hashed_address,
            StorageTrieUpdatesSorted { storage_nodes, is_deleted: storage_changeset.is_deleted },
        );
    }

    Ok(TrieUpdatesSorted::new(account_nodes, storage_tries))
}

/// A pending changeset computation that other threads can wait on.
///
/// When an eager changeset producer starts computing changesets for a block, it registers a
/// pending entry. If another thread needs the same changeset before the computation finishes, it
/// waits on this entry instead of falling back to the expensive DB-based computation.
struct PendingChangeset {
    /// `None` when cancelled (e.g. due to panic), `Some(..)` when resolved with data.
    result: OnceLock<Option<Arc<TrieUpdatesSorted>>>,
}

impl PendingChangeset {
    const fn new() -> Self {
        Self { result: OnceLock::new() }
    }

    /// Blocks until the computation finishes. Returns `Some` if resolved with data,
    /// `None` if the computation was cancelled.
    fn wait(&self) -> Option<Arc<TrieUpdatesSorted>> {
        let _span =
            debug_span!(target: "trie::changeset_cache", "waiting_for_pending_changeset").entered();
        self.result.wait().clone()
    }

    /// Resolves the pending computation with the given result, waking all waiters.
    fn resolve(&self, changesets: Arc<TrieUpdatesSorted>) {
        let _ = self.result.set(Some(changesets));
    }

    /// Cancels the pending computation, waking all waiters so they fall through
    /// to the DB fallback.
    fn cancel(&self) {
        let _ = self.result.set(None);
    }
}

impl fmt::Debug for PendingChangeset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let is_resolved = self.result.get().is_some();
        f.debug_struct("PendingChangeset").field("resolved", &is_resolved).finish()
    }
}

/// Guard for a pending changeset computation.
///
/// Returned by [`ChangesetCache::register_pending`]. Must be resolved via [`Self::resolve`] to
/// insert the computed changesets into the cache and wake waiting threads.
///
/// If dropped without resolving (e.g. due to a panic), the pending entry is automatically
/// cancelled so waiters fall through to the DB fallback.
#[must_use = "call .resolve() to insert changesets into the cache"]
pub struct PendingChangesetGuard {
    cache: ChangesetCache,
    key: ChangesetRangeKey,
    /// `None` after [`Self::resolve`] has been called.
    pending: Option<Arc<PendingChangeset>>,
}

impl PendingChangesetGuard {
    /// Resolves the pending computation by inserting the changesets into the cache
    /// and waking all waiting threads.
    pub fn resolve(mut self, changesets: Arc<TrieUpdatesSorted>) {
        self.cache.insert_resolved(self.key, changesets);
        self.pending = None;
    }
}

impl fmt::Debug for PendingChangesetGuard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PendingChangesetGuard").field("key", &self.key).finish()
    }
}

impl Drop for PendingChangesetGuard {
    fn drop(&mut self) {
        let Some(pending) = self.pending.take() else {
            // Guard was resolved successfully already, no-op
            return
        };

        let mut inner = self.cache.inner.write();
        let Some(removed) = inner.pending.remove(&self.key) else { return };

        if Arc::ptr_eq(&removed, &pending) {
            drop(inner);
            debug!(
                target: "trie::changeset_cache",
                key = ?self.key,
                "Pending changeset dropped without resolution, cancelling"
            );
            removed.cancel();
        } else {
            // Put it back — it belongs to a different registration.
            inner.pending.insert(self.key, removed);
        }
    }
}

/// Thread-safe changeset cache.
///
/// This type wraps a shared, mutable reference to the cache inner.
/// The `RwLock` enables concurrent reads while ensuring exclusive access for writes.
#[derive(Debug, Clone)]
pub(crate) struct ChangesetCache {
    inner: Arc<RwLock<ChangesetCacheInner>>,
}

impl Default for ChangesetCache {
    fn default() -> Self {
        Self::new()
    }
}

impl ChangesetCache {
    /// Creates a new cache.
    ///
    /// The cache has no capacity limit and relies on explicit eviction
    /// via the `evict()` method to manage memory usage.
    pub(crate) fn new() -> Self {
        Self { inner: Arc::new(RwLock::new(ChangesetCacheInner::new())) }
    }

    /// Registers a pending changeset computation for the given block.
    ///
    /// Call this before starting an eager changeset computation so that concurrent readers wait
    /// for the result instead of falling back to the expensive DB-based computation.
    ///
    /// The returned [`PendingChangesetGuard`] must be used to resolve the pending entry. If it is
    /// dropped without resolving (e.g. because the computing task panicked), the pending entry is
    /// removed so waiters fall through to the DB fallback.
    pub(crate) fn register_pending(
        &self,
        block_number: BlockNumber,
        block_hash: B256,
    ) -> PendingChangesetGuard {
        let key = ChangesetRangeKey::single(block_number, block_hash);
        let pending = Arc::new(PendingChangeset::new());
        self.inner.write().pending.insert(key, Arc::clone(&pending));
        PendingChangesetGuard { cache: self.clone(), key, pending: Some(pending) }
    }

    /// Inserts eagerly computed changesets and wakes any threads waiting on them.
    fn insert_resolved(&self, key: ChangesetRangeKey, changesets: Arc<TrieUpdatesSorted>) {
        let pending = {
            let mut inner = self.inner.write();
            inner.insert(key, Arc::clone(&changesets));
            inner.pending.remove(&key)
        };

        // Resolve outside the write lock so waking waiters cannot contend on it.
        if let Some(pending) = pending {
            pending.resolve(changesets);
        }
    }

    /// Evicts changesets for blocks below the given block number.
    ///
    /// This should be called after blocks are persisted to the database to free
    /// memory for changesets that are no longer needed in the cache.
    ///
    /// # Arguments
    ///
    /// * `up_to_block` - Evict blocks with number < this value. Blocks with number >= this value
    ///   are retained.
    pub(crate) fn evict(&self, up_to_block: BlockNumber) {
        self.inner.write().evict(up_to_block)
    }

    /// Gets changesets from cache, or computes them on-the-fly if missing.
    ///
    /// This is the primary API for retrieving changesets. It checks the cache first, then falls
    /// back to computing from database state if missing.
    ///
    /// # Arguments
    ///
    /// * `block_number` - Block number (for cache insertion and logging)
    /// * `provider` - Database provider for DB access
    ///
    /// # Returns
    ///
    /// Changesets for the block, either from cache or computed on-the-fly.
    pub(crate) fn get_or_compute<N, P>(
        &self,
        overlay_manager: &OverlayManager<N>,
        provider: &P,
        block_number: BlockNumber,
        partial_state_trie: BlockNumHash,
        finish: BlockNumHash,
    ) -> ProviderResult<Arc<TrieUpdatesSorted>>
    where
        N: NodePrimitives,
        P: DBProvider
            + ChangeSetReader
            + StorageChangeSetReader
            + PruneCheckpointReader
            + BlockNumReader
            + StorageSettingsCache,
    {
        self.get_or_compute_range(
            overlay_manager,
            provider,
            block_number..=block_number,
            partial_state_trie,
            finish,
        )
    }

    /// Gets or computes trie reverts for a range of blocks.
    ///
    /// If all blocks in the range are cached, this method retrieves and accumulates those
    /// per-block trie changesets (reverts) in reverse order (newest to oldest), so that older
    /// values take precedence when there are conflicts.
    ///
    /// If any block is missing from cache, this falls back to one aggregate database computation
    /// for the whole range. The aggregate result restores the trie to the state before the range
    /// and is inserted into the range cache.
    ///
    /// # Arguments
    ///
    /// * `provider` - Database provider for DB access and block lookups
    /// * `range` - Block range to accumulate reverts for (inclusive)
    ///
    /// # Returns
    ///
    /// Accumulated trie reverts for all blocks in the specified range
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Any block in the range is beyond the database tip
    /// - Database access fails
    /// - Block hash lookup fails
    /// - Changeset computation fails
    pub(crate) fn get_or_compute_range<N, P>(
        &self,
        overlay_manager: &OverlayManager<N>,
        provider: &P,
        range: RangeInclusive<BlockNumber>,
        partial_state_trie: BlockNumHash,
        finish: BlockNumHash,
    ) -> ProviderResult<Arc<TrieUpdatesSorted>>
    where
        N: NodePrimitives,
        P: DBProvider
            + ChangeSetReader
            + StorageChangeSetReader
            + PruneCheckpointReader
            + BlockNumReader
            + StorageSettingsCache,
    {
        let start_block = *range.start();
        let end_block = *range.end();
        let timer = Instant::now();

        if end_block > finish.number {
            return Err(ProviderError::InsufficientChangesets {
                requested: end_block,
                available: 0..=finish.number,
            });
        }

        debug!(
            target: "trie::changeset_cache",
            start_block,
            end_block,
            ?partial_state_trie,
            ?finish,
            "Starting get_or_compute_range"
        );

        if start_block > end_block {
            debug!(
                target: "trie::changeset_cache",
                start_block,
                end_block,
                "Empty changeset range requested"
            );
            return Ok(Arc::new(TrieUpdatesSorted::default()))
        }

        let end_block_hash = provider.block_hash(end_block)?.ok_or_else(|| {
            ProviderError::other(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("block hash not found for block number {}", end_block),
            ))
        })?;
        let range_key = ChangesetRangeKey::new(start_block, end_block, end_block_hash);

        if let Some(accumulated_reverts) = self.inner.read().get(&range_key) {
            let elapsed = timer.elapsed();

            debug!(
                target: "trie::changeset_cache",
                ?elapsed,
                start_block,
                end_block,
                ?end_block_hash,
                num_blocks = end_block.saturating_sub(start_block).saturating_add(1),
                "Changeset cache HIT for block range"
            );

            return Ok(accumulated_reverts)
        }

        let mut cached_reverts =
            Vec::with_capacity(end_block.saturating_sub(start_block).saturating_add(1) as usize);
        let mut all_cached = true;

        for block_number in range.rev() {
            // Get the block hash for this block number
            let block_hash = if block_number == end_block {
                end_block_hash
            } else {
                provider.block_hash(block_number)?.ok_or_else(|| {
                    ProviderError::other(std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        format!("block hash not found for block number {}", block_number),
                    ))
                })?
            };

            debug!(
                target: "trie::changeset_cache",
                block_number,
                ?block_hash,
                "Looked up block hash for block number in range"
            );

            let block_key = ChangesetRangeKey::single(block_number, block_hash);

            // Take the cached entry, or a handle to an in-flight eager computation for it. The
            // pending map is only ever populated when eager changeset caching is enabled, so this
            // is a plain map lookup otherwise.
            let pending = {
                let inner = self.inner.read();
                match inner.get(&block_key) {
                    Some(changesets) => {
                        cached_reverts.push(changesets);
                        continue
                    }
                    None => inner.pending.get(&block_key).cloned(),
                }
            };

            // Waiting on the producer is far cheaper than the aggregate DB fallback below.
            if let Some(pending) = pending {
                let wait_start = Instant::now();
                if let Some(changesets) = pending.wait() {
                    debug!(
                        target: "trie::changeset_cache",
                        block_number,
                        ?block_hash,
                        elapsed = ?wait_start.elapsed(),
                        "Pending changeset resolved for block in range"
                    );
                    cached_reverts.push(changesets);
                    continue
                }

                debug!(
                    target: "trie::changeset_cache",
                    block_number,
                    ?block_hash,
                    elapsed = ?wait_start.elapsed(),
                    "Pending changeset was cancelled, falling through to DB computation"
                );
            }

            all_cached = false;
            break
        }

        // Merging per-block reverts cannot reproduce a storage wipe across more than one block.
        // `TrieUpdatesSorted::extend_ref_and_sort`, which backs `merge_slice` below the k-way
        // threshold, applies oldest-wins precedence to node values but leaves `is_deleted` set by
        // whichever revert seeded the accumulator. An account destroyed and recreated inside the
        // range then merges to "delete the storage trie" when the correct revert restores it. Defer
        // to the aggregate DB computation, which derives the wipe from reverts in one pass.
        if all_cached && start_block != end_block {
            let merged_wipe = cached_reverts
                .iter()
                .any(|revert| revert.storage_tries_ref().values().any(|trie| trie.is_deleted));

            if merged_wipe {
                debug!(
                    target: "trie::changeset_cache",
                    start_block,
                    end_block,
                    "Cached range contains a storage wipe, using aggregate DB computation"
                );
                all_cached = false;
            }
        }

        if all_cached {
            // `merge_slice` gives precedence to earlier items, so pass reverts oldest-to-newest.
            cached_reverts.reverse();
            let accumulated_reverts = Arc::new(TrieUpdatesSorted::merge_slice(&cached_reverts));
            let elapsed = timer.elapsed();

            let num_account_nodes = accumulated_reverts.account_nodes_ref().len();
            let num_storage_tries = accumulated_reverts.storage_tries_ref().len();

            debug!(
                target: "trie::changeset_cache",
                ?elapsed,
                start_block,
                end_block,
                num_blocks = end_block.saturating_sub(start_block).saturating_add(1),
                num_account_nodes,
                num_storage_tries,
                "Finished accumulating cached trie reverts for block range"
            );

            self.inner.write().insert(range_key, Arc::clone(&accumulated_reverts));
            return Ok(accumulated_reverts)
        }

        warn!(
            target: "trie::changeset_cache",
            start_block,
            end_block,
            "Changeset cache MISS in range, falling back to aggregate DB-based computation"
        );

        let overlay = overlay_manager
            .overlay_builder(finish.hash)
            .with_no_reverts()
            .build_overlay_at_frontiers(provider, partial_state_trie, finish)?;
        let state_trie_provider = OverlayStateProvider::new(
            provider,
            overlay,
            provider.cached_storage_settings().is_v2(),
        );

        let accumulated_reverts = Arc::new(reth_trie_db::compute_range_trie_changesets(
            provider,
            &state_trie_provider,
            start_block..=end_block,
            finish.number,
        )?);

        let elapsed = timer.elapsed();

        let num_account_nodes = accumulated_reverts.account_nodes_ref().len();
        let num_storage_tries = accumulated_reverts.storage_tries_ref().len();

        debug!(
            target: "trie::changeset_cache",
            ?elapsed,
            start_block,
            end_block,
            ?end_block_hash,
            num_blocks = end_block.saturating_sub(start_block).saturating_add(1),
            num_account_nodes,
            num_storage_tries,
            "Finished accumulating trie reverts for block range"
        );

        self.inner.write().insert(range_key, Arc::clone(&accumulated_reverts));

        Ok(accumulated_reverts)
    }
}

/// Cache key for one contiguous range of canonical trie changesets.
///
/// The end block hash disambiguates canonical rewrites where the same block numbers later refer to
/// a different chain. For a single block, `start_block == end_block` and `end_block_hash` is that
/// block's hash.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct ChangesetRangeKey {
    start_block: BlockNumber,
    end_block: BlockNumber,
    end_block_hash: B256,
}

impl ChangesetRangeKey {
    const fn new(start_block: BlockNumber, end_block: BlockNumber, end_block_hash: B256) -> Self {
        Self { start_block, end_block, end_block_hash }
    }

    const fn single(block_number: BlockNumber, block_hash: B256) -> Self {
        Self::new(block_number, block_number, block_hash)
    }
}

/// In-memory cache for trie changesets with explicit eviction policy.
///
/// Holds changesets for blocks or block ranges that have been validated but not yet persisted.
/// Keyed by canonical block range. Eviction is controlled
/// explicitly by the engine API tree handler when persistence completes.
///
/// ## Eviction Policy
///
/// Unlike traditional caches with automatic eviction, this cache requires explicit
/// eviction calls. The engine API tree handler calls `evict(block_number)` after
/// blocks are persisted to the database, ensuring changesets remain available
/// until their corresponding blocks are safely on disk.
///
/// ## Metrics
///
/// The cache maintains several metrics for observability:
/// - `hits`: Number of successful cache lookups
/// - `misses`: Number of failed cache lookups
/// - `evictions`: Number of blocks evicted
/// - `size`: Current number of cached blocks
#[derive(Debug)]
struct ChangesetCacheInner {
    /// Cache entries keyed by inclusive block range plus the range's canonical end hash.
    entries: HashMap<ChangesetRangeKey, Arc<TrieUpdatesSorted>>,

    /// Range start block to cache keys mapping for eviction.
    range_starts: BTreeMap<BlockNumber, Vec<ChangesetRangeKey>>,

    /// In-flight eager changeset computations, keyed by single-block range.
    ///
    /// Only populated when eager changeset caching is enabled. Threads that need an entry while
    /// it is being computed wait here instead of running the aggregate DB fallback.
    pending: HashMap<ChangesetRangeKey, Arc<PendingChangeset>>,

    /// Metrics for monitoring cache behavior
    metrics: ChangesetCacheMetrics,
}

/// Metrics for the changeset cache.
///
/// These metrics provide visibility into cache performance and help identify
/// potential issues like high miss rates.
#[derive(Metrics, Clone)]
#[metrics(scope = "trie.changeset_cache")]
struct ChangesetCacheMetrics {
    /// Cache hit counter
    hits: Counter,

    /// Cache miss counter
    misses: Counter,

    /// Eviction counter
    evictions: Counter,

    /// Current cache size (number of entries)
    size: Gauge,
}

impl Default for ChangesetCacheInner {
    fn default() -> Self {
        Self::new()
    }
}

impl ChangesetCacheInner {
    /// Creates a new empty changeset cache.
    ///
    /// The cache has no capacity limit and relies on explicit eviction
    /// via the `evict()` method to manage memory usage.
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
            range_starts: BTreeMap::new(),
            pending: HashMap::new(),
            metrics: Default::default(),
        }
    }

    fn get(&self, key: &ChangesetRangeKey) -> Option<Arc<TrieUpdatesSorted>> {
        match self.entries.get(key) {
            Some(changesets) => {
                self.metrics.hits.increment(1);
                Some(Arc::clone(changesets))
            }
            None => {
                self.metrics.misses.increment(1);
                None
            }
        }
    }

    fn insert(&mut self, key: ChangesetRangeKey, changesets: Arc<TrieUpdatesSorted>) {
        debug!(
            target: "trie::changeset_cache",
            ?key,
            cache_size_before = self.entries.len(),
            "Inserting changeset into cache"
        );

        let is_new_entry = self.entries.insert(key, changesets).is_none();

        if is_new_entry {
            self.range_starts.entry(key.start_block).or_default().push(key);
        }

        // Update size metric
        self.metrics.size.set(self.entries.len() as f64);

        debug!(
            target: "trie::changeset_cache",
            ?key,
            cache_size_after = self.entries.len(),
            "Changeset inserted into cache"
        );
    }

    fn evict(&mut self, up_to_block: BlockNumber) {
        debug!(
            target: "trie::changeset_cache",
            up_to_block,
            cache_size_before = self.entries.len(),
            "Starting cache eviction"
        );

        // Find all block numbers that should be evicted (< up_to_block)
        let range_starts_to_evict: Vec<u64> =
            self.range_starts.range(..up_to_block).map(|(num, _)| *num).collect();

        // Remove entries for each block number below threshold
        let mut evicted_count = 0;

        for start_block in &range_starts_to_evict {
            if let Some(keys) = self.range_starts.remove(start_block) {
                debug!(
                    target: "trie::changeset_cache",
                    start_block,
                    num_ranges = keys.len(),
                    "Evicting ranges from cache"
                );
                for key in keys {
                    if self.entries.remove(&key).is_some() {
                        evicted_count += 1;
                    }
                }
            }
        }

        debug!(
            target: "trie::changeset_cache",
            up_to_block,
            evicted_count,
            cache_size_after = self.entries.len(),
            "Finished cache eviction"
        );

        // Update metrics if we evicted anything
        if evicted_count > 0 {
            self.metrics.evictions.increment(evicted_count as u64);
            self.metrics.size.set(self.entries.len() as f64);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Overlay;
    use alloy_consensus::Header;
    use alloy_primitives::{
        keccak256,
        map::{B256Map, HashMap},
        Address, U256,
    };
    use reth_db::{
        models::{AccountBeforeTx, BlockNumberAddress},
        tables,
        transaction::DbTxMut,
    };
    use reth_primitives_traits::{Account, StorageEntry};
    use reth_provider::{
        test_utils::create_test_provider_factory, StaticFileProviderFactory, StaticFileSegment,
        StaticFileWriter,
    };
    use reth_stages_types::{StageCheckpoint, StageId};
    use reth_storage_api::{StageCheckpointWriter, TrieWriter};
    use reth_trie::{BranchNodeCompact, Nibbles, StateRoot};

    // Helper function to create empty TrieUpdatesSorted for testing
    fn create_test_changesets() -> Arc<TrieUpdatesSorted> {
        Arc::new(TrieUpdatesSorted::new(vec![], B256Map::default()))
    }

    fn empty_overlay() -> Overlay {
        Overlay { trie_updates: Arc::default(), hashed_post_state: Arc::default() }
    }

    fn insert_test_changesets(
        cache: &mut ChangesetCacheInner,
        block_hash: B256,
        block_number: BlockNumber,
        changesets: Arc<TrieUpdatesSorted>,
    ) {
        cache.insert(ChangesetRangeKey::single(block_number, block_hash), changesets);
    }

    fn get_test_changesets(
        cache: &ChangesetCacheInner,
        block_hash: B256,
        block_number: BlockNumber,
    ) -> Option<Arc<TrieUpdatesSorted>> {
        cache.get(&ChangesetRangeKey::single(block_number, block_hash))
    }

    fn test_account(balance: u64) -> Account {
        Account { balance: U256::from(balance), ..Default::default() }
    }

    fn test_storage(slot: u64, value: u64) -> StorageEntry {
        StorageEntry { key: B256::from(U256::from(slot)), value: U256::from(value) }
    }

    fn seed_headers(
        factory: &impl StaticFileProviderFactory<
            Primitives: reth_primitives_traits::NodePrimitives<BlockHeader = Header>,
        >,
        end_block: BlockNumber,
    ) {
        let static_file_provider = factory.static_file_provider();
        let mut header_writer =
            static_file_provider.latest_writer(StaticFileSegment::Headers).unwrap();
        for block_number in 0..=end_block {
            let header = Header { number: block_number, ..Default::default() };
            header_writer
                .append_header(&header, &B256::with_last_byte(block_number as u8))
                .unwrap();
        }
        header_writer.commit().unwrap();
    }

    fn legacy_compute_range_trie_changesets<Provider>(
        provider: &Provider,
        range: RangeInclusive<BlockNumber>,
    ) -> TrieUpdatesSorted
    where
        Provider: DBProvider
            + ChangeSetReader
            + StorageChangeSetReader
            + BlockNumReader
            + StorageSettingsCache,
    {
        let mut accumulated_reverts = TrieUpdatesSorted::default();
        for block_number in range.rev() {
            let changesets = legacy_compute_block_trie_changesets(provider, block_number);
            accumulated_reverts.extend_ref_and_sort(&changesets);
        }
        accumulated_reverts
    }

    fn legacy_compute_block_trie_changesets<Provider>(
        provider: &Provider,
        block_number: BlockNumber,
    ) -> TrieUpdatesSorted
    where
        Provider: DBProvider
            + ChangeSetReader
            + StorageChangeSetReader
            + BlockNumReader
            + StorageSettingsCache,
    {
        reth_trie_db::with_adapter!(provider, |A| {
            legacy_compute_block_trie_changesets_inner::<_, A>(provider, block_number)
        })
    }

    fn legacy_compute_block_trie_changesets_inner<Provider, A>(
        provider: &Provider,
        block_number: BlockNumber,
    ) -> TrieUpdatesSorted
    where
        Provider: DBProvider
            + ChangeSetReader
            + StorageChangeSetReader
            + BlockNumReader
            + StorageSettingsCache,
        A: TrieTableAdapter,
    {
        let individual_state_revert =
            HashedPostStateSorted::from_reverts(provider, block_number..=block_number).unwrap();
        let cumulative_state_revert =
            HashedPostStateSorted::from_reverts(provider, (block_number + 1)..).unwrap();

        let mut cumulative_state_revert_prev = cumulative_state_revert.clone();
        cumulative_state_revert_prev.extend_ref_and_sort(&individual_state_revert);

        type DbStateRoot<'a, TX, A> =
            StateRoot<DatabaseTrieCursorFactory<&'a TX, A>, DatabaseHashedCursorFactory<&'a TX>>;

        let input_prev = TrieInputSorted::new(
            Arc::default(),
            Arc::new(cumulative_state_revert_prev.clone()),
            cumulative_state_revert_prev.construct_prefix_sets(),
        );
        let cumulative_trie_updates_prev =
            DbStateRoot::<_, A>::overlay_root_from_nodes_with_updates(
                provider.tx_ref(),
                input_prev,
            )
            .unwrap()
            .1
            .into_sorted();

        let input = TrieInputSorted::new(
            Arc::new(cumulative_trie_updates_prev.clone()),
            Arc::new(cumulative_state_revert),
            individual_state_revert.construct_prefix_sets(),
        );
        let trie_updates =
            DbStateRoot::<_, A>::overlay_root_from_nodes_with_updates(provider.tx_ref(), input)
                .unwrap()
                .1
                .into_sorted();

        let db_cursor_factory = DatabaseTrieCursorFactory::<_, A>::new(provider.tx_ref());
        let overlay_factory =
            InMemoryTrieCursorFactory::new(db_cursor_factory, &cumulative_trie_updates_prev);

        compute_trie_changesets(&overlay_factory, &trie_updates).unwrap()
    }

    fn seed_tip_trie_tables<Provider, A>(provider: &Provider)
    where
        Provider: DBProvider + TrieWriter,
        A: TrieTableAdapter,
    {
        type DbStateRoot<'a, TX, A> =
            StateRoot<DatabaseTrieCursorFactory<&'a TX, A>, DatabaseHashedCursorFactory<&'a TX>>;

        let (_, trie_updates) =
            DbStateRoot::<_, A>::from_tx(provider.tx_ref()).root_with_updates().unwrap();
        provider.write_trie_updates(trie_updates).unwrap();
    }

    #[test]
    fn cached_range_merge_keeps_oldest_revert_values() {
        let factory = create_test_provider_factory();
        seed_headers(&factory, 2);

        let provider = factory.provider_rw().unwrap();
        provider.save_stage_checkpoint(StageId::Finish, StageCheckpoint::new(2)).unwrap();

        let cache = ChangesetCache::new();
        let path = Nibbles::from_nibbles([0x1, 0x2]);
        let older_node = BranchNodeCompact::new(0b0001, 0, 0, vec![], None);
        let newer_node = BranchNodeCompact::new(0b0010, 0, 0, vec![], None);

        {
            let mut cache = cache.inner.write();
            insert_test_changesets(
                &mut cache,
                B256::with_last_byte(1),
                1,
                Arc::new(TrieUpdatesSorted::new(
                    vec![(path, Some(older_node.clone()))],
                    B256Map::default(),
                )),
            );
            insert_test_changesets(
                &mut cache,
                B256::with_last_byte(2),
                2,
                Arc::new(TrieUpdatesSorted::new(
                    vec![(path, Some(newer_node))],
                    B256Map::default(),
                )),
            );
        }

        let overlay_manager = OverlayManager::<reth_ethereum_primitives::EthPrimitives>::default();
        let (partial_state_trie, finish) = database_state_frontiers(&*provider).unwrap();
        let accumulated = cache
            .get_or_compute_range(&overlay_manager, &*provider, 1..=2, partial_state_trie, finish)
            .unwrap();
        assert_eq!(accumulated.account_nodes_ref(), &[(path, Some(older_node))]);
    }

    #[test]
    fn aggregate_range_reverts_to_pre_range_state() {
        let factory = create_test_provider_factory();
        seed_headers(&factory, 3);

        let provider = factory.provider_rw().unwrap();
        let address = Address::with_last_byte(1);
        let hashed_address = keccak256(address);
        let slot1 = B256::from(U256::from(1));
        let slot2 = B256::from(U256::from(2));
        let account1 = test_account(10);
        let account2 = test_account(20);
        let account3 = test_account(30);

        provider.tx_ref().put::<tables::HashedAccounts>(hashed_address, account3).unwrap();
        provider
            .tx_ref()
            .put::<tables::HashedStorages>(
                hashed_address,
                StorageEntry { key: keccak256(slot1), value: U256::from(25) },
            )
            .unwrap();
        provider
            .tx_ref()
            .put::<tables::HashedStorages>(
                hashed_address,
                StorageEntry { key: keccak256(slot2), value: U256::from(20) },
            )
            .unwrap();

        provider
            .tx_ref()
            .put::<tables::AccountChangeSets>(1, AccountBeforeTx { address, info: None })
            .unwrap();
        provider
            .tx_ref()
            .put::<tables::AccountChangeSets>(2, AccountBeforeTx { address, info: Some(account1) })
            .unwrap();
        provider
            .tx_ref()
            .put::<tables::AccountChangeSets>(3, AccountBeforeTx { address, info: Some(account2) })
            .unwrap();

        provider
            .tx_ref()
            .put::<tables::StorageChangeSets>(BlockNumberAddress((1, address)), test_storage(1, 0))
            .unwrap();
        provider
            .tx_ref()
            .put::<tables::StorageChangeSets>(BlockNumberAddress((1, address)), test_storage(2, 0))
            .unwrap();
        provider
            .tx_ref()
            .put::<tables::StorageChangeSets>(
                BlockNumberAddress((2, address)),
                StorageEntry { key: slot1, value: U256::from(10) },
            )
            .unwrap();
        provider
            .tx_ref()
            .put::<tables::StorageChangeSets>(
                BlockNumberAddress((3, address)),
                StorageEntry { key: slot1, value: U256::from(15) },
            )
            .unwrap();

        provider.save_stage_checkpoint(StageId::Finish, StageCheckpoint::new(3)).unwrap();
        reth_trie_db::with_adapter!(provider, |A| seed_tip_trie_tables::<_, A>(&*provider));

        let overlay = empty_overlay();
        let state_trie_provider = OverlayStateProvider::new(
            &*provider,
            overlay,
            provider.cached_storage_settings().is_v2(),
        );
        let actual =
            reth_trie_db::compute_range_trie_changesets(&*provider, &state_trie_provider, 1..=3, 3)
                .unwrap();
        let storage_revert = actual
            .storage_tries_ref()
            .get(&hashed_address)
            .expect("created account storage trie should be deleted by range revert");
        assert!(storage_revert.is_deleted());
        assert!(storage_revert.storage_nodes_ref().is_empty());

        let cache = ChangesetCache::new();
        let overlay_manager = OverlayManager::<reth_ethereum_primitives::EthPrimitives>::default();
        let (partial_state_trie, finish) = database_state_frontiers(&*provider).unwrap();
        let from_cache_api = cache
            .get_or_compute_range(&overlay_manager, &*provider, 1..=3, partial_state_trie, finish)
            .unwrap();
        assert_eq!(*from_cache_api, actual);
        assert_eq!(cache.inner.read().entries.len(), 1);

        let block_changesets = cache
            .get_or_compute(&overlay_manager, &*provider, 2, partial_state_trie, finish)
            .unwrap();
        assert_eq!(*block_changesets, legacy_compute_block_trie_changesets(&*provider, 2));
        assert_eq!(cache.inner.read().entries.len(), 2);
    }

    #[test]
    fn aggregate_range_matches_legacy_per_block_merge_with_storage_wipe() {
        let factory = create_test_provider_factory();
        seed_headers(&factory, 3);

        let provider = factory.provider_rw().unwrap();
        let address = Address::with_last_byte(1);
        let slot1 = B256::from(U256::from(1));
        let slot2 = B256::from(U256::from(2));
        let account1 = test_account(10);
        let account2 = test_account(20);

        provider
            .tx_ref()
            .put::<tables::AccountChangeSets>(1, AccountBeforeTx { address, info: None })
            .unwrap();
        provider
            .tx_ref()
            .put::<tables::AccountChangeSets>(2, AccountBeforeTx { address, info: Some(account1) })
            .unwrap();
        provider
            .tx_ref()
            .put::<tables::AccountChangeSets>(3, AccountBeforeTx { address, info: Some(account2) })
            .unwrap();

        provider
            .tx_ref()
            .put::<tables::StorageChangeSets>(BlockNumberAddress((1, address)), test_storage(1, 0))
            .unwrap();
        provider
            .tx_ref()
            .put::<tables::StorageChangeSets>(BlockNumberAddress((1, address)), test_storage(2, 0))
            .unwrap();
        provider
            .tx_ref()
            .put::<tables::StorageChangeSets>(
                BlockNumberAddress((2, address)),
                StorageEntry { key: slot1, value: U256::from(10) },
            )
            .unwrap();
        provider
            .tx_ref()
            .put::<tables::StorageChangeSets>(
                BlockNumberAddress((3, address)),
                StorageEntry { key: slot1, value: U256::from(15) },
            )
            .unwrap();
        provider
            .tx_ref()
            .put::<tables::StorageChangeSets>(
                BlockNumberAddress((3, address)),
                StorageEntry { key: slot2, value: U256::from(20) },
            )
            .unwrap();

        provider.save_stage_checkpoint(StageId::Finish, StageCheckpoint::new(3)).unwrap();
        reth_trie_db::with_adapter!(provider, |A| seed_tip_trie_tables::<_, A>(&*provider));

        let expected = legacy_compute_range_trie_changesets(&*provider, 2..=3);
        let overlay = empty_overlay();
        let state_trie_provider = OverlayStateProvider::new(
            &*provider,
            overlay,
            provider.cached_storage_settings().is_v2(),
        );
        let actual =
            reth_trie_db::compute_range_trie_changesets(&*provider, &state_trie_provider, 2..=3, 3)
                .unwrap();
        assert_eq!(actual, expected);
    }

    /// Seeds a three-block fixture whose trie is large enough to hold stored branch nodes.
    ///
    /// A handful of accounts is not enough: the account trie only persists nodes once it branches,
    /// so a small fixture makes every changeset empty and any comparison over it vacuous. This
    /// seeds `ACCOUNT_COUNT` accounts at their post-block-3 values, builds the tip trie from them,
    /// and then records per-block reverts for a few of those accounts.
    fn seed_eager_fixture<Provider>(provider: &Provider)
    where
        Provider:
            DBProvider<Tx: DbTxMut> + StageCheckpointWriter + TrieWriter + StorageSettingsCache,
    {
        /// Enough accounts for the hashed keys to diverge in the first nibble, which is what
        /// forces branch nodes to be written to `AccountsTrie`.
        const ACCOUNT_COUNT: u64 = 256;

        let address_at = |i: u64| Address::from_word(keccak256(i.to_be_bytes()));

        // Tip state: every account present with a distinct balance.
        for i in 0..ACCOUNT_COUNT {
            provider
                .tx_ref()
                .put::<tables::HashedAccounts>(keccak256(address_at(i)), test_account(1_000 + i))
                .unwrap();
        }
        reth_trie_db::with_adapter!(provider, |A| seed_tip_trie_tables::<_, A>(provider));

        // Per-block reverts: the value each account held before that block changed it. Reverting
        // blocks 2..=3 therefore has to rewrite the trie nodes covering these two accounts.
        provider
            .tx_ref()
            .put::<tables::AccountChangeSets>(
                1,
                AccountBeforeTx { address: address_at(3), info: Some(test_account(1)) },
            )
            .unwrap();
        provider
            .tx_ref()
            .put::<tables::AccountChangeSets>(
                2,
                AccountBeforeTx { address: address_at(5), info: Some(test_account(2)) },
            )
            .unwrap();
        provider
            .tx_ref()
            .put::<tables::AccountChangeSets>(
                3,
                AccountBeforeTx { address: address_at(7), info: Some(test_account(3)) },
            )
            .unwrap();

        provider.save_stage_checkpoint(StageId::Finish, StageCheckpoint::new(3)).unwrap();
    }

    /// The aggregate DB fallback for the same range, i.e. what a cache miss would return.
    fn db_fallback_range<Provider>(
        provider: &Provider,
        range: RangeInclusive<BlockNumber>,
        db_tip: BlockNumber,
    ) -> TrieUpdatesSorted
    where
        Provider: DBProvider
            + ChangeSetReader
            + StorageChangeSetReader
            + BlockNumReader
            + StorageSettingsCache,
    {
        let state_trie_provider = OverlayStateProvider::new(
            provider,
            empty_overlay(),
            provider.cached_storage_settings().is_v2(),
        );
        reth_trie_db::compute_range_trie_changesets(provider, &state_trie_provider, range, db_tip)
            .unwrap()
    }

    /// The eager producer computes each block's changesets from that block's own trie updates
    /// against a parent-anchored view. Merging those cached per-block entries must reconstruct
    /// exactly what the aggregate DB fallback computes for the same range — anything else would
    /// silently corrupt reorg unwinds, which are the cache's only consumer.
    #[test]
    fn eagerly_cached_changesets_match_db_fallback() {
        let factory = create_test_provider_factory();
        seed_headers(&factory, 3);

        let provider = factory.provider_rw().unwrap();
        seed_eager_fixture(&*provider);

        let expected = db_fallback_range(&*provider, 2..=3, 3);

        // Populate the cache the way the eager producer does: register the pending entry, then
        // resolve it with the changesets computed from the block's own trie updates.
        let cache = ChangesetCache::new();
        for block_number in 2..=3u64 {
            let block_hash = B256::with_last_byte(block_number as u8);
            let changesets = legacy_compute_block_trie_changesets(&*provider, block_number);
            cache.register_pending(block_number, block_hash).resolve(Arc::new(changesets));
        }

        let overlay_manager = OverlayManager::<reth_ethereum_primitives::EthPrimitives>::default();
        let (partial_state_trie, finish) = database_state_frontiers(&*provider).unwrap();
        let actual = cache
            .get_or_compute_range(&overlay_manager, &*provider, 2..=3, partial_state_trie, finish)
            .unwrap();

        assert!(
            !expected.account_nodes_ref().is_empty(),
            "fixture must exercise real trie nodes, otherwise this compares two empty values"
        );
        assert_eq!(actual.as_ref(), &expected);
    }

    /// Same differential check, but over a range that destroys an account holding storage.
    ///
    /// A destroyed account wipes its whole storage trie, which the eager path and the aggregate
    /// fallback reach by different routes: the eager path takes the `is_deleted` branch of
    /// `compute_trie_changesets` off the block's own trie updates, while the fallback rebuilds the
    /// wipe from database reverts. This is the case where the two are most likely to diverge, and a
    /// divergence here corrupts the unwind of any self-destruct block.
    #[test]
    fn eagerly_cached_changesets_match_db_fallback_across_storage_wipe() {
        let factory = create_test_provider_factory();
        seed_headers(&factory, 3);

        let provider = factory.provider_rw().unwrap();
        seed_eager_fixture(&*provider);

        // An account with storage that block 2 brought into existence: reverting 2..=3 therefore
        // has to remove the account and wipe its storage trie.
        let wiped = Address::with_last_byte(0xee);
        let hashed_wiped = keccak256(wiped);
        let slot1 = B256::from(U256::from(1));
        let slot2 = B256::from(U256::from(2));

        provider.tx_ref().put::<tables::HashedAccounts>(hashed_wiped, test_account(500)).unwrap();
        for slot in [slot1, slot2] {
            provider
                .tx_ref()
                .put::<tables::HashedStorages>(
                    hashed_wiped,
                    StorageEntry { key: keccak256(slot), value: U256::from(7) },
                )
                .unwrap();
        }
        reth_trie_db::with_adapter!(provider, |A| seed_tip_trie_tables::<_, A>(&*provider));

        // `info: None` at block 2 means the account did not exist before block 2.
        provider
            .tx_ref()
            .put::<tables::AccountChangeSets>(2, AccountBeforeTx { address: wiped, info: None })
            .unwrap();
        for slot in [slot1, slot2] {
            provider
                .tx_ref()
                .put::<tables::StorageChangeSets>(
                    BlockNumberAddress((2, wiped)),
                    StorageEntry { key: slot, value: U256::ZERO },
                )
                .unwrap();
        }

        let expected = db_fallback_range(&*provider, 2..=3, 3);

        let cache = ChangesetCache::new();
        for block_number in 2..=3u64 {
            let block_hash = B256::with_last_byte(block_number as u8);
            let changesets = legacy_compute_block_trie_changesets(&*provider, block_number);
            cache.register_pending(block_number, block_hash).resolve(Arc::new(changesets));
        }

        let overlay_manager = OverlayManager::<reth_ethereum_primitives::EthPrimitives>::default();
        let (partial_state_trie, finish) = database_state_frontiers(&*provider).unwrap();
        let actual = cache
            .get_or_compute_range(&overlay_manager, &*provider, 2..=3, partial_state_trie, finish)
            .unwrap();

        assert!(
            expected.storage_tries_ref().contains_key(&hashed_wiped),
            "fixture must actually revert the destroyed account's storage trie"
        );
        assert_eq!(actual.as_ref(), &expected);
    }

    /// Seeds `address` at the tip holding `slot_count` storage slots.
    ///
    /// Pass a large `slot_count` when the storage trie needs to persist branch nodes of its own:
    /// a trie with only a few slots stores none, which is what hid the lost wipe marker.
    fn seed_account_with_storage<Provider>(provider: &Provider, address: Address, slot_count: u64)
    where
        Provider: DBProvider<Tx: DbTxMut>,
    {
        let hashed_address = keccak256(address);
        provider.tx_ref().put::<tables::HashedAccounts>(hashed_address, test_account(500)).unwrap();
        for i in 0..slot_count {
            provider
                .tx_ref()
                .put::<tables::HashedStorages>(
                    hashed_address,
                    StorageEntry {
                        key: keccak256(B256::from(U256::from(i))),
                        value: U256::from(i + 1),
                    },
                )
                .unwrap();
        }
    }

    /// Records that `address` and its `slot_count` slots did not exist before `block`, so that
    /// reverting `block` destroys the account and wipes its storage trie.
    fn record_account_created_at<Provider>(
        provider: &Provider,
        block: BlockNumber,
        address: Address,
        slot_count: u64,
    ) where
        Provider: DBProvider<Tx: DbTxMut>,
    {
        provider
            .tx_ref()
            .put::<tables::AccountChangeSets>(block, AccountBeforeTx { address, info: None })
            .unwrap();
        for i in 0..slot_count {
            provider
                .tx_ref()
                .put::<tables::StorageChangeSets>(
                    BlockNumberAddress((block, address)),
                    StorageEntry { key: B256::from(U256::from(i)), value: U256::ZERO },
                )
                .unwrap();
        }
    }

    /// Populates `cache` for `range` exactly the way the eager producer does.
    fn populate_cache_eagerly<Provider>(
        cache: &ChangesetCache,
        provider: &Provider,
        range: RangeInclusive<BlockNumber>,
    ) where
        Provider: DBProvider
            + ChangeSetReader
            + StorageChangeSetReader
            + BlockNumReader
            + StorageSettingsCache,
    {
        for block_number in range {
            let block_hash = B256::with_last_byte(block_number as u8);
            let changesets = legacy_compute_block_trie_changesets(provider, block_number);
            cache.register_pending(block_number, block_hash).resolve(Arc::new(changesets));
        }
    }

    /// Reads `range` back through `cache`.
    fn read_range_through_cache<Provider>(
        cache: &ChangesetCache,
        provider: &Provider,
        range: RangeInclusive<BlockNumber>,
    ) -> Arc<TrieUpdatesSorted>
    where
        Provider: DBProvider
            + ChangeSetReader
            + StorageChangeSetReader
            + PruneCheckpointReader
            + StageCheckpointReader
            + BlockNumReader
            + StorageSettingsCache,
    {
        let overlay_manager = OverlayManager::<reth_ethereum_primitives::EthPrimitives>::default();
        let (partial_state_trie, finish) = database_state_frontiers(provider).unwrap();
        cache
            .get_or_compute_range(&overlay_manager, provider, range, partial_state_trie, finish)
            .unwrap()
    }

    /// The wipe case again, but over a storage trie large enough to hold persisted branch nodes.
    ///
    /// Both paths report a wipe wholesale, as `is_deleted` with no node changesets, however large
    /// the trie was. This pins that down: the deletion marker must survive on the eager side for a
    /// big trie just as it must for the empty one, and neither path may start enumerating nodes.
    #[test]
    fn eagerly_cached_changesets_match_db_fallback_across_wipe_with_persisted_nodes() {
        let factory = create_test_provider_factory();
        seed_headers(&factory, 3);

        let provider = factory.provider_rw().unwrap();
        seed_eager_fixture(&*provider);

        let wiped = Address::with_last_byte(0xee);
        let hashed_wiped = keccak256(wiped);
        seed_account_with_storage(&*provider, wiped, 256);
        reth_trie_db::with_adapter!(provider, |A| seed_tip_trie_tables::<_, A>(&*provider));
        record_account_created_at(&*provider, 2, wiped, 256);

        let expected = db_fallback_range(&*provider, 2..=3, 3);

        let cache = ChangesetCache::new();
        populate_cache_eagerly(&cache, &*provider, 2..=3);
        let actual = read_range_through_cache(&cache, &*provider, 2..=3);

        let wipe = expected
            .storage_tries_ref()
            .get(&hashed_wiped)
            .expect("fixture must revert the destroyed account's storage trie");
        assert!(wipe.is_deleted, "the revert must mark the storage trie deleted");
        assert_eq!(actual.as_ref(), &expected);
    }

    /// An account destroyed and then recreated inside the same range.
    ///
    /// The range revert has to restore the account as it was before the destruction, which means
    /// the older per-block revert must win over the newer one. This is the case where the cache's
    /// `merge_slice` precedence and the fallback's single aggregate computation could disagree.
    #[test]
    fn eagerly_cached_changesets_match_db_fallback_across_wipe_then_recreate() {
        let factory = create_test_provider_factory();
        seed_headers(&factory, 4);

        let provider = factory.provider_rw().unwrap();
        seed_eager_fixture(&*provider);

        let churned = Address::with_last_byte(0xdd);
        let hashed_churned = keccak256(churned);
        seed_account_with_storage(&*provider, churned, 64);
        reth_trie_db::with_adapter!(provider, |A| seed_tip_trie_tables::<_, A>(&*provider));

        // Block 2 destroyed the account: before it, the account held these slot values.
        provider
            .tx_ref()
            .put::<tables::AccountChangeSets>(
                2,
                AccountBeforeTx { address: churned, info: Some(test_account(900)) },
            )
            .unwrap();
        for i in 0..64u64 {
            provider
                .tx_ref()
                .put::<tables::StorageChangeSets>(
                    BlockNumberAddress((2, churned)),
                    StorageEntry { key: B256::from(U256::from(i)), value: U256::from(i + 1_000) },
                )
                .unwrap();
        }
        // Block 3 recreated it, so before block 3 it did not exist.
        record_account_created_at(&*provider, 3, churned, 64);
        provider.save_stage_checkpoint(StageId::Finish, StageCheckpoint::new(4)).unwrap();

        let expected = db_fallback_range(&*provider, 2..=3, 4);

        let cache = ChangesetCache::new();
        populate_cache_eagerly(&cache, &*provider, 2..=3);
        let actual = read_range_through_cache(&cache, &*provider, 2..=3);

        assert!(
            expected.storage_tries_ref().contains_key(&hashed_churned),
            "fixture must revert the churned account's storage trie"
        );
        assert_eq!(actual.as_ref(), &expected);
    }

    /// A range whose oldest block has already been evicted must fall back, not answer from the
    /// partially populated cache.
    ///
    /// This is the retention edge: the cache keeps a bounded window, so a consumer asking across
    /// it will find some blocks cached and some gone. Answering from the surviving subset would
    /// silently return a revert that stops short of the range start.
    #[test]
    fn partially_evicted_range_falls_back_and_still_matches_db() {
        let factory = create_test_provider_factory();
        seed_headers(&factory, 3);

        let provider = factory.provider_rw().unwrap();
        seed_eager_fixture(&*provider);

        let expected = db_fallback_range(&*provider, 2..=3, 3);

        let cache = ChangesetCache::new();
        populate_cache_eagerly(&cache, &*provider, 2..=3);

        // Drop block 2's entry, leaving block 3 cached.
        cache.evict(3);
        assert!(
            cache
                .inner
                .read()
                .get(&ChangesetRangeKey::single(2, B256::with_last_byte(2)))
                .is_none(),
            "block 2 must be evicted for this to test the retention edge"
        );

        let actual = read_range_through_cache(&cache, &*provider, 2..=3);
        assert_eq!(actual.as_ref(), &expected);
    }

    /// A reader that arrives while the producer is still computing must wait for it rather than
    /// run the aggregate DB fallback. Resolving with a value the DB could never produce proves
    /// which path the reader took.
    #[test]
    fn pending_changeset_is_awaited_instead_of_db_fallback() {
        let factory = create_test_provider_factory();
        seed_headers(&factory, 3);

        let provider = factory.provider_rw().unwrap();
        seed_eager_fixture(&*provider);

        let path = Nibbles::from_nibbles([0xa, 0xb]);
        let sentinel_node = BranchNodeCompact::new(0b0100, 0, 0, vec![], None);
        let sentinel = Arc::new(TrieUpdatesSorted::new(
            vec![(path, Some(sentinel_node.clone()))],
            B256Map::default(),
        ));

        let cache = ChangesetCache::new();
        let guard = cache.register_pending(3, B256::with_last_byte(3));

        let reader = std::thread::scope(|scope| {
            let handle = scope.spawn(|| {
                let overlay_manager =
                    OverlayManager::<reth_ethereum_primitives::EthPrimitives>::default();
                let (partial_state_trie, finish) = database_state_frontiers(&*provider).unwrap();
                cache
                    .get_or_compute_range(
                        &overlay_manager,
                        &*provider,
                        3..=3,
                        partial_state_trie,
                        finish,
                    )
                    .unwrap()
            });

            // Give the reader a chance to reach the wait. If it does not get there first the
            // entry is simply a cache hit, so this cannot flake either way.
            std::thread::sleep(std::time::Duration::from_millis(50));
            guard.resolve(Arc::clone(&sentinel));

            handle.join().unwrap()
        });

        assert_eq!(reader.account_nodes_ref(), &[(path, Some(sentinel_node))]);
    }

    /// If the producer dies its guard drops unresolved. Waiters must then be released to the DB
    /// fallback and still get the correct answer, so a failed eager computation degrades to
    /// current behavior instead of hanging or returning something wrong.
    #[test]
    fn cancelled_pending_changeset_falls_back_to_db() {
        let factory = create_test_provider_factory();
        seed_headers(&factory, 3);

        let provider = factory.provider_rw().unwrap();
        seed_eager_fixture(&*provider);

        let expected = db_fallback_range(&*provider, 3..=3, 3);

        let cache = ChangesetCache::new();
        drop(cache.register_pending(3, B256::with_last_byte(3)));

        let overlay_manager = OverlayManager::<reth_ethereum_primitives::EthPrimitives>::default();
        let (partial_state_trie, finish) = database_state_frontiers(&*provider).unwrap();
        let actual = cache
            .get_or_compute_range(&overlay_manager, &*provider, 3..=3, partial_state_trie, finish)
            .unwrap();

        assert_eq!(actual.as_ref(), &expected);
    }

    #[test]
    fn test_insert_and_retrieve_single_entry() {
        let mut cache = ChangesetCacheInner::new();
        let hash = B256::random();
        let changesets = create_test_changesets();

        insert_test_changesets(&mut cache, hash, 100, Arc::clone(&changesets));

        // Should be able to retrieve it
        let retrieved = get_test_changesets(&cache, hash, 100);
        assert!(retrieved.is_some());
        assert_eq!(cache.entries.len(), 1);
    }

    #[test]
    fn test_insert_multiple_entries() {
        let mut cache = ChangesetCacheInner::new();

        // Insert 10 blocks
        let mut hashes = Vec::new();
        for i in 0..10 {
            let hash = B256::random();
            insert_test_changesets(&mut cache, hash, 100 + i, create_test_changesets());
            hashes.push((100 + i, hash));
        }

        // Should be able to retrieve all
        assert_eq!(cache.entries.len(), 10);
        for (block_number, hash) in hashes {
            assert!(get_test_changesets(&cache, hash, block_number).is_some());
        }
    }

    #[test]
    fn test_eviction_when_explicitly_called() {
        let mut cache = ChangesetCacheInner::new();

        // Insert 15 blocks (0-14)
        let mut hashes = Vec::new();
        for i in 0..15 {
            let hash = B256::random();
            insert_test_changesets(&mut cache, hash, i, create_test_changesets());
            hashes.push((i, hash));
        }

        // All blocks should be present (no automatic eviction)
        assert_eq!(cache.entries.len(), 15);

        // Explicitly evict blocks < 4
        cache.evict(4);

        // Blocks 0-3 should be evicted
        assert_eq!(cache.entries.len(), 11); // blocks 4-14 = 11 blocks

        // Verify blocks 0-3 are evicted
        for i in 0..4 {
            assert!(
                get_test_changesets(&cache, hashes[i as usize].1, i).is_none(),
                "Block {} should be evicted",
                i
            );
        }

        // Verify blocks 4-14 are still present
        for i in 4..15 {
            assert!(
                get_test_changesets(&cache, hashes[i as usize].1, i).is_some(),
                "Block {} should be present",
                i
            );
        }
    }

    #[test]
    fn test_eviction_with_persistence_watermark() {
        let mut cache = ChangesetCacheInner::new();

        // Insert blocks 100-165
        let mut hashes = HashMap::new();
        for i in 100..=165 {
            let hash = B256::random();
            insert_test_changesets(&mut cache, hash, i, create_test_changesets());
            hashes.insert(i, hash);
        }

        // All blocks should be present (no automatic eviction)
        assert_eq!(cache.entries.len(), 66);

        // Simulate persistence up to block 164, with 64-block retention window
        // Eviction threshold = 164 - 64 = 100
        cache.evict(100);

        // Blocks 100-165 should remain (66 blocks)
        assert_eq!(cache.entries.len(), 66);

        // Simulate persistence up to block 165
        // Eviction threshold = 165 - 64 = 101
        cache.evict(101);

        // Blocks 101-165 should remain (65 blocks)
        assert_eq!(cache.entries.len(), 65);
        assert!(get_test_changesets(&cache, hashes[&100], 100).is_none());
        assert!(get_test_changesets(&cache, hashes[&101], 101).is_some());
    }

    #[test]
    fn test_out_of_order_inserts_with_explicit_eviction() {
        let mut cache = ChangesetCacheInner::new();

        // Insert blocks in random order
        let hash_10 = B256::random();
        insert_test_changesets(&mut cache, hash_10, 10, create_test_changesets());

        let hash_5 = B256::random();
        insert_test_changesets(&mut cache, hash_5, 5, create_test_changesets());

        let hash_15 = B256::random();
        insert_test_changesets(&mut cache, hash_15, 15, create_test_changesets());

        let hash_3 = B256::random();
        insert_test_changesets(&mut cache, hash_3, 3, create_test_changesets());

        // All blocks should be present (no automatic eviction)
        assert_eq!(cache.entries.len(), 4);

        // Explicitly evict blocks < 5
        cache.evict(5);

        assert!(get_test_changesets(&cache, hash_3, 3).is_none(), "Block 3 should be evicted");
        assert!(get_test_changesets(&cache, hash_5, 5).is_some(), "Block 5 should be present");
        assert!(get_test_changesets(&cache, hash_10, 10).is_some(), "Block 10 should be present");
        assert!(get_test_changesets(&cache, hash_15, 15).is_some(), "Block 15 should be present");
    }

    #[test]
    fn test_multiple_blocks_same_number() {
        let mut cache = ChangesetCacheInner::new();

        // Insert multiple blocks with same number (side chains)
        let hash_1a = B256::random();
        let hash_1b = B256::random();
        insert_test_changesets(&mut cache, hash_1a, 100, create_test_changesets());
        insert_test_changesets(&mut cache, hash_1b, 100, create_test_changesets());

        // Both should be retrievable
        assert!(get_test_changesets(&cache, hash_1a, 100).is_some());
        assert!(get_test_changesets(&cache, hash_1b, 100).is_some());
        assert_eq!(cache.entries.len(), 2);
    }

    #[test]
    fn test_ranges_with_same_numbers_and_different_end_hashes_are_distinct() {
        let mut cache = ChangesetCacheInner::new();
        let path = Nibbles::from_nibbles_unchecked([0x01]);
        let hash_a = B256::with_last_byte(1);
        let hash_b = B256::with_last_byte(2);
        let key_a = ChangesetRangeKey::new(10, 20, hash_a);
        let key_b = ChangesetRangeKey::new(10, 20, hash_b);
        let changesets_a = Arc::new(TrieUpdatesSorted::new(
            vec![(path, Some(BranchNodeCompact::new(0b0001, 0, 0, vec![], None)))],
            B256Map::default(),
        ));
        let changesets_b = Arc::new(TrieUpdatesSorted::new(
            vec![(path, Some(BranchNodeCompact::new(0b0010, 0, 0, vec![], None)))],
            B256Map::default(),
        ));

        cache.insert(key_a, Arc::clone(&changesets_a));
        cache.insert(key_b, Arc::clone(&changesets_b));

        assert_eq!(cache.entries.len(), 2);
        assert_eq!(
            cache.get(&key_a).unwrap().account_nodes_ref(),
            changesets_a.account_nodes_ref()
        );
        assert_eq!(
            cache.get(&key_b).unwrap().account_nodes_ref(),
            changesets_b.account_nodes_ref()
        );

        cache.evict(11);
        assert!(cache.get(&key_a).is_none());
        assert!(cache.get(&key_b).is_none());
    }

    #[test]
    fn test_eviction_removes_all_side_chains() {
        let mut cache = ChangesetCacheInner::new();

        // Insert multiple blocks at same height (side chains)
        let hash_10a = B256::random();
        let hash_10b = B256::random();
        let hash_10c = B256::random();
        insert_test_changesets(&mut cache, hash_10a, 10, create_test_changesets());
        insert_test_changesets(&mut cache, hash_10b, 10, create_test_changesets());
        insert_test_changesets(&mut cache, hash_10c, 10, create_test_changesets());

        let hash_20 = B256::random();
        insert_test_changesets(&mut cache, hash_20, 20, create_test_changesets());

        assert_eq!(cache.entries.len(), 4);

        // Evict blocks < 15 - should remove all three side chains at height 10
        cache.evict(15);

        assert_eq!(cache.entries.len(), 1);
        assert!(get_test_changesets(&cache, hash_10a, 10).is_none());
        assert!(get_test_changesets(&cache, hash_10b, 10).is_none());
        assert!(get_test_changesets(&cache, hash_10c, 10).is_none());
        assert!(get_test_changesets(&cache, hash_20, 20).is_some());
    }
}
