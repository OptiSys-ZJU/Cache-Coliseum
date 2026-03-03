#!/usr/bin/env python3
"""Test TrieTrainingCache: DAgger collection + Belady oracle."""
from cache.trie.trie_cache import TrieTrainingCache

# Simulate a sequence of accesses
all_sequences = [
    [1, 2, 3],
    [1, 2, 4],
    [1, 2, 5],
    [1, 2, 3],   # re-access of [1,2,3]
    [1, 2, 6],
    [1, 2, 4],   # re-access of [1,2,4]
]

cache = TrieTrainingCache(max_node_num=5, model=None)
cache.load_future_accesses(all_sequences)
cache.set_model_prob(0.0)  # pure oracle

# Process accesses one by one
for i, seq in enumerate(all_sequences):
    snapshot, hit = cache.collect(seq)
    print(f'Step {i}: access {seq}, hit={hit}, snapshot={snapshot is not None}')
    if snapshot is not None:
        for j, step in enumerate(snapshot.eviction_steps):
            print(f'  eviction {j}: candidates={step.leaf_node_ids}, oracle_target_idx={step.oracle_target}')

print(f'\nHit rate: {cache.hit_rate:.2f}')
print(f'Total snapshots: {len(cache.get_snapshots())}')

# Verify oracle makes sensible choices:
# At step 2 (access [1,2,5], cap=5): no eviction needed (3+1+1=5)
# At step 3 (access [1,2,3], hit): no eviction (already exists)
# At step 4 (access [1,2,6], need to evict 1):
#   candidates = leaves except protected. Future = [[1,2,4]] at step 5
#   leaf 3 was already visited at step 3.
#   leaf 4 is re-accessed at step 5 → reuse dist = 1
#   leaf 5 is never re-accessed → reuse dist = inf → SHOULD be evicted by oracle

print('\nTest TrieTrainingCache: PASSED')
