#!/usr/bin/env python3
"""Test SequenceTrieCache with different eviction algorithms."""
from cache.trie.trie_cache import SequenceTrieCache
from cache.trie.trie_algorithms import (
    TrieLRUAlgorithm, TrieRandAlgorithm, 
    TrieModelPredictAlgorithm, TrieModelGuard,
)

sequences = [
    [1, 2, 3],
    [1, 2, 4],
    [1, 2, 5],
    [1, 2, 3],  # hit
    [1, 2, 6],
    [1, 2, 4],  # hit
    [10, 20, 30],
    [10, 20, 31],
]

# Test 1: LRU
print('=== LRU ===')
cache = SequenceTrieCache(max_node_num=8, evict_type=TrieLRUAlgorithm)
for seq in sequences:
    cache.access(seq)
cache.pretty_stat()
t, h, m, rate = cache.stat()
print(f'  stat: total={t}, hits={h}, miss={m}, rate={rate}')

# Test 2: Random
print('\n=== Random ===')
cache = SequenceTrieCache(max_node_num=8, evict_type=TrieRandAlgorithm)
for seq in sequences:
    cache.access(seq)
cache.pretty_stat()
t, h, m, rate = cache.stat()
assert t == 24 and h + m == t

# Test 3: Model-based (no model fallback)
print('\n=== ModelPredict (no model) ===')
cache = SequenceTrieCache(max_node_num=8, evict_type=TrieModelPredictAlgorithm)
for seq in sequences:
    cache.access(seq)
cache.pretty_stat()
t, h, m, rate = cache.stat()
assert t == 24 and h + m == t

# Test 4: ModelGuard (no model)
print('\n=== ModelGuard (no model) ===')
cache = SequenceTrieCache(max_node_num=8, evict_type=TrieModelGuard, variance_threshold=0.01)
for seq in sequences:
    cache.access(seq)
cache.pretty_stat()
t, h, m, rate = cache.stat()
assert t == 24 and h + m == t

print('\nSequenceTrieCache verification passed!')
