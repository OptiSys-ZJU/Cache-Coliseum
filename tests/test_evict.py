#!/usr/bin/env python3
"""Test path protection, model-based eviction, and guard fallback."""
from cache.trie.trie_algorithms import TrieModelPredictAlgorithm, TrieModelGuard

# ========== Test 1: TrieModelPredictAlgorithm (no model fallback) ==========
alg = TrieModelPredictAlgorithm(max_node_num=5, model=None)

result = alg.access([1, 2, 3])
assert result == (3, 0, 3), f'Expected (3,0,3), got {result}'
result2 = alg.access([1, 2, 4])
assert result2 == (3, 2, 1), f'Expected (3,2,1), got {result2}'
result3 = alg.access([1, 2, 5])
assert alg.cur_node_num <= 5
result4 = alg.access([1, 2, 6])
assert alg.cur_node_num <= 5

# Protected leaves: only leaf on path
protected = alg._get_protected_leaves([1, 2, 6])
assert len(protected) == 1 and all(n.is_leaf() for n in protected)
print('Test 1 (TrieModelPredictAlgorithm): PASSED')

# ========== Test 2: TrieModelGuard (no model → all guarded) ==========
guard = TrieModelGuard(max_node_num=5, model=None, variance_threshold=0.01)

guard.access([1, 2, 3])
guard.access([1, 2, 4])
guard.access([1, 2, 5])  # triggers eviction
guard.access([1, 2, 6])  # triggers eviction

# Without model, all evictions should be guarded (LRU fallback)
assert guard.total_evictions >= 1, f'Expected >=1 evictions, got {guard.total_evictions}'
assert guard.guarded_evictions == guard.total_evictions, \
    f'Without model, all evictions should be guarded: {guard.guarded_evictions}/{guard.total_evictions}'
assert guard.guard_rate == 1.0
print(f'Test 2 (TrieModelGuard no model): PASSED, guard_rate={guard.guard_rate}')

# ========== Test 3: TrieModelGuard incremental candidate update ==========
guard2 = TrieModelGuard(max_node_num=4, model=None)

guard2.access([1, 2, 3])      # 3 nodes
guard2.access([1, 2, 4])      # 4 nodes (full)
guard2.access([1, 2, 5])      # evict 1, insert 1
guard2.access([10, 20, 30])   # evict 3, insert 3 (completely new branch)
assert guard2.cur_node_num <= 4, f'Node count {guard2.cur_node_num} exceeds capacity'
print(f'Test 3 (incremental candidates): PASSED, cur_node_num={guard2.cur_node_num}')

print('\nAll eviction/path-protection tests passed!')
