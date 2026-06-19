"""Regression checks for recent trie-model fixes."""
import os
import sys

import torch
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model.trie_model.model import TrieParrotModel

# Test 1: forward() returns raw eviction logits (not softmax'd)
model = TrieParrotModel(vocab_size=100, node_embed_dim=16, hidden_size=32)
model.eval()

h_state = torch.randn(1, 32)
request_state = torch.randn(1, 32)
leaf_states = [torch.randn(1, 32) for _ in range(5)]
lru_features = [(1.0, 1.0, 1.0, 1.0, 1.0) for _ in leaf_states]

with torch.no_grad():
    logits, reuse_dist = model.forward(
        h_state,
        request_state,
        lru_features,
        candidate_states=leaf_states,
    )

assert logits.shape == (1, 5), f"Expected (1,5), got {logits.shape}"
# Logits should NOT sum to 1 (not softmax'd)
logit_sum = logits.sum().item()
assert abs(logit_sum - 1.0) > 1e-3, f"Logits should be raw, not softmax'd, sum={logit_sum}"
print(f"Test 1 PASSED: forward logits shape={logits.shape}, argmax={logits.squeeze(0).argmax().item()}")

# Test 2: loss() still computes gradients correctly
model.train()
snap = SimpleNamespace()
snap.eviction_steps = [SimpleNamespace(
    leaf_paths=[(1, 2), (3, 4), (5, 6)],
    oracle_target=1,
    microstep_history_paths=((10,), (10, 11), (10, 11, 12)),
    request_history_paths=((1, 2),),
    lru_features=(
        (1.0, 1.0, 1.0, 1.0, 2.0),
        (2.0, 2.0, 2.0, 2.0, 2.0),
        (3.0, 3.0, 3.0, 3.0, 2.0),
    ),
    oracle_distances=[1.0, float("inf"), 2.0],
    num_candidates=3,
)]
losses = model.loss([snap])
assert losses['ranking'].requires_grad, "Ranking loss should require grad"
assert torch.isfinite(losses['ranking']), "Ranking loss should be finite"
assert torch.isfinite(losses['reuse']), "Reuse loss should be finite"
sum(losses.values()).backward()
print(
    f"Test 2 PASSED: ranking={losses['ranking'].item():.4f}, "
    f"reuse={losses['reuse'].item():.4f}, backward OK"
)

# Test 3: TrieModelPredictAlgorithm.__evict__ fallback path
from cache.trie.trie_algorithms import TrieModelPredictAlgorithm, TrieNode
alg = TrieModelPredictAlgorithm(max_node_num=10, model=None)
# Add some nodes
for seq in [[1,2,3], [4,5,6], [7,8,9]]:
    this_node, insert_list = alg.__match__(seq)
    alg.__insert__(this_node, insert_list)

before = alg.cur_node_num
# Call __evict__ directly (the previously empty pass method)
alg.__evict__(1, alg.root_node)
after = alg.cur_node_num
assert after == before - 1, f"Expected {before-1} nodes, got {after}"
print(f"Test 3 PASSED: __evict__ fallback works ({before} -> {after} nodes)")

# Test 4: SequenceTrieCache uses isinstance dispatch
from cache.trie.trie_cache import SequenceTrieCache
from cache.trie.trie_algorithms import TrieLRUAlgorithm
cache = SequenceTrieCache(max_node_num=20, evict_type=TrieLRUAlgorithm)
cache.access([1, 2, 3])
cache.access([4, 5, 6])
total, hit, miss = cache.stat_info
assert total > 0, "Should have processed some accesses"
print(f"Test 4 PASSED: SequenceTrieCache isinstance dispatch, total={total}")

# Test 5: stat() returns (total, hit, miss, rate) consistently
stat = cache.stat()
assert stat[0] == total, f"stat()[0] should be total={total}, got {stat[0]}"
assert stat[1] == hit, f"stat()[1] should be hit={hit}, got {stat[1]}"
assert stat[2] == miss, f"stat()[2] should be miss={miss}, got {stat[2]}"
print(f"Test 5 PASSED: stat() returns consistent (total, hit, miss, rate) = {stat}")

print("\n=== ALL REGRESSION TESTS PASSED ===")
