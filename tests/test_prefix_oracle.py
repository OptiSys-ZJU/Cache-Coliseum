#!/usr/bin/env python3
"""Tests for trie path recovery and prefix future oracle."""

from functools import partial
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cache.trie.oracle import PrefixFutureOracle
from cache.trie.trie_algorithms import (
    TrieLRUAlgorithm,
    TrieNode,
    TrieOracleAlgorithm,
)
from cache.trie.trie_cache import SequenceTrieCache


sequences = [
    [1, 2, 3],
    [1, 2, 4],
    [9],
    [1, 2, 3],
]

oracle = PrefixFutureOracle(sequences)
assert oracle.next_request_index((1,)) == 0
assert oracle.next_request_index((1, 2, 3)) == 0
assert oracle.reuse_distance((1,), 0) == 0
assert oracle.reuse_distance((1,), 0, include_current=False) == 1
oracle.consume_current(sequences[0], 0)
assert oracle.next_request_index((1,)) == 1
assert oracle.next_request_index((1, 2, 3)) == 3
assert oracle.reuse_distance((1, 2, 3), 0) == 3
assert oracle.next_request_index((7,)) == float("inf")


alg = TrieLRUAlgorithm(max_node_num=10)
alg.access(None, [1, 2, 3])
leaf = alg.__leaves__()[0]
assert TrieNode.get_path_tuple_from_node(leaf) == (1, 2, 3)


long_cache = SequenceTrieCache(max_node_num=2, evict_type=TrieLRUAlgorithm)
assert long_cache.access([1, 2, 3]) == (3, 0, 3)
assert long_cache.alg.cur_node_num == 2
assert long_cache.access([1, 2, 4]) == (3, 2, 1)


oracle_cache = SequenceTrieCache(
    max_node_num=3,
    evict_type=partial(
        TrieOracleAlgorithm,
        future_oracle=PrefixFutureOracle(sequences, max_prefix_len=3),
    ),
)
for seq in sequences:
    oracle_cache.access(seq)
kv_stats = oracle_cache.kv_stat()
assert kv_stats["requests"] == len(sequences)
assert kv_stats["total_blocks"] == sum(len(seq) for seq in sequences)

print("Prefix oracle tests passed")
