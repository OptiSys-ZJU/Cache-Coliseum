from collections import defaultdict
from typing import List, Tuple, Type

from cache.cache import BaseCache
from cache.hash import HashFunction, OneHashFunction
from cache.trie.trie_algorithms import TrieEvictAlgorithm, TrieLRUAlgorithm, TrieRandAlgorithm
from utils.aligner import Aligner, ListAligner

class TrieCache(BaseCache):
    dummy_pc = 0
    def __init__(self, trace_path, aligner_type: Type[Aligner], hash_type:Type[HashFunction], evict_type: Type[TrieEvictAlgorithm], cache_line_size, cache_capacity, associativity):
        self.trace_path = trace_path
        self._trace_path = trace_path

        self.stat_info = [0, 0, 0] # hit, miss, count

        num_cache_lines = cache_capacity // cache_line_size
        num_sets = num_cache_lines // associativity
        if (cache_capacity % cache_line_size != 0 or num_cache_lines % associativity != 0):
            raise ValueError(
                ("Cache capacity ({}) must be an even multiple of "
                "cache_line_size ({}) and associativity ({})").format(
                    cache_capacity, cache_line_size, associativity))
        if num_sets == 0:
            raise ValueError(
                ("Cache capacity ({}) is not great enough for {} cache lines per set "
                "and cache lines of size {}").format(cache_capacity, associativity, cache_line_size))

        assert num_sets == 1

        assert aligner_type == ListAligner
        self._aligner = aligner_type(cache_line_size)
        assert hash_type == OneHashFunction
        self.hash_func = hash_type(num_sets)
        self.evict_algs = []
        for _ in range(num_sets):
            evict_alg = evict_type(associativity)  
            self.evict_algs.append(evict_alg)

    def pretty_print(self):
        for i, evict_alg in enumerate(self.evict_algs):
            print('---------------------------')
            print(f'Tree [{i}]')
            evict_alg.pretty_print()

    def access(self, pc, address: List):
        aligned_address = self._aligner.get_aligned_addr(address)
        stat = self.evict_algs[self.hash_func.get_bucket_index(aligned_address, TrieCache.dummy_pc)].access(TrieCache.dummy_pc, aligned_address)
        self.stat_info = [x + y for x, y in zip(self.stat_info, stat)]

if __name__ == "__main__":
    # tree = TrieCache('', ListAligner, OneHashFunction, TrieLRUAlgorithm, 1, 5, 5)
    tree = TrieCache('', ListAligner, OneHashFunction, TrieRandAlgorithm, 1, 5, 5)
    tree.access(TrieCache.dummy_pc, [1,2,3,4])
    tree.pretty_print()
    tree.access(TrieCache.dummy_pc, [1,2,3,5])
    tree.pretty_print()
    tree.access(TrieCache.dummy_pc, [1,2,3,4])
    tree.access(TrieCache.dummy_pc, [1,2,3,4])
    tree.access(TrieCache.dummy_pc, [1,2,3,4])
    tree.access(TrieCache.dummy_pc, [1,2,3,4])
    tree.access(TrieCache.dummy_pc, [3])
    tree.pretty_print()
