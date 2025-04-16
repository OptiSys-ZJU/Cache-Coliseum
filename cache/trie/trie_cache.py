from functools import partial
from typing import List, Type

import tqdm

from cache.cache import BaseCache
from cache.evict.evictor import ReuseDistanceEvictor
from cache.evict.predictor import OracleReuseDistancePredictor
from cache.hash import HashFunction, OneHashFunction
from cache.trie.trie_algorithms import TrieEvictAlgorithm, TrieGuard, TrieLRUAlgorithm, TriePredictAlgorithm, TrieRandAlgorithm
from data_trace.trie_data_trace import OracleTrieDataTrace, TrieDataTrace
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
        ###################################################################
        oracle = False
        for _ in range(num_sets):
            evict_alg = evict_type(associativity)  
            if hasattr(evict_alg, 'oracle_access'):
                oracle = True
            self.evict_algs.append(evict_alg)
        if oracle:
            self.__handle_oracle(trace_path)
    
    def pretty_stat(self):
        total, hit, miss = self.stat_info
        print(f'[Total/Hit/Miss]: [{total}/{hit}/{miss}]')
        if total > 0:
            hit_rate = hit / total
            print(f'[Hit Rate]: {hit_rate:.4f}')
        else:
            print('[Hit Rate]: N/A (Total is 0)')


    def pretty_print(self):
        for i, evict_alg in enumerate(self.evict_algs):
            print('---------------------------')
            print(f'Tree [{i}]')
            evict_alg.pretty_print()
        self.pretty_stat()

    def __handle_oracle(self, trace_path):
        with OracleTrieDataTrace(trace_path, self._aligner, self.hash_func, scale_times=1, offset=1) as sim_trace:
            while not sim_trace.done():
                pc, address = sim_trace.next()
                aligned_address = self._aligner.get_aligned_addr(address)
                self.evict_algs[self.hash_func.get_bucket_index(aligned_address, TrieCache.dummy_pc)].oracle_access(TrieCache.dummy_pc, aligned_address, sim_trace.next_bucket_access_time_by_address(address))

    def access(self, pc, address: List):
        aligned_address = self._aligner.get_aligned_addr(address)
        stat = self.evict_algs[self.hash_func.get_bucket_index(aligned_address, TrieCache.dummy_pc)].access(TrieCache.dummy_pc, aligned_address)
        self.stat_info = [x + y for x, y in zip(self.stat_info, stat)]

if __name__ == "__main__":
    file_path = 'traces/oass1_val.csv'
    # size = 3226
    size = 4096
    reuse_dis_noise_sigma = 10

    relax_times = 0

    # file_path = 'traces/oass1_train.csv'
    # size = 12129

    print('--------------')
    print('LRU')
    alg = TrieLRUAlgorithm
    cache = TrieCache(file_path, ListAligner, OneHashFunction, alg, 1, size, size)
    with TrieDataTrace(file_path) as trace:
        with tqdm.tqdm(desc="Producing cache on MemoryTrace") as pbar:
            while not trace.done():
                pc, address = trace.next()
                cache.access(pc, address)
                pbar.update(1)
    cache.pretty_stat()

    print('--------------')
    print('RAND')
    alg = TrieRandAlgorithm
    cache = TrieCache(file_path, ListAligner, OneHashFunction, alg, 1, size, size)
    with TrieDataTrace(file_path) as trace:
        with tqdm.tqdm(desc="Producing cache on MemoryTrace") as pbar:
            while not trace.done():
                pc, address = trace.next()
                cache.access(pc, address)
                pbar.update(1)
    cache.pretty_stat()

    print('--------------')
    print('Belady')
    alg = partial(TriePredictAlgorithm, evictor_type=ReuseDistanceEvictor, predictor_type=partial(OracleReuseDistancePredictor, reuse_dis_noise_sigma=reuse_dis_noise_sigma, lognormal=True))
    cache = TrieCache(file_path, ListAligner, OneHashFunction, alg, 1, size, size)
    with TrieDataTrace(file_path) as trace:
        with tqdm.tqdm(desc="Producing cache on MemoryTrace") as pbar:
            while not trace.done():
                pc, address = trace.next()
                cache.access(pc, address)
                pbar.update(1)
    cache.pretty_stat()

    print('--------------')
    print('Guard')
    alg = partial(TrieGuard, evictor_type=ReuseDistanceEvictor, predictor_type=partial(OracleReuseDistancePredictor, reuse_dis_noise_sigma=reuse_dis_noise_sigma, lognormal=True), follow_if_guarded=False, relax_times=relax_times, relax_prob=0)
    cache = TrieCache(file_path, ListAligner, OneHashFunction, alg, 1, size, size)
    with TrieDataTrace(file_path) as trace:
        with tqdm.tqdm(desc="Producing cache on MemoryTrace") as pbar:
            while not trace.done():
                pc, address = trace.next()
                cache.access(pc, address)
                # cache.pretty_print()
                pbar.update(1)
    cache.pretty_stat()
