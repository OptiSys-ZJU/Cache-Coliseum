from cache.evict_algorithms import BeladyEvictAlgorithm, EvictAlgorithm, OracleEvictAlgorithm, OracleFollowBinaryPredictionEvictAlgorithm
from cache.hash import HashFunction
from memory_trace.memtrace import MemoryTrace
from utils.aligner import Aligner
from typing import Type, Callable
from functools import partial

import numpy as np
import tqdm

class Cache:
    def __init__(self, trace_path, aligner: Aligner, evict_type: Type[EvictAlgorithm], hash_type:Type[HashFunction], cache_capacity, associativity=16):
        def is_pow_of_two(x):
            return (x & (x - 1)) == 0
        
        self.evict_algs = None
        self.hits = 0
        self.miss = 0
        self.counts = 0
        self._trace_path = trace_path
        self._aligner = aligner
        align_size = self._aligner.get_align_size()

        num_cache_lines = cache_capacity // align_size
        num_sets = num_cache_lines // associativity
        if (cache_capacity % align_size != 0 or num_cache_lines % associativity != 0):
            raise ValueError(
                ("Cache capacity ({}) must be an even multiple of "
                "cache_line_size ({}) and associativity ({})").format(
                    cache_capacity, align_size, associativity))

        if not is_pow_of_two(num_sets):
            raise ValueError("Number of cache sets ({}) must be a power of two.".format(num_sets))

        if num_sets == 0:
            raise ValueError(
                ("Cache capacity ({}) is not great enough for {} cache lines per set "
                "and cache lines of size {}").format(cache_capacity, associativity, align_size))
        
        self.hash_func = hash_type(num_sets)
        with MemoryTrace(self._trace_path, aligner) as trace:
            if evict_type == OracleFollowBinaryPredictionEvictAlgorithm:
                with MemoryTrace(self._trace_path, aligner) as sim_trace:
                    evict_cls = partial(evict_type, next_access_func=sim_trace.next_access_time_by_aligned_address)
                    self.evict_algs = [evict_cls(associativity) for i in range(num_sets)]
                    with tqdm.tqdm(desc="Simulating cache on MemoryTrace") as pbar:
                        while not sim_trace.done():
                            _, address = sim_trace.next()
                            self.sim_access(address)
                            pbar.update(1)
            elif evict_type == BeladyEvictAlgorithm:
                evict_cls = partial(evict_type, next_access_func=trace.next_access_time_by_aligned_address)
            else:
                evict_cls = evict_type
        
            if self.evict_algs is None:
                self.evict_algs = [evict_cls(associativity) for i in range(num_sets)]

            with tqdm.tqdm(desc="Producing cache on MemoryTrace") as pbar:
                while not trace.done():
                    pc, address = trace.next()
                    self.access(address)
                    pbar.update(1)
        

    def sim_access(self, address):
        aligned_address = self._aligner.get_aligned_addr(address)
        self.evict_algs[self.hash_func.get_bucket_index(aligned_address)].prepare_pred(aligned_address)

    def access(self, address):
        aligned_address = self._aligner.get_aligned_addr(address)
        hit = self.evict_algs[self.hash_func.get_bucket_index(aligned_address)].access(aligned_address)
        
        if hit:
            self.hits += 1
        else:
            self.miss += 1
        self.counts += 1
    
    def stat(self):
        print("miss count:", self.miss, 'counts:', self.counts, 'hits:', self.hits)
        print('hit rate:', self.hits / self.counts)