from cache.evict_algorithms import EvictAlgorithm, OracleAlgorithm
from cache.hash import HashFunction
from data_trace.data_trace import OracleDataTrace
from utils.aligner import Aligner
from typing import Type, Callable
from functools import partial

import numpy as np
import tqdm

class Cache:
    def __init__(self, trace_path, aligner_type: Type[Aligner], evict_type: Type[EvictAlgorithm], hash_type:Type[HashFunction], cache_line_size, cache_capacity, associativity=16):
        def is_pow_of_two(x):
            return (x & (x - 1)) == 0
        
        self.evict_algs = None
        self.hits = 0
        self.miss = 0
        self.counts = 0
        self._trace_path = trace_path
        self._aligner = aligner_type(cache_line_size)

        num_cache_lines = cache_capacity // cache_line_size
        num_sets = num_cache_lines // associativity
        if (cache_capacity % cache_line_size != 0 or num_cache_lines % associativity != 0):
            raise ValueError(
                ("Cache capacity ({}) must be an even multiple of "
                "cache_line_size ({}) and associativity ({})").format(
                    cache_capacity, cache_line_size, associativity))
        if not is_pow_of_two(num_sets):
            raise ValueError("Number of cache sets ({}) must be a power of two.".format(num_sets))
        if num_sets == 0:
            raise ValueError(
                ("Cache capacity ({}) is not great enough for {} cache lines per set "
                "and cache lines of size {}").format(cache_capacity, associativity, cache_line_size))
        
        self.hash_func = hash_type(num_sets)

        self.evict_algs = []
        oracle = False
        for _ in range(num_sets):
            evict_alg = evict_type(associativity)
            if hasattr(evict_alg, 'oracle_access'):
                oracle = True
            self.evict_algs.append(evict_alg)

        if oracle:
            with OracleDataTrace(trace_path, self._aligner) as sim_trace:
                # with tqdm.tqdm(desc="Oracle cache on MemoryTrace") as pbar:
                while not sim_trace.done():
                    _, address = sim_trace.next()
                    aligned_address = self._aligner.get_aligned_addr(address)
                    self.evict_algs[self.hash_func.get_bucket_index(aligned_address)].oracle_access(aligned_address, sim_trace.next_access_time_by_aligned_address(aligned_address))
                    # pbar.update(1)

    def access(self, address):
        aligned_address = self._aligner.get_aligned_addr(address)
        hit = self.evict_algs[self.hash_func.get_bucket_index(aligned_address)].access(aligned_address)
        
        if hit:
            self.hits += 1
        else:
            self.miss += 1
        self.counts += 1
    
    def stat(self):
        return (self.hits, self.miss, self.counts, round(self.hits / self.counts, 4))