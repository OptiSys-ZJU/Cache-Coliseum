from cache.evict import EvictAlgorithm
from cache.hash import HashFunction
from data_trace.data_trace import OracleDataTrace
from utils.aligner import Aligner
from typing import Type
from types import SimpleNamespace

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
        ###################################################################
        oracle = False
        for _ in range(num_sets):
            evict_alg = evict_type(associativity)
            if hasattr(evict_alg, 'oracle_access'):
                oracle = True
            self.evict_algs.append(evict_alg)
        if oracle:
            self.__handle_oracle(trace_path)

    def __handle_oracle(self, trace_path):
        with OracleDataTrace(trace_path, self._aligner) as sim_trace:
            # with tqdm.tqdm(desc="Oracle cache on MemoryTrace") as pbar:
            while not sim_trace.done():
                pc, address = sim_trace.next()
                aligned_address = self._aligner.get_aligned_addr(address)
                self.evict_algs[self.hash_func.get_bucket_index(aligned_address)].oracle_access(pc, aligned_address, sim_trace.next_access_time_by_aligned_address(aligned_address))
                # pbar.update(1)

    def access(self, pc, address):
        aligned_address = self._aligner.get_aligned_addr(address)
        hit = self.evict_algs[self.hash_func.get_bucket_index(aligned_address)].access(pc, aligned_address)
        
        if hit:
            self.hits += 1
        else:
            self.miss += 1
        self.counts += 1
    
    def stat(self):
        return (self.hits, self.miss, self.counts, round(self.hits / self.counts, 4))



class TrainingCache(Cache):
    def __init__(self, trace_path, aligner_type, evict_type, hash_type, cache_line_size, cache_capacity, associativity=16):
        super().__init__(trace_path, aligner_type, evict_type, hash_type, cache_line_size, cache_capacity, associativity)

    def reset(self, model_prob):
        for alg in self.evict_algs:
            alg.reset([1-model_prob, model_prob])

    def snapshot(self, pc, address):
        snapshot = SimpleNamespace()
        snapshot.pc = pc
        snapshot.address = address

        aligned_address = self._aligner.get_aligned_addr(address)
        idx = self.hash_func.get_bucket_index(aligned_address)
        alg = self.evict_algs[idx]

        cache_lines, scores = alg.snapshot()
        snapshot.cache_lines = cache_lines
        snapshot.cache_line_scores = scores
        hit = alg.access(pc, aligned_address)
        return snapshot, hit