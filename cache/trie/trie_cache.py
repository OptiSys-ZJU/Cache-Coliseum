from functools import partial
from types import SimpleNamespace
from typing import List, Type, Optional
import random

try:
    import torch
except ModuleNotFoundError:
    torch = None
import tqdm

try:
    from cache.cache import BaseCache
except ModuleNotFoundError:
    class BaseCache:
        pass
from cache.evict.evictor import ReuseDistanceEvictor
from cache.evict.predictor import OracleReuseDistancePredictor
from cache.hash import HashFunction, OneHashFunction
from cache.trie.trie_algorithms import (
    TrieEvictAlgorithm, TrieGuard, TrieLRUAlgorithm, 
    TrieModelPredictAlgorithm, TrieModelGuard, TrieNode, 
    TrieOracleAlgorithm, TriePredictAlgorithm, TrieRandAlgorithm,
)
from cache.trie.oracle import PrefixFutureOracle
try:
    from data_trace.trie_data_trace import OracleTrieDataTrace, TrieDataTrace
except ModuleNotFoundError:
    OracleTrieDataTrace = None
    TrieDataTrace = None
from utils.aligner import Aligner, ListAligner

class TrieCache(BaseCache):
    dummy_pc = 0
    def __init__(self, trace_path, aligner_type: Type[Aligner], hash_type:Type[HashFunction], evict_type: Type[TrieEvictAlgorithm], cache_line_size, cache_capacity, associativity):
        self.trace_path = trace_path
        self._trace_path = trace_path

        self.stat_info = [0, 0, 0] # total, hit, miss

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
        if OracleTrieDataTrace is None:
            raise ImportError("OracleTrieDataTrace dependencies are not available")
        with OracleTrieDataTrace(trace_path, self._aligner, self.hash_func, scale_times=1, offset=1) as sim_trace:
            while not sim_trace.done():
                pc, address = sim_trace.next()
                aligned_address = self._aligner.get_aligned_addr(address)
                self.evict_algs[self.hash_func.get_bucket_index(aligned_address, TrieCache.dummy_pc)].oracle_access(TrieCache.dummy_pc, aligned_address, sim_trace.next_bucket_access_time_by_address(address))

    def access(self, pc, address: List):
        aligned_address = self._aligner.get_aligned_addr(address)
        stat = self.evict_algs[self.hash_func.get_bucket_index(aligned_address, TrieCache.dummy_pc)].access(TrieCache.dummy_pc, aligned_address)
        self.stat_info = [x + y for x, y in zip(self.stat_info, stat)]


#############################################
# Task 3.6: TrieTrainingCache for DAgger training
#############################################

class TrieTrainingCache:
    """
    Training cache for TrieParrotModel using DAgger (Dataset Aggregation).
    
    Wraps a TrieModelPredictAlgorithm and collects request-level cache-state
    snapshots with Belady oracle labels before each request is applied.
    
    Usage:
        cache = TrieTrainingCache(max_node_num=1024, model=my_model)
        cache.load_future_accesses(all_sequences)  # for oracle reuse distance
        cache.set_model_prob(0.0)  # pure oracle at start
        
        for seq in sequences:
            snapshot, hit = cache.collect(seq)
            # snapshot.eviction_steps contains request-level training steps.
    """
    
    def __init__(self, max_node_num: int, model=None):
        """
        Args:
            max_node_num: Maximum trie capacity
            model: TrieParrotModel (can be None, set later via set_model)
        """
        self.max_node_num = max_node_num
        self.model = model
        
        # Internal trie algorithm
        self.alg = TrieModelPredictAlgorithm(max_node_num, model)
        
        # DAgger mixing
        self.model_prob = 0.0  # fraction of evictions using model policy
        
        # Oracle: precomputed future access list
        self._future_accesses: Optional[List[List[int]]] = None
        self._future_oracle: Optional[PrefixFutureOracle] = None
        self._current_step = 0
        
        # Collected training snapshots
        self.snapshots: List[SimpleNamespace] = []
        
        # Statistics
        self.total_count = 0
        self.hit_count = 0
    
    def set_model(self, model):
        """Set or replace the model."""
        self.model = model
        self.alg.set_model(model)
    
    def set_model_prob(self, prob: float):
        """Set DAgger model probability (0 = pure oracle, 1 = pure model)."""
        self.model_prob = prob
    
    def load_future_accesses(self, sequences: List[List[int]]):
        """
        Load all future access sequences for Belady oracle computation.
        Must be called before collect() to enable oracle labeling.
        """
        self._future_accesses = sequences
        self._future_oracle = PrefixFutureOracle(
            sequences,
            max_prefix_len=self.max_node_num,
        )
        self._current_step = 0
        self.alg._reset_history()
    
    def _reuse_distance(self, path: tuple, include_current: bool = False) -> float:
        """
        Belady oracle: how many steps until a future access matches this leaf's path.
        
        A leaf's path matches a future sequence if the path is a prefix of that sequence.
        Returns float('inf') if this path is never re-accessed.
        """
        if self._future_accesses is None:
            return float('inf')

        if self._future_oracle is not None:
            return self._future_oracle.reuse_distance(
                path,
                self._current_step,
                include_current=include_current,
            )
        
        path_list = list(path)
        path_len = len(path_list)
        start = self._current_step if include_current else self._current_step + 1
        for offset in range(start, len(self._future_accesses)):
            future_seq = self._future_accesses[offset]
            if len(future_seq) >= path_len and future_seq[:path_len] == path_list:
                return offset - self._current_step
        return float('inf')
    
    def _oracle_target(self, candidates: List[TrieNode]) -> int:
        """
        Belady's optimal: return index of candidate with max reuse distance 
        (the one that won't be needed for the longest time).
        """
        distances = self._oracle_distances(candidates)
        return max(range(len(distances)), key=lambda idx: distances[idx])

    def _oracle_distances(
        self,
        candidates: List[TrieNode],
        include_current: bool = False,
    ) -> List[float]:
        """Return request-clock future reuse distance for each candidate path."""
        return [
            self._reuse_distance(
                TrieNode.get_path_tuple_from_node(leaf),
                include_current=include_current,
            )
            for leaf in candidates
        ]

    def _oracle_target_from_distances(self, distances: List[float]) -> int:
        """Belady target index from a precomputed distance vector."""
        if not distances:
            raise ValueError("Cannot choose oracle target from an empty candidate list")
        return max(range(len(distances)), key=lambda idx: distances[idx])

    def _request_state_snapshot(
        self,
        cache_sequence: List[int],
        pre_request_history_paths,
    ) -> Optional[SimpleNamespace]:
        """
        Build a request-level cache-state snapshot before mutating the trie.

        The field name eviction_steps is kept for compatibility with the
        existing loss/training pipeline; entries are generic training steps.
        """
        candidates = [
            leaf for leaf in self.alg.__leaves__()
            if leaf != self.alg.root_node
            and self.alg.__is_live_leaf__(leaf)
        ]
        if not candidates:
            return None

        oracle_distances = self._oracle_distances(candidates, include_current=True)
        step_data = SimpleNamespace()
        step_data.step_kind = "request_state"
        step_data.leaf_node_ids = [c.node_id for c in candidates]
        step_data.leaf_paths = [
            TrieNode.get_node_id_path(c) for c in candidates
        ]
        step_data.current_path = tuple(cache_sequence)
        step_data.step_index = -1
        step_data.history_paths = tuple(pre_request_history_paths)
        step_data.oracle_distances = oracle_distances
        step_data.oracle_target = self._oracle_target_from_distances(oracle_distances)
        step_data.num_candidates = len(candidates)

        snapshot = SimpleNamespace()
        snapshot.sequence = tuple(cache_sequence)
        snapshot.eviction_steps = [step_data]
        return snapshot
    
    def collect(self, sequence: List[int]):
        """
        Collect a pre-request cache-state snapshot, then process one access.
        
        Returns:
            (snapshot, hit): 
                snapshot: request-level training snapshot (None if no candidates)
                hit: bool, whether the entire sequence was a cache hit
        """
        cache_sequence = sequence[:self.max_node_num]
        pre_request_history_paths = tuple(self.alg.history_path_window)

        snapshot = self._request_state_snapshot(
            cache_sequence,
            pre_request_history_paths,
        )

        if self._future_oracle is not None:
            self._future_oracle.consume_current(cache_sequence, self._current_step)
        hit_nodes = 0
        this_node = self.alg.root_node

        current_prefix = []
        for step_index, node_id in enumerate(cache_sequence):
            current_prefix.append(node_id)
            if node_id in this_node.children:
                this_node = this_node.children[node_id]
                self.alg.__visit_node__(this_node)
                hit_nodes += 1
            else:
                evict_num = self.alg.cur_node_num + 1 - self.alg.max_node_num
                if evict_num > 0:
                    self._evict_and_collect(
                        evict_num,
                        this_node,
                        current_prefix,
                        step_index,
                    )

                new_node = TrieNode()
                new_node.key = node_id
                new_node.node_id = node_id
                new_node.parent = this_node
                self.alg.__add_node__(new_node)
                self.alg.__mark_as_non_leaf__(this_node)
                this_node.children[node_id] = new_node
                this_node = new_node
                self.alg.cur_node_num += 1
                self.alg.__mark_as_leaf__(this_node)

        self.alg._record_history_leaf(this_node)
        hit = hit_nodes == len(sequence)
        if hit and len(cache_sequence) == len(sequence):
            self.hit_count += 1
        self.total_count += 1

        if snapshot is not None:
            self.snapshots.append(snapshot)

        self.alg.timestamp += 1
        self._current_step += 1
        
        return snapshot, hit

    def _evict_and_collect(
        self,
        evict_num: int,
        this_node: TrieNode,
        current_path: List[int],
        step_index: int,
    ) -> SimpleNamespace:
        """
        Perform eviction with DAgger mixing.

        The returned prefix-level eviction-decision snapshot is for direct
        diagnostics/tests only. collect() trains on pre-request cache-state
        snapshots and does not buffer this helper's micro-steps.
        
        For each eviction:
        1. Compute oracle target (Belady's)
        2. Record training sample
        3. Use DAgger to decide actual eviction (oracle or model)
        """
        protected_leaves = self.alg._get_protected_leaves(current_path)
        candidates = [
            leaf for leaf in self.alg.__leaves__()
            if leaf not in protected_leaves
            and leaf != self.alg.root_node
            and leaf != this_node
            and self.alg.__is_live_leaf__(leaf)
        ]
        
        # Collect one snapshot per eviction batch (all evictions for this access)
        snapshot = SimpleNamespace()
        snapshot.sequence = tuple(current_path)  # metadata only
        snapshot.eviction_steps = []
        
        for _ in range(evict_num):
            candidates = [
                leaf for leaf in candidates
                if leaf not in protected_leaves
                and leaf != self.alg.root_node
                and leaf != this_node
                and self.alg.__is_live_leaf__(leaf)
            ]
            if not candidates:
                raise ValueError("No eviction candidates available")
            
            # Oracle target, aligned to this micro-step's live candidate set.
            oracle_distances = self._oracle_distances(candidates)
            oracle_idx = self._oracle_target_from_distances(oracle_distances)
            
            # Record per-eviction step data
            step_data = SimpleNamespace()
            step_data.step_kind = "eviction_decision"
            step_data.leaf_node_ids = [c.node_id for c in candidates]
            step_data.leaf_paths = [
                TrieNode.get_node_id_path(c) for c in candidates
            ]
            step_data.current_path = tuple(current_path)
            step_data.step_index = step_index
            step_data.history_paths = tuple(self.alg.history_path_window)
            step_data.oracle_distances = oracle_distances
            step_data.oracle_target = oracle_idx
            step_data.num_candidates = len(candidates)
            snapshot.eviction_steps.append(step_data)
            
            # DAgger: choose actual eviction target
            if random.random() < self.model_prob and self.model is not None and self.alg.history_hidden_states:
                if torch is None:
                    raise ImportError("TrieTrainingCache requires torch when model is set")
                # Model policy
                leaf_states = []
                for c in candidates:
                    if c.hidden_state is not None:
                        leaf_states.append(c.hidden_state[0])
                    else:
                        leaf_states.append(torch.zeros(1, self.model.hidden_size))
                
                with torch.no_grad():
                    scores, _ = self.model.forward(
                        self.alg._history_memory(),
                        candidate_states=leaf_states,
                    )
                target_idx = scores.squeeze(0).argmax().item()
            else:
                # Oracle policy
                target_idx = oracle_idx
            
            # Evict
            target_leaf = candidates.pop(target_idx)
            parent = target_leaf.parent
            self.alg.__delete_leaf_node__(target_leaf)
            
            # Incremental candidate update
            if parent is not None and parent != self.alg.root_node and parent.is_leaf():
                if parent not in protected_leaves:
                    candidates.append(parent)
        
        return snapshot
    
    def get_snapshots(self) -> List[SimpleNamespace]:
        """Return all collected training snapshots and clear the buffer."""
        result = self.snapshots
        self.snapshots = []
        return result
    
    @property
    def hit_rate(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.hit_count / self.total_count


class SequenceTrieCache:
    """
    Simplified Trie cache for sequence-based access patterns (e.g., YooChoose).
    
    Unlike TrieCache which uses pc/aligner/hash, this takes raw sequences directly.
    Supports TrieModelPredictAlgorithm, TrieModelGuard, TrieLRUAlgorithm, TrieRandAlgorithm.
    
    Usage:
        cache = SequenceTrieCache(max_node_num=1024, evict_type=TrieModelPredictAlgorithm, model=my_model)
        # or: cache = SequenceTrieCache(max_node_num=1024, evict_type=TrieLRUAlgorithm)
        
        for seq in data_trace:
            cache.access(seq)
        cache.pretty_stat()
    """
    
    def __init__(self, max_node_num: int, evict_type=None, model=None, **evict_kwargs):
        """
        Args:
            max_node_num: Maximum number of nodes in the trie
            evict_type: Algorithm class (default: TrieModelPredictAlgorithm)
            model: TrieParrotModel instance (only for model-based algorithms)
            **evict_kwargs: Extra kwargs passed to evict_type constructor
        """
        if evict_type is None:
            evict_type = TrieModelPredictAlgorithm
        
        # Instantiate algorithm
        if evict_type in (TrieModelPredictAlgorithm, TrieModelGuard):
            self.alg = evict_type(max_node_num, model=model, **evict_kwargs)
        else:
            # LRU, Rand, etc. don't take model param
            self.alg = evict_type(max_node_num, **evict_kwargs)
        
        self.model = model
        self.stat_info = [0, 0, 0]  # total, hit, miss
        self.request_count = 0
        self.request_full_hits = 0
        self.prefix_hit_sum = 0
        self.uncacheable_block_count = 0
    
    def set_model(self, model):
        """Set or update the model (for model-based algorithms)."""
        self.model = model
        if hasattr(self.alg, 'set_model'):
            self.alg.set_model(model)
    
    def access(self, sequence: List[int]):
        """
        Process a sequence access.
        
        Args:
            sequence: List of token IDs
            
        Returns:
            Tuple of (total_nodes, hit_nodes, miss_nodes)
        """
        cache_sequence = sequence[:self.alg.max_node_num]
        uncacheable_blocks = max(0, len(sequence) - len(cache_sequence))

        if isinstance(self.alg, (TrieModelPredictAlgorithm, TrieModelGuard)):
            # Model-based algorithms: access(sequence)
            cache_stat = self.alg.access(cache_sequence)
        else:
            # Legacy algorithms: access(pc, aligned_address)
            cache_stat = self.alg.access(None, cache_sequence)

        _, cache_hit, _ = cache_stat
        total = len(sequence)
        hit = cache_hit
        miss = total - hit
        stat = (total, hit, miss)
        
        self.stat_info = [x + y for x, y in zip(self.stat_info, stat)]
        self.request_count += 1
        self.prefix_hit_sum += hit
        self.uncacheable_block_count += uncacheable_blocks
        if miss == 0:
            self.request_full_hits += 1
        return stat
    
    def pretty_stat(self):
        total, hit, miss = self.stat_info
        print(f'[Total/Hit/Miss]: [{total}/{hit}/{miss}]')
        if total > 0:
            print(f'[Hit Rate]: {hit / total:.4f}')
        else:
            print('[Hit Rate]: N/A')
    
    def stat(self):
        """Return (total, hit, miss, hit_rate) — same order as stat_info."""
        total, hit, miss = self.stat_info
        if total == 0:
            return (total, hit, miss, 0.0)
        return (total, hit, miss, round(hit / total, 4))

    def kv_stat(self, block_size: int = 1):
        """Return KV-cache oriented aggregate metrics."""
        total, hit, miss = self.stat_info
        block_hit_rate = hit / total if total else 0.0
        request_full_hit_rate = (
            self.request_full_hits / self.request_count
            if self.request_count
            else 0.0
        )
        avg_prefix_hit_len = (
            self.prefix_hit_sum / self.request_count
            if self.request_count
            else 0.0
        )
        return {
            "requests": self.request_count,
            "request_full_hits": self.request_full_hits,
            "prefix_hit_sum": self.prefix_hit_sum,
            "total_blocks": total,
            "hit_blocks": hit,
            "miss_blocks": miss,
            "block_hit_rate": block_hit_rate,
            "request_full_hit_rate": request_full_hit_rate,
            "avg_prefix_hit_len": avg_prefix_hit_len,
            "recompute_blocks": miss,
            "saved_prefill_tokens": hit * block_size,
            "uncacheable_blocks": self.uncacheable_block_count,
            "evictions": getattr(self.alg, "eviction_count", 0),
            "resident_blocks": getattr(self.alg, "cur_node_num", 0),
        }


