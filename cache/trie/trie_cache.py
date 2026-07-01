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
    candidate_lcp_diagnostics,
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
    
    Wraps a TrieModelPredictAlgorithm and collects microstep cache-state
    snapshots with Belady oracle labels before each access prefix is applied.
    
    Usage:
        cache = TrieTrainingCache(max_node_num=1024, model=my_model)
        cache.load_future_accesses(all_sequences)  # for oracle reuse distance
        cache.set_model_prob(0.0)  # pure oracle at start
        
        for seq in sequences:
            snapshot, hit = cache.collect(seq)
            # snapshot.eviction_steps contains access-prefix training steps.
    """
    
    def __init__(
        self,
        max_node_num: int,
        model=None,
        train_on_eviction_decision: bool = False,
    ):
        """
        Args:
            max_node_num: Maximum trie capacity
            model: TrieParrotModel (can be None, set later via set_model)
        """
        self.max_node_num = max_node_num
        self.model = model
        self.train_on_eviction_decision = bool(train_on_eviction_decision)
        
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
        self._collected_training_steps = 0
        
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
        self.alg.counter = 0
        self.alg.timestamp = 0
        self.snapshots = []
        self._collected_training_steps = 0
    
    def _reuse_distance(self, path: tuple, include_current: bool = True) -> float:
        """
        Belady oracle: how many request steps until an access matches this leaf's path.

        TrieTrainingCache consumes the current request only after all snapshots,
        evictions, and inserts for that request are processed. While a request
        is being handled, oracle queries should see it by default; protected
        leaves, not strict-future labeling, prevent current-path eviction.
        
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
        include_current: bool = True,
    ) -> List[float]:
        """Return request-clock reuse distance for each candidate path."""
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

    @staticmethod
    def _oracle_top_set_from_distances(distances: List[float]):
        if not distances:
            return tuple()
        max_distance = max(distances)
        return tuple(
            idx for idx, distance in enumerate(distances)
            if distance == max_distance
        )

    def _lcp_diagnostics(self, candidate_paths, current_path):
        return candidate_lcp_diagnostics(candidate_paths, current_path)

    @staticmethod
    def _lru_target_from_features(lru_features) -> Optional[int]:
        if not lru_features:
            return None
        return max(range(len(lru_features)), key=lambda idx: lru_features[idx][0])

    def _attach_candidate_metadata(self, step_data, current_path, oracle_distances):
        step_data.oracle_target = self._oracle_target_from_distances(oracle_distances)
        step_data.oracle_top_set = self._oracle_top_set_from_distances(
            oracle_distances
        )
        step_data.lru_target = self._lru_target_from_features(step_data.lru_features)
        step_data.lcp_diagnostics = self._lcp_diagnostics(
            step_data.leaf_paths,
            current_path,
        )

    def _microstep_access_snapshot(
        self,
        cache_sequence: List[int],
        current_prefix: List[int],
        step_index: int,
        pre_request_history_paths,
    ) -> Optional[SimpleNamespace]:
        """
        Build an access-prefix cache-state snapshot before mutating the trie.

        Entries are generic microstep training states sampled before the trie
        mutates for the current prefix.
        """
        candidates = [
            leaf for leaf in self.alg.__leaves__()
            if leaf != self.alg.root_node
            and self.alg.__is_live_leaf__(leaf)
        ]
        if not candidates:
            return None

        oracle_distances = self._oracle_distances(candidates)
        required_candidate_indices = [
            idx for idx, distance in enumerate(oracle_distances)
            if distance == 0
        ]
        step_data = SimpleNamespace()
        step_data.step_kind = "microstep_access"
        step_data.leaf_node_ids = [c.node_id for c in candidates]
        step_data.leaf_paths = [
            TrieNode.get_node_id_path(c) for c in candidates
        ]
        step_data.request_path = tuple(cache_sequence)
        step_data.current_path = tuple(current_prefix)
        step_data.step_index = step_index
        step_data.microstep_history_paths = self.alg._score_time_microstep_history_paths(
            current_prefix
        )
        step_data.request_history_paths = tuple(pre_request_history_paths)
        step_data.lru_features = tuple(
            tuple(row) for row in self.alg._candidate_lru_features(candidates)
        )
        step_data.lru_feature_names = (
            "leaf_age",
            "path_min_age",
            "path_mean_age",
            "path_max_age",
            "path_depth",
        )
        step_data.oracle_distances = oracle_distances
        self._attach_candidate_metadata(step_data, current_prefix, oracle_distances)
        step_data.required_candidate_indices = tuple(required_candidate_indices)
        step_data.num_candidates = len(candidates)
        return step_data
    
    def collect(self, sequence: List[int]):
        """
        Collect pre-microstep cache-state snapshots, then process one request.
        
        Returns:
            (snapshot, hit): 
                snapshot: request container with microstep training steps
                    (None if no microstep has candidates)
                hit: bool, whether the entire sequence was a cache hit
        """
        cache_sequence = sequence[:self.max_node_num]
        self.alg.counter += 1
        snapshot = SimpleNamespace()
        snapshot.sequence = tuple(cache_sequence)
        snapshot.eviction_steps = []
        hit_nodes = 0
        this_node = self.alg.root_node
        pre_request_history_paths = tuple(self.alg.request_history_path_window)

        current_prefix = []
        for step_index, node_id in enumerate(cache_sequence):
            current_prefix.append(node_id)
            step_data = self._microstep_access_snapshot(
                cache_sequence,
                current_prefix,
                step_index,
                pre_request_history_paths,
            )
            if step_data is not None:
                snapshot.eviction_steps.append(step_data)

            if node_id in this_node.children:
                this_node = this_node.children[node_id]
                self.alg.__visit_node__(this_node)
                hit_nodes += 1
            else:
                evict_num = self.alg.cur_node_num + 1 - self.alg.max_node_num
                if evict_num > 0:
                    step_snapshot = self._evict_and_collect(
                        evict_num,
                        this_node,
                        current_prefix,
                        step_index,
                    )
                    if (
                        self.train_on_eviction_decision
                        and step_snapshot is not None
                    ):
                        snapshot.eviction_steps.extend(step_snapshot.eviction_steps)

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

            self.alg._record_microstep_history_path(
                current_prefix,
                this_node.hidden_state,
            )

        if self._future_oracle is not None:
            self._future_oracle.consume_current(cache_sequence, self._current_step)
        hit = hit_nodes == len(sequence)
        if hit and len(cache_sequence) == len(sequence):
            self.hit_count += 1
        self.total_count += 1

        if snapshot.eviction_steps:
            self.snapshots.append(snapshot)
            self._collected_training_steps += len(snapshot.eviction_steps)
        else:
            snapshot = None

        self.alg.timestamp += 1
        self.alg._record_request_history_path(cache_sequence, this_node.hidden_state)
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
            step_data.microstep_history_paths = (
                self.alg._score_time_microstep_history_paths(current_path)
            )
            step_data.request_history_paths = tuple(
                self.alg.request_history_path_window
            )
            step_data.lru_features = tuple(
                tuple(row) for row in self.alg._candidate_lru_features(candidates)
            )
            step_data.lru_feature_names = (
                "leaf_age",
                "path_min_age",
                "path_mean_age",
                "path_max_age",
                "path_depth",
            )
            step_data.oracle_distances = oracle_distances
            self._attach_candidate_metadata(step_data, current_path, oracle_distances)
            step_data.required_candidate_indices = tuple(
                idx for idx, distance in enumerate(oracle_distances)
                if distance == 0
            )
            step_data.num_candidates = len(candidates)
            snapshot.eviction_steps.append(step_data)
            
            # DAgger: choose actual eviction target
            if (
                random.random() < self.model_prob
                and self.model is not None
            ):
                if torch is None:
                    raise ImportError("TrieTrainingCache requires torch when model is set")
                # Model policy
                leaf_states = []
                for c in candidates:
                    if c.hidden_state is not None:
                        leaf_states.append(c.hidden_state[0])
                    else:
                        leaf_states.append(torch.zeros(1, self.model.hidden_size))
                
                forward_kwargs = {"candidate_states": leaf_states}
                if getattr(self.model, "use_lcp_features", False):
                    forward_kwargs["lcp_features"] = (
                        self.alg._candidate_lcp_features(candidates, current_path)
                    )
                with torch.no_grad():
                    scores, _ = self.model.forward(
                        self.alg._score_time_microstep_history_memory(current_path),
                        self.alg._request_history_memory(),
                        self.alg._candidate_lru_features(candidates),
                        **forward_kwargs,
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
        self._collected_training_steps = 0
        return result

    @property
    def collected_training_steps(self) -> int:
        """Number of buffered microstep training steps, not request containers."""
        return self._collected_training_steps
    
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
