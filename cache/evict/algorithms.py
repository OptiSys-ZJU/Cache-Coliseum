from abc import ABC, abstractmethod
from functools import partial
from typing import List, Union
from cache.evict.evictor import *
from cache.evict.predictor import *
import numpy as np
import types
import copy
import random

class EvictAlgorithm(ABC):
    """Evict an entry from one cache line
    
    Max size is associativity
    """
    def __init__(self, associativity) -> None:
        self.cache = [None] * associativity
        self.pcs = [None] * associativity
        self.associativity = associativity
    
    def snapshot(self):
        return list(zip(self.cache, self.pcs))
    
    @abstractmethod
    def access(self, pc, address) -> bool:
        pass

class OracleAlgorithm():
    def oracle_access(self, pc, address, next_access_time):
        self.predictor.oracle_access(pc, address, next_access_time)

class PredictAlgorithm(EvictAlgorithm):
    def __init__(self, associativity, evictor: Evictor, predictor: Predictor) -> None:
        super().__init__(associativity)
        self.timestamp = 0

        if isinstance(predictor, ReuseDistancePredictor):
            self.preds = [np.inf] * associativity
        elif isinstance(predictor, BinaryPredictor):
            self.preds = [0] * associativity
        elif isinstance(predictor, PhasePredictor):
            self.preds = [1] * associativity
        else:
            self.preds = None
        self.evictor = evictor
        self.predictor = predictor
    
    def snapshot(self):
        return (list(zip(self.cache, self.pcs)), self.preds)
    
    def before_pred(self, pc, address):
        preds = self.predictor.pred_before_evict(self.timestamp, pc, address, self.snapshot()[0])
        if preds is not None:
            self.preds = preds
    
    def after_pred(self, pc ,address, target_index):
        pred = self.predictor.pred_after_evict(self.timestamp, pc, address)
        if pred is not None:
            self.preds[target_index] = pred
        self.timestamp += 1

    def access(self, pc, address):
        target_index = -1
        hit = False

        self.before_pred(pc, address)
        if address in self.cache:
            target_index = self.cache.index(address)
            hit = True
        elif None in self.cache:
            target_index = self.cache.index(None)
        else:
            target_index = self.evictor.evict(list(enumerate(self.preds)))
        self.cache[target_index], self.pcs[target_index] = address, pc
        self.after_pred(pc, address, target_index)
        return hit

class PredictiveMarker(PredictAlgorithm):
    """
    PredictiveMarker algorithm

    Designed by Thodoris Lykouris and Sergei Vassilvitskii. 2018. Competitive Caching with Machine Learned Advice.
    https://dl.acm.org/doi/10.1145/3447579
    """
    def __init__(self, associativity, evictor: Evictor, predictor: Predictor) -> None:
        def harmonic_number(k):
            return sum(1 / i for i in range(1, k + 1))
        super().__init__(associativity, evictor, predictor)
        self.marked = [0] * associativity
        self.tracking_set = []
        self.h_k = harmonic_number(associativity)
        self.chains_len = []
        self.chains_rep = []
    
    def access(self, pc, address):
        hit = False
        self.before_pred(pc, address)
        target_index = -1

        if address in self.cache:
            target_index = self.cache.index(address)
            hit = True
        elif None in self.cache:
            target_index = self.cache.index(None)
        else:
            if all(mark == 1 for mark in self.marked):
                # new phase
                self.tracking_set = copy.deepcopy(self.cache)
                self.marked = [0] * self.associativity
            if address not in self.tracking_set:
                target_index = self.evictor.evict([(i, self.preds[i]) for i, mark in enumerate(self.marked) if mark == 0])
                self.chains_len.append(1)
                self.chains_rep.append(self.cache[target_index])
            if address in self.tracking_set:
                index = self.chains_rep.index(address)
                if self.chains_len[index] <= self.h_k:
                    target_index = self.evictor.evict([(i, self.preds[i]) for i, mark in enumerate(self.marked) if mark == 0])
                else:
                    target_index = random.choice([i for i, mark in enumerate(self.marked) if mark == 0])
                self.chains_rep[index] = self.cache[target_index]

        self.cache[target_index], self.pcs[target_index] = address, pc
        self.marked[target_index] = 1
        self.after_pred(pc, address, target_index)
        return hit

class LMarkerAlgorithm(PredictAlgorithm):
    """
    LMARKER Algorithm

    Designed by Dhruv Rohatgi. 2020. Near-Optimal Bounds for Online Caching with Machine Learned Advice
    https://epubs.siam.org/doi/10.1137/1.9781611975994.112
    """
    def __init__(self, associativity, evictor: Evictor, predictor: Predictor) -> None:
        super().__init__(associativity, evictor, predictor)

        self.stale = []
        self.marked = [0] * associativity
    
    def access(self, pc, address):
        target_index = -1
        hit = False

        self.before_pred(pc, address)
        if address in self.cache:
            target_index = self.cache.index(address)
            hit = True
        elif None in self.cache:
            target_index = self.cache.index(None)
        else:
            if all(mark == 1 for mark in self.marked):
                self.stale = copy.deepcopy(self.cache)
                self.marked = [0] * self.associativity
            
            if address in self.stale:
                target_index = random.choice([i for i, mark in enumerate(self.marked) if mark == 0])
            else:
                target_index = self.evictor.evict([(i, self.preds[i]) for i, mark in enumerate(self.marked) if mark == 0])
        
        self.cache[target_index], self.pcs[target_index] = address, pc
        self.marked[target_index] = 1
        self.after_pred(pc, address, target_index)
        return hit

class LNonMarkerAlgorithm(PredictAlgorithm):
    """
    LNONMARKER Algorithm

    Designed by Dhruv Rohatgi. 2020. Near-Optimal Bounds for Online Caching with Machine Learned Advice
    https://epubs.siam.org/doi/10.1137/1.9781611975994.112
    """
    def __init__(self, associativity, evictor: Evictor, predictor: Predictor) -> None:
        super().__init__(associativity, evictor, predictor)

        self.phase = set()
        self.stale = []
        self.marked = [0] * associativity
        self.evicts = {}
    
    def access(self, pc, address):
        target_index = -1
        hit = False
        self.before_pred(pc, address)

        if len(self.phase) == self.associativity:
            self.stale = copy.deepcopy(self.cache)
            self.marked = [0] * self.associativity
            self.evicts = {}
            self.phase = set()

        if address in self.cache:
            target_index = self.cache.index(address)
            hit = True
        elif None in self.cache:
            target_index = self.cache.index(None)
        else:
            if address in self.stale:
                if self.evicts[address] not in self.stale:
                    target_index = random.choice(range(self.associativity))
                else:
                    target_index = random.choice([i for i, mark in enumerate(self.marked) if mark == 0])
            else:
                target_index = self.evictor.evict([(i, self.preds[i]) for i, mark in enumerate(self.marked) if mark == 0])
        
        self.evicts[self.cache[target_index]] = address
        self.cache[target_index], self.pcs[target_index] = address, pc
        self.marked[target_index] = 1
        self.phase.add(address)
        self.after_pred(pc, address, target_index)
        return hit

class Mark0Algorithm(PredictAlgorithm):
    """
    MARK0 Eviction Strategy

    Designed by Antonios Antoniadis, Joan Boyar, Marek Eliáš, Lene M. Favrholdt, Ruben Hoeksma, Kim S. Larsen, Adam Polak, and Bertrand Simon. 2023. Paging with Succinct Prediction.
    https://dl.acm.org/doi/10.5555/3618408.3618447
    """
    def __init__(self, associativity, evictor, predictor):
        if not isinstance(predictor, BinaryPredictor):
            raise ValueError('Mark0Algorithm: predictor must be a BinaryPredictor')
        super().__init__(associativity, evictor, predictor)
        self.marked = [0] * associativity
        self.S_address = [None] * associativity
        self.S_visited = [0] * associativity
    
    def access(self, pc, address):
        target_index = -1
        hit = False

        self.before_pred(pc, address)
        if address in self.cache:
            target_index = self.cache.index(address)
            hit = True
        elif None in self.cache:
            if address in self.S_address and 0 in self.S_visited:
                target_index = random.choice([i for i, visited in enumerate(self.S_visited) if visited == 0])
            else:
                target_index = self.cache.index(None)
        else:
            if all(visited == 1 for visited in self.S_visited):
                self.marked = [0] * self.associativity
                self.S_address = copy.deepcopy(self.cache)
                self.S_visited = [0] * self.associativity

            if address in self.S_address and 0 in self.S_visited:
                target_index = random.choice([i for i, visited in enumerate(self.S_visited) if visited == 0])
            else:
                target_index = random.choice([i for i, mark in enumerate(self.marked) if mark == 0])
        
        self.S_address[target_index] = None
        self.S_visited[target_index] = 1
        self.marked[target_index] = 1
        self.cache[target_index], self.pcs[target_index] = address, pc
        self.after_pred(pc, address, target_index)
        if self.preds[target_index] == 1:
            self.cache[target_index], self.pcs[target_index] = None, None
        return hit

class MarkAndPredictAlgorithm(PredictAlgorithm):
    """
    MARK&PREDICT Eviction Strategy

    Designed by Antonios Antoniadis, Joan Boyar, Marek Eliáš, Lene M. Favrholdt, Ruben Hoeksma, Kim S. Larsen, Adam Polak, and Bertrand Simon. 2023. Paging with Succinct Prediction.
    https://dl.acm.org/doi/10.5555/3618408.3618447
    """
    def __init__(self, associativity, evictor, predictor):
        if not isinstance(predictor, PhasePredictor):
            raise ValueError('MarkAndPredictAlgorithm: predictor must be a PhasePredictor')
        if not isinstance(evictor, BinaryEvictor):
            raise ValueError('MarkAndPredictAlgorithm: evictor must be a BinaryEvictor')
        super().__init__(associativity, evictor, predictor)
        self.marked = [0] * associativity
    
    def access(self, pc, address):
        target_index = -1
        hit = False

        self.before_pred(pc, address)
        if address in self.cache:
            target_index = self.cache.index(address)
            hit = True
        elif None in self.cache:
            target_index = self.cache.index(None)
        else:
            if all(mark == 1 for mark in self.marked):
                self.marked = [0] * self.associativity
            target_index = self.evictor.evict([(i, self.preds[i]) for i, mark in enumerate(self.marked) if mark == 0])
        
        self.cache[target_index], self.pcs[target_index] = address, pc
        self.marked[target_index] = 1
        self.after_pred(pc, address, target_index)
        return hit

class GuardAlgorithm(PredictAlgorithm):
    """
    Guard algorithm

    Our work
    """
    def __init__(self, associativity, evictor: Evictor, predictor: Predictor, follow_if_guarded=False, relax_times=0, relax_prob=0) -> None:
        super().__init__(associativity, evictor, predictor)
        self.old_unvisited_set = []
        self.unguarded_set = []
        self.phase_evicted_set = set()

        self.follow_if_guarded = follow_if_guarded
        self.error_times = 0
        self.relax_times = relax_times
        self.relax_prob = relax_prob
    
    def access(self, pc, address):
        to_guard = False
        target_index = -1
        hit = False

        self.before_pred(pc, address)
        if address in self.cache:
            target_index = self.cache.index(address)
            hit = True
        elif None in self.cache:
            target_index = self.cache.index(None)
        else:
            if not self.old_unvisited_set:
                self.old_unvisited_set = list(range(self.associativity))
                self.unguarded_set = list(range(self.associativity))
                self.phase_evicted_set = set()
                self.error_times = 0
            
            if address in self.phase_evicted_set:
                if self.relax_times != 0:
                    self.error_times += 1
                    if self.error_times >= self.relax_times:
                        to_guard = True
                else:
                    if random.random() > self.relax_prob:
                        to_guard = True

            if to_guard and not self.follow_if_guarded:
                target_index = random.choice(self.old_unvisited_set)
            else:
                target_index = self.evictor.evict([(i, self.preds[i]) for i in self.unguarded_set])
            
            self.phase_evicted_set.add(self.cache[target_index])

        if target_index in self.old_unvisited_set:
            self.old_unvisited_set.remove(target_index)

        if to_guard:
            self.unguarded_set.remove(target_index)
        
        self.cache[target_index], self.pcs[target_index] = address, pc
        self.after_pred(pc, address, target_index)
        return hit

class CombineAlgorithm(EvictAlgorithm):
    def __init__(self, associativity, candidate_algorithms: List[Union[EvictAlgorithm, partial]], lazy_evictor_type: Union[LRUEvictor, RandEvictor, None] = LRUEvictor):
        if lazy_evictor_type is not None and not issubclass(lazy_evictor_type, Evictor):
            raise ValueError('CombineAlgorithm: Invalid Evictor')
        
        super().__init__(associativity)
        self.oracle_algs = []
        self.candidate_algs = []
        self.center = 0
        self.timestamp = 0
        self.lazy_evictor = lazy_evictor_type() if lazy_evictor_type is not None else None
        self.key_scores = {} if lazy_evictor_type == LRUEvictor else None

        for alg_type in candidate_algorithms:
            alg_instance = alg_type(associativity)
            self.candidate_algs.append([alg_instance, 0])
            alg_class = alg_type.func if hasattr(alg_type, 'func') else alg_type
            if hasattr(alg_class, 'oracle_access'):
                self.oracle_algs.append(alg_instance)

        if len(self.oracle_algs) != 0:
            def oracle_access(self, pc, address, next_access_time):
                for oracle_alg in self.oracle_algs:
                    oracle_alg.oracle_access(pc, address, next_access_time)
            self.oracle_access = types.MethodType(oracle_access, self)
        
        if len(self.candidate_algs) < 2:
            raise ValueError('CombineAlgorithm: Algorithm Count < 2')

    def __push_candidates__(self, pc, address):
        for i, (alg, _) in enumerate(self.candidate_algs):
            if not alg.access(pc, address):
                self.candidate_algs[i][1] += 1
                self.__trigger_miss__(i, address)
        
        if self.key_scores is not None:
            self.key_scores[address] = self.timestamp
        self.timestamp += 1
    
    def __trigger_miss__(self, i, address):
        pass

    @abstractmethod
    def __trigger_elect_center__(self):
        pass

    def access(self, pc, address):
        self.__push_candidates__(pc, address)

        target_index = -1
        hit = False
        if address in self.cache:
            target_index = self.cache.index(address)
            hit = True
        elif None in self.cache:
            target_index = self.cache.index(None)
        else:
            self.__trigger_elect_center__()
            center_cache = self.candidate_algs[self.center][0].cache
            if self.lazy_evictor is None:
                self.cache = copy.deepcopy(center_cache)
                self.pcs = copy.deepcopy(self.candidate_algs[self.center][0].pcs)
                target_index = self.cache.index(address)
            else:
                diff_keys = set(center_cache) - set(self.cache)
                target_index = self.lazy_evictor.evict([(center_cache.index(k), self.key_scores[k] if self.key_scores is not None else 0) for k in diff_keys])
        
        self.cache[target_index] = address
        self.pcs[target_index] = pc
        return hit

class CombineDeterministicAlgorithm(CombineAlgorithm):
    """
    black-box algorithm

    Designed by Thodoris Lykouris and Sergei Vassilvitskii. 2018. Competitive Caching with Machine Learned Advice.
    https://dl.acm.org/doi/10.1145/3447579
    """
    def __init__(self, associativity, candidate_algorithms: List[Union[EvictAlgorithm, partial]], switch_bound=2, lazy_evictor_type: Union[LRUEvictor, RandEvictor, None] = LRUEvictor):
        super().__init__(associativity, candidate_algorithms, lazy_evictor_type)
        self.switch_bound = switch_bound

    def __trigger_elect_center__(self):
        this_cost = self.candidate_algs[self.center][1]
        min_center, (_, min_cost) = min(enumerate(self.candidate_algs), key=lambda x: x[1][1])
        if this_cost >= self.switch_bound * min_cost:
            self.center = min_center

class CombineRandomAlgorithm(CombineAlgorithm):
    """
    Algorithm THRESH

    Designed by Avrim Blum and Carl Burch. 1997. On-line learning and the metrical task system problem.
    https://dl.acm.org/doi/10.1145/267460.267475
    """
    def __init__(self, associativity, candidate_algorithms: List[Union[EvictAlgorithm, partial]], alpha=0.0, beta=0.99, lazy_evictor_type: Union[LRUEvictor, RandEvictor, None] = LRUEvictor):
        super().__init__(associativity, candidate_algorithms, lazy_evictor_type)
        self.alpha = alpha
        self.beta = beta
        self.n = len(self.candidate_algs)
        self.weights = [1] * self.n
    
    def __trigger_miss__(self, i, key):
        self.weights[i] *= self.beta
    
    def __trigger_elect_center__(self):
        W = sum(self.weights)
        threshold = self.alpha * W / self.n
        valid_index, valid_weights = zip(*[(i, weight) for i, weight in enumerate(self.weights) if weight > threshold])
        if valid_weights:
            self.center = random.choices(valid_index, weights=valid_weights)[0]

class CombineWeightsAlgorithm(CombineAlgorithm):
    """
    Imitation learing for Parrot
    """
    def __init__(self, associativity, candidate_algorithms: List[Union[EvictAlgorithm, partial]], weights: Union[List[float], None], lazy_evictor_type: Union[LRUEvictor, RandEvictor, None] = LRUEvictor):
        super().__init__(associativity, candidate_algorithms, lazy_evictor_type)
        self.n = len(self.candidate_algs)
        if weights is not None:
            self.weights = weights
        else:
            self.weights = [1] * self.n
    
    def snapshot(self):
        return (list(zip(self.cache, self.pcs)), self.candidate_algs[self.center][0].preds)

    def reset(self, weights):
        self.weights = weights

    def __trigger_elect_center__(self):
        self.center = random.choices(list(range(self.n)), weights=self.weights)[0]

#######################################################################

class RandAlgorithm(EvictAlgorithm):
    def __init__(self, associativity):
        super().__init__(associativity)
        self.evictor = RandEvictor()
    
    def access(self, pc, address):
        target_index = -1
        hit = False
        if address in self.cache:
            target_index = self.cache.index(address)
            hit = True
        elif None in self.cache:
            target_index = self.cache.index(None)
        else:
            target_index = self.evictor.evict(list(enumerate(self.cache)))
        
        self.cache[target_index] = address
        self.pcs[target_index] = pc
        return hit

class LRUAlgorithm(EvictAlgorithm):
    def __init__(self, associativity):
        super().__init__(associativity)
        self.evictor = LRUEvictor()
        self.scores = [0] * associativity
        self.timestamp = 0
    
    def access(self, pc, address):
        target_index = -1
        hit = False
        if address in self.cache:
            target_index = self.cache.index(address)
            hit = True
        elif None in self.cache:
            target_index = self.cache.index(None)
        else:
            target_index = self.evictor.evict(list(enumerate(self.scores)))
        
        self.cache[target_index] = address
        self.pcs[target_index] = pc
        self.scores[target_index] = self.timestamp
        self.timestamp += 1
        return hit

class MarkerAlgorithm(EvictAlgorithm):
    def __init__(self, associativity):
        super().__init__(associativity)
        self.evictor = MarkerEvictor()
        self.scores = [0] * associativity
    
    def access(self, pc, address):
        if all(x == 1 for x in self.scores):
            self.scores = [0] * self.associativity

        target_index = -1
        hit = False
        if address in self.cache:
            target_index = self.cache.index(address)
            hit = True
        elif None in self.cache:
            target_index = self.cache.index(None)
        else:
            target_index = self.evictor.evict(list(enumerate(self.scores)))
        
        self.cache[target_index] = address
        self.pcs[target_index] = pc
        self.scores[target_index] = 1
        return hit

#######################################################################

class BeladyAlgorithm(PredictAlgorithm, OracleAlgorithm, ReuseDistancePredition):
    def __init__(self, associativity, reuse_dis_noise_sigma=0) -> None:
        super().__init__(associativity, ReuseDistanceEvictor(), OracleReuseDistancePredictor(reuse_dis_noise_sigma))

class FollowBinaryPredictAlgorithm(PredictAlgorithm, OracleAlgorithm, BinaryPredition):
    def __init__(self, associativity, reuse_dis_noise_sigma=0, bin_noise_prob=0):
        super().__init__(associativity, BinaryEvictor(), OracleBinaryPredictor(associativity, reuse_dis_noise_sigma, bin_noise_prob))

class PredictiveMarkerBeladyAlgorithm(PredictiveMarker, OracleAlgorithm, ReuseDistancePredition):
    def __init__(self, associativity, reuse_dis_noise_sigma=0):
        super().__init__(associativity, ReuseDistanceEvictor(), OracleReuseDistancePredictor(reuse_dis_noise_sigma))

class PredictiveMarkerFollowBinaryPredictAlgorithm(PredictiveMarker, OracleAlgorithm, BinaryPredition):
    def __init__(self, associativity, reuse_dis_noise_sigma=0, bin_noise_prob=0):
        super().__init__(associativity, BinaryEvictor(), OracleBinaryPredictor(associativity, reuse_dis_noise_sigma, bin_noise_prob))

class LMarkerBeladyAlgorithm(LMarkerAlgorithm, OracleAlgorithm, ReuseDistancePredition):
    def __init__(self, associativity, reuse_dis_noise_sigma=0):
        super().__init__(associativity, ReuseDistanceEvictor(), OracleReuseDistancePredictor(reuse_dis_noise_sigma))

class LNonMarkerBeladyAlgorithm(LNonMarkerAlgorithm, OracleAlgorithm, ReuseDistancePredition):
    def __init__(self, associativity, reuse_dis_noise_sigma=0):
        super().__init__(associativity, ReuseDistanceEvictor(), OracleReuseDistancePredictor(reuse_dis_noise_sigma))

class Mark0FollowBinaryPredictAlgorithm(Mark0Algorithm, OracleAlgorithm, BinaryPredition):
    def __init__(self, associativity, reuse_dis_noise_sigma=0, bin_noise_prob=0):
        super().__init__(associativity, BinaryEvictor(), OracleBinaryPredictor(associativity, reuse_dis_noise_sigma, bin_noise_prob))

class MarkAndPredictOracleAlgorithm(MarkAndPredictAlgorithm, OracleAlgorithm, PhasePredition):
    def __init__(self, associativity, reuse_dis_noise_sigma=0, bin_noise_prob=0):
        super().__init__(associativity, BinaryEvictor(), OraclePhasePredictor(associativity, reuse_dis_noise_sigma, bin_noise_prob))

class GuardBeladyAlgorithm(GuardAlgorithm, OracleAlgorithm, ReuseDistancePredition):
    def __init__(self, associativity, reuse_dis_noise_sigma=0, follow_if_guarded=False, relax_times=0, relax_prob=0):
        super().__init__(associativity, ReuseDistanceEvictor(), OracleReuseDistancePredictor(reuse_dis_noise_sigma), follow_if_guarded, relax_times, relax_prob)

class GuardFollowBinaryPredictAlgorithm(GuardAlgorithm, OracleAlgorithm, BinaryPredition):
    def __init__(self, associativity, reuse_dis_noise_sigma=0, follow_if_guarded=False, bin_noise_prob=0, relax_times=0, relax_prob=0):
        super().__init__(associativity, ReuseDistanceEvictor(), OracleBinaryPredictor(associativity, reuse_dis_noise_sigma, bin_noise_prob), follow_if_guarded, relax_times, relax_prob)

class GuardParrotAlgorithm(GuardAlgorithm):
    def __init__(self, associativity, shared_model, follow_if_guarded=False, relax_times=0, relax_prob=0):
        super().__init__(associativity, MaxEvictor(), ParrotPredictor(shared_model), follow_if_guarded, relax_times, relax_prob)

class ParrotAlgorithm(PredictAlgorithm):
    def __init__(self, associativity, shared_model):
        super().__init__(associativity, MaxEvictor(), ParrotPredictor(shared_model))

####################################################################
def format_guard(relax_times, relax_prob):
    if relax_times == 0 and relax_prob == 0:
        return "-no-relax"
    elif relax_times == 0 and relax_prob != 0:
        return f"-relax-prob-{relax_prob}"
    elif relax_times != 0 and relax_prob == 0:
        return f"-relax-times-{relax_times}"
    else:
        raise ValueError('relax_times and relax_prob invaild')

def format_oracle(reuse_dis_noise_sigma, bin_noise_prob):
    if reuse_dis_noise_sigma == 0 and bin_noise_prob == 0:
        return "-oracle"
    elif reuse_dis_noise_sigma == 0 and bin_noise_prob != 0:
        return f"-bin-{bin_noise_prob}"
    elif reuse_dis_noise_sigma != 0 and bin_noise_prob == 0:
        return f"-dis-{reuse_dis_noise_sigma}"
    else:
        return f"-dis-{reuse_dis_noise_sigma}-bin-{bin_noise_prob}"

def pretty_print(callable: Union[EvictAlgorithm, partial], verbose=False) -> str:
    this_cls = callable
    if hasattr(callable, 'func'):
        this_cls = callable.func
    this_cls_name = this_cls.__name__.replace("Algorithm", '').replace("FollowBinaryPredict", 'FBP').replace("MarkAndPredict", "Mark&Predict")
    metadata = this_cls_name
    if hasattr(callable, 'keywords'):
        kw = callable.keywords
        if issubclass(this_cls, CombineAlgorithm):
            algs = kw['candidate_algorithms']
            alg_names = []
            for alg in algs:
                alg_names.append(pretty_print(alg, verbose))
            metadata += ("[" + (", ".join(alg_names)) + "]")
        if issubclass(this_cls, GuardAlgorithm):
            follow_if_guarded = False
            relax_times = relax_prob = 0
            if 'follow_if_guarded' in kw:
                follow_if_guarded = kw['follow_if_guarded']
            if follow_if_guarded:
                metadata += '-unv'
            else:
                metadata += '-f-pred'
            if 'relax_times' in kw:
                relax_times = kw['relax_times']
            if 'relax_prob' in kw:
                relax_prob = kw['relax_prob']
            metadata += format_guard(relax_times, relax_prob)

        if issubclass(this_cls, OracleAlgorithm) and verbose:
            reuse_dis_noise_sigma = bin_noise_prob = 0
            if 'reuse_dis_noise_sigma' in kw:
                reuse_dis_noise_sigma = kw['reuse_dis_noise_sigma']
            if 'bin_noise_prob' in kw:
                bin_noise_prob = kw['bin_noise_prob']
            metadata += format_oracle(reuse_dis_noise_sigma, bin_noise_prob) 
    
    return metadata
        


