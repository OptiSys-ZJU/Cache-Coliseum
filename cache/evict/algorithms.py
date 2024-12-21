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
        self.associativity = associativity
    
    @abstractmethod
    def access(self, key) -> bool:
        pass

class OracleAlgorithm():
    def oracle_access(self, key, next_access_time):
        self.predictor.oracle_access(key, next_access_time)

class PredictAlgorithm(EvictAlgorithm):
    def __init__(self, associativity, evictor: Evictor, predictor: Predictor) -> None:
        super().__init__(associativity)
        self.timestamp = 0

        if isinstance(predictor, ReuseDistancePredictor):
            self.preds = [np.inf] * associativity
        elif isinstance(predictor, BinaryPredictor):
            self.preds = [0] * associativity
        else:
            self.preds = None
        self.evictor = evictor
        self.predictor = predictor
    
    def trigger_pred(self, key):
        pred = self.predictor.pred(key, self.timestamp)
        self.timestamp += 1
        return pred

    def access(self, key):
        target_index = -1
        hit = False
        if key in self.cache:
            target_index = self.cache.index(key)
            hit = True
        elif None in self.cache:
            target_index = self.cache.index(None)
        else:
            target_index = self.evictor.evict(list(enumerate(self.preds)))
        
        self.cache[target_index] = key
        self.preds[target_index] = self.trigger_pred(key)
        return hit

class GuardAlgorithm(PredictAlgorithm):
    def __init__(self, associativity, evictor: Evictor, predictor: Predictor, relax_times=0, relax_prob=0) -> None:
        super().__init__(associativity, evictor, predictor)
        self.old_unvisited_set = []
        self.unguarded_set = []
        self.phase_evicted_set = set()

        self.error_times = 0
        self.relax_times = relax_times
        self.relax_prob = relax_prob
    
    def access(self, key):
        to_guard = 0
        target_index = -1
        hit = False
        if key in self.cache:
            target_index = self.cache.index(key)
            hit = True
        elif None in self.cache:
            target_index = self.cache.index(None)
        else:
            if not self.old_unvisited_set:
                self.old_unvisited_set = list(range(self.associativity))
                self.unguarded_set = list(range(self.associativity))
                self.phase_evicted_set = set()
                self.error_times = 0
            
            if key in self.phase_evicted_set:
                if self.relax_times != 0:
                    self.error_times += 1
                    if self.error_times >= self.relax_times:
                        to_guard = 1
                else:
                    if random.random() > self.relax_prob:
                        to_guard = 1

            if to_guard:
                target_index = random.choice(self.old_unvisited_set)
            else:
                target_index = self.evictor.evict([(i, self.preds[i]) for i in self.unguarded_set])
            
            self.phase_evicted_set.add(self.cache[target_index])

        if target_index in self.old_unvisited_set:
            self.old_unvisited_set.remove(target_index)

        self.cache[target_index] = key
        self.preds[target_index] = self.trigger_pred(key)
        if to_guard == 1:
            self.unguarded_set.remove(target_index)

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
            def oracle_access(self, key, next_access_time):
                for oracle_alg in self.oracle_algs:
                    oracle_alg.oracle_access(key, next_access_time)
            self.oracle_access = types.MethodType(oracle_access, self)
        
        if len(self.candidate_algs) < 2:
            raise ValueError('CombineAlgorithm: Algorithm Count < 2')

    def __push_candidates__(self, key):
        for i, (alg, _) in enumerate(self.candidate_algs):
            if not alg.access(key):
                self.candidate_algs[i][1] += 1
                self.__trigger_miss__(i, key)
        
        if self.key_scores is not None:
            self.key_scores[key] = self.timestamp
        self.timestamp += 1
    
    def __trigger_miss__(self, i, key):
        pass

    @abstractmethod
    def __trigger_elect_center__(self):
        pass

    def access(self, key):
        self.__push_candidates__(key)

        target_index = -1
        hit = False
        if key in self.cache:
            target_index = self.cache.index(key)
            hit = True
        elif None in self.cache:
            target_index = self.cache.index(None)
        else:
            self.__trigger_elect_center__()
            center_cache = self.candidate_algs[self.center][0].cache
            if self.lazy_evictor is None:
                self.cache = copy.deepcopy(center_cache)
                target_index = self.cache.index(key)
            else:
                diff_keys = set(center_cache) - set(self.cache)
                target_index = self.lazy_evictor.evict([(center_cache.index(k), self.key_scores[k] if self.key_scores is not None else 0) for k in diff_keys])
        
        self.cache[target_index] = key
        return hit

class CombineDeterministicAlgorithm(CombineAlgorithm):
    """
    black-box algorithm

    Designed by Thodoris Lykouris and Sergei Vassilvitskii. 2021. Competitive Caching with Machine Learned Advice.
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

#######################################################################

class RandAlgorithm(EvictAlgorithm):
    def __init__(self, associativity):
        super().__init__(associativity)
        self.evictor = RandEvictor()
    
    def access(self, key):
        target_index = -1
        hit = False
        if key in self.cache:
            target_index = self.cache.index(key)
            hit = True
        elif None in self.cache:
            target_index = self.cache.index(None)
        else:
            target_index = self.evictor.evict(list(enumerate(self.cache)))
        
        self.cache[target_index] = key
        return hit

class LRUAlgorithm(EvictAlgorithm):
    def __init__(self, associativity):
        super().__init__(associativity)
        self.evictor = LRUEvictor()
        self.scores = [0] * associativity
        self.timestamp = 0
    
    def access(self, key):
        target_index = -1
        hit = False
        if key in self.cache:
            target_index = self.cache.index(key)
            hit = True
        elif None in self.cache:
            target_index = self.cache.index(None)
        else:
            target_index = self.evictor.evict(list(enumerate(self.scores)))
        
        self.cache[target_index] = key
        self.scores[target_index] = self.timestamp
        self.timestamp += 1
        return hit

class MarkerAlgorithm(EvictAlgorithm):
    def __init__(self, associativity):
        super().__init__(associativity)
        self.evictor = MarkerEvictor()
        self.scores = [0] * associativity
    
    def access(self, key):
        if all(x == 1 for x in self.scores):
            self.scores = [0] * self.associativity

        target_index = -1
        hit = False
        if key in self.cache:
            target_index = self.cache.index(key)
            hit = True
        elif None in self.cache:
            target_index = self.cache.index(None)
        else:
            target_index = self.evictor.evict(list(enumerate(self.scores)))
        
        self.cache[target_index] = key
        self.scores[target_index] = 1
        return hit

#######################################################################

class BeladyAlgorithm(PredictAlgorithm, OracleAlgorithm, ReuseDistancePredition):
    def __init__(self, associativity, reuse_dis_noise_sigma=0) -> None:
        super().__init__(associativity, ReuseDistanceEvictor(), OracleReuseDistancePredictor(reuse_dis_noise_sigma))

class FollowBinaryPredictAlgorithm(PredictAlgorithm, OracleAlgorithm, BinaryPredition):
    def __init__(self, associativity, reuse_dis_noise_sigma=0, bin_noise_prob=0):
        super().__init__(associativity, BinaryEvictor(), OracleBinaryPredictor(associativity, reuse_dis_noise_sigma, bin_noise_prob))

class GuardBeladyAlgorithm(GuardAlgorithm, OracleAlgorithm, ReuseDistancePredition):
    def __init__(self, associativity, reuse_dis_noise_sigma=0, relax_times=0, relax_prob=0):
        super().__init__(associativity, ReuseDistanceEvictor(), OracleReuseDistancePredictor(reuse_dis_noise_sigma), relax_times, relax_prob)

class GuardFollowBinaryPredictAlgorithm(GuardAlgorithm, OracleAlgorithm, BinaryPredition):
    def __init__(self, associativity, reuse_dis_noise_sigma=0, bin_noise_prob=0, relax_times=0, relax_prob=0):
        super().__init__(associativity, ReuseDistanceEvictor(), OracleBinaryPredictor(associativity, reuse_dis_noise_sigma, bin_noise_prob), relax_times, relax_prob)