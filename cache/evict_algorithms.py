from abc import ABC, abstractmethod
import collections
from typing import Type

import random
import numpy as np

class Evictor(ABC):
    @abstractmethod
    def evict(self, candidates):
        pass

class ReuseDistanceEvictor(Evictor):
    def evict(self, candidates):
        return max(candidates, key=lambda x: x[1])[0]

class BinaryEvictor(Evictor):
    def evict(self, candidates):
        indices_with_1 = [i for i, pred in candidates if pred == 1]
        if indices_with_1:
            chosen_index = random.choice(indices_with_1)
        else:
            chosen_index = random.choice(candidates)[0]
        return chosen_index

class LRUEvictor(Evictor):
    def evict(self, candidates):
        return min(candidates, key=lambda x: x[1])[0]

class MarkerEvictor(Evictor):
    def evict(self, candidates):
        # 1 marked, 0 unmarked
        indices_with_0 = [i for i, mark in candidates if mark == 0]
        if indices_with_0:
            chosen_index = random.choice(indices_with_0)
        else:
            raise ValueError('MarkerEvictor: all marked')

        return chosen_index

class RandEvictor(Evictor):
    def evict(self, candidates):
        return random.choice([i for i, _ in candidates])

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

class Predictor(ABC):
    @abstractmethod
    def pred(self, key, ts):
        pass

class ReuseDistancePredictor(Predictor):
    def pred(self, key, ts):
        pass 

class BinaryPredictor(Predictor):
    def pred(self, key, ts):
        pass 

class OraclePredictor(ABC):
    def __init__(self, reuse_dis_noise_sigma=0) -> None:
        self.reuse_dis_noise_sigma = reuse_dis_noise_sigma
    
    def oracle_access(self, key, next_access_time):
        if self.reuse_dis_noise_sigma == 0:
            self.__oracle_access__(key, next_access_time)
        else:
            self.__oracle_access__(key, np.random.normal(next_access_time, self.reuse_dis_noise_sigma))

    @abstractmethod
    def __oracle_access__(self, key, next_access_time):
        pass

class OracleReuseDistancePredictor(ReuseDistancePredictor, OraclePredictor):
    def __init__(self, reuse_dis_noise_sigma=0, oracle_check=True) -> None:
        super().__init__(reuse_dis_noise_sigma)
        self.oracle_preds = collections.deque()
        self.oracle_check = oracle_check
    
    def pred(self, key, ts):
        oracle_key, next_access_time = self.oracle_preds.popleft()
        if self.oracle_check and oracle_key != key:
            raise ValueError("OracleReuseDistancePredictAlgorithm: oracle key not equals to key")
        return next_access_time

    def __oracle_access__(self, key, next_access_time):
        self.oracle_preds.append((key, next_access_time))

class OracleBinaryPredictor(BinaryPredictor, OraclePredictor):
    def __init__(self, associativity, reuse_dis_noise_sigma=0, bin_noise_prob=0, oracle_check=True) -> None:
        super().__init__(reuse_dis_noise_sigma)
        self.oracle_cache = [None] * associativity
        self.oracle_preds = collections.deque()
        self.oracle_check = oracle_check
        self.oracle_next_access_times = [np.inf] * associativity
        self.oracle_last_time = {}
        self.oracle_t = 0
        self.bin_noise_prob = bin_noise_prob
    
    def pred(self, key, ts):
        oracle_key, bin_pred = self.oracle_preds.popleft()
        if self.oracle_check and oracle_key != key:
            raise ValueError("OracleBinaryPredictAlgorithm: oracle key not equals to key")

        if self.bin_noise_prob == 0:
            return bin_pred
        else:
            if random.random() < self.bin_noise_prob:
                return 1 - bin_pred
            else:
                return bin_pred

    def __oracle_access__(self, key, next_access_time):
        if key in self.oracle_cache:
            target_index = self.oracle_cache.index(key)
        elif None in self.oracle_cache:
            target_index = self.oracle_cache.index(None)
        else:
            target_index = self.oracle_next_access_times.index(max(self.oracle_next_access_times))
            self.oracle_preds[self.oracle_last_time[self.oracle_cache[target_index]]][1] = 1
        
        self.oracle_cache[target_index] = key
        self.oracle_next_access_times[target_index] = next_access_time 
        self.oracle_preds.append([key, 0])
        self.oracle_last_time[key] = self.oracle_t
        self.oracle_t += 1

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

class ReuseDistancePredition:
    pass
class BinaryPredition:
    pass

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