from abc import ABC, abstractmethod
import collections
from typing import Callable, Union

import random
import numpy as np

class EvictAlgorithm(ABC):
    """Evict an entry from one cache line
    
    Max size is associativity
    """
    def __init__(self, associativity) -> None:
        self.cache = [None] * associativity
    
    @abstractmethod
    def access(self, key) -> bool:
        pass

class OracleAlgorithm(ABC):
    def __init__(self, associativity) -> None:
        self.oracle_cache = [None] * associativity
    
    @abstractmethod
    def oracle_access(self, key, next_access_time):
        pass


class PredictAlgorithm(EvictAlgorithm):
    def __init__(self, associativity) -> None:
        super().__init__(associativity)
        self.timestamp = -1
    
    @abstractmethod
    def pred(self, key, ts):
        pass

    def access(self, key):
        self.timestamp += 1
        return self.access_with_pred(key, self.pred(key, self.timestamp))

    @abstractmethod
    def access_with_pred(self, key, pred):
        pass

class ReuseDistancePredictAlgorithm(PredictAlgorithm):
    def __init__(self, associativity) -> None:
        super().__init__(associativity)
        self.reuse_dis = [np.inf] * associativity
    
    def access_with_pred(self, key, pred):
        target_index = -1
        hit = False
        if key in self.cache:
            target_index = self.cache.index(key)
            hit = True
        elif None in self.cache:
            target_index = self.cache.index(None)
        else:
            target_index = self.reuse_dis.index(max(self.reuse_dis))
        self.cache[target_index] = key
        self.reuse_dis[target_index] = pred
        return hit

class BinaryPredictAlgorithm(PredictAlgorithm):
    def __init__(self, associativity) -> None:
        super().__init__(associativity)
        self.binary_pred = [0] * associativity
    
    def __evict__(self):
        indices_with_1 = [i for i, val in enumerate(self.binary_pred) if val == 1]
        if indices_with_1:
            chosen_index = random.choice(indices_with_1)
        else:
            chosen_index = random.randint(0, len(self.binary_pred) - 1)
        return chosen_index
    
    def access_with_pred(self, key, pred):
        target_index = -1
        hit = False
        if key in self.cache:
            target_index = self.cache.index(key)
            hit = True
        elif None in self.cache:
            target_index = self.cache.index(None)
        else:
            target_index = self.__evict__()
        self.cache[target_index] = key
        self.binary_pred[target_index] = pred
        return hit


class OracleReuseDistancePredictAlgorithm(ReuseDistancePredictAlgorithm, OracleAlgorithm):
    def __init__(self, associativity, oracle_check=True) -> None:
        super().__init__(associativity)
        self.oracle_preds = collections.deque()
        self.oracle_check = oracle_check
    
    def pred(self, key, ts):
        oracle_key, next_access_time = self.oracle_preds.popleft()
        if self.oracle_check and oracle_key != key:
            raise ValueError("OracleReuseDistancePredictAlgorithm: oracle key not equals to key")
        return next_access_time

    def oracle_access(self, key, next_access_time):
        self.oracle_preds.append((key, next_access_time))

BeladyAlgorithm = OracleReuseDistancePredictAlgorithm


class OracleBinaryPredictAlgorithm(BinaryPredictAlgorithm, OracleAlgorithm):
    def __init__(self, associativity, oracle_check=True) -> None:
        BinaryPredictAlgorithm.__init__(self, associativity)
        OracleAlgorithm.__init__(self, associativity)
        self.oracle_preds = collections.deque()
        self.oracle_check = oracle_check
        self.oracle_next_access_times = [np.inf] * associativity
        self.oracle_last_time = {}
        self.oracle_t = 0
    
    def pred(self, key, ts):
        oracle_key, bin_pred = self.oracle_preds.popleft()
        if self.oracle_check and oracle_key != key:
            raise ValueError("OracleBinaryPredictAlgorithm: oracle key not equals to key")

        return bin_pred

    def oracle_access(self, key, next_access_time):
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

FollowBinaryPredictAlgorithm = OracleBinaryPredictAlgorithm
