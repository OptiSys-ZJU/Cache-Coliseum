from abc import ABC, abstractmethod
from typing import Callable, Union

import random

class EvictAlgorithm(ABC):
    """Evict an entry from one cache line
    
    Max size is associativity
    """
    def __init__(self, associativity) -> None:
        super().__init__()
        self.max_size = associativity
        self.cache = [None] * associativity
    
    @abstractmethod
    def access(self, key) -> bool:
        pass


class OracleEvictAlgorithm(EvictAlgorithm):
    def __init__(self, associativity, next_access_func: Callable[[int], Union[int, float]], **kwargs) -> None:
        super().__init__(associativity, **kwargs)
        self._next_access_func = next_access_func
    
    @abstractmethod
    def access(self, key) -> bool:
        pass


class BeladyEvictAlgorithm(OracleEvictAlgorithm):
    def __init__(self, associativity, next_access_func: Callable[[int], Union[int, float]]) -> None:
        super().__init__(associativity, next_access_func)
    
    def __evict__(self) -> int:
        next_access_times = [self._next_access_func(item) for item in self.cache]
        return next_access_times.index(max(next_access_times))
    
    def access(self, key):
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
        return hit


class PredictEvictAlgorithm(EvictAlgorithm):
    def __init__(self, associativity, **kwargs) -> None:
        super().__init__(associativity, **kwargs)
    
    @abstractmethod
    def pred(self, key):
        pass

    def access(self, key):
        return self.access_with_pred(key, self.pred(key))

    @abstractmethod
    def access_with_pred(self, key, pred):
        pass


class FollowBinaryPredictionEvictAlgorithm(PredictEvictAlgorithm):
    def __init__(self, associativity, **kwargs) -> None:
        super().__init__(associativity, **kwargs)

        # key --> binary pred
        self.binary_pred = {}

        self.averse_set = []
        self.friend_set = []
    
    @abstractmethod
    def pred(self, key):
        pass

    def __evict__(self):
        bin_preds = [self.binary_pred[key] for key in self.cache]
        indices_with_1 = [i for i, val in enumerate(bin_preds) if val == 1]
        if indices_with_1:
            chosen_index = random.choice(indices_with_1)
        else:
            chosen_index = random.randint(0, len(bin_preds) - 1)
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
        self.binary_pred[key] = pred
        return hit


class OracleFollowBinaryPredictionEvictAlgorithm(FollowBinaryPredictionEvictAlgorithm, OracleEvictAlgorithm):
    def __init__(self, associativity, next_access_func: Callable[[int], Union[int, float]]) -> None:
        super().__init__(associativity=associativity, next_access_func=next_access_func)
        self.oracle_cache = [None] * associativity
        self.oracle_bin_pred = []
        self.oracle_last_time = {}
        self.oracle_t = 0
        self.fbp_t = 0

    def prepare_pred(self, key):
        if key in self.oracle_cache:
            target_index = self.oracle_cache.index(key)
        elif None in self.oracle_cache:
            target_index = self.oracle_cache.index(None)
        else:
            next_access_times = [self._next_access_func(item) for item in self.oracle_cache]
            target_index = next_access_times.index(max(next_access_times))
            self.oracle_bin_pred[self.oracle_last_time[self.oracle_cache[target_index]]] = 1
        
        self.oracle_cache[target_index] = key
        self.oracle_bin_pred.append(0)
        self.oracle_last_time[key] = self.oracle_t
        self.oracle_t += 1

    def pred(self, key):
        if len(self.oracle_bin_pred) == 0:
            raise ValueError("OracleFollowBinaryPrediction: prepare_pred must be called first")
        bin_pred = self.oracle_bin_pred[self.fbp_t]
        self.fbp_t += 1
        return bin_pred
