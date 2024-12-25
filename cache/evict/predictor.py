from abc import ABC, abstractmethod
from typing import Union, List
from types import SimpleNamespace
import numpy as np
import collections
import random
import torch

class Predictor(ABC):
    def pred_before_evict(self, ts, pc, address, cache_state) -> Union[List[Union[int, float]], None]:
        return None

    def pred_after_evict(self, ts, pc, address) -> Union[int, float, None]:
        return None

class ReuseDistancePredictor(Predictor):
    pass
class BinaryPredictor(Predictor):
    pass
class PhasePredictor(Predictor):
    pass

class ReuseDistancePredition:
    pass
class BinaryPredition:
    pass
class PhasePredition:
    pass

class OraclePredictor(ABC):
    def __init__(self, reuse_dis_noise_sigma=0) -> None:
        self.reuse_dis_noise_sigma = reuse_dis_noise_sigma
    
    def oracle_access(self, pc, address, next_access_time):
        if self.reuse_dis_noise_sigma == 0:
            self.__oracle_access__(pc, address, next_access_time)
        else:
            self.__oracle_access__(pc, address, next_access_time + np.random.lognormal(0, self.reuse_dis_noise_sigma))

    @abstractmethod
    def __oracle_access__(self, pc, address, next_access_time):
        pass

class OracleReuseDistancePredictor(ReuseDistancePredictor, OraclePredictor):
    def __init__(self, reuse_dis_noise_sigma=0, oracle_check=True) -> None:
        super().__init__(reuse_dis_noise_sigma)
        self.oracle_preds = collections.deque()
        self.oracle_check = oracle_check
    
    def pred_after_evict(self, ts, pc, address):
        oracle_key, next_access_time = self.oracle_preds.popleft()
        if self.oracle_check and oracle_key != address:
            raise ValueError("OracleReuseDistancePredictAlgorithm: oracle key not equals to key")
        return next_access_time

    def __oracle_access__(self, pc, address, next_access_time):
        self.oracle_preds.append((address, next_access_time))

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
    
    def pred_after_evict(self, ts, pc, address):
        oracle_key, bin_pred = self.oracle_preds.popleft()
        if self.oracle_check and oracle_key != address:
            raise ValueError("OracleBinaryPredictAlgorithm: oracle key not equals to key")

        if self.bin_noise_prob == 0:
            return bin_pred
        else:
            if random.random() < self.bin_noise_prob:
                return 1 - bin_pred
            else:
                return bin_pred

    def __oracle_access__(self, pc, address, next_access_time):
        if address in self.oracle_cache:
            target_index = self.oracle_cache.index(address)
        elif None in self.oracle_cache:
            target_index = self.oracle_cache.index(None)
        else:
            target_index = self.oracle_next_access_times.index(max(self.oracle_next_access_times))
            self.oracle_preds[self.oracle_last_time[self.oracle_cache[target_index]]][1] = 1
        
        self.oracle_cache[target_index] = address
        self.oracle_next_access_times[target_index] = next_access_time 
        self.oracle_preds.append([address, 0])
        self.oracle_last_time[address] = self.oracle_t
        self.oracle_t += 1

class OraclePhasePredictor(PhasePredictor, OraclePredictor):
    def __init__(self, associativity, reuse_dis_noise_sigma=0, bin_noise_prob=0, oracle_check=True):
        super().__init__(reuse_dis_noise_sigma)
        self.associativity = associativity

        self.oracle_t = 0
        self.oracle_preds = collections.deque()
        self.oracle_check = oracle_check
        self.bin_noise_prob = bin_noise_prob

        self.oracle_prev_phase = []
        self.oracle_curr_phase = []
        self.oracle_curr_key_set = set()
    
    def pred_after_evict(self, ts, pc, address):
        oracle_key, bin_pred = self.oracle_preds.popleft()
        if self.oracle_check and oracle_key != address:
            raise ValueError("OraclePhasePredictorm: oracle key not equals to key")

        if self.bin_noise_prob == 0:
            return bin_pred
        else:
            if random.random() < self.bin_noise_prob:
                return 1 - bin_pred
            else:
                return bin_pred

    def __oracle_access__(self, pc, address, next_access_time):
        if len(self.oracle_curr_key_set) == self.associativity:
            for key, t in self.oracle_prev_phase:
                if key in self.oracle_curr_key_set:
                    self.oracle_preds[t][1] = 0
            self.oracle_prev_phase = self.oracle_curr_phase
            self.oracle_curr_phase = []
            self.oracle_curr_key_set = set()
        self.oracle_curr_phase.append((address, self.oracle_t))
        self.oracle_curr_key_set.add(address)
        self.oracle_preds.append([address, 1])
        self.oracle_t += 1
        
class ParrotPredictor(Predictor):
    def __init__(self, shared_model):
        self._model = shared_model

    def pred_before_evict(self, ts, pc, address, cache_state) -> Union[List[Union[int, float]], None]:
        cache_access = SimpleNamespace()
        cache_access.pc = pc
        cache_access.address = address
        cache_access.cache_lines = cache_state
        scores = self._model(cache_access)
        return [scores[0, i].item() for i in range(len(cache_state))]