from abc import ABC, abstractmethod
import numpy as np
import collections
import random

class Predictor(ABC):
    @abstractmethod
    def pred(self, pc, address, ts):
        pass

class ReuseDistancePredictor(Predictor):
    def pred(self, pc, address, ts):
        pass 

class BinaryPredictor(Predictor):
    def pred(self, pc, address, ts):
        pass

class ReuseDistancePredition:
    pass
class BinaryPredition:
    pass

class OraclePredictor(ABC):
    def __init__(self, reuse_dis_noise_sigma=0) -> None:
        self.reuse_dis_noise_sigma = reuse_dis_noise_sigma
    
    def oracle_access(self, pc, address, next_access_time):
        if self.reuse_dis_noise_sigma == 0:
            self.__oracle_access__(pc, address, next_access_time)
        else:
            self.__oracle_access__(pc, address, np.random.normal(next_access_time, self.reuse_dis_noise_sigma))

    @abstractmethod
    def __oracle_access__(self, pc, address, next_access_time):
        pass

class OracleReuseDistancePredictor(ReuseDistancePredictor, OraclePredictor):
    def __init__(self, reuse_dis_noise_sigma=0, oracle_check=True) -> None:
        super().__init__(reuse_dis_noise_sigma)
        self.oracle_preds = collections.deque()
        self.oracle_check = oracle_check
    
    def pred(self, pc, address, ts):
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
    
    def pred(self, pc, address, ts):
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

class ParrotPredictor(ABC):
    def pred(self, pc, address, ts):
        pass