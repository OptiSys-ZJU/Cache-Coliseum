from abc import ABC, abstractmethod
import numpy as np

class HashFunction(ABC):
    def __init__(self, num_buckets):
        self._num_buckets = num_buckets

    @abstractmethod
    def get_bucket_index(self, aligned_address):
        pass

class ShiftHashFunction(HashFunction):
    def __init__(self, num_buckets):
        if num_buckets & (num_buckets - 1) != 0:
            raise ValueError("Cache line size must be power of two.")

        super().__init__(num_buckets)
        self._set_bits = int(np.log2(num_buckets))
    
    def get_bucket_index(self, aligned_address):
        return aligned_address & ((1 << self._set_bits) - 1)