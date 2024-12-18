from memory_trace.memtrace import MemoryTrace
from utils.aligner import ShiftAligner
from cache.cache import Cache
from cache.evict_algorithms import OracleEvictAlgorithm, BeladyEvictAlgorithm, OracleFollowBinaryPredictionEvictAlgorithm
from cache.hash import ShiftHashFunction
from functools import partial
import tqdm

if __name__ == "__main__":
    file_path = 'traces/sphinx3_test.csv'
    cache_line_size = 64
    capacity = 2097152
    associativity = 16
    aligner = ShiftAligner(cache_line_size)

    #evict_type = BeladyEvictAlgorithm
    evict_type = OracleFollowBinaryPredictionEvictAlgorithm

    hash_type = ShiftHashFunction

    cache = Cache(file_path, aligner, evict_type, hash_type, capacity, associativity)        
    cache.stat()

            