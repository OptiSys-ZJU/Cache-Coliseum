from data_trace.data_trace import DataTrace
from utils.aligner import ShiftAligner
from cache.cache import Cache
from cache.evict_algorithms import BeladyAlgorithm, FollowBinaryPredictAlgorithm, GuardBeladyAlgorithm, GuardFollowBinaryPredictAlgorithm
from cache.hash import ShiftHashFunction
from functools import partial
import tqdm
import numpy as np

if __name__ == "__main__":
    file_path = 'traces/sphinx3_test.csv'
    cache_line_size = 64
    capacity = 2097152
    associativity = 16

    reuse_dis_noise = 0
    bin_pred_noise = 0

    align_type = ShiftAligner

    #evict_type = partial(BeladyAlgorithm, reuse_dis_noise_sigma=reuse_dis_noise)
    #evict_type = partial(FollowBinaryPredictAlgorithm, reuse_dis_noise_sigma=reuse_dis_noise, bin_noise_prob=bin_pred_noise)

    evict_type = partial(GuardBeladyAlgorithm, reuse_dis_noise_sigma=reuse_dis_noise)
    #evict_type = partial(GuardFollowBinaryPredictAlgorithm, reuse_dis_noise_sigma=reuse_dis_noise, bin_noise_prob=bin_pred_noise)

    hash_type = ShiftHashFunction

    np.random.seed(42)

    cache = Cache(file_path, align_type, evict_type, hash_type, cache_line_size, capacity, associativity, reuse_dis_noise)
    with DataTrace(file_path) as trace:
        with tqdm.tqdm(desc="Producing cache on MemoryTrace") as pbar:
            while not trace.done():
                pc, address = trace.next()
                cache.access(address)
                pbar.update(1)      
    cache.stat()

            