from data_trace.data_trace import DataTrace
from utils.aligner import ShiftAligner
from cache.cache import Cache
from cache.evict_algorithms import *
from cache.hash import ShiftHashFunction
from functools import partial
import tqdm
import numpy as np
from prettytable import PrettyTable

if __name__ == "__main__":
    file_path = 'traces/sphinx3_test.csv'

    cache_line_size = 64
    capacity = 2097152
    associativity = 16
    align_type = ShiftAligner
    hash_type = ShiftHashFunction

    evict_types = [
        RandAlgorithm,
        LRUAlgorithm,
        MarkerAlgorithm,
        BeladyAlgorithm,
        FollowBinaryPredictAlgorithm,
        GuardBeladyAlgorithm,
        GuardFollowBinaryPredictAlgorithm
    ]

###########################################################
    reuse_dis_noises = [0, 100, 1000, 10000]
    bin_pred_noises = [0, 0.1, 0.2, 0.3, 0.5, 0.8, 1]

    guard_relax_times = [0, 1, 2, 3, 4, 5]
    guard_relax_probs = [0, 0.1, 0.2, 0.3, 0.5, 0.8]

    reuse_dis_noises = [0, 1000]
    bin_pred_noises = [0, 0.1]

    guard_relax_times = [0, 1]
    guard_relax_probs = [0, 0.1]
##############################################################
    def mask_guard_func(func):
        cur_funcs = []
        for guard_relax_time in guard_relax_times:
            func = partial(func, relax_times=guard_relax_time)
            cur_funcs.append(func)
        for guard_relax_prob in guard_relax_probs:
            if guard_relax_prob != 0:
                func = partial(func, relax_prob=guard_relax_prob, relax_times=0)
                cur_funcs.append(func)
        return cur_funcs
    
    def mask_oracle_dis(func):
        cur_funcs = []
        for reuse_dis_noise in reuse_dis_noises:
            func = partial(func, reuse_dis_noise_sigma=reuse_dis_noise)
            cur_funcs.append(func)
        return cur_funcs
    
    def mask_oracle_bin(func):
        cur_funcs = []
        for reuse_dis_noise in reuse_dis_noises:
            func = partial(func, reuse_dis_noise_sigma=reuse_dis_noise)
            cur_funcs.append(func)
        for bin_pred_noise in bin_pred_noises:
            if bin_pred_noise != 0:
                func = partial(func, bin_noise_prob=bin_pred_noise, reuse_dis_noise_sigma=0)
                cur_funcs.append(func)
        return cur_funcs

    funcs = []
    for evict_type in evict_types:
        is_oracle = issubclass(evict_type, OracleAlgorithm)
        is_reused_dis = issubclass(evict_type, ReuseDistancePredition)
        is_binary = issubclass(evict_type, BinaryPredition)
        is_guard = issubclass(evict_type, GuardAlgorithm)
        
        func = evict_type

        if is_guard:
            guard_funcs = mask_guard_func(func)
        else:
            guard_funcs = [func]
        for g_func in guard_funcs:
            if is_oracle:
                if is_reused_dis:
                    funcs.extend(mask_oracle_dis(g_func))
                if is_binary:
                    funcs.extend(mask_oracle_bin(g_func))
            else:
                funcs.append(g_func)


###############################################################
    bin = 1
    funcs = [
        partial(GuardFollowBinaryPredictAlgorithm, bin_noise_prob=bin, reuse_dis_noise_sigma=0, relax_times=5),
        partial(CombineRandomAlgorithm, candidate_algorithms=[MarkerAlgorithm, partial(FollowBinaryPredictAlgorithm, bin_noise_prob=bin, reuse_dis_noise_sigma=0)], beta=0.99, lazy_evictor_type=LRUEvictor),
        partial(CombineDeterministicAlgorithm, candidate_algorithms=[MarkerAlgorithm, partial(FollowBinaryPredictAlgorithm, bin_noise_prob=bin, reuse_dis_noise_sigma=0)], switch_bound=1, lazy_evictor_type=LRUEvictor),
        partial(FollowBinaryPredictAlgorithm, bin_noise_prob=bin, reuse_dis_noise_sigma=0),
        MarkerAlgorithm,
        LRUAlgorithm
    ]

    caches = []
    with tqdm.tqdm(desc="Init caches for benchmark", total=len(funcs)) as pbar:
        for evict_type in funcs:
            caches.append(Cache(file_path, align_type, evict_type, hash_type, cache_line_size, capacity, associativity)) 
            pbar.update(1)   

    # np.random.seed(42)
    with DataTrace(file_path) as trace:
        with tqdm.tqdm(desc="Producing cache on MemoryTrace") as pbar:
            while not trace.done():
                pc, address = trace.next()

                for cache in caches:
                    cache.access(address)
                pbar.update(1)     

    table = PrettyTable() 
    table.field_names = ["Name", "Hit", "Miss", "Total", "Hit Rate"]
    for i, this_func in enumerate(funcs):
        if hasattr(this_func, 'func'):
            # partial
            exp_name = this_func.func.__name__
            if 'bin_noise_prob' in this_func.keywords:
                assert this_func.keywords.get('reuse_dis_noise_sigma') == 0
                exp_name = f"{exp_name}_bin-noise-{str(this_func.keywords.get('bin_noise_prob'))}"
            elif 'reuse_dis_noise_sigma' in this_func.keywords:
                noise = this_func.keywords.get('reuse_dis_noise_sigma')
                if noise != 0:
                    exp_name = f"{exp_name}_dis-noise-{str(noise)}"
                else:
                    exp_name = f"{exp_name}_no-noise"
            
            if 'relax_prob' in this_func.keywords:
                assert this_func.keywords.get('relax_times') == 0
                exp_name = f"{exp_name}_relax-prob-{str(this_func.keywords.get('relax_prob'))}"
            elif 'relax_times' in this_func.keywords:
                relax_times = this_func.keywords.get('relax_times')
                if relax_times != 0:
                    exp_name = f"{exp_name}_relax-time-{str(this_func.keywords.get('relax_times'))}"
                else:
                    exp_name = f"{exp_name}_no-relaxed"
        else:
            # class
            exp_name = this_func.__name__

        exp_name = exp_name.replace("Algorithm", '').replace("FollowBinaryPredict", 'FBP')

        hit, miss, total, rate = caches[i].stat()
        table.add_row([exp_name, hit, miss, total, rate])

    print(table)
            