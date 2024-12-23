from data_trace.data_trace import DataTrace
from model.models import ParrotModel
from utils.aligner import ShiftAligner
from cache.cache import Cache
from cache.evict import *
from cache.hash import ShiftHashFunction
from functools import partial
from prettytable import PrettyTable
import tqdm
import os
import json
import copy

if __name__ == "__main__":
    file_path = 'traces/bzip_test.csv'

    cache_line_size = 64
    capacity = 2097152
    associativity = 16
    align_type = ShiftAligner
    hash_type = ShiftHashFunction

    online_types = [
        RandAlgorithm,
        LRUAlgorithm,
        MarkerAlgorithm,
    ]

    ## mask noises
    oracle_dis_noise_mask = [0, 100, 1000, 10000]
    oracle_bin_noise_mask = [0, 0.1, 0.2, 0.5, 0.8, 1]

    #noise_type = 'dis'
    noise_type = 'bin'

    oracle_types = [
        partial(BeladyAlgorithm),
        partial(FollowBinaryPredictAlgorithm),
        partial(GuardBeladyAlgorithm, follow_if_guarded=False, relax_prob=0.2),
        partial(GuardFollowBinaryPredictAlgorithm, follow_if_guarded=True, relax_times=5),
    ]

    combiner_types = [
        (CombineDeterministicAlgorithm, [BeladyAlgorithm, LRUAlgorithm]),
        (CombineRandomAlgorithm, [FollowBinaryPredictAlgorithm, MarkerAlgorithm]),
    ]

##############################################################
    funcs = []

    funcs.extend(online_types)

    for oracle_alg_type in oracle_types:
        if noise_type == 'dis':
            for noise in oracle_dis_noise_mask:
                this_partial = copy.deepcopy(oracle_alg_type)
                this_partial.keywords['reuse_dis_noise_sigma'] = noise
                funcs.append(this_partial)
        elif noise_type == 'bin':
            if issubclass(oracle_alg_type.func, BinaryPredition):
                for noise in oracle_bin_noise_mask:
                    this_partial = copy.deepcopy(oracle_alg_type)
                    oracle_alg_type.keywords['bin_noise_prob'] = noise
                    funcs.append(this_partial)
        else:
            raise ValueError('Invalid noise type')

    def mask_combiner(noise):
        for combiner, algs in combiner_types:
            candidate_algorithms = []
            for alg in algs:
                if issubclass(alg, OracleAlgorithm):
                    if noise_type == 'dis':
                        this_partial = partial(alg, reuse_dis_noise_sigma=noise)
                        candidate_algorithms.append(this_partial)
                    elif noise_type == 'bin':
                        if issubclass(alg, BinaryPredition):
                            this_partial = partial(alg, bin_noise_prob=noise)
                            candidate_algorithms.append(this_partial)
                    else:
                        raise ValueError('Invalid noise type')
                else:
                    candidate_algorithms.append(alg)
            
            if len(candidate_algorithms) > 1:
                funcs.append(partial(combiner, candidate_algorithms=candidate_algorithms))
    
    if noise_type == 'dis':
        for noise in oracle_dis_noise_mask:
            mask_combiner(noise)
    elif noise_type == 'bin':
        for noise in oracle_bin_noise_mask:
            mask_combiner(noise)
    else:
        raise ValueError('Invalid noise type')

###############################################################
    # with open(os.path.join("", "tmp/model_config.json"), "r") as f:
    #     model_config = json.load(f)
    #     shared_model = ParrotModel(model_config)

    pretty_names = [pretty_print(func) for func in funcs]
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
                    cache.access(pc, address)
                pbar.update(1)     

    table = PrettyTable() 
    #table.field_names = ["Name", "Hit", "Miss", "Total", "Hit Rate"] + [f'Dis-{noise}' for noise in oracle_dis_noise_mask]
    table.field_names = ["Name", "Hit", "Miss", "Total", "Hit Rate"]
    for i, pretty_name in enumerate(pretty_names):
        hit, miss, total, rate = caches[i].stat()
        table.add_row([pretty_name, hit, miss, total, rate])

    print(table)
            