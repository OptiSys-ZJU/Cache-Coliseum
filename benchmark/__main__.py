from data_trace.data_trace import DataTrace
from model.models import ParrotModel
from model import device_manager
from utils.aligner import ShiftAligner
from cache.cache import Cache
from cache.evict import *
from cache.hash import ShiftHashFunction
from functools import partial
from prettytable import PrettyTable
import tqdm
import copy

if __name__ == "__main__":
    file_path = 'traces/sphinx3_test.csv'
    print(file_path)
    verbose = True
    verbose = False
    sorted = False

    device = 'cuda:1'

    device_manager.set_device(device)

    cache_line_size = 64
    capacity = 2097152
    associativity = 16
    align_type = ShiftAligner
    hash_type = ShiftHashFunction

    #parrot_gen = lambda : ParrotModel.from_config("tmp/model_config.json", 'tmp/xalanc/250000.ckpt')

    online_types = [
        RandAlgorithm,
        LRUAlgorithm,
        MarkerAlgorithm,
        # partial(ParrotAlgorithm, shared_model=parrot_gen()),
        # partial(GuardParrotAlgorithm, shared_model=parrot_gen(), follow_if_guarded=False, relax_times=0, relax_prob=0)
    ]

    combiner_types = [
        (partial(CombineDeterministicAlgorithm, switch_bound=1, lazy_evictor_type=LRUEvictor), [BeladyAlgorithm, LRUAlgorithm]),
        (partial(CombineRandomAlgorithm, alpha=0.0, beta=0.99, lazy_evictor_type=LRUEvictor), [FollowBinaryPredictAlgorithm, MarkerAlgorithm]),
        #(partial(CombineDeterministicAlgorithm, switch_bound=1, lazy_evictor_type=LRUEvictor), [partial(ParrotAlgorithm, shared_model=parrot_gen()), MarkerAlgorithm])
    ]

    #noise_type = None
    #noise_type = 'dis'
    noise_type = 'bin'

    if noise_type is not None:
        ## mask noises
        oracle_dis_noise_mask = [0, 10, 20, 50, 100, 1000, 10000, 100000]
        oracle_bin_noise_mask = [0, 0.1, 0.2, 0.5, 0.8, 1]
    oracle_types = [
        partial(BeladyAlgorithm),
        partial(FollowBinaryPredictAlgorithm),
        partial(PredictiveMarkerBeladyAlgorithm),
        partial(LMarkerBeladyAlgorithm),
        partial(LNonMarkerBeladyAlgorithm),
        partial(Mark0FollowBinaryPredictAlgorithm),
        partial(MarkAndPredictOracleAlgorithm),
        partial(GuardBeladyAlgorithm, follow_if_guarded=False, relax_prob=0.2),
        partial(GuardFollowBinaryPredictAlgorithm, follow_if_guarded=True, relax_times=0),
        partial(GuardFollowBinaryPredictAlgorithm, follow_if_guarded=True, relax_times=5),
    ]

##############################################################
    func_dict = {}
    def register_func(this_partial, noise):
        pretty_name = pretty_print(this_partial, verbose)
        if pretty_name not in func_dict:
            func_dict[pretty_name] = {}
        func_dict[pretty_name][noise] = this_partial

    for online_type in online_types:
        register_func(online_type, 0)

    if noise_type is not None:
        for oracle_alg_type in oracle_types:
            if noise_type == 'dis':
                for noise in oracle_dis_noise_mask:
                    this_partial = copy.deepcopy(oracle_alg_type)
                    this_partial.keywords['reuse_dis_noise_sigma'] = noise
                    register_func(this_partial, noise)
            elif noise_type == 'bin':
                if issubclass(oracle_alg_type.func, BinaryPredition) or issubclass(oracle_alg_type.func, PhasePredition):
                    for noise in oracle_bin_noise_mask:
                        this_partial = copy.deepcopy(oracle_alg_type)
                        this_partial.keywords['bin_noise_prob'] = noise
                        register_func(this_partial, noise)
            else:
                raise ValueError('Invalid noise type')
    else:
        for oracle_alg_type in oracle_types:
            register_func(oracle_alg_type, 0)

    def mask_combiner(noise):
        for combiner, algs in combiner_types:
            candidate_algorithms = []
            for alg in algs:
                if isinstance(alg, type) and issubclass(alg, OracleAlgorithm):
                    if noise_type == 'dis':
                        this_partial = partial(alg, reuse_dis_noise_sigma=noise)
                        candidate_algorithms.append(this_partial)
                    elif noise_type == 'bin':
                        if issubclass(alg, BinaryPredition) or issubclass(alg, PhasePredition):
                            this_partial = partial(alg, bin_noise_prob=noise)
                            candidate_algorithms.append(this_partial)
                    else:
                        raise ValueError('Invalid noise type')
                else:
                    candidate_algorithms.append(alg)
            
            if len(candidate_algorithms) > 1:
                this_partial = copy.deepcopy(combiner)
                this_partial.keywords['candidate_algorithms'] = candidate_algorithms
                register_func(this_partial, noise)
    
    if noise_type is not None:
        if noise_type == 'dis':
            for noise in oracle_dis_noise_mask:
                mask_combiner(noise)
        elif noise_type == 'bin':
            for noise in oracle_bin_noise_mask:
                mask_combiner(noise)
        else:
            raise ValueError('Invalid noise type')
    else:
        mask_combiner(0)

###############################################################
    cache_dict = {}
    caches = []
    funcs = [(outer_key, inner_key, value) for outer_key, inner_dict in func_dict.items() for inner_key, value in inner_dict.items()]

    with tqdm.tqdm(desc="Init caches for benchmark", total=len(funcs)) as pbar:
        def register_cache(pretty_name, noise, cache):
            if pretty_name not in cache_dict:
                cache_dict[pretty_name] = {}
            cache_dict[pretty_name][noise] = cache
        
        for pretty_name, noise, this_partial in funcs:
            cache = Cache(file_path, align_type, this_partial, hash_type, cache_line_size, capacity, associativity)
            register_cache(pretty_name, noise, cache)
            caches.append(cache)
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
    if verbose:
        table.field_names = ["Name", "Hit", "Miss", "Total", "Hit Rate"]
        for i, (pretty_name, _, _) in enumerate(funcs):
            hit, miss, total, rate = caches[i].stat()
            table.add_row([pretty_name, hit, miss, total, rate])
        if sorted:
            table.sortby = "Hit Rate"
            table.reversesort = True
    else:
        if noise_type == 'dis':
            table.field_names = ["Name"] + [f'Dis-{noise}' for noise in oracle_dis_noise_mask]
        elif noise_type == 'bin':
            table.field_names = ["Name"] + [f'Bin-{noise}' for noise in oracle_bin_noise_mask]
        for pretty_name in cache_dict:
            lst = [pretty_name]
            for noise in cache_dict[pretty_name]:
                _, _, _, rate = cache_dict[pretty_name][noise].stat()
                lst.append(rate)
            if len(lst) == 2:
                lst.extend([lst[-1]] * (len(table.field_names) - 2))
            table.add_row(lst)

    print(table)
            