from data_trace.data_trace import DataTrace
from model.models import ParrotModel
from model import device_manager
from utils.aligner import ShiftAligner, NormalAligner
from cache.cache import Cache
from cache.evict import *
from cache.hash import ShiftHashFunction, BrightKiteHashFunction, CitiHashFunction
from functools import partial
from typing import Tuple
from prettytable import PrettyTable
import tqdm
import copy
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='xalanc')
    parser.add_argument("--device", type=str, default='cpu')

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--oracle', action='store_true')
    mode_group.add_argument('--real', action='store_true')

    parser.add_argument("--noise_type", type=str, default='logdis', choices=['dis', 'bin', 'logdis'])
    parser.add_argument("--output_root_dir", type=str, default='res')
    args = parser.parse_args()
    file_path = f'traces/{args.dataset}_test.csv'
    if args.dataset == 'brightkite':
        cache_line_size = 1
        capacity = 1000
        associativity = 10
        align_type = NormalAligner
        hash_type = BrightKiteHashFunction
    elif args.dataset == 'citi':
        cache_line_size = 1
        capacity = 1200
        associativity = 100
        align_type = NormalAligner
        hash_type = CitiHashFunction
    else:
        cache_line_size = 64
        capacity = 2097152
        associativity = 16
        align_type = ShiftAligner
        hash_type = ShiftHashFunction

    print('Use Trace:', file_path)
    device = args.device
    device_manager.set_device(device)

    online_types = [
        RandAlgorithm,
        LRUAlgorithm,
        MarkerAlgorithm,
    ]

    sorted = False
    verbose = False

    func_dict = {}
    def register_func(this_partial, noise, baseline=False):
        if baseline:
            func_dict['OPT'] = {}
            func_dict['OPT'][0] = this_partial
        else:
            pretty_name = pretty_print(this_partial, verbose)
            if pretty_name not in func_dict:
                func_dict[pretty_name] = {}
            func_dict[pretty_name][noise] = this_partial
    
    register_func(PredictAlgorithmFactory.generate_predictive_algorithm(PredictAlgorithm, 'OracleDis'), 0, True)

    if args.real:
        verbose = True

        parrot_gen = lambda : ParrotModel.from_config("tmp/model_config.json", 'tmp/xalanc/250000.ckpt')

        online_types.extend([
            # predictive_algorithm_generator(PredictAlgorithm, 'PARROT', parrot_gen()),
            # predictive_algorithm_generator(partial(GuardAlgorithm, follow_if_guarded=False, relax_times=0, relax_prob=0), 'PARROT', parrot_gen()),
            # predictive_algorithm_generator(partial(GuardAlgorithm, follow_if_guarded=False, relax_times=5, relax_prob=0), 'PARROT', parrot_gen()),
            PredictAlgorithmFactory.generate_predictive_algorithm(PredictAlgorithm, 'PLECO'),
            PredictAlgorithmFactory.generate_predictive_algorithm(PredictiveMarker, 'PLECO'),
            PredictAlgorithmFactory.generate_predictive_algorithm(LMarker, 'PLECO'),
            PredictAlgorithmFactory.generate_predictive_algorithm(LNonMarker, 'PLECO'),
            PredictAlgorithmFactory.generate_predictive_algorithm(FollowerRobust, 'PLECO-State', associativity=associativity),
            PredictAlgorithmFactory.generate_predictive_algorithm(partial(Guard, follow_if_guarded=False, relax_times=0, relax_prob=0), 'PLECO'),
            PredictAlgorithmFactory.generate_predictive_algorithm(partial(Guard, follow_if_guarded=False, relax_times=5, relax_prob=0), 'PLECO'),
            PredictAlgorithmFactory.generate_predictive_algorithm(PredictAlgorithm, 'POPU'),
            PredictAlgorithmFactory.generate_predictive_algorithm(PredictiveMarker, 'POPU'),
            PredictAlgorithmFactory.generate_predictive_algorithm(LMarker, 'POPU'),
            PredictAlgorithmFactory.generate_predictive_algorithm(LNonMarker, 'POPU'),
            PredictAlgorithmFactory.generate_predictive_algorithm(FollowerRobust, 'POPU-State', associativity=associativity),
            PredictAlgorithmFactory.generate_predictive_algorithm(partial(Guard, follow_if_guarded=False, relax_times=0, relax_prob=0), 'POPU'),
            PredictAlgorithmFactory.generate_predictive_algorithm(partial(Guard, follow_if_guarded=False, relax_times=5, relax_prob=0), 'POPU'),
        ])

        for online_type in online_types:
            register_func(online_type, 0)

        oracle_types = [
            (PredictAlgorithm, "OracleDis")
        ]

        for oracle_type, pred_type_str in oracle_types:
            register_func(PredictAlgorithmFactory.generate_predictive_algorithm(oracle_type, pred_type_str), 0)

        combiner_types = [
            # (partial(CombineDeterministicAlgorithm, switch_bound=1, lazy_evictor_type=LRUEvictor), [predictive_algorithm_generator(PredictAlgorithm, 'PARROT', parrot_gen()), MarkerAlgorithm]),
            (partial(CombineDeterministicAlgorithm, switch_bound=1, lazy_evictor_type=LRUEvictor), [PredictAlgorithmFactory.generate_predictive_algorithm(PredictAlgorithm, 'PLECO'), MarkerAlgorithm]),
            (partial(CombineDeterministicAlgorithm, switch_bound=1, lazy_evictor_type=LRUEvictor), [PredictAlgorithmFactory.generate_predictive_algorithm(PredictAlgorithm, 'POPU'), MarkerAlgorithm]),
        ]

        for combiner, algs in combiner_types:
            if len(algs) > 1:
                this_partial = copy.deepcopy(combiner)
                this_partial.keywords['candidate_algorithms'] = algs
                register_func(this_partial, 0)
    else:
        verbose = False
        noise_type = args.noise_type

        for online_type in online_types:
            register_func(online_type, 0)

        oracle_dis_noise_mask = [0, 10, 20, 50, 100, 1000, 10000, 100000]
        oracle_bin_noise_mask = [0, 0.1, 0.2, 0.5, 0.8, 1]

        oracle_types = [
            (PredictAlgorithm, 'OracleDis'),
            (PredictAlgorithm, 'OracleBin'),

            (PredictiveMarker, 'OracleDis'),
            (LMarker, 'OracleDis'),
            (LNonMarker, 'OracleDis'),
            (Mark0, 'OracleBin'),
            (MarkAndPredict, 'OraclePhase'),
            # (FollowerRobust, 'OracleState'),

            (partial(Guard, follow_if_guarded=False, relax_times=0, relax_prob=0), 'OracleDis'),
            (partial(Guard, follow_if_guarded=False, relax_times=5, relax_prob=0), 'OracleDis'),
            (partial(Guard, follow_if_guarded=False, relax_times=0, relax_prob=0), 'OracleBin'),
            (partial(Guard, follow_if_guarded=False, relax_times=5, relax_prob=0), 'OracleBin'),
        ]

        for oracle_type, pred_type_str in oracle_types:
            if noise_type == 'dis':
                for noise in oracle_dis_noise_mask:
                    register_func(PredictAlgorithmFactory.generate_predictive_algorithm(oracle_type, pred_type_str, reuse_dis_noise_sigma=noise, lognormal=False, associativity=associativity), noise)
            elif noise_type == 'logdis':
                for noise in oracle_dis_noise_mask:
                    register_func(PredictAlgorithmFactory.generate_predictive_algorithm(oracle_type, pred_type_str, reuse_dis_noise_sigma=noise, lognormal=True, associativity=associativity), noise)
            elif noise_type == 'bin':
                if pred_type_str == 'OracleBin' or pred_type_str == 'OraclePhase':
                    for noise in oracle_bin_noise_mask:
                        register_func(PredictAlgorithmFactory.generate_predictive_algorithm(oracle_type, pred_type_str, bin_noise_prob=noise, associativity=associativity), noise)
            else:
                raise ValueError('Invalid noise type')

        combiner_types = [
            (partial(CombineDeterministicAlgorithm, switch_bound=1, lazy_evictor_type=LRUEvictor), [(PredictiveMarker, 'OracleDis'), LRUAlgorithm]),
            (partial(CombineDeterministicAlgorithm, switch_bound=1, lazy_evictor_type=LRUEvictor), [(PredictAlgorithm, 'OracleBin'), LRUAlgorithm]),
            (partial(CombineDeterministicAlgorithm, switch_bound=1, lazy_evictor_type=LRUEvictor), [(PredictiveMarker, 'OracleDis'), MarkerAlgorithm]),
            (partial(CombineDeterministicAlgorithm, switch_bound=1, lazy_evictor_type=LRUEvictor), [(PredictAlgorithm, 'OracleBin'), MarkerAlgorithm]),
            (partial(CombineRandomAlgorithm, alpha=0.0, beta=0.99, lazy_evictor_type=LRUEvictor), [(PredictiveMarker, 'OracleDis'), LRUAlgorithm]),
            (partial(CombineRandomAlgorithm, alpha=0.0, beta=0.99, lazy_evictor_type=LRUEvictor), [(PredictAlgorithm, 'OracleBin'), LRUAlgorithm]),
            (partial(CombineRandomAlgorithm, alpha=0.0, beta=0.99, lazy_evictor_type=LRUEvictor), [(PredictiveMarker, 'OracleDis'), MarkerAlgorithm]),
            (partial(CombineRandomAlgorithm, alpha=0.0, beta=0.99, lazy_evictor_type=LRUEvictor), [(PredictAlgorithm, 'OracleBin'), MarkerAlgorithm]),
        ]

        def mask_combiner(noise):
            for combiner, algs in combiner_types:
                candidate_algorithms = []
                for alg in algs:
                    if isinstance(alg, Tuple):
                        alg_type, pred_type_str = alg
                        if noise_type == 'dis':
                            candidate_algorithms.append(PredictAlgorithmFactory.generate_predictive_algorithm(alg_type, pred_type_str, associativity=associativity, reuse_dis_noise_sigma=noise, lognormal=False))
                        elif noise_type == 'logdis':
                            candidate_algorithms.append(PredictAlgorithmFactory.generate_predictive_algorithm(alg_type, pred_type_str, associativity=associativity, reuse_dis_noise_sigma=noise, lognormal=True))
                        elif noise_type == 'bin':
                            if pred_type_str == 'OracleBin' or pred_type_str == 'OraclePhase':
                                candidate_algorithms.append(PredictAlgorithmFactory.generate_predictive_algorithm(alg_type, pred_type_str, associativity=associativity, bin_noise_prob=noise))
                        else:
                            raise ValueError('Invalid noise type')
                    else:
                        candidate_algorithms.append(alg)
                
                if len(candidate_algorithms) > 1:
                    this_partial = copy.deepcopy(combiner)
                    this_partial.keywords['candidate_algorithms'] = candidate_algorithms
                    register_func(this_partial, noise)

        if noise_type == 'bin':
            for noise in oracle_bin_noise_mask:
                mask_combiner(noise)
        else:
            for noise in oracle_dis_noise_mask:
                mask_combiner(noise)

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

    opt_miss = np.inf
    if 'OPT' in cache_dict:
        hit, opt_miss, total, rate = cache_dict['OPT'][0].stat()

    if verbose:
        table.field_names = ["Name", "Hit", "Miss", "Total", "Hit Rate", "Competitive Ratio"]
        for i, (pretty_name, _, _) in enumerate(funcs):
            hit, miss, total, rate = caches[i].stat()
            table.add_row([pretty_name, hit, miss, total, rate, f"{miss / opt_miss:.3f}"])
        if sorted:
            table.sortby = "Hit Rate"
            table.reversesort = True
    else:
        if noise_type == 'dis':
            table.field_names = ["Name"] + [f'Dis-{noise}' for noise in oracle_dis_noise_mask]
        elif noise_type == 'logdis':
            table.field_names = ["Name"] + [f'LogDis-{noise}' for noise in oracle_dis_noise_mask]
        elif noise_type == 'bin':
            table.field_names = ["Name"] + [f'Bin-{noise}' for noise in oracle_bin_noise_mask]
        for pretty_name in cache_dict:
            if pretty_name == 'OPT':
                continue
            lst = [pretty_name]
            for noise in cache_dict[pretty_name]:
                _, miss, _, rate = cache_dict[pretty_name][noise].stat()
                lst.append(f'{rate:.3f}/{miss / opt_miss:.3f}')
            if len(lst) == 2:
                lst.extend([lst[-1]] * (len(table.field_names) - 2))
            table.add_row(lst)


    res_dir = os.path.join(args.output_root_dir, args.dataset)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    if args.noise_type is None:
        with open(os.path.join(res_dir, "real.csv"), "w", encoding="utf-8") as file:
            file.write(table.get_csv_string())
    else:
        with open(os.path.join(res_dir, f"{args.noise_type}.csv"), "w", encoding="utf-8") as file:
            file.write(table.get_csv_string())
    print(table)
            