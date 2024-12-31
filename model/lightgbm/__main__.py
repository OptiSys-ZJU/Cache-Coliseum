import lightgbm as lgb
import argparse
import numpy as np
import pandas as pd
import os
import tqdm
import hashlib
from cache.evict.algorithms import PredictAlgorithm, PredictAlgorithmFactory
from data_trace.data_trace import DataTrace
from utils.aligner import ShiftAligner
from cache.hash import ShiftHashFunction
from cache.cache import TrainingCache
from utils.aligner import ShiftAligner, NormalAligner
from cache.hash import ShiftHashFunction, BrightKiteHashFunction, CitiHashFunction

from model import device_manager

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='xalanc')
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--model_fraction", type=str, default='1')
    parser.add_argument("--checkpoints_root_dir", type=str, default='checkpoints')
    parser.add_argument("--traces_root_dir", type=str, default='traces')
    args = parser.parse_args()
    device_manager.set_device(args.device)

    traces_dir = os.path.join(args.traces_root_dir, args.dataset)
    if not os.path.exists(traces_dir):
        raise ValueError(f'LightGBM: {traces_dir} not found')

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
    
    def generate_fraction_train_file(train_all_file_path, fraction):
        this_train_file_path = os.path.join(traces_dir, f'{args.dataset}_train_{args.model_fraction}.csv')
        with open(train_all_file_path, "r") as infile:
            lines = infile.readlines()
        
        total_lines = len(lines)
        num_lines_to_write = int(total_lines * float(fraction))
        with open(this_train_file_path, "w") as outfile:
            outfile.writelines(lines[:num_lines_to_write])
        print(f"Generate Fraction File: Written {num_lines_to_write} out of {total_lines} lines to {this_train_file_path}.")
    
    if args.model_fraction == '1':
        # all
        train_file_path = os.path.join(traces_dir, f'{args.dataset}_train.csv')
        if not os.path.exists(train_file_path):
            raise ValueError(f'LightGBM: {train_file_path} not found')
    else:
        train_file_path = os.path.join(traces_dir, f'{args.dataset}_train_{args.model_fraction}.csv')
        if not os.path.exists(train_file_path):
            print(f'LightGBM: {args.model_fraction} Train File not found, try to generate')
            train_all_file_path = os.path.join(traces_dir, f'{args.dataset}_train.csv')
            if not os.path.exists(train_all_file_path):
                raise ValueError(f'LightGBM: {train_all_file_path} not found')
            generate_fraction_train_file(train_all_file_path, args.model_fraction)
            if not os.path.exists(train_file_path):
                raise ValueError(f'LightGBM: {train_file_path} not found, generate failed')

    valid_file_path = os.path.join(traces_dir, f'{args.dataset}_valid.csv')
    test_file_path = os.path.join(traces_dir, f'{args.dataset}_test.csv')
    
    def generate_binary_path(trace_path, label_path):
        bins = []
        evict_type = PredictAlgorithmFactory.generate_predictive_algorithm(PredictAlgorithm, 'OracleBin', associativity=associativity)
        cache = TrainingCache(trace_path, align_type, evict_type, hash_type, cache_line_size, capacity, associativity)
        with DataTrace(trace_path) as trace:
            with tqdm.tqdm(desc="Collecting bin labels on DataTrace") as pbar:
                while not trace.done():
                    pc, address = trace.next()
                    bin_label = cache.collect(pc, address)
                    bins.append(bin_label)
                    pbar.update(1) 
        with open(label_path, 'w') as f:
            for item in bins:
                f.write(f"{item}\n")
        print(f"Generate Bin Label File: Written to {label_path}, Len[{len(bins)}]")

    
    bin_label_dir = os.path.join(traces_dir, 'labels')
    if not os.path.exists(bin_label_dir):
        os.makedirs(bin_label_dir)
    
    valid_bin_file_path = os.path.join(bin_label_dir, 'valid.txt')
    if not os.path.exists(valid_bin_file_path):
        generate_binary_path(valid_file_path, valid_bin_file_path)
    if not os.path.exists(valid_bin_file_path):
        raise ValueError(f'LightGBM: {valid_bin_file_path} not found, generate failed')
    
    test_bin_file_path = os.path.join(bin_label_dir, 'test.txt')
    if not os.path.exists(test_bin_file_path):
        generate_binary_path(test_file_path, test_bin_file_path)
    if not os.path.exists(test_bin_file_path):
        raise ValueError(f'LightGBM: {test_bin_file_path} not found, generate failed')

    if args.model_fraction == '1':
        train_bin_file_path = os.path.join(bin_label_dir, 'train.txt')
    else:
        train_bin_file_path = os.path.join(bin_label_dir, f'train_{args.model_fraction}.txt')
    if not os.path.exists(train_bin_file_path):
        generate_binary_path(train_file_path, train_bin_file_path)
    if not os.path.exists(train_bin_file_path):
        raise ValueError(f'LightGBM: {train_bin_file_path} not found, generate failed')

    print(f'LightGBM: Train Path[{train_file_path}], Label[{train_bin_file_path}]')
    print(f'LightGBM: Valid Path[{valid_file_path}], Label[{valid_bin_file_path}]')
    print(f'LightGBM: Test Path[{test_file_path}], Label[{test_bin_file_path}]')

    def load_dataset(trace_path, bin_path, dataset):
        if dataset == 'brightkite':
            df = pd.read_csv(trace_path, header=None)
            df[0] = df[0].apply(int)
            df[1] = df[1].apply(lambda x: int(hashlib.sha256(x.lstrip().encode()).hexdigest(), 16))
            data = df.to_numpy().astype(np.float64)
            bins = np.loadtxt(bin_path, dtype=int)
            data = lgb.Dataset(data, label=bins)
            return data
        elif dataset == 'citi':
            df = pd.read_csv(trace_path, header=None)
            df = df.applymap(lambda x: int(x))
            data = df.to_numpy()
            bins = np.loadtxt(bin_path, dtype=int)
            data = lgb.Dataset(data, label=bins)
            return data
        else:
            aligner = align_type(cache_line_size)
            df = pd.read_csv(trace_path, header=None)
            df = df.applymap(lambda x: aligner.get_aligned_addr(int(x, 16)))
            data = df.to_numpy()
            bins = np.loadtxt(bin_path, dtype=int)
            data = lgb.Dataset(data, label=bins)
            return data
        
    train_data = load_dataset(train_file_path, train_bin_file_path, args.dataset)
    valid_data = load_dataset(valid_file_path, valid_bin_file_path, args.dataset)
    test_data = load_dataset(test_file_path, test_bin_file_path, args.dataset)

    param = {'num_leaves':31, 'objective':'binary', 'metric': 'auc'}
    bst = lgb.train(param, train_data, valid_sets=[valid_data])
    print('Light GBM: Trained finished, generate Threshold...')
    ypred = bst.predict(test_data.data)
    def accuracy_score(test_bins, predictions):
        success = 0
        for i in range(test_bins.size):
            if test_bins[i] == predictions[i]:
                success += 1
        return success
    threshold_range = np.arange(0.0, 1.0, 0.01)
    best_accuracy = 0
    for threshold in threshold_range:
        predictions = (ypred > threshold).astype(int)
        accuracy = accuracy_score(test_data.label, predictions)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    
    checkpoint_dir = os.path.join(args.checkpoints_root_dir, 'lightgbm', args.dataset, args.model_fraction)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    bst.save_model(os.path.join(checkpoint_dir, f'{args.dataset}_{args.model_fraction}.txt'))
    
    with open(os.path.join(checkpoint_dir, 'threshold'), 'w') as f:
        f.write(str(best_threshold))
    print(f"LightGBM: Best Threshold [{best_threshold}]")
    print(f"LightGBM: Best Accuracy [{best_accuracy/test_data.label.size}]")
