#!/usr/bin/env python3
"""Sample a small subset of YooChoose data for CPU-friendly experiments."""
import pickle
import random
import json
import os
import shutil

random.seed(42)

src_dir = 'data/yoochoose'
dst_dir = 'data/yoochoose_small'
os.makedirs(dst_dir, exist_ok=True)

sizes = {'train': 5000, 'valid': 500, 'test': 500}

for split, n in sizes.items():
    with open(os.path.join(src_dir, f'{split}.pkl'), 'rb') as f:
        data = pickle.load(f)
    
    sampled = random.sample(data, min(n, len(data)))
    
    with open(os.path.join(dst_dir, f'{split}.pkl'), 'wb') as f:
        pickle.dump(sampled, f)
    
    avg_len = sum(len(s) for s in sampled) / len(sampled)
    print(f'{split}: {len(sampled)} sequences, avg_len={avg_len:.1f}')

# Copy vocab
shutil.copy(os.path.join(src_dir, 'vocab.json'), os.path.join(dst_dir, 'vocab.json'))

# Metadata
metadata = {'vocab_size': 51850, 'num_train': sizes['train'], 'num_valid': sizes['valid'], 'num_test': sizes['test']}
with open(os.path.join(dst_dir, 'metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)

print(f'Small dataset created in {dst_dir}/')
