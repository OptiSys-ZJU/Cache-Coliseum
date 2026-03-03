"""
Training script for TrieParrotModel using DAgger (Dataset Aggregation).

Usage:
    python -m model.trie_model --dataset yoochoose --device cpu
    python -m model.trie_model --dataset yoochoose --device cuda:0
"""
import os
import io
import json
import argparse

import torch
import tqdm

from model.trie_model.model import TrieParrotModel
from cache.trie.trie_cache import TrieTrainingCache, SequenceTrieCache
from cache.trie.trie_algorithms import TrieModelPredictAlgorithm, TrieLRUAlgorithm
from data_trace.trie_data_trace import SequenceTrieDataTrace


def get_model_prob(step: int, dagger_init: float, dagger_final: float, dagger_steps: int) -> float:
    """DAgger schedule: linear interpolation from init to final over dagger_steps."""
    fraction = min(float(step) / max(dagger_steps, 1), 1.0)
    return dagger_init + fraction * (dagger_final - dagger_init)


def collect_snapshots(
    data_path: str, 
    vocab_path: str,
    model: TrieParrotModel,
    max_node_num: int,
    model_prob: float,
    max_examples: int = None,
):
    """
    Run one pass over data, collecting DAgger training snapshots.
    
    Returns:
        (snapshots, hit_rate)
    """
    cache = TrieTrainingCache(max_node_num=max_node_num, model=model)
    
    with SequenceTrieDataTrace(data_path, vocab_path) as trace:
        # Load all sequences for oracle
        all_seqs = list(trace.iter_sequences())
    
    cache.load_future_accesses(all_seqs)
    cache.set_model_prob(model_prob)
    
    for seq in all_seqs:
        cache.collect(seq)
        if max_examples is not None and len(cache.snapshots) >= max_examples:
            break
    
    return cache.get_snapshots(), cache.hit_rate


def evaluate(
    data_path: str,
    vocab_path: str,
    model: TrieParrotModel,
    max_node_num: int,
):
    """Evaluate model hit rate on a dataset (pure model policy)."""
    cache = SequenceTrieCache(
        max_node_num=max_node_num, 
        evict_type=TrieModelPredictAlgorithm, 
        model=model,
    )
    with SequenceTrieDataTrace(data_path, vocab_path) as trace:
        while not trace.done():
            seq = trace.next()
            cache.access(seq)
    
    _, hit, miss = cache.stat_info
    total = hit + miss
    return hit / total if total > 0 else 0.0


def evaluate_lru(data_path: str, vocab_path: str, max_node_num: int):
    """Baseline: LRU hit rate."""
    cache = SequenceTrieCache(
        max_node_num=max_node_num, 
        evict_type=TrieLRUAlgorithm,
    )
    with SequenceTrieDataTrace(data_path, vocab_path) as trace:
        while not trace.done():
            seq = trace.next()
            cache.access(seq)
    
    _, hit, miss = cache.stat_info
    total = hit + miss
    return hit / total if total > 0 else 0.0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train TrieParrotModel with DAgger')
    parser.add_argument("--dataset", type=str, default='yoochoose')
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("-p", "--model_config_path", type=str, 
                        default='checkpoints/trie_model/model_config.json')
    parser.add_argument("--checkpoints_root_dir", type=str, default='checkpoints')
    parser.add_argument("--data_root_dir", type=str, default='data')
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load config
    if not os.path.exists(args.model_config_path):
        raise ValueError(f'Config not found: {args.model_config_path}')
    with open(args.model_config_path, 'r') as f:
        config = json.load(f)

    lr = config['lr']
    total_steps = config['total_steps']
    eval_freq = config['eval_freq']
    save_freq = config['save_freq']
    batch_size = config['batch_size']
    collection_multiplier = config.get('collection_multiplier', 4)
    max_node_num = config['max_node_num']
    dagger_init = config['dagger_init']
    dagger_final = config['dagger_final']
    dagger_steps = config['dagger_steps']
    dagger_update_freq = config['dagger_update_freq']

    print(f'TrieParrot: lr={lr}, total_steps={total_steps}, eval_freq={eval_freq}, '
          f'save_freq={save_freq}, batch_size={batch_size}')
    print(f'TrieParrot: DAgger init={dagger_init}, final={dagger_final}, '
          f'steps={dagger_steps}, update_freq={dagger_update_freq}')
    print(f'TrieParrot: max_node_num={max_node_num}')

    # Data paths
    data_dir = os.path.join(args.data_root_dir, args.dataset)
    train_path = os.path.join(data_dir, 'train.pkl')
    valid_path = os.path.join(data_dir, 'valid.pkl')
    test_path = os.path.join(data_dir, 'test.pkl')
    vocab_path = os.path.join(data_dir, 'vocab.json')

    for p in [train_path, vocab_path]:
        if not os.path.exists(p):
            raise ValueError(f'Data file not found: {p}')

    # Read vocab size
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)
    vocab_size = vocab_data['vocab_size']
    print(f'TrieParrot: vocab_size={vocab_size}')

    # Override config vocab_size with actual data vocab_size
    config['vocab_size'] = vocab_size

    # Checkpoint dir
    checkpoint_dir = os.path.join(args.checkpoints_root_dir, 'trie_model', args.dataset)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save effective config
    with open(os.path.join(checkpoint_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Create model
    model = TrieParrotModel(
        vocab_size=vocab_size,
        node_embed_dim=config.get('node_embed_dim', 64),
        history_embed_dim=config.get('history_embed_dim', 64),
        hidden_size=config.get('hidden_size', 128),
        max_attention_history=config.get('max_attention_history', 30),
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f'TrieParrot: {total_params} parameters')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Baseline
    if os.path.exists(valid_path):
        lru_hit_rate = evaluate_lru(valid_path, vocab_path, max_node_num)
        print(f'Baseline LRU hit rate (valid): {lru_hit_rate:.4f}')

    # Training loop
    step = 0
    best_eval_hit_rate = 0.0
    
    with tqdm.tqdm(total=total_steps, desc='Training') as pbar:
        postfix = {
            'loss': 0.0,
            'train_hr': 0.0,
            'eval_hr': 0.0,
            'model_prob': 0.0,
        }
        
        while step < total_steps:
            model_prob = get_model_prob(step, dagger_init, dagger_final, dagger_steps)
            postfix['model_prob'] = f'{model_prob:.2f}'
            
            # Collect DAgger snapshots
            max_examples = dagger_update_freq * collection_multiplier * batch_size
            model.eval()
            snapshots, train_hit_rate = collect_snapshots(
                train_path, vocab_path, model, max_node_num, model_prob, max_examples
            )
            postfix['train_hr'] = f'{train_hit_rate:.4f}'
            
            if not snapshots:
                print('WARNING: No snapshots collected, skipping batch')
                continue
            
            # Train on collected snapshots in mini-batches
            model.train()
            for batch_start in range(0, len(snapshots), batch_size):
                if step >= total_steps:
                    break
                
                batch = snapshots[batch_start:batch_start + batch_size]
                
                # Eval
                if step > 0 and step % eval_freq == 0:
                    model.eval()
                    if os.path.exists(valid_path):
                        eval_hit_rate = evaluate(valid_path, vocab_path, model, max_node_num)
                    else:
                        eval_hit_rate = evaluate(test_path, vocab_path, model, max_node_num)
                    postfix['eval_hr'] = f'{eval_hit_rate:.4f}'
                    
                    if eval_hit_rate > best_eval_hit_rate:
                        best_eval_hit_rate = eval_hit_rate
                        best_path = os.path.join(checkpoint_dir, 'best.ckpt')
                        torch.save(model.state_dict(), best_path)
                        print(f'\n  New best: {eval_hit_rate:.4f}, saved to {best_path}')
                    model.train()
                
                # Save checkpoint
                if step > 0 and step % save_freq == 0:
                    save_path = os.path.join(checkpoint_dir, f'step_{step}.ckpt')
                    torch.save(model.state_dict(), save_path)
                    print(f'\n  Checkpoint saved: {save_path}')
                
                # Forward + backward
                optimizer.zero_grad()
                losses = model.loss(batch)
                total_loss = sum(losses.values())
                total_loss.backward()
                optimizer.step()
                
                postfix['loss'] = f'{total_loss.item():.4f}'
                pbar.set_postfix(postfix)
                pbar.update(1)
                step += 1
                
                if step >= total_steps:
                    break
    
    # Final save
    final_path = os.path.join(checkpoint_dir, f'final_{step}.ckpt')
    torch.save(model.state_dict(), final_path)
    print(f'Training complete. Final checkpoint: {final_path}')
    print(f'Best eval hit rate: {best_eval_hit_rate:.4f}')
