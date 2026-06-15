"""
Training script for TrieParrotModel using DAgger (Dataset Aggregation).

Usage:
    python -m model.trie_model --dataset oasst1_timed_global_b16 --device cpu
    python -m model.trie_model --dataset oasst1_timed_global_b16 --device cuda:0
"""
import os
import json
import argparse
import glob

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
    max_requests: int = None,
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
    
    for request_idx, seq in enumerate(all_seqs):
        if max_requests is not None and request_idx >= max_requests:
            break
        cache.collect(seq)
        if max_examples is not None and len(cache.snapshots) >= max_examples:
            break
    
    return cache.get_snapshots(), cache.hit_rate


def evaluate(
    data_path: str,
    vocab_path: str,
    model: TrieParrotModel,
    max_node_num: int,
    max_requests: int = None,
):
    """Evaluate model hit rate on a dataset (pure model policy)."""
    cache = SequenceTrieCache(
        max_node_num=max_node_num, 
        evict_type=TrieModelPredictAlgorithm, 
        model=model,
    )
    with SequenceTrieDataTrace(data_path, vocab_path) as trace:
        request_count = 0
        while not trace.done():
            if max_requests is not None and request_count >= max_requests:
                break
            seq = trace.next()
            cache.access(seq)
            request_count += 1
    
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


def save_training_checkpoint(path: str, model, optimizer, step: int, best_eval_hit_rate: float):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "best_eval_hit_rate": best_eval_hit_rate,
    }, path)


def load_training_checkpoint(path: str, model, optimizer, device):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return int(checkpoint.get("step", 0)), float(checkpoint.get("best_eval_hit_rate", 0.0))

    model.load_state_dict(checkpoint)
    return 0, 0.0


def latest_training_checkpoint(checkpoint_dir: str):
    candidates = glob.glob(os.path.join(checkpoint_dir, "training_step_*.pt"))
    if not candidates:
        return None

    def step_number(path):
        name = os.path.basename(path)
        stem = os.path.splitext(name)[0]
        return int(stem.rsplit("_", 1)[-1])

    return max(candidates, key=step_number)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train TrieParrotModel with DAgger')
    parser.add_argument("--dataset", type=str, default='oasst1_timed_global_b16')
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("-p", "--model_config_path", type=str, 
                        default='checkpoints/trie_model/model_config.json')
    parser.add_argument("--checkpoints_root_dir", type=str, default='checkpoints')
    parser.add_argument("--data_root_dir", type=str, default='data')
    parser.add_argument("--resume_checkpoint_path", type=str, default=None)
    parser.add_argument("--resume_auto", action="store_true")
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
    max_collection_requests = config.get('max_collection_requests')
    max_eval_requests = config.get('max_eval_requests')
    max_loss_candidates = config.get('max_loss_candidates')
    max_loss_steps_per_snapshot = config.get('max_loss_steps_per_snapshot')
    max_node_num = config['max_node_num']
    dagger_init = config['dagger_init']
    dagger_final = config['dagger_final']
    dagger_steps = config['dagger_steps']
    dagger_update_freq = config['dagger_update_freq']

    print(f'TrieParrot: lr={lr}, total_steps={total_steps}, eval_freq={eval_freq}, '
          f'save_freq={save_freq}, batch_size={batch_size}')
    if max_collection_requests is not None or max_loss_candidates is not None:
        print(
            "TrieParrot: collection/loss caps "
            f"max_collection_requests={max_collection_requests} "
            f"max_eval_requests={max_eval_requests} "
            f"max_loss_candidates={max_loss_candidates} "
            f"max_loss_steps_per_snapshot={max_loss_steps_per_snapshot}"
        )
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

    # Resume after optimizer creation so optimizer state can be restored.
    step = 0
    best_eval_hit_rate = 0.0
    resume_path = args.resume_checkpoint_path
    if args.resume_auto and resume_path is None:
        resume_path = latest_training_checkpoint(checkpoint_dir)
    if resume_path:
        step, best_eval_hit_rate = load_training_checkpoint(
            resume_path,
            model,
            optimizer,
            device,
        )
        print(
            f"TrieParrot: resumed from {resume_path} "
            f"at step={step} best_eval_hit_rate={best_eval_hit_rate:.4f}"
        )

    # Baseline
    if os.path.exists(valid_path):
        lru_hit_rate = evaluate_lru(valid_path, vocab_path, max_node_num)
        print(f'Baseline LRU hit rate (valid): {lru_hit_rate:.4f}')

    # Training loop
    remaining_steps = max(total_steps - step, 0)
    with tqdm.tqdm(total=remaining_steps, desc='Training') as pbar:
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
                train_path,
                vocab_path,
                model,
                max_node_num,
                model_prob,
                max_examples,
                max_collection_requests,
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
                        eval_hit_rate = evaluate(
                            valid_path,
                            vocab_path,
                            model,
                            max_node_num,
                            max_eval_requests,
                        )
                    else:
                        eval_hit_rate = evaluate(
                            test_path,
                            vocab_path,
                            model,
                            max_node_num,
                            max_eval_requests,
                        )
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
                    training_state_path = os.path.join(
                        checkpoint_dir,
                        f'training_step_{step}.pt',
                    )
                    save_training_checkpoint(
                        training_state_path,
                        model,
                        optimizer,
                        step,
                        best_eval_hit_rate,
                    )
                    print(f'\n  Checkpoint saved: {save_path}')
                
                # Forward + backward
                optimizer.zero_grad()
                losses = model.loss(
                    batch,
                    max_candidates=max_loss_candidates,
                    max_steps_per_snapshot=max_loss_steps_per_snapshot,
                )
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
    final_training_path = os.path.join(checkpoint_dir, f'training_final_{step}.pt')
    save_training_checkpoint(
        final_training_path,
        model,
        optimizer,
        step,
        best_eval_hit_rate,
    )
    print(f'Training complete. Final checkpoint: {final_path}')
    print(f'Best eval hit rate: {best_eval_hit_rate:.4f}')
