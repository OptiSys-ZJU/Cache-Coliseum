from data_trace.data_trace import DataTrace
from model.parrot import utils
from model.models import ParrotModel
from model import device_manager
from utils.aligner import ShiftAligner
from cache.hash import ShiftHashFunction
from cache.cache import TrainingCache
from cache.evict import *
from typing import Callable
import os
import torch
import io
import tqdm
import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='xalanc')
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--eval", action='store_true')
    args = parser.parse_args()
    device_manager.set_device(args.device)

    train_file_path = f'traces/{args.dataset}_train.csv'
    valid_file_path = f'traces/{args.dataset}_valid.csv'
    test_file_path = f'traces/{args.dataset}_test.csv'

    cache_line_size = 64
    capacity = 2097152
    associativity = 16
    align_type = ShiftAligner
    hash_type = ShiftHashFunction
#################################################################################################
    total_steps = 120000

    lr = 0.001

    batch_size = 32
    collection_multiplier = 5
    
    dagger_init = 1
    dagger_final = 1
    dagger_steps = 200000
    dagger_update_freq = 50000
    
    exp_root_dir = 'tmp'
    eval_freq = 5000
    save_freq = 5000

    res_dir = os.path.join(exp_root_dir, args.dataset)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)


    # total_steps = 1000
    # batch_size = 8
    # collection_multiplier = 2
    # dagger_steps = 800
    # dagger_update_freq = 5
    # save_freq = 200
#################################################################################################
    model_config_path = os.path.join(exp_root_dir, "model_config.json")
    with open(model_config_path, "r") as f:
        model_config = json.load(f)
    parrot_model = ParrotModel.from_config(model_config_path, None)
    optimizer = torch.optim.Adam(parrot_model._model.parameters(), lr=lr)
#################################################################################################
    evict_type = partial(CombineWeightsAlgorithm, 
                         candidate_algorithms=[PredictAlgorithmFactory.generate_predictive_algorithm(PredictAlgorithm, 'OracleDis'), 
                                               PredictAlgorithmFactory.generate_predictive_algorithm(PredictAlgorithm, 'Parrot', shared_model=parrot_model)], 
                         weights=[1, 0], lazy_evictor_type=None)
    
    def get_model_prob(get_step_lambda):
        fraction = min(float(get_step_lambda()) / dagger_steps, 1.0)
        model_prob = dagger_init + fraction * (dagger_final - dagger_init)
        return model_prob

    def generate_snapshots(file_path: str, max_examples=None, model_prob_gen=lambda:0):
        if max_examples is None:
            max_examples = np.inf
        cache = TrainingCache(file_path, align_type, evict_type, hash_type, cache_line_size, capacity, associativity)
        with DataTrace(file_path) as trace:
            with tqdm.tqdm(desc=f'Collecting data on DataTrace [{file_path}]') as pbar:
                while not trace.done():
                    model_prob = model_prob_gen()
                    pbar.set_postfix({"Model Prob": model_prob})
                    cache.reset(model_prob)
                    data = []
                    total_cnt = 0
                    hit_cnt = 0
                    while len(data) < max_examples and not trace.done():
                        pc, address = trace.next()
                        snapshot, hit = cache.snapshot(pc, address)
                        if hit:
                            hit_cnt += 1
                        total_cnt += 1
                        data.append(snapshot)
                        pbar.update(1)
                    
                    if total_cnt == 0:
                        hit_rate = 0
                    else:
                        hit_rate = hit_cnt / total_cnt
                    yield data, hit_rate

    oracle_data, oracle_hit_rate = next(generate_snapshots(valid_file_path))
    print('oracle hit rate: ', oracle_hit_rate, flush=True)

    step = 0
    get_step = lambda: step
    eval_hit_rate = 0
    with tqdm.tqdm(total=total_steps, desc='Training Process: ') as pbar:
        postfix_dict = {
            "train_hit_rate": 0,
            "eval_now_hit_rate": 0,
            "eval_oracle_hit_rate": oracle_hit_rate,
            "loss/total": 0,
        }
        
        while True:
            max_examples = (dagger_update_freq * collection_multiplier * batch_size)
            train_data_generator = generate_snapshots(train_file_path, max_examples, partial(get_model_prob, get_step_lambda=get_step))
            for train_data, train_hit_rate in train_data_generator:
                postfix_dict['train_hit_rate'] = train_hit_rate
                for batch_num, batch in enumerate(utils.as_batches([train_data], batch_size, model_config.get("sequence_length"))):
                    if step % eval_freq == 0 and step != 0:
                        eval_data, eval_hit_rate = next(generate_snapshots(valid_file_path, None, lambda:1))
                        postfix_dict['eval_now_hit_rate'] = eval_hit_rate
                    
                    if step % save_freq == 0 and step != 0:
                        save_path = os.path.join(res_dir, f"{step}_{eval_hit_rate}.ckpt")
                        with open(save_path, "wb") as save_file:
                            checkpoint_buffer = io.BytesIO()
                            torch.save(parrot_model._model.state_dict(), checkpoint_buffer)
                            print(f"Saving model checkpoint to: {save_path}", flush=True)
                            save_file.write(checkpoint_buffer.getvalue())
                    optimizer.zero_grad()
                    losses = parrot_model._model.loss(batch, model_config.get("sequence_length") // 2)
                    total_loss = sum(losses.values())
                    total_loss.backward()
                    optimizer.step()
                    pbar.update(1)
                    step += 1
                    postfix_dict["loss/total"] = total_loss.item()
                    pbar.set_postfix(postfix_dict)
                    # for loss_name, loss_value in losses.items():
                    #     pbar.set_postfix({f"loss/{loss_name}": loss_value})
                    if step == total_steps:
                        exit(0)
                    if batch_num == dagger_update_freq:
                        break