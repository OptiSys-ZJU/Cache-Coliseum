from data_trace.data_trace import DataTrace
from model.parrot import utils, model
from model.models import ParrotModel
from utils.aligner import ShiftAligner
from cache.hash import ShiftHashFunction
from cache.cache import TrainingCache
from cache.evict import *
import os
import torch
from torch.nn.parallel import DataParallel
import tensorflow._api.v2.compat.v1 as tf
import io
import logging
import json
import tqdm
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()

    file_path = f'traces/{args.dataset}_train.csv'

    cache_line_size = 64
    capacity = 2097152
    associativity = 16
    align_type = ShiftAligner
    hash_type = ShiftHashFunction
#################################################################################################
    total_steps = 1e6

    batch_size = 32
    collection_multiplier = 5
    
    dagger_init = 0.0
    dagger_final = 1.0
    dagger_steps = 200000
    dagger_update_freq = 10000
    
    exp_root_dir = 'tmp'
    save_freq = 20000
    tb_freq = 100

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
    device = torch.device("cpu")
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        device = torch.device("cuda:0")
    print("Device:", device, flush=True)
    with open(os.path.join(exp_root_dir, "model_config.json"), "r") as f:
        model_config = json.load(f)
    parrot_model = ParrotModel(model_config, device)
    optimizer = torch.optim.Adam(parrot_model._model.parameters(), lr=model_config.get("lr"))
#################################################################################################
    def get_model_prob(get_step_lambda):
        fraction = min(float(get_step_lambda()) / dagger_steps, 1.0)
        model_prob = dagger_init + fraction * (dagger_final - dagger_init)
        return model_prob
    
    def generate_snapshots(file_path: str, get_step_lambda, max_examples=None):
        if max_examples is None:
            max_examples = np.inf
        cache = TrainingCache(file_path, align_type, evict_type, hash_type, cache_line_size, capacity, associativity)
        with DataTrace(file_path) as trace:
            with tqdm.tqdm(desc='Collecting data on DataTrace') as pbar:
                while not trace.done():
                    model_prob = get_model_prob(get_step_lambda)
                    pbar.set_postfix({"Model Prob": model_prob})
                    cache.reset(model_prob)
                    data = []
                    while len(data) < max_examples and not trace.done():
                        pc, address = trace.next()
                        snapshot = cache.snapshot(pc, address)
                        data.append(snapshot)
                        pbar.update(1)
                    yield data


    evict_type = partial(CombineWeightsAlgorithm, 
                         candidate_algorithms=[BeladyAlgorithm, 
                                               partial(ParrotAlgorithm, shared_model=parrot_model)], 
                         weights=[1, 0], lazy_evictor_type=None)

    step = 0
    get_step = lambda: step
    with tqdm.tqdm(total=total_steps, desc='Training Process: ') as pbar:
        while True:
            max_examples = (dagger_update_freq * collection_multiplier * batch_size)
            train_data_generator = generate_snapshots(file_path, get_step, max_examples)
            for train_data in train_data_generator:                
                for batch_num, batch in enumerate(utils.as_batches([train_data], batch_size, model_config.get("sequence_length"))):
                    if step % save_freq == 0 and step != 0:
                        save_path = os.path.join(res_dir, "{}.ckpt".format(step))
                        with open(save_path, "wb") as save_file:
                            checkpoint_buffer = io.BytesIO()
                            torch.save(parrot_model._model.state_dict(), checkpoint_buffer)
                            print(f"Saving model checkpoint to: {save_path}", flush=True)
                            save_file.write(checkpoint_buffer.getvalue())
                    optimizer.zero_grad()
                    losses = parrot_model._model.loss(batch, model_config.get("sequence_length") // 2)
                    #losses = parrot_model._model.loss(batch, 1)
                    total_loss = sum(losses.values())
                    total_loss.backward()
                    optimizer.step()
                    pbar.update(1)
                    step += 1
                    pbar.set_postfix({"loss/total": total_loss.item()})
                    # for loss_name, loss_value in losses.items():
                    #     pbar.set_postfix({f"loss/{loss_name}": loss_value})
                    if step == total_steps:
                        exit(0)
                    # Break out of inner-loop to get next set of k * update_freq batches
                    if batch_num == dagger_update_freq:
                        break