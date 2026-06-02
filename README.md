# Cache-Coliseum

[![Paper](https://img.shields.io/badge/arxiv-2507.16242-1)](https://arxiv.org/abs/2507.16242)
[![nips](https://img.shields.io/badge/NeurIPS%202025-Poster-blueviolet)](https://neurips.cc/virtual/2025/poster/116615)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black)](https://github.com/OptiSys-ZJU/cache-coliseum)

A benchmark for evaluating learning-augmented caching algorithms, featuring multiple algorithm implementations and a variety of predictors. This benchmark is used in the following paper:

- NeurIPS'25 [Robustifying Learning-Augmented Caching Efficiently without Compromising 1-Consistency](https://papers.nips.cc/paper_files/paper/2025/hash/b1e7f61f40d68b2177857bfcb195a507-Abstract-Conference.html) [[arXiv](https://arxiv.org/pdf/2507.16242)]

## Experimental Code & Acknowledgements

The experimental scripts, training/evaluation pipelines, data preprocessing, and result analysis were implemented and maintained by [Jiaji Zhang](https://github.com/Freesia810).

Some parts of the project code were inspired by or adapted from existing open-source repositories, including:
- [cache_replacement](https://github.com/google-research/google-research/tree/master/cache_replacement)
- [ML caching with guarantees](https://github.com/chledowski/ml_caching_with_guarantees)

The algorithm design and refinement of Guard were contributed by [Peng Chen](https://github.com/Natureal).

## Traces and Checkpoints

We provide SPEC CPU2006 memory traces in the repository, sourced from the Dropbox of [ML caching with guarantees](https://github.com/chledowski/ml_caching_with_guarantees)<sup>[1]</sup>. Each trace is located in the `traces/` directory, e.g. `traces/xalanc/xalanc_test.csv`. In addition, we reorganize the Brightkite and Citi datasets into the same format; for these two datasets a `*_all.csv` is also provided and is used by `--test_all` to evaluate on the entire trace instead of the test split.

We also ship the **boost traces** under `boost_traces/` and the pre-trained **GBM / Parrot checkpoints** under `checkpoints/`, so the benchmark can be run out of the box with no additional downloads. The boost traces are the per-step predictor outputs consumed by `--boost` in `real` mode (see the **Boost** subsection in Usage below): one pickle per `(dataset, predictor, model_fraction)` triple, e.g. `boost_traces/astar_PLECO_1.pkl`. Their presence lets `--boost` skip the one-time predictor pre-pass entirely. If a pickle is missing — or if the shipped one fails to load on your Python/library setup (see Environment below) — it will be regenerated and saved on first use, which for `gbm` / `parrot` requires the corresponding checkpoint under `checkpoints/`.

> **Note on Parrot checkpoints:** Due to the large size of Parrot model checkpoints, the `checkpoints/` directory in this repository only includes pre-trained LRB (LightGBM) models. To test the Parrot predictor, download the full `checkpoints` archive from the [Releases page](https://github.com/OptiSys-ZJU/Cache-Coliseum/releases), extract it, and replace the `checkpoints/` folder in the repository.

## Environment

We use an environment based on Python 3.10.16 with Anaconda3 (Anaconda3-2021.05-Linux-x86_64.sh), we strongly recommend using Anaconda for Python package management and virtual environment.

You can also install your own environment based on another Python Version but only pay attention that:
- If you want to use the `Parrot` Model, you should install your torch env(in our benchmark, we use CUDA 12.4 to enable Torch GPU)
- Because of different Python pickle rules, the boost traces shipped under `boost_traces/` may fail to load on a different Python/library setup. In that case, delete the affected pickles and the benchmark will regenerate them locally on first use.

## Usage

### Benchmark

Quick start with different predictors on the `xalanc` dataset, e.g.,:

```bash
python -m benchmark --dataset xalanc --real --pred pleco --boost --boost_fr

python -m benchmark --dataset xalanc --real --pred lrb --boost --boost_fr

# Parrot predictor (requires Parrot checkpoint under checkpoints/parrot/, see note above)
python -m benchmark --dataset xalanc --real --pred parrot --boost --boost_fr --device cuda:0

# Oracle with reuse-distance noise
python -m benchmark --dataset xalanc --oracle --noise_type dis

# Oracle with binary-flipping noise
python -m benchmark --dataset xalanc --oracle --noise_type bin
```

You will see the benchmark results of algorithms on the console. Add `--dump_file` to save results as CSV files under `stat/` (e.g., `stat/xalanc_lrb_1.csv`).

To run all SPEC CPU2006 datasets at once, use the batch scripts under `scripts/`, e.g.,:

```bash
# Real-mode predictors (all 13 datasets, parallel)
scripts/run_pleco.sh          # PLECO
scripts/run_lrb.sh            # LRB (default fraction=1)
scripts/run_parrot.sh         # Parrot (default fraction=1)
```

Afterward (e.g., via `scripts/run_lrb.sh`), the following command automatically aggregates the per-dataset results into a single table:

```bash
# Aggregate LRB results (fraction=1)
python scripts/aggregate_results.py --name lrb
```

#### All Usages
```python
python -m benchmark [--dataset DATASET] [--test_all] [--device DEVICE] (--oracle | --real)
                   [--pred {parrot,pleco,popu,pleco-bin,lrb,oracle_bin,oracle_dis}]
                   [--noise_type {dis,bin,logdis}] [--dump_file] [--output_root_dir OUTPUT_ROOT_DIR] [--verbose]
                   [--boost] [--num_workers NUM_WORKERS] [--boost_fr] [--boost_preds_dir BOOST_PREDS_DIR]
                   [--model_fraction MODEL_FRACTION] [--checkpoints_root_dir CHECKPOINTS_ROOT_DIR]
                   [--parrot_config_path PARROT_CONFIG_PATH] [--lightgbm_config_path LIGHTGBM_CONFIG_PATH]
```

- Dataset

  `--dataset`: Benchmark source dataset name, supported name:
    + `brightkite`
    + `citi`
    + `astar`
    + `bwaves`
    + `bzip`
    + `cactusadm`
    + `gems`
    + `lbm`
    + `leslie3d`
    + `libq`
    + `mcf`
    + `milc`
    + `omnetpp`
    + `sphinx3`
    + `xalanc`
    
    You can also support your own dataset.

  `--split`: Selects which trace split to use (`test`, `train`, `valid`, or `all`). Defaults to `test` for SPEC CPU2006 datasets, and automatically uses `all` for `brightkite` and `citi` (which don't require a train/test split).
- Device

  `--device`: Model target device, like `cpu`, `cuda:0` and `cuda:1`...

  This option only works for the `parrot` predictor, in other cases we only use CPU.
  
- Mode

  Exactly one of `--real` / `--oracle` is required.

  - **Real Mode** — supports `parrot`, `pleco`, `popu`, `pleco-bin`, `gbm`. `--verbose` defaults to True; `--noise_type` is ignored.
  - **Oracle Mode** — supports `oracle_dis`, `oracle_bin`. `--verbose` defaults to False; `--noise_type` selects how the oracle is perturbed. Model-loading flags (`--model_fraction`, `--checkpoints_root_dir`, `--parrot_config_path`, `--lightgbm_config_path`) and `--boost_preds_dir` are ignored since no predictor checkpoint is loaded.

  `--boost` and `--num_workers` behave differently in the two modes; see the Boost subsection below.

- Noise Type
  + `logdis`: Log Normal Noise on Next Request Time (Reuse Distance)
  + `dis`: Normal Noise on Next Request Time (Reuse Distance)
  + `bin`: Binary Flipping Noise on Belady's label (FBP label)
  
- Algorithm

  We implemented most of the existing algorithms in our benchmark, you can also implement and test your own algorithm in the benchmark.

  #### Algorithms and Predictors compatibility

  | Algorithm                          | PLECO    |   POPU   |  Parrot  |   LRB    |Oracle-Dis|Oracle-Bin|
  |:----------|:------:|:-----:|:---------:|:---:|:----------:|:----------:|
  | Rand                               | &#10060; | &#10060; | &#10060; | &#10060; | &#10060; | &#10060; |
  | LRU                                | &#10060; | &#10060; | &#10060; | &#10060; | &#10060; | &#10060; |
  | Marker                             | &#10060; | &#10060; | &#10060; | &#10060; | &#10060; | &#10060; |
  | Predict (BlindOracle)              | &#10004; | &#10004; | &#10004; | &#10004; | &#10004; | &#10004; |
  | PredictiveMarker<sup>[2]</sup>     | &#10004; | &#10004; | &#10060; | &#10060; | &#10004; | &#10060; |
  | LMarker<sup>[3]</sup>              | &#10004; | &#10004; | &#10060; | &#10060; | &#10004; | &#10060; |
  | LNonMarker<sup>[3]</sup>           | &#10004; | &#10004; | &#10060; | &#10060; | &#10004; | &#10060; |
  | Follower&Robust(F&R)<sup>[4]</sup> | &#10004; | &#10004; | &#10060; | &#10060; | &#10004; | &#10060; |
  | Mark0<sup>[5]</sup>                | &#10060; | &#10060; | &#10060; | &#10004; | &#10060; | &#10004; |
  | Mark&Predict<sup>[5]</sup>         | &#10060; | &#10060; | &#10060; | &#10060; | &#10060; | &#10004; |
  | CombineDeterministic<sup>[6]</sup> | &#10004; | &#10004; | &#10004; | &#10004; | &#10004; | &#10004; |
  | CombineRandom<sup>[6]</sup>        | &#10004; | &#10004; | &#10004; | &#10004; | &#10004; | &#10004; |
  | The Guard Framework                | &#10004; | &#10004; | &#10004; | &#10004; | &#10004; | &#10004; |
  | Guard&BlindOracle                  | &#10004; | &#10004; | &#10060; | &#10060; | &#10004; | &#10060; |
  | Guard&LRB                          | &#10060; | &#10060; | &#10060; | &#10004; | &#10060; | &#10004; |
  | Guard&Parrot                       | &#10060; | &#10060; | &#10004; | &#10060; | &#10060; | &#10060; |

- Predictor

  `--pred` accepts one or more predictor names (it is a multi-valued argument). If omitted, the benchmark runs every predictor compatible with the current mode — all real predictors under `--real`, all oracle predictors under `--oracle`.

  | Flag         | Output                                                  | Needs checkpoint              |
  |:-------------|:--------------------------------------------------------|:------------------------------|
  | `pleco`      | next request time (NRT) from a PLECO model              | no                            |
  | `popu`       | next request time (NRT) from a popularity model         | no                            |
  | `pleco-bin`  | Belady binary label from PLECO with a probability threshold (default 0.5) | no |
  | `lrb`        | Belady binary label from a GBM on Delta / EDC features  | yes (`checkpoints/lightgbm/`) |
  | `parrot`     | per-page eviction priority from a Parrot imitation model | yes (`checkpoints/parrot/`)  |
  | `oracle_dis` | offline NRT, optionally perturbed by `dis` noise        | no                            |
  | `oracle_bin` | offline Belady binary label, optionally perturbed by `logdis` / `dis` noise | no |

- Dump and Verbose

  `dump_file`: Enables result dumping. Results are written to `$output_root_dir/<dataset>_<predictor>_<model_fraction>.csv` in `real` mode, and to `$output_root_dir/<dataset>_<noise_type>_<model_fraction>.csv` in `oracle` mode.

  `verbose`: Enables detailed output. When enabled, all statistical data (hits, misses, hit rates, and competitive ratios) will be displayed; otherwise, only the hit rate and competitive ratio will be shown.
  
- Boost

  `--boost` and `--boost_fr` are two independent speed-ups; either or both can be enabled.

  - **`--boost`** — shares predictor work across algorithms.
    - `real` mode: each predictor is run over the trace once, and its per-step outputs are pickled into `boost_preds_dir` (default `boost_traces/`). All algorithms in the run that share that predictor (and any subsequent run) read from the pickle instead of re-running the model. Effectively required for `parrot` and `gbm`, whose inference dominates wall-clock time.
    - `oracle` mode: no pickle. Every algorithm's cache is instead evaluated in parallel via a multiprocess pool of size `--num_workers` (default `-1` = all cores).

  - **`--boost_fr`** — speeds up the Follower & Robust algorithm. By default F&R recomputes the offline Belady cache from the remaining future trace at every step (quadratic, and dominant whenever F&R is in the run). With this flag, all Belady snapshots are precomputed in a single pre-pass and looked up in O(1). Recommended whenever the run includes F&R (i.e., with `pleco`, `popu`, `parrot`, or `oracle_dis`).
  
- Model Config

  To use the Parrot or GBM predictors, you must specify the following parameters (or use the defaults):

  + `model_fraction` (default to 1): Specifies the fraction of the training set to be used for training. (In benchmarking, you must select a fraction corresponding to an existing pre-trained model.)
  + `checkpoints_root_dir`: Checkpoints dir.
  + `parrot_config_path`: The `parrot` predictor's config file.
  + `lightgbm_config_path`: The `gbm` predictor's config file.


### Model Training

#### GBM Model

```python
python -m model.lightgbm [--dataset DATASET] [--device DEVICE]
                    [--model_fraction MODEL_FRACTION] [--model_config_path MODEL_CONFIG_PATH]
                    [--checkpoints_root_dir CHECKPOINTS_ROOT_DIR] [--traces_root_dir TRACES_ROOT_DIR]
                    [--iter_threshold] [--real_test]
```

You can use our GBM training tool to generate and test the cache model.

- Params:
  + `dataset`: Source dataset
  + `device`: Model load device (in GBM, only `cpu` can work)
  + `model_fraction`: The proportion of the training set used for training. The default value is 1, indicating that the entire training set will be used.
  + `model_config_path`: Model's config
  + `checkpoints_root_dir`: The directory where the checkpoint files are saved after training is completed.
  + `traces_root_dir`: The directory where the datasets are stored.
  + `iter_threshold`: When enabled, this iteratively adjusts the binary splitting threshold from 0 to 1 to determine the optimal threshold for binary classification. If disabled, the threshold defaults to 0.5.
  + `real_test`: After training is completed, the test set will be loaded to evaluate the model's performance.

- Config Example:
  ```json
  {
      "delta_nums": 10,
      "edc_nums": 10,
      "training": {
          "boosting_type": "gbdt", 
          "objective": "binary", 
          "metric": "l2",
          "learning_rate": 0.01,
          "num_boost_round": 8000,
          "num_leaves": 31, 
          "max_depth": 6,
          "subsample": 0.8, 
          "colsample_bytree": 0.8
      }
  }
  ```

  `delta_nums` and `edc_nums`: The dimensions of the **Delta** and **EDC** features, respectively. For more details, refer to [webcachesim2](https://github.com/sunnyszy/lrb) and [Learning relaxed Belady for content distribution network caching](https://dl.acm.org/doi/10.5555/3388242.3388281)<sup>[7]</sup>.

  `training`: The training parameters that will be used by LightGBM.

#### Parrot Model

```python
python -m model.parrot [--dataset DATASET] [--device DEVICE]
                    [--model_fraction MODEL_FRACTION] [--model_config_path MODEL_CONFIG_PATH]
                    [--checkpoints_root_dir CHECKPOINTS_ROOT_DIR] [--traces_root_dir TRACES_ROOT_DIR]
                    [--real_test]
```

- Params:
  + `dataset`: Source dataset
  + `device`: Model load device (Parrot use `CUDA_VISIBLE_DEVICES` to set target device)
  + `model_fraction`: The proportion of the training set used for training. The default value is 1, indicating that the entire training set will be used.
  + `model_config_path`: Model's config
  + `checkpoints_root_dir`: The directory where the checkpoint files are saved after training is completed.
  + `traces_root_dir`: The directory where the datasets are stored.
  + `real_test`: After training is completed, the test set will be loaded to evaluate the model's performance.
 
- Config Example:
  ```json
  {
    "address_embedder": {
        "embed_dim": 64,
        "max_vocab_size": 5000,
        "type": "dynamic-vocab"
    },
    "cache_line_embedder": "address_embedder",
    "cache_pc_embedder": "none",
    "loss": [
        "ndcg",
        "reuse_dist"
    ],
    "lstm_hidden_size": 128,
    "max_attention_history": 30,
    "pc_embedder": {
        "embed_dim": 64,
        "max_vocab_size": 5000,
        "type": "dynamic-vocab"
    },
    "positional_embedder": {
        "embed_dim": 128,
        "type": "positional"
    },
    "sequence_length": 80,
    "lr": 0.001,
    "total_steps": 25000,
    "eval_freq": 5000,
    "save_freq": 5000,
    "batch_size": 32,
    "collection_multiplier": 5,
    "dagger_init": 1,
    "dagger_final": 1,
    "dagger_steps": 200000,
    "dagger_update_freq": 50000
  }
  ```
  
  You can find more details about Parrot and its training in [cache_replacement](https://github.com/google-research/google-research/tree/master/cache_replacement)<sup>[8]</sup>.


### Script

We also provide some scripts for training and testing, for reference purposes only. You can find them in the `scripts` folder.


## Citation

Please cite the following paper if you find this benchmark useful. Feel free to contact [Peng Chen](https://github.com/Natureal) at pgchen@zju.edu.cn for any questions.

```bibtex
@misc{chen2025robustifyinglearningaugmentedcachingefficiently,
      title={Robustifying Learning-Augmented Caching Efficiently without Compromising 1-Consistency},
      author={Peng Chen and Hailiang Zhao and Jiaji Zhang and Xueyan Tang and Yixuan Wang and Shuiguang Deng},
      year={2025},
      eprint={2507.16242},
      archivePrefix={arXiv},
      primaryClass={cs.DS},
      url={https://arxiv.org/abs/2507.16242},
}
```

## References

[1] Chłędowski, Jakub Polak, Adam Szabucki, Bartosz Zolna, Konrad. 2021. Robust Learning-Augmented Caching: An Experimental Study. 10.48550/arXiv.2106.14693. 

[2] Thodoris Lykouris and Sergei Vassilvitskii. 2021. Competitive Caching with Machine Learned Advice. J. ACM 68, 4, Article 24 (August 2021), 25 pages. https://doi.org/10.1145/3447579

[3] Dhruv Rohatgi. 2020. Near-optimal bounds for online caching with machine-learned advice. In Proceedings of the Thirty-First Annual ACM-SIAM Symposium on Discrete Algorithms (SODA '20). Society for Industrial and Applied Mathematics, USA, 1834–1845.

[4] Sadek, K. A. and Elias, M. Algorithms for caching and MTS with reduced number of predictions. arXiv preprint arXiv:2404.06280, 2024.

[5] Antonios Antoniadis, Joan Boyar, Marek Eliáš, Lene M. Favrholdt, Ruben Hoeksma, Kim S. Larsen, Adam Polak, and Bertrand Simon. 2023. Paging with succinct predictions. In Proceedings of the 40th International Conference on Machine Learning (ICML'23), Vol. 202. JMLR.org, Article 39, 952–968.

[6] Wei, A. Better and simpler learning-augmented online caching. arXiv preprint arXiv:2005.13716, 2020.

[7] Zhenyu Song, Daniel S. Berger, Kai Li, and Wyatt Lloyd. 2020. Learning relaxed Belady for content distribution network caching. In Proceedings of the 17th USENIX Conference on Networked Systems Design and Implementation (NSDI'20). USENIX Association, USA, 529–544.

[8] Evan Zheran Liu, Milad Hashemi, Kevin Swersky, Parthasarathy Ranganathan, and Junwhan Ahn. 2020. An imitation learning approach for cache replacement. In Proceedings of the 37th International Conference on Machine Learning (ICML'20), Vol. 119. JMLR.org, Article 579, 6237–6247.


