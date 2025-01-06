# cache_alps

A benchmark about Cache Algorithms and Predictor training with Parrot and LightGBM

## Trace

Based on xxx's work, we have reorganized the Brightkite and Cite datasets to conform to a unified format.

You can easily download all traces file, boost traces pickle and GBM Model Checkpoints from the Releases page.

## Environment

We use a environment based on Python 3.8.8 with Anaconda3 (Anaconda3-2021.05-Linux-x86_64.sh), we strongly recommend using Anaconda for Python package management and virtual environment.

You can also install your own environment based on another Python Version but only pay attention that:
- If you want to use the `Parrot` Model, you should install your torch env(in our benchmark, we use CUDA 11.4 to enbale Torch GPU)
- Because of different Python pickle rules, the released `boost-traces.zip` maybe doesn't work in your env, so you need to generate your own boost trace file.

## Usage

### Benchmark

A Simple usage:
```python
python -m benchmark --dataset xalanc --real --pred pleco --boost --boost_fr 
```

You will see the benchmark results of all supported algorithms on the console.

#### All Usages
```python
python -m benchmark [--dataset DATASET] [--test_all] [--device DEVICE] (--oracle | --real)
                   [--pred {parrot,pleco,popu,pleco-bin,gbm,oracle_bin,oracle_dis} [{parrot,pleco,popu,pleco-bin,gbm,oracle_bin,oracle_dis} ...]]
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

  `--test_all`: This option only works for `brightkite` and `citi` datasets, meaning the entire traces will be used in these datasets. It cannot be used with `parrot` or `gbm` predictors as they require a training set.
- Device

  `--device`: Model target device, like `cpu`, `cuda:0` and `cuda:1`...

  This option only works for the `parrot` predcitor, in other cases we only use CPU.
  
- Mode

  `real` and `oracle`: You must specify one of them in command.

  - Real Mode

    Supported Predictors:

    + `parrot`
    + `pleco`
    + `popu`
    + `pleco-bin`
    + `gbm`

    `verbose` default to True

    `boost` will use pickle traces (`num_workers` doesn't work)

    `noise_type` don't work
    
  - Oracle Mode

    Supported Predictors:

    + `oracle_dis`
    + `oracle_bin`
 
    `verbose` default to False

    `boost` will use multiprocess pool (`num_workers` works)

    `boost_preds_dir` `model_fraction` `checkpoints_root_dir` `parrot_config_path` `lightgbm_config_path` don't work

- Noise Type
  + `logdis`: Log Normal Noise on Next Arrival Time (Reuse Distance)
  + `dis`: Normal Noise on Next Arrival Time (Reuse Distance)
  + `bin`: Binary Flipping Noise on Belady's label (FBP label)
  
- Algorithm

  We implemented most of real algorithms in our benchmark, you can also implement and test your own algorithm in the benchmark

  #### Algorithms and Predcitors compatibility

  | Algorithm | Parrot | PLECO | POPU | Pleco-Bin | GBM | Oracle-Dis (Belady) | Oracle-Bin (FBP) |
  |:----------|:------:|:-----:|:----:|:---------:|:---:|:----------:|:----------:|
  | Rand                 | &#10060; | &#10060; | &#10060; | &#10060; | &#10060; | &#10060; | &#10060; |
  | LRU                  | &#10060; | &#10060; | &#10060; | &#10060; | &#10060; | &#10060; | &#10060; |
  | Marker               | &#10060; | &#10060; | &#10060; | &#10060; | &#10060; | &#10060; | &#10060; |
  | Predict              | &#10004; | &#10004; | &#10004; | &#10004; | &#10004; | &#10004; | &#10004; |
  | PredictiveMarker     | &#10004; | &#10004; | &#10004; | &#10060; | &#10060; | &#10004; | &#10060; |
  | LMarker              | &#10004; | &#10004; | &#10004; | &#10060; | &#10060; | &#10004; | &#10060; |
  | LNonMarker           | &#10004; | &#10004; | &#10004; | &#10060; | &#10060; | &#10004; | &#10060; |
  | Follower&Robust      | &#10004; | &#10004; | &#10004; | &#10060; | &#10060; | &#10004; | &#10060; |
  | Mark0                | &#10060; | &#10060; | &#10060; | &#10004; | &#10004; | &#10060; | &#10004; |
  | Mark&Predict         | &#10060; | &#10060; | &#10060; | &#10060; | &#10060; | &#10060; | &#10004; |
  | CombineDeterministic | &#10004; | &#10004; | &#10004; | &#10004; | &#10004; | &#10004; | &#10004; |
  | CombineRandom        | &#10004; | &#10004; | &#10004; | &#10004; | &#10004; | &#10004; | &#10004; |
  | Guard                | &#10004; | &#10004; | &#10004; | &#10004; | &#10004; | &#10004; | &#10004; |

- Predictor

  + `parrot`: A Parrot Imitation Model, gives a page's score when predicting (regarded as reuse distance)
  + `pleco`: A PLECO Model, gives a page's next arrival time (reuse distance) when predicting.
  + `popu`: A Popularity Model, gives a page's next arrival time (reuse distance) when predicting.
  + `pleco-bin`: A PLECO Binary Model based on PLECO, gives a page's **belady's label** when predicting
  + `gbm`: A Grandient Boosting Machine based on Delta and EDC features, gives a page's **belady's label** when predicting
  + `oracle_bin`: An offline predictor that gives the predicted next arrival time (reuse distance) of a page during prediction, potentially affected by noise (logdis or dis).
  + `oracle_dis`: An offline predictor that gives  a page's **belady's label**, potentially affected by noise (logdis, dis or bin).

### Model Training
