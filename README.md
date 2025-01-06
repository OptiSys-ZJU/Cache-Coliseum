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


### Model Training