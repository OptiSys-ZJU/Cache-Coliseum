# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在本仓库工作时提供指引。

## 沟通约定

在本项目中与用户沟通时,一律使用中文。

## 本仓库是什么

Cache-Coliseum 是一个研究**学习增强型缓存淘汰(learning-augmented cache eviction)**的基准测试项目。它包含两个基本独立的模拟器,二者共享部分基础组件(`cache.evict`、`cache.hash`、`utils.aligner`):

1. **经典分页模拟器**(`cache/cache.py`、`benchmark/__main__.py`)——基于 `(pc, address)` 内存/CDN 访问轨迹的组相联缓存。这是 `README.md` 中描述的、原始 `main` 分支的系统,用于将 LRU、Marker、PredictiveMarker、Follower&Robust 等算法与 ML 预测器(Parrot、LightGBM/GBM、PLECO)进行对比。
2. **前缀树 KV 缓存模拟器**(`cache/trie/`、`benchmark/trie_kv.py`)——当前 `trie-cache` 分支的重点。它对多轮 LLM 服务建模:每个请求是一串 KV block id 序列,被缓存的 prompt 前缀对应前缀树中从根到某节点的路径,淘汰则是删除叶子 block。完整设计见 `DESIGN_DOCUMENT.md`——在改动 trie 相关代码前请先阅读它。

当任务提到序列、KV block、前缀、`max_node_num` 或 LLM 服务时,属于 trie 系统;当任务提到 `pc`/`address`、相联度(associativity)、组(set),或 README 表格中的数据集(brightkite、xalanc 等)时,属于经典系统。

## 常用命令

本项目没有构建步骤、linter 配置或测试框架。测试是直接运行的普通脚本,大多需要 `PYTHONPATH=.`(仓库根目录)才能导入相应包。

```bash
# Trie / KV 缓存测试(当前分支)
python tests/test_prefix_oracle.py
PYTHONPATH=. python tests/test_seq_cache.py
PYTHONPATH=. python tests/test_evict.py
PYTHONPATH=. python tests/test_training_cache.py

# 仅做语法检查(torch 不可用时很有用)
python -m py_compile cache/trie/oracle.py cache/trie/trie_algorithms.py cache/trie/trie_cache.py
```

每个 `tests/test_*.py` 都是独立脚本,包含 `print(...)`/`assert` 检查和 `__main__` 块——直接运行该文件即可,没有 pytest 框架。当脚本打印出成功信息并以 0 退出时,即视为"通过"。

### 运行 trie KV 基准测试

```bash
python -m benchmark.trie_kv --dataset oasst1_timed_global_b16 --split valid \
  --capacity 256 512 1024 2048 4096 --policy lru rand oracle \
  [--output_csv res/oasst1_timed_global_b16_valid_kv.csv]
```
`--data_root_dir`(默认 `data`)目录下必须有 `<dataset>/<split>.pkl`(pickle 序列化的 `List[List[int]]`)以及 `metadata.json`。`model`/`guard` 策略还额外需要 `--model_config_path`、`--model_checkpoint_path`、`--device` 以及已安装的 torch。

### 运行经典基准测试

```bash
python -m benchmark --dataset xalanc --real --pred pleco --boost --boost_fr
```
读取 `traces/<dataset>/<dataset>_test.csv`。完整的参数矩阵以及算法/预测器兼容性表见 `README.md`。`scripts/*.sh` 是供参考的批量运行封装脚本(如 `run_bin.sh`、`train_parrot.sh`)。

### 数据预处理

```bash
# OASST1 timed KV 轨迹 -> data/<dataset>/{train,valid}.pkl + vocab.json + metadata.json
python scripts/data_process/preprocess_oasst1_timed.py --output_dir data/oasst1_timed_global_b16 --block_token_size 16 --identity_scope global --event_role all
```
其他预处理脚本:`preprocess_yoochoose.py`、`scripts/oasst1.py`、`scripts/make_small_dataset.py`。

### 环境

Python 3.10(推荐 Anaconda)。`torch` 和 `lightgbm` 是较重的可选依赖:trie 的 LRU/Rand/Oracle 路径以及 `cache/trie/trie_algorithms.py` 都采用懒加载导入 torch(`try: import torch except ModuleNotFoundError`),以便在没有 torch 时也能运行非模型的模拟器。**请保留这种可选性**——不要在 trie 模拟或 evict 核心中加入无条件的 torch/lightgbm 导入。

## 架构

### 共享的淘汰核心(`cache/evict/`)

经典系统由三个可插拔的部件组合而成,在构造时通过 `partial(...)` 按类型装配:

- **EvictAlgorithm**(`algorithms.py`)——管理一个缓存组(大小 = 相联度)。其中关键的是 `PredictAlgorithm`:它持有一个 **Evictor** 和一个 **Predictor**,并通过检查预测器的基类来决定预测语义(`ReuseDistancePredictor` → 重用距离浮点数,`BinaryPredictor` → Belady 0/1 标签,等等)。若预测器是 `OraclePredictor` 的子类,则会动态挂上 `oracle_access` 方法。
- **Evictor**(`evictor.py`)——在 `(index, score)` 候选项上执行的纯策略:`MinEvictor`/`LRUEvictor`、`MaxEvictor`/`ReuseDistanceEvictor`、`BinaryEvictor`、`MarkerEvictor`、`RandEvictor`。
- **Predictor**(`predictor.py`)——为每条缓存行产生分数;oracle 预测器会在对轨迹的预扫描阶段被预先填充。

`Cache`(`cache/cache.py`)将容量切分为 `num_sets` 个相互独立的 `EvictAlgorithm` 实例,通过 `HashFunction.get_bucket_index(addr, pc)` 索引。一旦通过 `hasattr(alg, 'oracle_access')` 检测到 oracle 能力,就会触发对整条轨迹的离线预扫描(`__handle_oracle`)。子类:`DumpCache`/`BoostCache`(把预测结果预先算好并存为 pickle,避免重复跑模型——即 "boost");`TrainingCache`/`ParrotTrainingCache`/`LightGBMTrainingCache`(为训练收集每次访问的特征快照 + Belady 标签)。

### Trie KV 系统(`cache/trie/`)

- **`trie_algorithms.py`**——`TrieNode`(children 字典、parent 指针、`node_id`、缓存的 LSTM `hidden_state`)以及各策略类:`TrieLRUAlgorithm`、`TrieRandAlgorithm`、`TrieOracleAlgorithm`(Belady)、`TriePredictAlgorithm`、`TrieModelPredictAlgorithm`、`TrieGuard`/`TrieModelGuard`。一次访问 = 匹配最长的已有前缀,将未命中的后缀作为新叶子节点插入,当 `cur_node_num + insert_len > max_node_num` 时淘汰叶子节点。**只有非受保护路径上的叶子才是淘汰候选**;当前请求的路径受保护。每个策略返回一个 `(total, hit, miss)` 的 block 统计元组。
- **`oracle.py`**——`PrefixFutureOracle`:预先计算 `prefix_tuple -> deque[future_request_indices]`,使 Belady 的"该叶子路径的下次使用时间"成为 O(1) 的队头查询。当前请求每一步都会通过 `consume_current(...)` 从其各前缀队列中消费掉;空 deque ⇒ 下次使用为 `inf` ⇒ 最佳淘汰候选。这是 trie 系统中规范的 Belady oracle——详见 `DESIGN_DOCUMENT.md` 的 §"Prefix Future Oracle"。
- **`trie_cache.py`**——三个缓存封装:
  - `SequenceTrieCache`——主模拟入口。直接接收原始 `List[int]` 序列(不涉及 pc/hash/aligner)。通过 `kv_stat()` 跟踪 KV 指标(block 命中率、请求全命中率、平均前缀命中长度、recompute/saved-prefill block、淘汰次数)。对**超容量请求**的处理方式是只缓存 `sequence[:max_node_num]`,其余部分计为 `uncacheable_blocks`/miss。
  - `TrieTrainingCache`——DAgger 训练数据收集器:每次淘汰时计算 Belady oracle 目标,并以概率 `model_prob` 混入模型自己的选择;输出每次淘汰的快照(候选叶子的 id/路径、`oracle_target`、历史 LSTM 状态)。
  - `TrieCache`——较旧的 pc/hash/aligner 形态的封装(断言 `num_sets == 1`、`ListAligner`、`OneHashFunction`);被本文件中的 `__main__` 演示在 `TrieDataTrace` 上使用。

编辑 trie 插入/淘汰逻辑时的关键不变量:必须在调用 `__add_node__` **之前**设置 `new_node.parent`(模型路径会读取 `parent.hidden_state` 来计算子节点的 Tree-LSTM 状态)。对插入逻辑的任何改动都要在 `TrieModelPredictAlgorithm.__insert__` 和 `TrieTrainingCache.collect` 两处同步。

### 模型(`model/`)

- `model/models.py`——轻量封装(`ParrotModel`、`LightGBMModel`、`CodexModel`),带 `from_config(config_path, ckpt)` 工厂方法,返回供 evict 算法消费的逐次访问预测。
- `model/parrot/`——Parrot 模仿学习淘汰模型(LSTM + attention,NDCG/重用距离损失),移植自 Google Research 的 `cache_replacement`。
- `model/trie_model/`——`TrieParrotModel`:在前缀树上运行的 Tree-LSTM,为各叶子打淘汰分;通过 `python -m model.trie_model` 用 DAgger 训练。
- `model/device.py`——全局 `device_manager` 单例;CUDA 选择通过 `CUDA_VISIBLE_DEVICES` 进行。只有 Parrot/trie 的 torch 模型使用 GPU,其余一律用 CPU。
- `model/lightgbm/`——GBM 训练入口(`python -m model.lightgbm`),基于 Delta+EDC 特征。

### 轨迹(`data_trace/`)

`DataTrace`/`OracleDataTrace`(经典,`(pc,address)` CSV,带供 oracle 用的未来访问查询)与 `TrieDataTrace`/`OracleTrieDataTrace`/`SequenceTrieDataTrace`(序列轨迹)。trie 基准测试绕过这些类,直接通过 `benchmark/trie_kv.load_data` 加载 pickle 化的序列列表。

## 约定

- **用 `partial` 构造**:算法/evictor/predictor 都以类型(通常是 `functools.partial`)的形式传入,在缓存内部实例化。请遵循此模式,而不要传入已构造好的实例。
- **预测器基类具有语义含义**:`PredictAlgorithm` 会根据你的预测器是否为 `ReuseDistancePredictor` / `BinaryPredictor` / `PhasePredictor` / `StatePredictor` / `OraclePredictor` 的子类来分支。新增预测器必须继承正确的基类,才能获得正确的预测初始化与 oracle 装配。
- **数据集不在 git 中**——轨迹、pickle、checkpoint 以及 `boost_traces/` 都从 Releases 下载或本地生成(已 `.gitignore`)。不要假设 `data/`、`traces/` 或 `checkpoints/` 存在。
- `trie-cache` 分支是活跃的研究分支;`DESIGN_DOCUMENT.md` 的 "Current Limitations" / "Suggested Next Steps" 记录了未完成的工作和已知的粗糙之处(例如历史更新只用了最后一个 block、长请求只采用前缀策略等)。
