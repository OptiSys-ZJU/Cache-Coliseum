# 面向多轮大模型服务的学习增强型 Prefix-Tree KV Cache 淘汰策略研究

硕士学位论文开题报告初稿

学生姓名：________

学号：________

专业：________

导师：________

学院：________

日期：________

## 一、选题依据

### 1. 课题来源

本课题来源于大模型推理服务优化与缓存替换策略研究方向。随着大语言模型在智能问答、代码生成、智能客服、Agent 工作流、检索增强生成、企业知识库问答等应用中的快速落地，推理阶段的计算成本和显存开销逐渐成为限制大模型规模化部署的重要瓶颈。尤其在多轮对话、长上下文问答、工具调用和多智能体协作场景中，不同请求之间往往存在大量重复的系统提示词、历史上下文、模板化指令、检索文档片段和共享任务描述。若每次请求都重新执行 prefill 计算，将造成大量重复计算，并带来较高的首 token 延迟和较低的 GPU 利用率。

KV Cache 是 Transformer 自回归推理过程中的关键中间状态。模型在生成每个 token 时，会复用历史 token 对应的 Key/Value 张量，从而避免重复计算历史上下文。进一步地，如果不同请求具有相同或部分相同的 prompt prefix，系统也可以复用已经计算好的 KV Cache，以减少 prefill 阶段开销。当前 vLLM、SGLang、OpenAI、Anthropic、Google Vertex AI 等系统和云服务均已支持不同形式的 prompt caching 或 prefix caching，说明该技术已经具有明确的工程价值和应用需求。

然而，KV Cache 通常存储于 GPU 显存或跨层缓存系统中，容量昂贵且有限。系统必须决定哪些 KV block 或 prefix path 应被保留，哪些应被淘汰。现有系统多采用 LRU 等传统启发式策略。LRU 实现简单、开销较低，但在多轮对话、多租户混排、RAG 负载和 Agent 工作流等复杂场景下，最近访问时间并不一定能够准确反映未来复用价值。系统可能在缓存命中前不久淘汰高价值前缀，导致重复 prefill、显存资源浪费和请求延迟上升。

本人前期已基于 Cache-Coliseum 仓库开展了 prefix-tree KV cache 仿真平台的初步实现，支持将请求表示为 KV block hash 序列，构建 trie 结构管理共享前缀，并实现 LRU、Random、Oracle 等基线策略及容量扫描实验。已有 OASST1 timed 初步结果表明，在中等 KV cache 压力区间 Oracle 策略相较 LRU 存在明显 block hit rate 差距，说明传统 LRU 策略仍存在可优化空间。因此，本课题拟在已有工作基础上，进一步研究面向多轮 LLM 服务的学习增强型 trie-cache 淘汰策略。

### 2. 课题的研究意义

大语言模型已成为人工智能技术发展的核心基础设施之一，在科学研究、工业生产、软件开发、教育医疗、政务服务和智能制造等领域具有广泛应用前景。随着模型参数规模、上下文长度和用户请求量不断增加，大模型推理成本快速上升。相比训练阶段，推理服务通常具有长期、持续、高并发的特点，其计算资源消耗和能耗成本直接影响大模型应用的普及能力。因此，提升大模型推理系统效率，不仅具有重要的学术研究价值，也具有明确的经济价值和工程意义。

从面向世界科技前沿角度看，LLM serving 已成为系统、体系结构、机器学习和数据库等多个领域交叉研究的热点。近年来，vLLM 的 PagedAttention、SGLang 的 RadixAttention、Prompt Cache、Hydragen、Mooncake、MemServe、KVFlow、Learned Prefix Caching 等工作不断涌现，说明 KV Cache 管理、prefix reuse 和缓存淘汰策略已成为大模型系统研究的重要方向。传统缓存替换策略主要针对页面访问或 Web 对象访问设计，而 LLM prefix cache 具有前缀共享、树形依赖、块级复用、显存约束和请求语义相关等新特征。因此，针对该类新型缓存对象设计更高效的管理策略，是大模型系统研究中的前沿问题。

从面向经济主战场角度看，大模型服务通常部署在昂贵的 GPU 集群上，推理成本直接影响企业级 AI 应用的商业可行性。对于智能客服、代码助手、企业知识库问答、办公自动化和 Agent 平台等应用，请求中存在大量重复上下文。若能够提升 prefix cache 命中率，减少 prefill 重算，就可以降低首 token 延迟、提升吞吐量、减少 GPU 使用时长，从而降低服务成本。OpenAI、Anthropic、Google Vertex AI、AWS Bedrock 等平台均已提供 prompt caching 或 context caching 能力，说明缓存复用已从学术方案进入产业系统，进一步优化其缓存管理策略具有现实应用价值。

从面向国家重大需求角度看，我国正在推进人工智能基础设施建设和大模型产业化应用。大模型推理系统的高效部署关系到国产算力集群利用率、企业智能化转型成本以及自主可控 AI 基础设施建设。GPU 显存资源昂贵且有限，如何在有限算力条件下支撑更高并发、更长上下文和更低延迟的大模型服务，是提升智能计算中心资源利用率的重要问题。本课题通过研究 KV Cache 的精细化管理策略，为降低推理服务成本、提升国产大模型服务效率提供技术参考。

从服务人民生命健康和社会民生角度看，医疗问答、临床辅助决策、法律咨询、教育辅导、政务服务等应用往往依赖长上下文、多轮交互和固定知识模板。这些场景中，系统提示词、病例资料、法规文本、课程资料等内容具有较强重复性，适合通过 prefix cache 复用减少延迟。更低的推理延迟和更低的部署成本，有助于让高质量智能服务覆盖更多用户。因此，本课题虽然属于系统优化研究，但其成果可间接支撑低成本、高可用的大模型应用落地。

### 3. 国内外研究现状分析

现有研究大体可以分为四类：KV Cache 基础管理、prefix reuse 机制、分布式或多级 KV Cache 系统，以及学习增强型缓存淘汰策略。

第一类是 KV Cache 基础管理。vLLM 提出的 PagedAttention 将操作系统分页思想引入 LLM 推理，把 KV Cache 划分为固定大小 block，减少显存碎片并支持跨请求共享。该工作奠定了后续高吞吐 LLM serving 系统的重要基础。vLLM 后续支持 Automatic Prefix Caching，利用 hash-based block cache 复用相同前缀，并在缓存满时优先淘汰引用计数为 0 的 block，再使用 LRU 等规则选择淘汰对象。这类工作解决了 KV Cache 如何高效分配、定位和复用的问题，但其淘汰策略仍以传统启发式方法为主。

第二类是 prefix reuse 和树形共享机制。SGLang 提出的 RadixAttention 使用 radix tree 管理请求前缀，使不同请求之间的共享 prompt、few-shot 示例和多轮对话历史能够自动复用。ChunkAttention 在 prefix-tree based KV cache 之上设计 attention kernel，提高共享系统提示词场景下的数据局部性。Prompt Cache 通过模块化 prompt schema 显式复用频繁出现的文本片段。Hydragen 将共享前缀和独有后缀的 attention 计算分解，在大 batch 且存在长共享前缀时获得显著吞吐提升。RelayAttention 则关注长 system prompt 场景，减少批处理请求中重复读取共享 KV 的内存访问。这些工作表明，prefix 复用已成为降低 LLM 推理成本的重要技术路线。然而，它们更多关注“如何识别和复用前缀”以及“如何优化 attention kernel”，对有限 cache 容量下的淘汰决策研究相对不足。

第三类是分布式和多级 KV Cache 系统。Mooncake 提出 KVCache-centric 的大模型服务架构，将 KV Cache 作为核心资源进行跨层调度，并结合 prefill/decode 分离、多级存储和全局调度服务真实在线负载。MemServe 将 context caching 与 disaggregated inference 结合，使用全局 prompt tree 提升跨实例 cache locality。LoongServe 面向长上下文 LLM 服务，研究弹性序列并行、KV Cache 迁移和碎片管理问题。CacheGen 则关注 KV Cache 压缩和流式加载，以降低长上下文加载成本。这些工作说明国内外均已认识到 KV Cache 不再只是单个请求内部的临时状态，而是 LLM serving 系统中的核心资源。但这些系统通常更关注架构、迁移、存储层次和调度问题，对 prefix-tree 节点级淘汰策略的可学习性、Oracle 上界和 workload 条件分析仍有进一步研究空间。

第四类是学习增强型或 workload-aware 淘汰策略。NeurIPS 2025 的 Learned Prefix Caching 明确指出 LRU 与最优策略之间存在显著差距，并提出使用对话内容分析指导 prefix cache eviction。该工作表明，在 LLM prefix cache 场景中，缓存对象未来是否复用与对话内容、会话状态和访问时间共同相关，仅依赖最近访问时间不足以刻画复用价值。KVFlow 面向多智能体工作流，指出 LRU 无法预判 agent 未来调用，因而容易在复用前淘汰有价值 KV Cache；该工作通过 Agent Step Graph 估计未来执行距离，指导 tree-structured cache 中的节点级淘汰和预取。SAECache 进一步提出并非所有 token 都同样值得缓存，尝试将 token 类型、语义价值和在线学习引入 prefix cache eviction。这些最新工作说明，prefix cache eviction 已从传统 LRU 策略走向学习增强、语义感知和工作流感知方向。

综合来看，国内外已经在 prefix caching 的数据结构、系统架构和工程实现方面取得较多成果，trie 或 radix tree 本身并不是新的研究点。若仅提出“使用 trie 管理共享前缀”，创新性不足。但当前仍存在以下不足：一是许多系统默认使用 LRU 或其变体，对 LRU 与 Oracle 之间差距的 workload 条件缺乏系统刻画；二是已有学习型方法通常依赖具体会话内容或工作流结构，缺少面向通用 KV block trace 的可复现实验框架；三是 prefix-tree cache 中节点深度、子树大小、访问频率、最近访问时间、未来复用距离、请求长度等结构特征如何共同影响淘汰价值，仍有研究空间；四是国内相关工作更多集中在长上下文服务、分布式 KV Cache 和多级存储架构，对轻量级学习增强淘汰策略的公开研究相对较少。因此，本课题以 trie-cache 为基础，重点研究 Oracle 上界驱动的性能分析和学习增强淘汰策略，仍具有一定创新意义。

### 4. 主要参考文献

[1] Kwon W, Li Z, Zhuang S, et al. Efficient Memory Management for Large Language Model Serving with PagedAttention. SOSP, 2023.

[2] vLLM Project. Automatic Prefix Caching. https://docs.vllm.ai/en/stable/design/prefix_caching/

[3] Zheng L, et al. SGLang: Efficient Execution of Structured Language Model Programs. 2024.

[4] Yang D, Li A, Li K, Lloyd W. Learned Prefix Caching for Efficient LLM Inference. NeurIPS, 2025.

[5] Pan Z, Patel A, Hu Z, et al. KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows. 2025.

[6] Qin R, Li Z, He W, et al. Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving. FAST, 2025.

[7] MemServe: Context Caching for Disaggregated LLM Serving with Elastic Memory Pool. 2024.

[8] LoongServe: Efficiently Serving Long-Context Large Language Models with Elastic Sequence Parallelism. 2024.

[9] Gim I, Chen G, Lee S, et al. Prompt Cache: Modular Attention Reuse for Low-Latency Inference. 2023.

[10] Juravsky J, Brown B, Ehrlich R, et al. Hydragen: High-Throughput LLM Inference with Shared Prefixes. 2024.

[11] ChunkAttention: Efficient Self-Attention with Prefix-Aware KV Cache and Two-Phase Partition. 2024.

[12] Zhu L, Wang X, Zhang W, Lau R W H. RelayAttention for Efficient Large Language Model Serving with Long System Prompts. 2024.

[13] Yao J, Li H, Liu Y, et al. CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion. 2024.

[14] Not All Tokens Are Worth Caching: Learning Semantic-Aware Eviction for LLM Prefix Caches. 2026.

[15] OpenAI. Prompt Caching. https://platform.openai.com/docs/guides/prompt-caching

[16] Anthropic. Prompt Caching. https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching

[17] Google Cloud. Vertex AI Context Caching. https://cloud.google.com/vertex-ai/generative-ai/docs/context-cache/context-cache-overview

## 二、研究方案

### 1. 研究目标、内容和拟解决的关键问题

本课题的总体目标是：面向多轮大模型推理服务中的 prefix-tree KV cache 场景，构建可复现的缓存仿真与评估框架，分析 LRU 与 Oracle 策略之间的性能差距，设计一种低开销、可解释的学习增强型缓存淘汰策略，以提升有限 KV Cache 容量下的 block hit rate，减少 prefill 重算开销和首 token 延迟。

主要研究内容包括以下几个方面。

第一，构建 prefix-tree KV cache 抽象模型。将 LLM 请求表示为 KV block hash 序列，将共享前缀映射为 trie 中的公共路径，将 GPU KV cache 容量抽象为可驻留节点或 block 数量。请求到达时，系统查找最长命中前缀，命中部分可复用，未命中后缀需要重新 prefill。缓存空间不足时，需要从 trie 的叶节点或可淘汰节点中选择对象进行删除。

第二，建立 LRU、Random、Oracle 等基线策略。LRU 表示当前系统中常见的低开销启发式策略，Random 用作弱基线，Oracle 使用未来请求信息估计每个 prefix path 的下一次复用时间，作为近似最优上界。通过比较 LRU 与 Oracle 的差距，判断不同工作负载下是否存在学习增强策略的收益空间。

第三，开展 workload 条件分析。不同 trace 形态下，LRU 与 Oracle 的差距可能显著不同。例如，连续单会话 trace 中最近使用时间可能较好预测未来复用，LRU 表现接近 Oracle；而多租户混排、Agent 工作流和共享模板场景中，请求间隔与未来复用关系更加复杂，LRU 可能明显失效。本课题将分析容量、block size、请求长度、trace 混排方式、前缀共享程度等因素对缓存命中率的影响。

第四，设计学习增强型 trie-cache 淘汰策略。基于 Oracle 产生的未来复用标签，提取 trie 节点或叶路径特征，如节点深度、路径长度、最近访问时间、访问频率、子树规模、历史复用间隔、请求长度、当前命中前缀长度等，训练轻量模型预测候选节点的复用价值或淘汰优先级。在线推理时，模型仅在缓存满并需要淘汰时对候选节点打分，选择低价值节点淘汰。

第五，评估策略性能与开销。实验指标包括 block hit rate、request full hit rate、average prefix hit length、recompute blocks、saved prefill tokens、eviction count 和模型推理开销等。通过与 LRU、Random、Oracle 比较，评估所提策略能否在较低开销下缩小 LRU 与 Oracle 的差距。

本课题拟解决的关键问题包括：如何在 prefix-tree KV cache 中定义合理的 Oracle 上界；如何从 trie 结构和访问历史中提取有效特征；如何在低开销约束下设计学习增强淘汰策略；如何判断该策略在哪些负载条件下优于 LRU；如何构建可复现实验流程证明方法有效。

### 2. 拟采取的研究方法、技术路线、试验方案及可行性分析

本课题拟采用“系统建模、仿真实验、Oracle 标注、模型学习、对比评估”的技术路线。

首先，进行系统建模。将 LLM 服务中的 prompt prefix reuse 抽象为序列缓存问题。每个请求由若干 KV block id 构成，trie 中一条 root-to-node 路径代表一个已缓存前缀。缓存命中定义为请求序列与 trie 中路径的最长公共前缀长度。缓存容量以 block 数或 trie 节点数表示。当请求超过容量时，可采用保守策略，仅缓存容量允许范围内的前缀，剩余部分计入不可缓存 miss。

其次，实现和完善仿真平台。基于已有 Cache-Coliseum 代码，继续完善 SequenceTrieCache、PrefixFutureOracle、TrieLRUAlgorithm、TrieOracleAlgorithm 等模块，确保 LRU、Random、Oracle 可在不依赖深度学习环境的情况下稳定运行。对 OASST1 timed 派生 trace 进行预处理，将原始 KV block identity 映射为紧凑整数序列，并保存 train、valid 划分及 metadata。

再次，构造 Oracle 标签。对于每个请求，在未来请求序列中建立 prefix 到未来访问位置的索引。缓存淘汰时，对候选叶路径查询其下一次作为请求前缀出现的位置，复用时间越远或不再出现的节点越适合淘汰。该 Oracle 策略不作为可部署算法，而作为性能上界和训练标签来源。

然后，设计学习增强策略。初步方案可分为两类：一是监督学习型 eviction scorer，即使用 Oracle 标签训练模型预测每个候选节点的复用距离或淘汰概率；二是 guard 型策略，即在 LRU 的基础上增加模型保护机制，当 LRU 候选节点被预测为高价值时，选择次优候选淘汰。前者可能收益更高，后者开销更低、风险更小。模型选择上，可先采用 LightGBM、逻辑回归或小型 MLP 等轻量方法，避免复杂模型推理开销抵消缓存收益。

最后，开展实验评估。实验将从以下方面展开：容量扫描实验，测试不同 cache capacity 下 LRU、Random、Oracle、Model 的表现；trace 形态实验，比较连续 trace 与 interleaved trace 的差异；block size 实验，分析不同 token block 粒度对命中率和重算量的影响；消融实验，分析各类特征对模型效果的贡献；开销实验，统计模型推理耗时和额外内存开销。

本课题具有较好的可行性。代码基础方面，本人已完成 prefix-trie cache simulator 的初步实现，并支持 LRU、Random、Oracle 和 KV-oriented metrics。数据基础方面，仓库中已有 OASST1 timed 派生数据，并已产生部分 CSV 实验结果。初步实验表明，在 OASST1 不同 block size 设置下，Oracle 相较 LRU 存在稳定提升，尤其在中等 KV cache 压力区间差距更明显。这说明课题不仅有工程实现基础，也有实验现象支撑。环境方面，非模型基线不依赖 GPU，前期可在本地 CPU 环境完成；后续轻量模型训练可根据条件使用 CPU 或单卡 GPU 完成。

### 3. 研究的创新点

本课题的创新点主要体现在以下三个方面。

第一，构建面向 LLM prefix-tree KV cache 的 Oracle 上界分析框架。不同于传统页面缓存，本课题关注树形前缀共享结构中路径级复用价值，并基于未来 trace 构建 prefix future oracle，用于刻画 LRU 与最优策略之间的差距。这有助于回答“何种负载下 LRU 足够好，何种负载下需要学习增强策略”这一问题。

第二，提出结合 trie 结构特征和访问历史特征的轻量学习增强淘汰策略。已有系统多使用 LRU，而本课题拟利用节点深度、前缀长度、访问频率、最近访问时间、子树规模、历史复用间隔等特征预测候选节点未来价值，在保持低推理开销的前提下改善淘汰决策。

第三，面向多轮和多租户混排场景进行系统性实验评估。本课题不仅比较单一命中率指标，还将综合评估 block hit rate、重算 block 数、saved prefill tokens、request full hit rate 和模型开销，并分析 trace 混排方式、容量、block size 等因素对策略收益的影响，从而形成对 trie-cache 适用边界的较完整认识。

需要强调的是，本课题不将“使用 trie 存储共享前缀”本身作为主要创新点，因为 SGLang/RadixAttention 等已有相关设计。本文的创新重点在于：以 trie-cache 为载体，对 LRU 与 Oracle gap 进行可复现刻画，并设计低开销学习增强淘汰策略。

### 4. 研究计划及预测进展

第一阶段：文献调研与问题建模。时间为第 1 至第 2 个月。完成 vLLM、SGLang、PagedAttention、RadixAttention、Learned Prefix Caching、Mooncake、MemServe、KVFlow 等相关工作的阅读和整理，明确本文研究边界，形成开题报告和技术路线。

第二阶段：仿真平台完善与数据预处理。时间为第 3 至第 4 个月。完善 prefix-tree KV cache simulator，修复长请求处理、trace reset、interleaving、指标统计等问题；完成 OASST1 timed 数据预处理；形成稳定的 LRU、Random、Oracle capacity sweep 实验流程。

第三阶段：Oracle gap 分析与特征工程。时间为第 5 至第 6 个月。系统分析不同容量、block size、trace 形态下 LRU 与 Oracle 的差距，提取影响 cache 命中的 workload 特征；构造训练样本和 Oracle 标签，为学习型策略设计提供依据。

第四阶段：学习增强淘汰策略设计与实现。时间为第 7 至第 8 个月。实现基于 LightGBM、MLP 或 guard 机制的 eviction scorer，完成模型训练、验证和初步调参；与 LRU、Random、Oracle 进行对比实验。

第五阶段：实验扩展、消融分析与论文撰写。时间为第 9 至第 11 个月。完成主要实验、消融实验和开销分析，整理图表，撰写论文主体内容。

第六阶段：论文修改与答辩准备。时间为第 12 个月。根据导师意见完善论文，补充实验，准备答辩材料和代码归档。

### 5. 预期研究成果

预期形成以下成果：第一，完成一个可复现的 prefix-tree KV cache 仿真与评估框架，支持 LRU、Random、Oracle 和学习增强策略；第二，形成对 LRU 与 Oracle gap 的系统分析结果，明确不同工作负载和 cache 配置下传统策略的适用边界；第三，提出并实现一种低开销学习增强型 trie-cache 淘汰策略，在部分多轮或混排场景中相较 LRU 提升 block hit rate、减少 recompute blocks；第四，完成硕士学位论文一篇，并整理实验代码、数据处理脚本和结果图表。

## 三、研究基础

### 1. 与本项目有关的研究工作积累和已取得的研究工作成绩

本人已围绕 Cache-Coliseum 仓库开展了与本课题相关的前期研究和工程实现。具体包括以下方面。

第一，完成了 prefix-tree KV cache simulator 的初步设计。将 LLM 请求抽象为 KV block hash 序列，将共享前缀存储于 trie 结构中，支持通过最长前缀匹配统计缓存命中，并将缓存容量抽象为最大可驻留节点数。

第二，实现了多种缓存淘汰基线。当前代码已支持 Trie LRU、Random 和 Belady-style Oracle 策略。其中 Oracle 策略通过 PrefixFutureOracle 建立 prefix 到未来请求索引的映射，可在淘汰时查询候选路径的下一次复用位置，为后续训练标签和性能上界分析提供支持。

第三，完成了面向 KV Cache 的指标统计。已有指标包括 requests、total blocks、hit blocks、miss blocks、block hit rate、request full hit rate、average prefix hit length、recompute blocks、saved prefill tokens、evictions、resident blocks 等，能够从多个角度评估缓存策略效果。

第四，完成了部分数据预处理和初步实验。仓库中已包含 OASST1 timed 派生 trace 及多组实验 CSV。初步结果显示，在 timestamp shared-cache 场景中，Oracle 相较 LRU 存在明显命中率提升，说明本课题存在实际可优化空间。同时，也观察到极小或极大容量下 LRU 与 Oracle 差距缩小，说明本课题需要进一步分析策略收益的适用条件。

第五，完成了基础测试验证。已有测试覆盖 prefix future oracle、path recovery、long request handling、OASST1 timed preprocessing 语义和 trie cache eviction 等模块，为后续扩展模型策略和实验流程提供了较可靠的代码基础。

### 2. 已具备的实验、资料等条件，尚缺少的实验、资料条件和拟解决的途径

已具备条件包括：第一，已有 Cache-Coliseum 代码仓库和 prefix-trie cache 实现基础；第二，已有 OASST1 timed 派生数据和预处理脚本；第三，已有 LRU、Random、Oracle baseline 和 capacity sweep 结果；第四，已有本地 Python 虚拟环境，可运行非深度学习仿真实验；第五，已有较明确的文献基础和研究问题定位。

尚缺少的条件包括：第一，真实大规模生产 trace 仍不充分，现阶段主要使用公开或派生数据；第二，学习增强模型尚未完成系统训练和对比；第三，实验尚未覆盖足够丰富的工作负载，如 Agent workflow、RAG 模板请求、多租户长上下文混排等；第四，当前仿真主要关注 block hit 与 prefill 重算，尚未完整建模 GPU、CPU、SSD 多级 KV Cache 传输成本和真实 TTFT。

拟解决途径为：继续扩展公开 trace 和合成 workload，构造可控的多租户混排与共享前缀场景；在已有 Oracle 标签基础上训练轻量模型，优先完成 CPU 可运行的策略验证；在指标层面加入近似 TTFT 和重算 token 成本估计；若实验条件允许，再进一步接入 vLLM 或 SGLang 小规模原型进行端到端验证。这样可以在保证硕士论文工作量可控的前提下，形成完整、可信的研究闭环。
