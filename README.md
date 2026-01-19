# ProGraph

基于**原子命题（Atomic Proposition）**和**意图驱动分组束搜索（Intent-Group Beam Search）**的多跳问答系统。

## 项目概述

ProGraph 是一个创新的知识图谱问答系统，具有以下核心特点：

1. **原子命题图**：将文档分解为原子命题，构建更细粒度的知识表示
2. **RST 修辞关系**：基于修辞结构理论（RST）建立命题间的逻辑关系
3. **意图驱动推理**：使用 LLM 识别推理意图，动态激活相关边类型
4. **分组束搜索**：同时进行逻辑主攻和探索兜底的分组搜索
5. **桥接评分**：基于新实体发现率的纯集合运算，高效且无需 LLM
6. **Agent 状态机**：基于状态图驱动的自主问答 Agent，支持多轮渐进式检索
7. **性能优化**：支持批处理、缓存、优先队列等多种优化策略

## 与传统方法的区别

| 方面 | 传统方法 | ProGraph |
|------|---------|----------|
| 图节点 | 文档/句子 | 原子命题 + 实体 |
| 图边类型 | 基本关系 | RST 骨架边 + 细节边 + 包含边 |
| 检索算法 | 固定策略 | Intent-Group Beam Search + Agent 状态机 |
| 意图识别 | 无 | LLM 驱动的动态意图识别 |
| 评分公式 | 单一相似度 | S_sem + S_bridge + S_intent 三因子 |
| 性能优化 | 无 | 批处理、缓存、优先队列 |

## 项目结构

```
prograph/
├── scripts/                       # 执行脚本（1-2-3 命名）
│   ├── 1-build_proposition_graph.py    # 离线图构建
│   ├── 2a-generate_candidates.py       # 实体链接阶段1：生成候选对（仅向量模型）
│   ├── 2b-link_and_fuse.py             # 实体链接阶段2：链接与融合（仅大模型）
│   ├── 2-link_entities_fuse_graph.py   # 实体链接与图融合（一体化，显存充足时使用）
│   ├── 2.5-filter_dataset.py           # 数据集过滤脚本
│   ├── 3-build_search_index.py         # 构建搜索索引
│   └── 4-run_multi_hop_qa.py           # 在线多跳问答
│
├── src/                           # 源代码
│   ├── config/                    # 配置管理
│   ├── llm/                       # LLM 抽象层
│   ├── proposition_graph/         # 图构建模块
│   ├── entity_linking/            # 实体链接模块
│   ├── retrieval/                 # 在线检索模块
│   │   ├── agent_states.py            # Agent 状态枚举
│   │   ├── agent_data_structures.py   # Agent 数据结构
│   │   ├── agent_context.py           # Agent 上下文
│   │   ├── agent_state_machine.py     # Agent 状态机核心
│   │   ├── agent_batch_executor.py    # Agent 批量执行器
│   │   └── agent_prompts.py           # Agent Prompt 模板
│   └── utils/                     # 工具函数
│
├── docs/                          # 文档
├── config.yaml                    # 配置文件
├── requirements.txt               # Python 依赖
└── README.md                      # 本文件
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置模型路径

编辑 `config.yaml`，设置你的模型路径：

```yaml
model:
  vllm:
    model_path: "/path/to/your/llm/model"
  vllm_embedding:
    model_path: "/path/to/your/embedding/model"
```

### 3. 构建知识图谱

```bash
# 从原始文档构建命题图
python scripts/1-build_proposition_graph.py \
  --dataset dataset/HotpotQA/full_docs.json \
  --output output/HotpotQA/proposition_graph/raw_graph

# 实体链接与图融合（两种方式）

# 方式一：两阶段拆分（推荐，显存不足时使用）
# 阶段1：生成候选对（仅加载向量模型）
python scripts/2a-generate_candidates.py \
  --graph output/HotpotQA/proposition_graph/raw_graph.pkl \
  --output output/HotpotQA/temp \
  --config config.yaml

# 阶段2：链接与融合（仅加载大模型）
python scripts/2b-link_and_fuse.py \
  --graph output/HotpotQA/proposition_graph/raw_graph.pkl \
  --temp_dir output/HotpotQA/temp \
  --output output/HotpotQA/proposition_graph/linked_graph \
  --config config.yaml

# 方式二：一体化（显存充足时使用）
python scripts/2-link_entities_fuse_graph.py \
  --graph output/HotpotQA/proposition_graph/raw_graph.pkl \
  --output output/HotpotQA/proposition_graph/linked_graph \
  --config config.yaml
```

### 4. 运行问答

```bash
# 单问题模式
python scripts/4-run_multi_hop_qa.py \
  --question "What is the capital of France?" \
  --graph output/HotpotQA/proposition_graph/linked_graph.pkl

# 批量模式
python scripts/4-run_multi_hop_qa.py \
  --dataset dataset/HotpotQA/test_data.json \
  --graph output/HotpotQA/proposition_graph/linked_graph.pkl \
  --output output/HotpotQA/qa_results.json
```

## 核心算法

### Intent-Group Beam Search

```
1. 意图识别：LLM 分析当前状态，预测下一步推理意图
2. 分组扩展：
   - Group A (逻辑主攻)：根据意图激活特定边类型
   - Group B (探索兜底)：始终激活实体跳转边
3. 评分剪枝：S = w1 * S_sem + w2 * S_bridge + w3 * S_intent
4. 迭代直到达到最大深度或信息饱和
```

### 评分公式

```
S(v) = w1 * S_sem(v) + w2 * S_bridge(v) + w3 * S_intent(v)

其中：
- S_sem: 余弦相似度（语义相关性）
- S_bridge: |新实体| / |关联实体|（桥接价值）
- S_intent: |激活边 ∩ 节点边| / |激活边|（意图匹配度）
```

### Agent 状态机

基于状态图驱动的多跳问答 Agent，实现 6 个状态的循环：

```
┌─────────────────────────────────────────┐
│                                         │
│   ┌───────┐    ┌─────┐    ┌──────────┐    │
└──▶│ CHECK │───▶│ PLAN │───▶│ RETRIEVE │───▶│  MAP  │
    └───────┘    └─────┘    └──────────┘    └───────┘
     ▲                                         │
     │                                         │
     │            ┌──────────┐                 │
     └───────────▶│ UPDATE   │◀────────────────┘
                  └──────────┘
                       │
                       ▼
                  ┌─────────┐
                  │ ANSWER  │
                  └─────────┘
```

**状态说明**：
- **CHECK**: 判断当前信息是否足够回答问题
- **PLAN**: 意图识别，识别信息缺口
- **RETRIEVE**: 确定起始锚点
- **MAP**: Beam Search 节点探索（渐进式）
- **UPDATE**: 路径级打分排序，提取证据
- **ANSWER**: 生成最终答案

### 性能优化

系统实现了多项性能优化策略：

| 优化项 | 技术方案 | 性能提升 |
|--------|----------|----------|
| **LLM 批处理** | 并发调用 + 合并请求 | 2-5x 延迟降低 |
| **BeamSearch 缓存** | LRU 缓存复用实例 | 减少 GC 压力 |
| **优先队列** | heapq 替代列表排序 | 10x+ 大规模队列 |
| **向量索引** | HNSW 索引检索 | 100x+ 初始锚点检索 |
| **内存优化** | 避免不必要的拷贝 | 减少内存分配 |

## 意图标签

| 意图 | 激活的边类型 |
|------|-------------|
| Trace_Process | NEXT_EVENT, SEQUENCE |
| Find_Reason | CAUSED_BY, MOTIVATION |
| Expand_Detail | ELABORATION, BACKGROUND |
| Bridge_Entity | MENTIONS_ENTITY |
| Check_Conflict | CONTRAST, CONCESSION |

## 图结构

### 节点类型
- **proposition**：原子命题节点
- **entity**：局部实体节点
- **global_entity**：全局实体节点（链接后）

### 边类型
- **骨架边**（Nucleus-Nucleus）：SEQUENCE, CONTRAST, CONCESSION
- **细节边**（Nucleus-Satellite）：CAUSED_BY, MOTIVATION, ELABORATION, BACKGROUND
- **包含边**：MENTIONS_ENTITY（命题 → 实体）

## 支持的数据集

- HotpotQA
- 2WikiMultihopQA
- MuSiQue

## 配置说明

主要配置项：

```yaml
# 模型配置
model:
  vllm:                      # 离线推理
    model_path: "..."
    tensor_parallel_size: 2
    gpu_memory_utilization: 0.7
  vllm_embedding:            # 离线嵌入
    model_path: "..."
    gpu_memory_utilization: 0.3  # 嵌入模型通常需要 0.2-0.4
  llm:                       # 在线推理
    base_url: "http://localhost:8901/v1"
    temperature: 0.1
  embedding:                 # 在线嵌入
    base_url: "http://localhost:8902/v1"

# 检索配置
retrieval:
  graph_path: "..."           # 图路径
  meta_dir: "..."             # 元数据目录
  index_dir: "..."            # 向量索引目录

  # 基础检索配置
  max_rounds: 5               # 最大检索轮数
  max_path_depth: 6           # 最大搜索深度
  beam_groups: 3              # 分组数量
  beam_width: 4               # 每组保留路径数
  neighbor_top_k: 10          # 邻居节点检索数量

  # 评分权重
  semantic_weight: 0.4        # 语义分数权重
  bridge_weight: 0.6          # 桥接分数权重
  intent_weight: 0.0          # 意图匹配权重（新增）

  # 意图识别配置
  intent_temperature: 0.3     # 意图识别温度
  intent_max_tokens: 512      # 意图识别最大 token 数

  # ========== Agent 状态机配置 ==========
  agent_max_rounds: 5         # Agent 最大轮数
  agent_max_evidence: 50      # 最大证据数量
  agent_max_total_hops: 30   # 最大总跳数

  # 锚点队列配置
  agent_anchor_queue_size: 100           # 锚点队列大小
  agent_anchor_duplicate_threshold: 0.9  # 重复检测阈值

  # MAP 阶段配置
  agent_map_max_iterations: 10            # MAP 最大迭代次数
  agent_map_beam_width: 5                # MAP 束宽度
  agent_map_score_plateau_threshold: 0.02 # 评分停滞阈值
  agent_map_score_plateau_window: 2       # 停滞检测窗口

  # 批量执行配置
  agent_batch_concurrency: 10    # 批量执行并发度
  agent_top_k_paths: 10          # Top-K 路径数量

# 实体链接配置
entity_linking:
  similarity_threshold: 0.7          # 向量相似度阈值
  vector_top_k: 20                  # 向量检索返回数量
  entity_similarity_threshold: 0.9  # 实体相似度阈值
  proposition_similarity_threshold: 0.85  # 命题相似度阈值
```

## 开发文档

详细的开发文档请参阅 `docs/` 目录：

- `docs/architecture.md` - 系统架构
- `docs/data_structures.md` - 数据结构定义
- `docs/api_spec.md` - API 规范
- `docs/plan.md` - 开发计划
- `docs/style_guide.md` - 代码风格规范
- `docs/common_pitfalls.md` - 常见陷阱与案例集

## 许可证

MIT License
# prograph
