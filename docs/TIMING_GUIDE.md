# Performance Timing Logs - 使用指南

## 概述

ProGraph 现在支持性能计时日志，可以帮助识别 Online QA 流程中的性能瓶颈。

## 功能特性

- ✅ 轻量级计时工具（开销 <1%）
- ✅ 层级化的计时输出
- ✅ 易于启用/禁用
- ✅ 结构化的控制台输出
- ✅ 跟踪主要操作：
  - 总 QA 时间
  - 每个状态的时间（CHECK_PLAN, RETRIEVE, MAP, UPDATE, ANSWER）
  - 主要操作的时间（LLM 调用、向量搜索、子图扩展）

## 使用方法

### 1. 启用 Timing（默认已启用）

在 `scripts/5-run_multi_hop_qa.py` 中，timing 已默认启用：

```python
agent_machine = AgentStateMachine(
    graph=graph,
    llm=llm_client,
    embedding_client=embedding_client,
    path_scorer=path_scorer,
    path_selector=path_selector,
    max_rounds=retrieval_config.max_rounds,
    # ... 其他参数 ...
    enable_timing=True,  # 启用性能计时
)
```

### 2. 查看 Timing 日志

运行问答时需要启用 `verbose=True` 才能看到 timing 日志：

```python
result = await agent_machine.run(
    question="What is the capital of France?",
    verbose=True,  # 必须启用 verbose
    debug=False
)
```

### 3. 输出示例

```
============================================================
ProGraph Agent 开始运行
问题: What is the capital of France?
最大轮数: 5
============================================================
⏱️  Total QA: What is the capital of France?: 2.456s
  ⏱️  Round 1 - CHECK_PLAN: 0.234s
    ⏱️  LLM: Check plan: 0.221s
  ⏱️  Round 1 - RETRIEVE: 0.156s
    ⏱️  Vector search: Propositions: 0.089s
    ⏱️  LLM: Validate anchors: 0.062s
  ⏱️  Round 1 - MAP: 1.234s
    ⏱️  Subgraph expansion (beam search): 1.198s
  ⏱️  Round 1 - UPDATE: 0.089s
    ⏱️  Path ranking: 0.045s
    ⏱️  Evidence extraction: 0.041s
  ⏱️  Round 2 - CHECK_PLAN: 0.198s
    ⏱️  LLM: Check plan: 0.187s
  ⏱️  ANSWER - Generate final answer: 0.543s
    ⏱️  LLM: Generate answer: 0.531s
============================================================
Agent 运行完成
终止原因: information_sufficient
总轮数: 2
收集证据: 5 条
============================================================
```

### 4. 禁用 Timing

如果需要禁用 timing（例如在生产环境中）：

```python
agent_machine = AgentStateMachine(
    # ... 其他参数 ...
    enable_timing=False,  # 禁用性能计时
)
```

## 性能分析

### 识别瓶颈

通过 timing 日志，可以快速识别哪些操作最耗时：

1. **MAP 阶段通常最耗时**：子图扩展（beam search）涉及大量图遍历和评分
2. **LLM 调用**：每次 LLM 调用通常需要 0.1-0.5 秒
3. **向量搜索**：取决于索引大小和查询复杂度

### 优化建议

根据 timing 日志，可以针对性优化：

- 如果 **MAP 阶段慢**：考虑减少 `beam_width` 或 `max_iterations`
- 如果 **LLM 调用慢**：考虑使用更快的模型或增加并发
- 如果 **向量搜索慢**：考虑优化索引或减少 `top_k`

## 测试

运行 timing 功能测试：

```bash
python tests/test_timing.py
```

运行所有测试（包括 timing）：

```bash
bash run_tests.sh
```

## 技术细节

### 实现

- **工具类**：`src/utils/timing.py`
  - `TimingContext`：上下文管理器，用于计时代码块
  - `TimingLogger`：全局计时日志器，支持启用/禁用

- **集成点**：`src/retrieval/agent_state_machine.py`
  - `run()` 方法：包装整个 QA 流程
  - 状态方法：包装每个状态和主要操作

### 开销

- 使用 `time.perf_counter()` 进行高精度计时（~100ns 开销）
- 上下文管理器开销可忽略不计
- 仅在启用时打印日志
- 总开销 <1% 的执行时间

## 未来增强（计划外）

1. **JSON 输出**：将 timing 数据写入 JSON 文件以便分析
2. **详细 Timing**：添加每次迭代的 timing（子图扩展）
3. **指标收集**：跟踪缓存命中率、批次大小、迭代次数
4. **可视化**：创建 timing 可视化工具（火焰图、时间线）
5. **Pytest 迁移**：将自定义测试脚本转换为 pytest

## 相关文件

- `src/utils/timing.py` - Timing 工具模块
- `src/retrieval/agent_state_machine.py` - 集成 timing 的主要文件
- `scripts/5-run_multi_hop_qa.py` - 启用 timing 的脚本
- `tests/test_timing.py` - Timing 功能测试
- `run_tests.sh` - 测试运行脚本
