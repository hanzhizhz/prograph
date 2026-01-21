"""
Agent Prompt 模板

定义基于状态图驱动的多跳问答系统的 Prompt 模板。
参考 ggagent3 的 gap-driven exploration 设计。
"""

from typing import List, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .agent_data_structures import InfoGap


# ============== CHECK_PLAN 合并阶段 Prompt ==============

CHECK_PLAN_PROMPT = """你是一个专业的多跳问答系统。请判断当前信息是否足够回答问题，如果不够，识别信息缺口并选择探索路径。

**问题：**
{question}

**已收集的上下文文档：**
{evidence}

**已访问实体：**
{visited_entities}

{gap_history_section}

**探索路径说明：**

请根据信息缺口选择合适的探索路径（active_edges）：

1. **SKELETON（骨干边）** - 追踪事件发展、时序关系、对比转折
   - 适用场景：需要了解"接下来发生什么"、"事件顺序"、"矛盾对比"
   - 意图标签示例：Trace_Process（追踪事件发展）、Check_Conflict（检查矛盾转折）

2. **DETAIL（细节边）** - 寻找原因、动机、背景、解释说明
   - 适用场景：需要了解"为什么"、"原因是什么"、"背景信息"、"详细说明"
   - 意图标签示例：Find_Reason（寻找原因动机）、Expand_Detail（扩展背景细节）

**路径选择建议：**
- 追踪事件发展、时序流程 → 选择 SKELETON
- 寻找原因、动机、背景 → 选择 DETAIL
- 检查矛盾、转折对比 → 选择 SKELETON
- 扩展详细说明 → 选择 DETAIL
- 可以组合多种边类型以覆盖不同推理角度

**输出格式（如果可以回答）：**
```json
{{
  "can_answer": true,
  "rationale": "判断理由"
}}
```

**输出格式（如果不能回答）：**
```json
{{
  "can_answer": false,
  "rationale": "判断理由",
  "info_gaps": [
    {{
      "gap_id": "gap_1",
      "gap_description": "需要查找的具体信息",
      "related_entities": ["实体1", "实体2"],
      "intent_label": "自定义意图描述（如 Trace_Process, Find_Reason, Expand_Detail, Check_Conflict 等）",
      "active_edges": ["SKELETON", "DETAIL"]
    }}
  ],
  "summary": "简要总结当前推理状态和下一步计划"
}}
```

**要求：**
1. 如果文档内容包含问题的直接答案，设置 can_answer=true
2. intent_label 用于记录推理意图（自由描述，不受限制）
3. active_edges 从 SKELETON 和 DETAIL 中选择，可以组合使用
4. 根据信息缺口的类型选择最合适的探索路径组合
5. gap_description请不要使用复合描述，多个信息缺口请使用多个gap_description，保持每个gap_description的独立性，方便检索
6. **重要**：参考历史缺口状态，避免提出与已耗尽(EXHAUSTED)缺口相同或高度相似的描述
7. 对于状态为"正在探索"(ACTIVE)的缺口，如有失败提示，请参考提示换一个角度描述
"""

# 历史缺口状态部分（可选）
GAP_HISTORY_SECTION = """**历史缺口状态：**
{gap_history}

**缺口状态说明：**
- 待探索(PENDING)：尚未开始检索
- 正在探索(ACTIVE)：已尝试但未补全，可参考失败提示改进
- 已补全(SATISFIED)：信息足够，已满足
- 部分满足(PARTIALLY_SATISFIED)：检索到部分相关信息，但不完整
- 手动关闭(MANUALLY_CLOSED)：虽未直接找到信息，但可通过现有信息推理得出
- 已耗尽(EXHAUSTED)：多次尝试无果，不应再次提出相同描述

**重要提醒：**
- 不要重复提出与"已耗尽"状态缺口相同或高度相似的 gap_description
- 对于"正在探索"状态的缺口，参考其失败提示，换一个完全不同的角度
- 如果某个信息确实无法获取，考虑放弃该方向，尝试其他推理路径
"""


def get_check_plan_prompt(
    question: str,
    documents: List[str],
    visited_entities: set,
    gap_history: str = ""
) -> str:
    """生成 CHECK_PLAN 阶段的 Prompt

    Args:
        question: 问题
        documents: 格式化的文档内容列表
        visited_entities: 已访问实体集合
        gap_history: 历史缺口状态信息

    Returns:
        Prompt 字符串
    """
    evidence_text = "\n".join(documents)
    entities_text = ", ".join(visited_entities) if visited_entities else "无"
    
    # 构建历史缺口状态部分
    if gap_history:
        gap_history_section = GAP_HISTORY_SECTION.format(gap_history=gap_history)
    else:
        gap_history_section = ""

    return CHECK_PLAN_PROMPT.format(
        question=question,
        evidence=evidence_text,
        visited_entities=entities_text,
        gap_history_section=gap_history_section
    )


# ============== 查询改写 Prompt ==============

QUERY_REWRITE_PROMPT = """你是一个专业的检索查询优化专家。请将信息缺口改写为适合向量检索的查询文本。

**原始问题：**
{question}

**信息缺口（按序号列出）：**
{info_gaps}

**改写要求：**

1. **主体明确**：查询的主体必须完整，不能使用代词（你、我、他、她、它、这、那等）
   - 错误："它的创刊时间是什么？"
   - 正确："Arthur's Magazine 的创刊时间是什么？"

2. **原子查询**：每个查询应该是单一、独立的查询单元
   - 错误："A 的属性和 B 与 C 的关系"（复合查询）
   - 正确：分解为 "A 的属性" 和 "B 与 C 的关系"（两个独立查询）

3. **无依赖性**：查询不应依赖上下文或前置条件
   - 错误："该公司"（依赖上文提到的公司）
   - 正确："Oberoi Hotels"（直接指明实体）

4. **信息丰富**：包含关键实体、属性、关系等信息
   - 简洁："杂志创刊时间"
   - 丰富："Arthur's Magazine 创刊时间 First for Women 创刊时间"

5. **自然表达**：使用自然语言表达，便于向量检索匹配
   - 避免过度简化的关键词
   - 使用完整句式或短语

**输出格式：**
```json
{{
  "queries": [
    {{
      "index": 0,
      "query": "改写后的查询文本",
      "target_gap": "对应的原始缺口描述（仅用于验证）"
    }}
  ],
  "rationale": "改写理由的简要说明"
}}
```

**要求：**
1. 必须按序号处理每个信息缺口
2. `index` 必须与输入的序号一致（从 0 开始）
3. 每个缺口生成 1 个改写查询
"""

# 重试缺口的增强版查询改写 Prompt
QUERY_REWRITE_RETRY_PROMPT = """你是一个专业的检索查询优化专家。请将信息缺口改写为适合向量检索的查询文本。

**原始问题：**
{question}

**信息缺口（按序号列出）：**
{info_gaps}

**历史尝试信息（重要！）：**
{retry_history}

**改写要求：**

1. **主体明确**：查询的主体必须完整，不能使用代词
2. **原子查询**：每个查询应该是单一、独立的查询单元
3. **无依赖性**：查询不应依赖上下文或前置条件
4. **信息丰富**：包含关键实体、属性、关系等信息
5. **自然表达**：使用自然语言表达，便于向量检索匹配

**重试缺口的特殊改写要求：**

对于标记为"重试"的缺口，你必须：
1. **完全换一个角度**：不要使用与上次查询相似的表述
2. **参考失败提示**：根据提示的建议方向进行改写
3. **尝试不同的关联**：
   - 如果之前搜索实体属性失败，尝试搜索实体关系
   - 如果之前搜索直接信息失败，尝试搜索背景或上下文信息
   - 如果之前搜索具体事实失败，尝试搜索相关事件或时间线

**改写角度建议：**
- 人物 → 尝试搜索其职业、所属组织、参与事件
- 事件 → 尝试搜索时间、地点、参与者、原因、结果
- 概念 → 尝试搜索定义、示例、相关概念
- 关系 → 尝试搜索关系的双方、关系类型、时间范围

**输出格式：**
```json
{{
  "queries": [
    {{
      "index": 0,
      "query": "改写后的查询文本",
      "target_gap": "对应的原始缺口描述",
      "strategy": "改写策略说明（对于重试缺口必填）"
    }}
  ],
  "rationale": "改写理由的简要说明"
}}
```

**要求：**
1. 必须按序号处理每个信息缺口
2. `index` 必须与输入的序号一致（从 0 开始）
3. 对于重试缺口，查询必须与上次查询有明显区别
"""


def get_query_rewrite_prompt(
    question: str,
    info_gaps: List['InfoGap']
) -> str:
    """生成查询改写 Prompt

    Args:
        question: 原始问题
        info_gaps: 信息缺口列表

    Returns:
        Prompt 字符串
    """
    gaps_text = "\n".join(
        f"{i}. {gap.gap_description}（实体：{', '.join(gap.related_entities)}）"
        for i, gap in enumerate(info_gaps)
    )

    return QUERY_REWRITE_PROMPT.format(
        question=question,
        info_gaps=gaps_text
    )


def get_query_rewrite_retry_prompt(
    question: str,
    info_gaps: List['InfoGap'],
    retry_info: Dict[int, Dict[str, Any]]
) -> str:
    """生成重试缺口的增强版查询改写 Prompt

    Args:
        question: 原始问题
        info_gaps: 信息缺口列表
        retry_info: 重试信息字典 {index: {last_query, failure_hints, retrieved_docs}}

    Returns:
        Prompt 字符串
    """
    # 构建缺口文本，标记重试缺口
    gaps_lines = []
    for i, gap in enumerate(info_gaps):
        line = f"{i}. {gap.gap_description}（实体：{', '.join(gap.related_entities)}）"
        if i in retry_info:
            line += " [重试]"
        gaps_lines.append(line)
    gaps_text = "\n".join(gaps_lines)
    
    # 构建重试历史信息
    retry_lines = []
    for idx, info in retry_info.items():
        retry_lines.append(f"缺口 {idx}:")
        if info.get("last_query"):
            retry_lines.append(f"  - 上次查询: {info['last_query']}")
        if info.get("failure_hints"):
            retry_lines.append(f"  - 失败提示: {'; '.join(info['failure_hints'])}")
        if info.get("retrieved_docs"):
            doc_count = len(info["retrieved_docs"])
            retry_lines.append(f"  - 上次检索到 {doc_count} 个文档，但未能满足需求")
    
    retry_history = "\n".join(retry_lines) if retry_lines else "无历史尝试"

    return QUERY_REWRITE_RETRY_PROMPT.format(
        question=question,
        info_gaps=gaps_text,
        retry_history=retry_history
    )


# ============== RETRIEVE 阶段 Prompt ==============

RETRIEVE_ANCHOR_JUDGE_PROMPT = """你是一个专业的信息检索专家。请判断给定的命题节点是否可以作为检索锚点。

**检索锚点说明：**
锚点是图谱搜索的起点或入口，并不要求它直接包含问题的最终答案。只要命题内容与问题主体、关系或背景具有潜在相关性，能够通过图索引扩展引导至正确答案，即可被视为有效锚点。

**问题：**
{question}

**当前信息缺口：**
{info_gaps}

**候选命题节点：**
{candidates}

**筛选建议（满足以下任一即可）：**
1. 命题涉及问题或信息缺口中的核心实体或其同义词。
2. 命题描述的关系（如所属、因果、动作等）与搜索意图相似。
3. 命题虽然不是直接答案，但能作为逻辑推导或图谱跳转的合理“跳板”。
4. 命题包含可能缩小搜索范围的关键属性或背景信息。

**输出格式：**
```json
{{
  "valid_anchors": ["节点ID1", "节点ID2", ...],
  "reason": "说明这些节点如何能作为‘跳板’辅助后续图谱搜索或填补信息缺口"
}}
```
"""


def get_anchor_judge_prompt(
    question: str,
    info_gaps: List[Dict[str, Any]],
    candidates: List[Dict[str, str]]
) -> str:
    """生成锚点判断 Prompt

    Args:
        question: 问题
        info_gaps: 信息缺口列表
        candidates: 候选节点列表

    Returns:
        Prompt 字符串
    """
    gaps_text = "\n".join(
        f"- {gap.get('gap_description', '')}: {', '.join(gap.get('related_entities', []))}"
        for gap in info_gaps
    )

    candidates_text = "\n".join(
        f"- ID: {c.get('node_id')}, 内容: {c.get('text', '')}"
        for c in candidates
    )

    return RETRIEVE_ANCHOR_JUDGE_PROMPT.format(
        question=question,
        info_gaps=gaps_text,
        candidates=candidates_text
    )


# ============== UPDATE 阶段 Prompt ==============

DOC_EXTRACT_PROMPT = """你是一个专业的信息提取专家。请从检索到的文档中提取有用的信息来回答问题。

**问题：**
{question}

**检索到的文档（按相关性排序）：**
{docs}

**请提取：**
1. 与问题直接相关的事实
2. 支撑推理的关键信息
3. 可以作为答案依据的内容

**输出格式：**
```json
{{
  "extracted_evidence": ["证据1", "证据2", ...],
  "key_insights": ["洞察1", "洞察2", ...],
  "confidence": 0.8
}}
```
"""


def get_extract_from_docs_prompt(
    question: str,
    docs: List[Dict[str, Any]]
) -> str:
    """生成信息提取 Prompt（基于文档）

    Args:
        question: 问题
        docs: 文档列表

    Returns:
        Prompt 字符串
    """
    docs_text = ""
    for i, doc in enumerate(docs):
        doc_id = doc.get("doc_id", f"doc_{i}")
        title = doc.get("title", "未知标题")
        content = doc.get("content", "")
        docs_text += f"\n[{i+1}] {title} (ID: {doc_id})\n{content}\n"
    
    if not docs_text:
        docs_text = "无检索到的文档"

    return DOC_EXTRACT_PROMPT.format(
        question=question,
        docs=docs_text
    )


# ============== 缺口补全评估 Prompt ==============

GAP_SATISFACTION_PROMPT = """你是一个专业的信息检索评估专家。请评估检索结果是否满足了信息缺口的需求。

**原始问题：**
{question}

**信息缺口描述：**
{gap_description}

**关联实体：**
{related_entities}

**检索到的文档：**
{retrieved_docs}

**评估任务：**
1. 判断检索结果是否包含了能够填补该信息缺口的信息
2. 如果包含，从文档中提取有效的证据（原始文本片段）
3. 如果未能满足，分析原因并给出下一步搜索方向的建议

**输出格式：**
```json
{{
  "is_satisfied": true/false,
  "selected_evidence": ["证据1", "证据2", ...],
  "satisfaction_reason": "满足/不满足的原因说明",
  "failure_hints": ["建议1", "建议2", ...]
}}
```

**failure_hints 示例（仅在 is_satisfied=false 时填写）：**
- "尝试搜索该人物的其他身份或职业信息"
- "尝试搜索相关事件的时间线"
- "尝试搜索与该实体有关联的其他组织或机构"
- "尝试从其他角度描述相同的信息需求"
- "尝试扩大搜索范围，包含相关背景信息"

**评估标准：**
- is_satisfied=true：检索结果中包含了能够直接或间接填补信息缺口的关键信息
- is_satisfied=false：检索结果与信息缺口无关，或信息不足以填补缺口
"""


def get_gap_satisfaction_prompt(
    question: str,
    gap_description: str,
    related_entities: List[str],
    retrieved_docs: List[Dict[str, Any]]
) -> str:
    """生成缺口补全评估 Prompt

    Args:
        question: 原始问题
        gap_description: 信息缺口描述
        related_entities: 关联实体列表
        retrieved_docs: 检索到的文档列表

    Returns:
        Prompt 字符串
    """
    entities_text = ", ".join(related_entities) if related_entities else "无"
    
    # 格式化文档
    docs_text = ""
    for i, doc in enumerate(retrieved_docs):
        doc_id = doc.get("doc_id", f"doc_{i}")
        title = doc.get("title", "未知标题")
        content = doc.get("content", "")
        docs_text += f"\n[{i+1}] {title} (ID: {doc_id})\n{content}\n"
    
    if not docs_text:
        docs_text = "无检索到的文档"

    return GAP_SATISFACTION_PROMPT.format(
        question=question,
        gap_description=gap_description,
        related_entities=entities_text,
        retrieved_docs=docs_text
    )


# ============== 统一评估与选择 Prompt ==============

UNIFIED_EVALUATE_PROMPT = """你是一个专业的信息检索评估专家。请评估检索到的文档对所有信息缺口的满足情况，并选择需要添加到上下文记忆的文档。

**原始问题：**
{question}

**当前信息缺口列表：**
{intent_list}

**已有上下文文档：**
{existing_docs}

**本轮检索到的文档（共{doc_count}个）：**
{documents}

**任务：**
1. 评估每个信息缺口是否被当前检索到的文档满足，并选择合适的状态
2. 对于状态为 continue 的缺口，给出下一步检索建议
3. 选择需要添加到上下文记忆的文档（选择对回答问题有价值的文档，包括对原始问题有帮助但不一定满足特定意图的文档，避免与已有文档冗余）

**输出格式：**
```json
{{
  "intent_evaluation": [
    {{
      "gap_id": "gap_1",
      "intent_label": "意图标签",
      "gap_description": "缺口描述",
      "status": "satisfied|partially_satisfied|manually_closed|continue",
      "reason": "判断理由",
      "next_hint": "下一步检索提示（仅 continue 时）"
    }}
  ],
  "docs_to_add": ["doc_id_1", "doc_id_2"],
  "add_reason": "选择这些文档的理由"
}
```

**状态选项说明：**
- satisfied：检索到的信息完全满足该缺口
- partially_satisfied：检索到部分相关信息，但不完整
- manually_closed：虽未直接找到信息，但可通过现有信息推理得出，建议关闭
- continue：信息不足，需要继续检索

**评估标准：**
- status=satisfied：检索到的文档中包含了能够完全填补该信息缺口的关键信息
- status=partially_satisfied：检索到部分相关信息，但不完整，可能需要继续检索
- status=manually_closed：虽未直接找到信息，但可通过现有信息推理得出，建议关闭该缺口
- status=continue：检索到的文档与信息缺口无关，或信息不足以填补缺口，需要继续检索
- docs_to_add：选择对回答问题最有价值的文档（包括对原始问题有帮助但不一定满足特定意图的文档），不需要区分它们属于哪个意图
- 避免选择与已有上下文文档重复的内容
"""


def get_unified_evaluate_prompt(
    question: str,
    intent_list: str,
    existing_docs: List[Dict[str, Any]],
    retrieved_docs: List[Dict[str, Any]]
) -> str:
    """生成统一评估与选择 Prompt
    
    Args:
        question: 原始问题
        intent_list: 格式化的信息缺口列表
        existing_docs: 已有的上下文文档
        retrieved_docs: 本轮检索到的文档
        
    Returns:
        Prompt 字符串
    """
    # 格式化已有文档
    existing_text = ""
    if existing_docs:
        for i, doc in enumerate(existing_docs):  # 显示所有已有文档
            doc_id = doc.get("doc_id", f"doc_{i}")
            title = doc.get("title", "未知标题")
            existing_text += f"[{i+1}] {title} (ID: {doc_id})\n"
    else:
        existing_text = "无"
    
    # 格式化本轮检索到的文档
    docs_text = ""
    for i, doc in enumerate(retrieved_docs):
        doc_id = doc.get("doc_id", f"doc_{i}")
        title = doc.get("title", "未知标题")
        content = doc.get("content", "")  # 不限制长度
        docs_text += f"\n[{i+1}] {title} (ID: {doc_id})\n{content}\n"
    
    if not docs_text:
        docs_text = "无检索到的文档"
    
    return UNIFIED_EVALUATE_PROMPT.format(
        question=question,
        intent_list=intent_list,
        existing_docs=existing_text,
        documents=docs_text,
        doc_count=len(retrieved_docs)
    )


# ============== ANSWER 阶段 Prompt ==============

ANSWER_GENERATE_PROMPT = """你是一个专业的问答助手。请根据收集的信息回答问题。

**问题：**
{question}

**收集到的证据：**
{evidence}

**探索历史：**
{history}

**要求：**
1. 基于证据生成准确答案
2. 如果证据不足，明确说明并基于常识推理
3. 保持答案简洁明了

**输出格式：**
```json
{{
  "answer": "答案内容",
  "confidence": 0.85,
  "reasoning": "推理过程",
  "sources": ["证据1", "证据2"]
}}
```
"""


def get_answer_prompt(
    question: str,
    evidence: List[str],
    history: str
) -> str:
    """生成答案生成 Prompt

    Args:
        question: 问题
        evidence: 证据列表（即文档格式化的内容）
        history: 探索历史

    Returns:
        Prompt 字符串
    """
    evidence_text = "\n".join(evidence)

    return ANSWER_GENERATE_PROMPT.format(
        question=question,
        evidence=evidence_text,
        history=history
    )


# ============== 意图类型描述 ==============

INTENT_TYPES_DESCRIPTION = {
    "Trace_Process": "追踪事件或过程的发展脉络",
    "Find_Reason": "查找事件发生的原因或理由",
    "Expand_Detail": "扩展和补充细节信息",
    "Bridge_Entity": "通过实体进行桥接推理",
    "Check_Conflict": "检查信息之间是否存在矛盾"
}


def get_intent_types() -> Dict[str, str]:
    """获取意图类型描述"""
    return INTENT_TYPES_DESCRIPTION.copy()
