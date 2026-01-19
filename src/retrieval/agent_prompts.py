"""
Agent Prompt 模板

定义基于状态图驱动的多跳问答系统的 Prompt 模板。
参考 ggagent3 的 gap-driven exploration 设计。
"""

from typing import List, Dict, Any


# ============== CHECK_PLAN 合并阶段 Prompt ==============

CHECK_PLAN_PROMPT = """你是一个专业的多跳问答系统。请判断当前信息是否足够回答问题，如果不够，识别信息缺口并选择探索路径。

**问题：**
{question}

**已收集的证据（最近 {evidence_count} 条）：**
{evidence}

**已访问实体：**
{visited_entities}

**探索路径说明：**

系统使用两种逻辑路径进行推理探索，请根据当前信息缺口选择最合适的路径：

1. **DETAIL（细节路径）**
   - 含义：寻找补充信息、解释说明、背景细节
   - 适用场景：需要了解"具体情况"、"原因是什么"、"更多细节"
   - 示例：事件的时间、地点、原因、方式、参与者背景
   - 设计理由：主从关系，目标节点对源节点进行补充，扩展细节

2. **SIMILARITY（相似路径）**
   - 含义：寻找内容相似、相关主题、跨文档关联
   - 适用场景：需要了解"类似信息"、"其他角度"、"相关描述"
   - 示例：相似事件的描述、不同文档对同一主题的讨论
   - 设计理由：软连接，通过语义相似性连接相关信息

**路径选择建议：**
- 问题询问细节、原因、背景 → 优先选择 DETAIL
- 问题询问类似信息、其他角度 → 优先选择 SIMILARITY
- 可以组合多条路径以覆盖不同推理角度

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
      "gap_description": "需要查找的具体信息",
      "related_entities": ["实体1", "实体2"],
      "intent_label": "自定义意图描述",
      "active_edges": ["DETAIL", "SIMILARITY"]
    }}
  ],
  "summary": "简要总结当前推理状态和下一步计划"
}}
```

**要求：**
1. 如果证据包含问题的直接答案，设置 can_answer=true
2. intent_label 用于记录推理意图（自由描述，不受限制）
3. active_edges 从上述两种路径中选择，可以选择多个
4. 根据信息缺口的类型选择最合适的探索路径组合
5.gap_description请不要使用复合描述，多个信息缺口请使用多个gap_description，保持每个gap_description的独立性，方便检索。
"""


def get_check_plan_prompt(
    question: str,
    evidence: List[str],
    visited_entities: set,
    evidence_count: int = 20
) -> str:
    """生成 CHECK_PLAN 阶段的 Prompt

    Args:
        question: 问题
        evidence: 证据列表
        visited_entities: 已访问实体集合
        evidence_count: 证据数量

    Returns:
        Prompt 字符串
    """
    evidence_text = "\n".join(f"- {e}" for e in evidence[-evidence_count:])
    entities_text = ", ".join(visited_entities) if visited_entities else "无"

    return CHECK_PLAN_PROMPT.format(
        question=question,
        evidence=evidence_text,
        visited_entities=entities_text,
        evidence_count=evidence_count
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


# ============== RETRIEVE 阶段 Prompt ==============

RETRIEVE_ANCHOR_JUDGE_PROMPT = """你是一个专业的信息检索专家。请判断给定的命题节点是否可以作为检索锚点。

**问题：**
{question}

**当前信息缺口：**
{info_gaps}

**候选命题节点：**
{candidates}

**判断标准：**
1. 命题内容与问题或信息缺口相关
2. 命题包含关键实体或属性
3. 命题能够作为推理的起点

**输出格式：**
```json
{{
  "valid_anchors": ["节点ID1", "节点ID2", ...],
  "reason": "选择理由"
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

UPDATE_EXTRACT_PROMPT = """你是一个专业的信息提取专家。请从检索到的路径中提取有用的信息来回答问题。

**问题：**
{question}

**检索到的路径（按相关性排序）：**
{paths}

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


def get_extract_prompt(
    question: str,
    paths: List[Dict[str, Any]],
    max_paths: int = 10
) -> str:
    """生成信息提取 Prompt

    Args:
        question: 问题
        paths: 路径列表
        max_paths: 最大路径数

    Returns:
        Prompt 字符串
    """
    paths_text = "\n".join(
        f"{i+1}. {p.get('text', '')} (分数: {p.get('score', 0):.2f})"
        for i, p in enumerate(paths[:max_paths])
    )

    return UPDATE_EXTRACT_PROMPT.format(
        question=question,
        paths=paths_text
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
    history: str,
    max_evidence: int = 30
) -> str:
    """生成答案生成 Prompt

    Args:
        question: 问题
        evidence: 证据列表
        history: 探索历史
        max_evidence: 最大证据数

    Returns:
        Prompt 字符串
    """
    evidence_text = "\n".join(f"- {e}" for e in evidence[-max_evidence:])

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
