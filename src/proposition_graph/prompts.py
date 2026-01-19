"""
LLM 提示词模板
用于命题提取、实体提取、RST 关系分析和统一文档提取
"""


# ==================== 统一文档提取提示词 ====================

UNIFIED_EXTRACTION_PROMPT = """你是一个专业的文档分析专家，负责从文档中提取结构化知识。

**文档内容：**
{document_content}

**任务：**

1. **提取原子命题**：将每个句子分解为最小、独立的语义单元。每个命题应该：
   - 表达一个完整的事实或信息
   - 不依赖上下文即可理解
   - 避免复合陈述（将 "X和Y" 拆分为两个命题）

2. **提取实体**：为每个命题识别所有实体。使用以下类型：
   - PERSON: 人物（如 "Barack Obama", "Marie Curie"）
   - ORGANIZATION: 组织/机构（如 "NASA", "Harvard University"）
   - LOCATION: 地点（如 "Hawaii", "Paris", "the White House"）
   - WORK_OF_ART: 艺术作品（如 "Mona Lisa", "To Kill a Mockingbird"）
   - EVENT: 事件（如 "World War II", "the Olympics"）
   - DATE: 时间/日期（如 "1961", "July 20, 1969", "the 1990s"）
   - CUSTOM: 自定义有价值的类型（仅当满足以下条件时使用）：
     * 必须是具体、命名的实体（不是通用术语）
     * 必须对问答有潜在价值（如特定产品、法律、科学概念）
     * 必须提供 custom_type 字段说明类型（如 "LAW", "PRODUCT", "CONCEPT"）
     * 必须提供 reason 字段解释其关键性
     * 不是：普通名词、通用引用、模糊术语

3. **识别关系**：分析命题之间的修辞关系，归类为以下两种类型：
   - **SKELETON（骨架边）**：两个命题同等重要，相互独立
     示例：SEQUENCE（顺序）、CONTRAST（对比）、CONCESSION（让步）
   - **DETAIL（细节边）**：主从依赖关系，一个命题支持另一个命题
     示例：CAUSED_BY（因果）、MOTIVATION（动机）、ELABORATION（详细说明）、BACKGROUND（背景）

**输出格式：**

Observation: <简要总结文档内容>

Thinking: <逐步说明你的分析过程>

Answer: ```json
{{
  "propositions": [
    {{
      "id": 0,
      "text": "命题的精确文本",
      "sentence_index": 0,
      "entities": [
        {{
          "text": "实体文本",
          "type": "PERSON|ORGANIZATION|LOCATION|WORK_OF_ART|EVENT|DATE|CUSTOM",
          "custom_type": "仅对CUSTOM类型：自定义类型名称（如 LAW, PRODUCT, CONCEPT）",
          "reason": "仅对CUSTOM类型：简要说明为什么这个实体对问答有价值"
        }}
      ]
    }}
  ],
  "relations": [
    {{
      "source": 0,
      "target": 1,
      "relation": "SKELETON 或 DETAIL",
      "direction": "forward|backward|bidirectional",
      "reason": "简要说明关系判断"
    }}
  ]
}}
```

**重要指南：**

1. **命题质量**：
   - 每个命题必须自包含（无需上下文即可理解）
   - 避免复合命题（拆分 "X and Y"）
   - 保留重要细节（日期、数字、名称）

2. **实体提取**：
   - 提取所有标准类型（PERSON, ORGANIZATION, LOCATION, WORK_OF_ART, EVENT, DATE）
   - 对 CUSTOM 保持谨慎：仅提取满足以下条件的实体：
     * 已命名且具体
     * 对问答有潜在价值
     * 不是通用术语（避免 "the company", "the project", "the study"）
   - 为 CUSTOM 实体提供 custom_type（类型名称）和 reason（价值说明）

3. **关系分析**：
   - 关注逻辑连接的命题
   - 仅在连接清晰且有明确意义时添加关系
   - 考虑前向和后向方向
   - 为每个关系提供简要理由

现在请处理上面的文档，并提供你的分析。
"""


def get_unified_extraction_prompt(document_content: str) -> str:
    """获取统一文档提取提示词"""
    return UNIFIED_EXTRACTION_PROMPT.format(document_content=document_content)


# ==================== 意图识别提示词 ====================

INTENT_RECOGNITION_PROMPT = """你是一个专业的意图识别专家。根据当前问题和已探索路径，预测下一步的推理意图。

**意图类型：**

1. **Trace_Process**（追踪过程）：寻找事件的后续、步骤或发展
   - 激活边：SKELETON（骨架边）
   - 适用场景：询问"发生了什么"、"接下来怎样"

2. **Find_Reason**（寻找原因）：寻找原因、动机或理由
   - 激活边：DETAIL（细节边）
   - 适用场景：询问"为什么"、"什么原因"

3. **Expand_Detail**（扩展细节）：寻找背景、解释或补充信息
   - 激活边：DETAIL（细节边）
   - 适用场景：询问"具体情况"、"背景信息"

4. **Bridge_Entity**（桥接实体）：跨越到新实体或寻找属性
   - 激活边：MENTIONS_ENTITY
   - 适用场景：需要跳转到相关实体

5. **Check_Conflict**（检查冲突）：寻找转折、矛盾或对比
   - 激活边：SKELETON（骨架边）
   - 适用场景：寻找相反观点或例外情况

**当前状态：**
- 问题：{question}
- 已探索路径：{current_path}
- 已访问实体：{visited_entities}

**输出格式：**
请以 JSON 格式输出，可以选择多个意图：
```json
{{
  "intents": [
    {{"intent": "意图类型", "confidence": 0.9, "reason": "选择理由"}},
    ...
  ],
  "summary": "简要总结当前推理状态和下一步计划"
}}
```
"""


def get_intent_recognition_prompt(
    question: str,
    current_path: str,
    visited_entities: str
) -> str:
    """获取意图识别提示词"""
    return INTENT_RECOGNITION_PROMPT.format(
        question=question,
        current_path=current_path,
        visited_entities=visited_entities
    )
