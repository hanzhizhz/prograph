"""
统一文档提取器
将命题提取、实体提取、关系分析合并为单个 LLM 调用
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from ..llm.base import BaseLLM
from .document_loader import Document
from .proposition_extractor import Proposition
from .entity_extractor import Entity
from .rst_analyzer import RSTRelation
from ..utils.json_parser import retry_parse_with_llm


# 实体类型定义
ENTITY_TYPES = {
    "PERSON": "人物（真实人物、虚构角色、职位名称）",
    "ORGANIZATION": "组织机构（公司、政党、乐队、运动队、政府机构、学校）",
    "LOCATION": "地点（国家、城市、州、山脉、河流、建筑）",
    "WORK_OF_ART": "艺术作品（电影、书籍、歌曲、专辑、绘画、电视节目）",
    "EVENT": "事件（战争、体育赛事、历史事件、节日）",
    "DATE": "时间/日期（具体日期、年份、时代）",
    "CUSTOM": "自定义类型（有价值的实体，需提供 custom_type 和 reason）"
}

# 关系类型定义
RELATION_TYPES = {
    # 骨架边（Nucleus-Nucleus，同等重要）
    "SEQUENCE": "顺序关系，事件按时间顺序发生",
    "CONTRAST": "对比关系，两个命题表达相反或对比的观点",
    "CONCESSION": "让步关系，尽管有某种情况，但另一种情况仍然成立",
    # 细节边（Nucleus-Satellite，主从关系）
    "CAUSED_BY": "因果关系（命题2导致命题1，命题1是核心）",
    "MOTIVATION": "动机关系（命题2解释命题1的动机）",
    "ELABORATION": "详细说明（命题2对命题1进行详细阐述）",
    "BACKGROUND": "背景关系（命题2提供命题1的背景信息）"
}


@dataclass
class EntityWithReason:
    """带理由的实体"""
    text: str
    type: str
    custom_type: Optional[str] = None  # CUSTOM 类型必须提供类型名称
    reason: Optional[str] = None  # CUSTOM 类型必须提供价值说明


@dataclass
class PropositionWithEntities:
    """带实体的命题"""
    id: int
    text: str
    sentence_index: int
    entities: List[EntityWithReason] = field(default_factory=list)


@dataclass
class UnifiedExtractionResult:
    """统一提取结果"""
    propositions: List[PropositionWithEntities]
    relations: List[dict]


def get_unified_extraction_prompt() -> str:
    """获取统一提取提示词"""
    return """你是一个专业的文档分析专家，负责从文档中提取结构化知识。

**文档内容：**
{document_content}

**任务：**

1. **提取原子命题**：将每个句子分解为最小、独立的语义单元。每个命题应该：
   - 表达一个完整的事实或信息
   - 不依赖上下文即可理解
   - 避免复合陈述（将 "X和Y" 拆分为两个命题）
   - 避免使用代词，如你我他，恢复完整的实体主体。

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


class UnifiedDocumentExtractor:
    """
    统一文档提取器

    将命题提取、实体提取、关系分析合并为单个 LLM 调用
    """

    def __init__(
        self,
        llm: BaseLLM,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ):
        self.llm = llm
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def extract_from_document(
        self,
        document: Document,
    ) -> Tuple[List[Proposition], List[Entity], List[RSTRelation]]:
        """
        从文档中提取命题、实体和关系

        Args:
            document: 文档对象

        Returns:
            (命题列表, 实体列表, 关系列表)

        Raises:
            Exception: 如果解析失败
        """
        # 1. 构建消息
        messages = self._build_messages(document)

        # 2. 调用 LLM
        response = await self.llm.generate(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        # 3. 解析响应
        result = self._parse_response(response, document.doc_id)

        # 4. 转换为现有数据结构
        propositions = self._to_propositions(result.propositions, document.doc_id)
        entities = self._to_entities(result.propositions, document.doc_id)
        relations = self._to_relations(result.relations)

        return propositions, entities, relations

    async def extract_from_documents(
        self,
        documents: List[Document],
    ) -> Tuple[
        List[List[Proposition]],   # 每个文档的命题列表
        List[List[Entity]],        # 每个文档的实体列表
        List[List[RSTRelation]],   # 每个文档的关系列表
        List[bool]                # 每个文档的解析成功标记
    ]:
        """
        批量从文档中提取命题、实体和关系

        利用 vLLM 的批量推理能力，一次性处理多个文档。

        Args:
            documents: 文档列表

        Returns:
            (命题列表的列表, 实体列表的列表, 关系列表的列表, 成功标记列表)
        """
        if not documents:
            return [], [], [], []

        # 1. 构建所有 messages
        messages_list = [self._build_messages(doc) for doc in documents]

        # 2. 批量调用 LLM
        try:
            responses = await self.llm.generate_batch(
                messages=messages_list,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        except Exception as e:
            print(f"批量 LLM 调用失败: {e}")
            # 返回全部失败
            empty = [[] for _ in documents]
            failed = [False] * len(documents)
            return empty, empty, empty, failed

        # 3. 解析所有响应
        all_propositions: List[List[Proposition]] = []
        all_entities: List[List[Entity]] = []
        all_relations: List[List[RSTRelation]] = []
        success_flags: List[bool] = []

        # responses 已经是字符串列表
        response_texts = responses
        parse_funcs = [lambda text, doc_id=doc.doc_id: self._parse_response(text, doc_id) for doc in documents]

        # 使用错误恢复机制
        parse_results, _ = await retry_parse_with_llm(
            llm_client=self.llm,
            messages=messages_list,
            responses=response_texts,
            parse_funcs=parse_funcs,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        # 处理解析结果
        for i, result in enumerate(parse_results):
            if result is not None:
                # 解析成功
                props = self._to_propositions(result.propositions, documents[i].doc_id)
                ents = self._to_entities(result.propositions, documents[i].doc_id)
                rels = self._to_relations(result.relations)

                all_propositions.append(props)
                all_entities.append(ents)
                all_relations.append(rels)
                success_flags.append(True)
            else:
                # 解析失败
                all_propositions.append([])
                all_entities.append([])
                all_relations.append([])
                success_flags.append(False)
                print(f"文档解析失败: {documents[i].title[:50]}...")

        return all_propositions, all_entities, all_relations, success_flags

    def _build_messages(self, document: Document) -> List[Dict[str, str]]:
        """构建统一提取消息"""
        # 将句子编号
        numbered_content = "\n".join(
            f"[{i}] {sent}"
            for i, sent in enumerate(document.content)
        )

        prompt = get_unified_extraction_prompt().format(
            document_content=numbered_content
        )

        return [{"role": "user", "content": prompt}]

    def _parse_response(
        self,
        response_text: str,
        doc_id: Optional[str] = None,
    ) -> UnifiedExtractionResult:
        """解析 LLM 响应"""
        _ = doc_id  # 保留参数用于调试
        import json
        # 提取 JSON
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # 宽松匹配：从第一个 { 到最后一个 }
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                raise ValueError(f"无法从响应中提取 JSON: {response_text}")

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON 解析失败: {e}")

        # 解析命题
        propositions = []
        for p in data.get("propositions", []):
            entities = [
                EntityWithReason(
                    text=e["text"],
                    type=e["type"],
                    custom_type=e.get("custom_type"),
                    reason=e.get("reason")
                )
                for e in p.get("entities", [])
            ]
            propositions.append(PropositionWithEntities(
                id=p["id"],
                text=p["text"],
                sentence_index=p["sentence_index"],
                entities=entities
            ))

        # 解析关系
        relations = data.get("relations", [])

        return UnifiedExtractionResult(
            propositions=propositions,
            relations=relations
        )

    def _to_propositions(
        self,
        props: List[PropositionWithEntities],
        doc_id: str,
    ) -> List[Proposition]:
        """转换为 Proposition 对象"""
        return [
            Proposition(
                text=p.text,
                sent_idx=p.sentence_index,
                prop_idx=p.id,
                doc_id=doc_id
            )
            for p in props
        ]

    def _to_entities(
        self,
        props: List[PropositionWithEntities],
        doc_id: str,
    ) -> List[Entity]:
        """转换为 Entity 对象"""
        entities = []
        for prop in props:
            for entity in prop.entities:
                entities.append(Entity(
                    text=entity.text,
                    type=entity.type,
                    custom_type=entity.custom_type,
                    reason=entity.reason,
                    prop_idx=prop.id,
                    doc_id=doc_id
                ))
        return entities

    def _to_relations(
        self,
        relations: List[dict],
    ) -> List[RSTRelation]:
        """转换为 RSTRelation 对象"""
        result = []
        for r in relations:
            # 映射 direction: "forward" -> "1->2", "backward" -> "2->1", "bidirectional" -> "1<->2"
            direction_map = {
                "forward": "1->2",
                "backward": "2->1",
                "bidirectional": "1<->2"
            }
            direction = direction_map.get(r.get("direction", "forward"), "1->2")

            result.append(RSTRelation(
                source_idx=r["source"],
                target_idx=r["target"],
                relation=r["relation"],
                direction=direction,
                reason=r.get("reason", "")
            ))
        return result
