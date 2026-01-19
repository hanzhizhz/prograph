"""
实体链接器
使用 LLM 进行精确的实体链接判断
"""

import asyncio
import json
import re
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass

from ..llm.base import BaseLLM, LLMResponse
from .candidate_generator import CandidatePair, EntityCandidateGroup
from ..utils.json_parser import retry_parse_with_llm


@dataclass
class LinkingDecision:
    """链接决策"""
    id1: str
    id2: str
    should_link: bool
    relation_type: Optional[str] = None  # for propositions
    direction: Optional[str] = "1->2"    # 边方向: "1->2", "2->1", or "1<->2"
    confidence: float = 0.0
    reason: str = ""

    def __repr__(self):
        return f"LinkingDecision({self.id1[:30]}... <-> {self.id2[:30]}..., link={self.should_link})"


@dataclass
class FusedEntity:
    """单个融合实体"""
    original_ids: List[str]  # 被融合的原始实体 ID 列表
    fused_text: str  # 融合后的实体文本
    fused_type: str  # 融合后的实体类型
    confidence: float  # 0.0-1.0
    reason: str  # 融合理由

    def __repr__(self):
        return f"FusedEntity({len(self.original_ids)} entities -> {self.fused_text})"


@dataclass
class EntityFusionResult:
    """实体融合结果 - LLM 可能返回多个融合实体"""
    fused_entities: List[FusedEntity]  # 融合后的实体列表

    def __repr__(self):
        return f"EntityFusionResult({len(self.fused_entities)} fused entities)"


# 实体融合提示词（三段式回答）
ENTITY_FUSION_PROMPT = """你是一个专业的实体融合专家。请分析以下实体是否指代同一个现实世界对象，并决定如何融合。

**候选实体列表（共 {count} 个）：**
{entities_info}

**判断原则：**
1. **同名或高度相似的实体**可能是同一对象（如 "Barack Obama"、"Obama"、"巴拉克·奥巴马"）
2. **必须仔细分析每个实体的文档上下文**：
   - 如果多个文档描述的是同一个人物/地点/事件，则这些实体应该融合
   - 如果明显描述不同的事物，则不应该融合
3. **融合策略**：
   - 可以返回多个融合实体（部分融合）
   - 可以完全不融合（返回空列表）
   - 可以全部融合为一个实体

**输出格式（三段式）：**

Observation:
[观察到的关键信息，如实体名称、类型、上下文摘要]

Thinking:
[思考过程，分析哪些实体可能指代同一对象，哪些是不同的对象]

Answer:
```json
{{
  "fused_entities": [
    {{
      "original_ids": ["实体ID1", "实体ID2", ...],
      "fused_text": "融合后的实体文本（选择最简洁准确的表示）",
      "fused_type": "PERSON/ORGANIZATION/LOCATION/DATE/NUMBER/MISC",
      "confidence": 0.0-1.0,
      "reason": "融合理由"
    }}
  ]
}}
```

注意：
- 如果没有实体应该融合，返回 `{{"fused_entities": []}}`
- 如果应该形成多个融合实体，在 `fused_entities` 数组中返回多个对象
"""


# 命题关系提示词
PROPOSITION_RELATION_PROMPT = """你是一个专业的命题关系分析专家。请判断两个命题之间的关系。

**命题 1:**
{proposition1}
来源文档内容:
{context1}

**命题 2:**
{proposition2}
来源文档内容:
{context2}

**关系类型（仅输出以下四种之一）：**

1. **SKELETON**（骨架边）- 两个命题同等重要，构成文档骨架
   - 包括：时序关系（SEQUENCE）、并列关系、对比关系（CONTRAST）、让步关系（CONCESSION）

2. **DETAIL**（细节边）- 主从依赖关系，一个命题对另一个起补充说明作用
   - 包括：因果关系（CAUSED_BY）、动机关系（MOTIVATION）、详细说明（ELABORATION）、背景关系（BACKGROUND）

3. **SIMILARITY**（软连接边）- 两个命题包含相似内容或相关实体，存在语义关联
   - 用于跨文档的相似命题连接

4. **NO_RELATION** - 无明确关系

**判断原则：**
- 优先判断骨架关系（SKELETON），保持文档结构完整性
- 补充说明性关系归为 DETAIL
- 跨文档的相似命题归为 SIMILARITY
- **参考完整文档内容**：理解命题在各自文档中的上下文

**输出格式：**
请以 JSON 格式输出：
```json
{{
  "relation": "SKELETON 或 DETAIL 或 SIMILARITY 或 NO_RELATION",
  "direction": "1->2" 或 "2->1",
  "confidence": 0.0-1.0,
  "reason": "判断理由"
}}
```
"""


class EntityLinker:
    """
    实体链接器

    使用 LLM 对候选对进行精确判断
    """

    def __init__(
        self,
        llm: BaseLLM,
    ):
        self.llm = llm

    async def link_propositions(
        self,
        candidates: List[CandidatePair],
    ) -> List[LinkingDecision]:
        """
        链接命题候选对（vLLM 离线处理，内部已有批量优化）

        Args:
            candidates: 候选对列表

        Returns:
            链接决策列表
        """
        if not candidates:
            return []

        # vLLM 离线处理不需要分 batch，直接一次性处理所有候选对
        return await self._process_proposition_batch(candidates)

    async def fuse_entity_groups(
        self,
        groups: List[EntityCandidateGroup],
    ) -> List[EntityFusionResult]:
        """
        批量处理实体组融合（使用三段式回答）

        Args:
            groups: 实体候选组列表

        Returns:
            融合结果列表
        """
        if not groups:
            return []

        # 构建消息
        messages_list = []
        parse_funcs = []

        for group in groups:
            entities_info = self._format_entities_for_fusion(group)
            prompt = ENTITY_FUSION_PROMPT.format(
                count=len(group.entity_ids),
                entities_info=entities_info
            )
            messages_list.append([{"role": "user", "content": prompt}])
            parse_funcs.append(lambda text, g=group: self._parse_fusion_response(text, g))

        # 批量调用
        responses = await self.llm.generate_batch(messages_list, temperature=0.1)
        response_texts = responses

        # 并行解析并重试失败的项目
        results, failed_indices = await retry_parse_with_llm(
            llm_client=self.llm,
            messages=messages_list,
            responses=response_texts,
            parse_funcs=parse_funcs,
            temperature=0.1
        )

        # 处理失败的项目
        for idx in failed_indices:
            group = groups[idx]
            print(f"实体融合解析失败: {group.group_id}")
            results[idx] = EntityFusionResult(fused_entities=[])

        return results

    def _format_entities_for_fusion(self, group: EntityCandidateGroup) -> str:
        """格式化实体信息用于融合提示词"""
        entities_info = ""
        for i, (eid, text, etype, context) in enumerate(zip(
            group.entity_ids, group.entity_texts, group.entity_types, group.contexts
        )):
            # 完整上下文
            context_preview = context
            entities_info += f"""
**实体 {i + 1}:**
- ID: {eid}
- 文本: {text}
- 类型: {etype}
- 来源文档内容:
{context_preview}
"""
        return entities_info

    async def _process_proposition_batch(
        self,
        batch: List[CandidatePair],
    ) -> List[LinkingDecision]:
        """处理一批命题候选对"""
        # 构建消息
        messages_list = []
        parse_funcs = []

        for pair in batch:
            # 准备文档内容
            context1 = pair.context1 if pair.context1 else "（无文档内容）"
            context2 = pair.context2 if pair.context2 else "（无文档内容）"

            prompt = PROPOSITION_RELATION_PROMPT.format(
                proposition1=pair.text1,
                proposition2=pair.text2,
                context1=context1,
                context2=context2,
            )
            messages_list.append([{"role": "user", "content": prompt}])
            parse_funcs.append(lambda text, p=pair: self._parse_proposition_response(text, p.id1, p.id2))

        # 批量调用
        responses = await self.llm.generate_batch(messages_list, temperature=0.2)
        response_texts = responses

        # 并行解析并重试失败的项目
        results, failed_indices = await retry_parse_with_llm(
            llm_client=self.llm,
            messages=messages_list,
            responses=response_texts,
            parse_funcs=parse_funcs,
            temperature=0.2
        )

        # 处理失败的项目
        for idx in failed_indices:
            pair = batch[idx]
            print(f"命题关系分析解析失败: {pair.id1} vs {pair.id2}")
            results[idx] = LinkingDecision(
                id1=pair.id1,
                id2=pair.id2,
                should_link=False,
                relation_type=None,
                confidence=0.0,
                reason="Parse error"
            )

        return results

    def _parse_proposition_response(
        self,
        response_text: str,
        id1: str,
        id2: str,
    ) -> LinkingDecision:
        """解析命题关系响应"""
        # 提取 JSON
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if not json_match:
            json_match = re.search(r'\{\s*"relation"\s*:\s*"[^"]+"', response_text)

        if json_match:
            try:
                json_str = json_match.group(1) if '```' in json_match.group(0) else json_match.group(0)
                # 补全 JSON
                if not json_str.strip().endswith('}'):
                    json_str += '}'

                data = json.loads(json_str)
                relation = data.get("relation", "NO_RELATION")
                direction = data.get("direction", "1->2")

                return LinkingDecision(
                    id1=id1,
                    id2=id2,
                    should_link=(relation != "NO_RELATION"),
                    relation_type=relation if relation != "NO_RELATION" else None,
                    direction=direction,
                    confidence=data.get("confidence", 0.0),
                    reason=data.get("reason", "")
                )
            except json.JSONDecodeError:
                pass

        # 默认无关系
        return LinkingDecision(
            id1=id1,
            id2=id2,
            should_link=False,
            relation_type=None,
            direction="1->2",
            confidence=0.0,
            reason="Parse failed"
        )

    def _parse_fusion_response(
        self,
        response_text: str,
        group: EntityCandidateGroup,
    ) -> EntityFusionResult:
        """解析实体融合响应（三段式回答）"""
        # 提取 JSON（支持三段式回答中的 Answer 部分）
        json_match = re.search(r'Answer:\s*```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if not json_match:
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if not json_match:
            json_match = re.search(r'\{\s*"fused_entities"\s*:\s*\[', response_text)
            if json_match:
                # 尝试提取完整的 JSON
                brace_count = 0
                start = json_match.start()
                for i, char in enumerate(response_text[start:], start):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_str = response_text[start:i + 1]
                            break
                else:
                    json_str = json_match.group(0)
            else:
                json_str = None
        else:
            json_str = json_match.group(1)

        if json_str:
            try:
                data = json.loads(json_str)
                fused_entities_list = data.get("fused_entities", [])

                fused_entities = []
                for fe_data in fused_entities_list:
                    original_ids = fe_data.get("original_ids", [])
                    # 验证 original_ids 是否都在 group 中
                    valid_ids = [eid for eid in original_ids if eid in group.entity_ids]

                    if len(valid_ids) >= 2:  # 至少 2 个实体才能融合
                        fused_entities.append(FusedEntity(
                            original_ids=valid_ids,
                            fused_text=fe_data.get("fused_text", valid_ids[0]),
                            fused_type=fe_data.get("fused_type", "MISC"),
                            confidence=fe_data.get("confidence", 0.0),
                            reason=fe_data.get("reason", "")
                        ))

                return EntityFusionResult(fused_entities=fused_entities)

            except (json.JSONDecodeError, KeyError) as e:
                print(f"融合响应解析失败: {e}")

        # 默认返回空融合结果
        return EntityFusionResult(fused_entities=[])
