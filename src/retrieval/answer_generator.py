"""
答案生成器
根据收集的证据生成最终答案
"""

import json
from typing import List, Dict, Optional
from dataclasses import dataclass

from ..llm.base import BaseLLM
from .path_structures import Path
from .path_selector import RankedPath
from ..utils.llm_response_parser import LLMResponseParser


# 答案生成提示词
ANSWER_GENERATION_PROMPT = """你是一个专业的问答助手。请根据以下证据回答问题。

**问题：**
{question}

**证据：**
{evidence}

**要求：**
1. 答案必须基于提供的证据
2. 如果证据不足，明确说明
3. 答案应该简洁、准确、完整
4. 如果问题涉及多个实体或事件，确保覆盖所有方面

**输出格式：**
请以 JSON 格式输出：
```json
{{
  "answer": "你的答案",
  "confidence": 0.0-1.0,
  "reasoning": "推理过程"
}}
```
"""


@dataclass
class AnswerResult:
    """答案结果"""
    answer: str
    confidence: float
    reasoning: str

    def __repr__(self):
        return f"AnswerResult(answer={self.answer[:50]}..., confidence={self.confidence:.2f})"


class AnswerGenerator:
    """
    答案生成器

    使用 LLM 根据收集的证据生成最终答案
    """

    def __init__(
        self,
        llm: BaseLLM,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ):
        self.llm = llm
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.response_parser = LLMResponseParser(llm)

    async def generate(
        self,
        question: str,
        evidence: List[str],
        paths: Optional[List[RankedPath]] = None,
        history: Optional[str] = None,
    ) -> str:
        """
        生成答案

        Args:
            question: 问题
            evidence: 证据列表
            paths: 排序后的路径（可选）
            history: 探索历史（可选）

        Returns:
            答案文本
        """
        # 格式化证据
        evidence_text = self._format_evidence(evidence)

        # 生成提示词
        prompt = ANSWER_GENERATION_PROMPT.format(
            question=question,
            evidence=evidence_text
        )

        # 如果有历史，添加到提示词中
        if history:
            prompt = f"{prompt}\n\n**探索历史：**\n{history}"

        try:
            messages = [{"role": "user", "content": prompt}]
            response = await self.llm.generate(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            # 解析答案
            result = await self._parse_response(response)
            return result.answer

        except Exception as e:
            print(f"答案生成失败: {e}")

            # 后备：直接返回证据摘要
            return self._fallback_answer(question, evidence)

    def _format_evidence(self, evidence: List[str]) -> str:
        """格式化证据"""
        if not evidence:
            return "（无证据）"

        # 限制证据数量
        max_evidence = 20
        if len(evidence) > max_evidence:
            evidence = evidence[:max_evidence]

        formatted = []
        for i, ev in enumerate(evidence, 1):
            formatted.append(f"{i}. {ev}")

        return "\n".join(formatted)

    async def _parse_response(self, response_text: str) -> AnswerResult:
        """解析 LLM 响应（使用 LLMResponseParser）"""
        def parse_answer(json_str: str) -> AnswerResult:
            """解析答案 JSON"""
            data = json.loads(json_str)
            return AnswerResult(
                answer=data.get("answer", ""),
                confidence=data.get("confidence", 0.5),
                reasoning=data.get("reasoning", "")
            )

        # 使用批量解析器
        results = await self.response_parser.parse_batch(
            responses=[response_text],
            parse_funcs=[parse_answer],
            enable_retry=True
        )

        result, error = results[0]
        if error:
            # 降级到文本解析
            return self._fallback_parse(response_text)

        return result

    def _fallback_parse(self, response_text: str) -> AnswerResult:
        """降级解析：文本提取"""
        import re

        # 尝试找到 "answer:" 后的内容
        answer_match = re.search(r'answer["\s:]+([^\n]+)', response_text, re.IGNORECASE)
        if answer_match:
            return AnswerResult(
                answer=answer_match.group(1).strip(),
                confidence=0.5,
                reasoning=""
            )

        # 默认返回整个响应
        return AnswerResult(
            answer=response_text.strip(),
            confidence=0.3,
            reasoning="解析失败，使用原始响应"
        )

    def _fallback_answer(self, question: str, evidence: List[str]) -> str:
        """后备答案生成"""
        if not evidence:
            return "抱歉，根据现有信息无法回答该问题。"

        # 简单的证据摘要
        return f"根据相关证据，{' '.join(evidence[:3])}"
