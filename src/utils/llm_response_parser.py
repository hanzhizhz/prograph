"""
LLM 响应统一批量解析器

简洁设计：仅一个批量解析接口，重试可选
"""

import json
import logging
from typing import List, Dict, Any, Callable, Tuple, Optional, TypeVar
from .json_parser import extract_json_with_fallback

logger = logging.getLogger(__name__)

T = TypeVar('T')


class LLMResponseParser:
    """LLM 响应批量解析器

    使用方法：
        parser = LLMResponseParser(llm_client)

        # 批量解析（默认启用重试）
        results = await parser.parse_batch(
            responses=responses,
            parse_funcs=parse_funcs
        )

        # 批量解析（禁用重试）
        results = await parser.parse_batch(
            responses=responses,
            parse_funcs=parse_funcs,
            enable_retry=False
        )

    返回: List[Tuple[result, error]]
        - result: 解析结果，失败为 None
        - error: 错误信息，成功为 None
    """

    def __init__(self, llm_client):
        self.llm = llm_client

    async def parse_batch(
        self,
        responses: List[str],
        parse_funcs: List[Callable[[str], T]],
        enable_retry: bool = True,
        retry_prompt: str = None
    ) -> List[Tuple[Optional[T], Optional[Dict[str, str]]]]:
        """批量解析 LLM 响应

        Args:
            responses: LLM 响应文本列表
            parse_funcs: 解析函数列表（与 responses 一一对应）
            enable_retry: 是否启用 LLM 纠错重试（默认 True）
            retry_prompt: 自定义重试提示（可选）

        Returns:
            List[Tuple[result, error]] - 每个元素是 (解析结果, 错误信息)
        """
        if len(responses) != len(parse_funcs):
            raise ValueError("responses 和 parse_funcs 长度必须相同")

        # 第一轮：串行解析所有响应
        results = []
        failed_indices = []

        for i, (response, parse_func) in enumerate(zip(responses, parse_funcs)):
            try:
                # 使用多层兜底提取 JSON
                json_str = extract_json_with_fallback(response)
                if json_str:
                    result = parse_func(json_str)
                    results.append((result, None))
                else:
                    raise ValueError("未能提取 JSON")
            except Exception as e:
                error_msg = str(e)[:200]  # 限制错误信息长度
                results.append((None, {"error": error_msg, "index": i}))
                failed_indices.append(i)
                logger.debug(f"[索引 {i}] 解析失败: {error_msg}")

        # 如果没有失败或禁用重试，直接返回
        if not failed_indices or not enable_retry:
            return results

        # 第二轮：LLM 批量纠错重试
        logger.info(f"开始对 {len(failed_indices)} 个失败项进行 LLM 纠错重试...")
        retry_results = await self._retry_failed(
            [responses[i] for i in failed_indices],
            [parse_funcs[i] for i in failed_indices],
            retry_prompt
        )

        # 更新失败项的结果
        for idx, (result, error) in zip(failed_indices, retry_results):
            results[idx] = (result, error)

        return results

    async def _retry_failed(
        self,
        failed_responses: List[str],
        failed_parse_funcs: List[Callable],
        custom_prompt: str = None
    ) -> List[Tuple[Optional[T], Optional[Dict[str, str]]]]:
        """对失败的响应进行 LLM 纠错重试

        Args:
            failed_responses: 失败的响应列表
            failed_parse_funcs: 对应的解析函数列表
            custom_prompt: 自定义重试提示

        Returns:
            重试结果列表
        """
        retry_results = []

        for response, parse_func in zip(failed_responses, failed_parse_funcs):
            try:
                # 获取第一次解析的错误信息
                try:
                    # 尝试解析以获取具体错误
                    extract_json_with_fallback(response)
                    json.loads(response)  # 触发以获取错误
                    error_msg = "未知错误"
                except Exception as e:
                    error_msg = str(e)[:200]

                # 构建纠错提示（包含错误信息）
                prompt = custom_prompt or self._build_default_retry_prompt(response, error_msg)

                # 调用 LLM 纠错
                retry_response = await self.llm.generate(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=2048
                )

                # 解析纠错后的响应
                json_str = extract_json_with_fallback(retry_response)
                if json_str:
                    result = parse_func(json_str)
                    retry_results.append((result, None))
                    logger.info("纠错重试成功")
                else:
                    raise ValueError("纠错后仍无法提取 JSON")

            except Exception as e:
                error_msg = f"纠错失败: {str(e)[:200]}"
                retry_results.append((None, {"error": error_msg}))
                logger.warning(error_msg)

        return retry_results

    def _build_default_retry_prompt(self, original_response: str, error_message: str) -> str:
        """构建默认纠错提示（包含错误信息）

        Args:
            original_response: 原始响应
            error_message: 解析错误信息

        Returns:
            纠错提示
        """
        return f"""你的上一次输出无法被正确解析为 JSON 格式。

**错误信息：**
{error_message}

你的输出：
{original_response}

请重新输出，确保：
1. JSON 格式正确，使用 ```json``` 代码块包裹
2. 所有字符串正确转义
3. 所有大括号正确配对
4. 不要添加其他解释

请直接输出修正后的 JSON。"""
