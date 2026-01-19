"""JSON解析工具 - 多层兜底解析 + LLM纠错重试"""

import json
import logging
import re
from typing import List, Any, Callable, Dict, Optional, Tuple
from dataclasses import dataclass

from .types import ParseErrorRecord

logger = logging.getLogger(__name__)


def extract_json_with_fallback(
    response: str,
    target_field: Optional[str] = None
) -> Optional[str]:
    """
    多层兜底提取JSON字符串

    五层解析策略：
    1. 匹配 ```json``` 代码块
    2. 匹配 ``` 代码块（不带 json 标记）
    3. 大括号计数法提取所有完整JSON对象，优先选择包含目标字段的
    4. 使用第一个完整的JSON对象
    5. 简单正则匹配兜底

    Args:
        response: LLM响应文本
        target_field: 期望包含的字段名（如 "entities", "propositions"）

    Returns:
        提取的JSON字符串，如果未找到则返回None
    """
    json_str = None

    # 第1层：优先匹配 ```json``` 代码块
    json_code_block_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
    if json_code_block_match:
        json_str = json_code_block_match.group(1).strip()
        logger.debug("使用第1层解析：```json``` 代码块")
    else:
        # 第2层：匹配 ``` 代码块（不带 json 标记）
        code_block_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response, re.DOTALL)
        if code_block_match:
            json_str = code_block_match.group(1).strip()
            logger.debug("使用第2层解析：``` 代码块")

    # 第3层：大括号计数法（正确处理嵌套）
    if json_str is None:
        potential_jsons = []
        brace_count = 0
        start_idx = -1

        for i, char in enumerate(response):
            if char == '{':
                if start_idx == -1:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    # 找到了一个完整的JSON对象
                    potential_json = response[start_idx:i + 1]
                    potential_jsons.append(potential_json)
                    start_idx = -1  # 重置，继续查找下一个

        # 尝试解析每个潜在的JSON对象，找到包含目标字段的
        if target_field:
            for potential_json in potential_jsons:
                try:
                    data = json.loads(potential_json)
                    if target_field in data:
                        json_str = potential_json
                        logger.debug(f"使用第3层解析：找到包含 '{target_field}' 字段的JSON")
                        break
                except json.JSONDecodeError:
                    continue

        # 如果没找到包含目标字段的，使用第一个完整的JSON对象
        if json_str is None and potential_jsons:
            json_str = potential_jsons[0]
            logger.debug("使用第3层解析：第一个完整的JSON对象")

    # 第4层：使用第一个完整的JSON对象（如果第3层没找到）
    if json_str is None:
        # 尝试简单的正则匹配
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            logger.debug("使用第4层解析：简单正则匹配")

    # 第5层：兜底，直接使用原始响应
    if json_str is None:
        json_str = response
        logger.debug("使用第5层解析：原始响应")

    return json_str


class JsonParserWithRetry:
    """带纠错重试的JSON解析器

    从老项目移植，提供：
    1. 多层兜底解析
    2. LLM纠错重试
    3. 完整错误记录
    """

    def __init__(self):
        """初始化JSON解析器"""
        pass

    async def parse_with_retry(
        self,
        llm,
        original_messages: List[Dict[str, str]],
        response: str,
        parse_func: Callable[[str], Any],
        expected_format: str,
        stage: str = "UNKNOWN",
        module: str = "unknown",
        target_field: Optional[str] = None
    ) -> Tuple[Any, Optional[ParseErrorRecord]]:
        """
        尝试解析，失败则进行一次纠错重试

        Args:
            llm: LLM客户端
            original_messages: 原始对话消息列表
            response: LLM响应文本
            parse_func: 解析函数，接受response字符串，返回解析结果
            expected_format: 期望的JSON格式描述
            stage: 发生错误的阶段（如 "EXTRACT", "LINK"）
            module: 发生错误的模块名
            target_field: 期望包含的字段名（用于多层解析）

        Returns:
            (解析结果, 错误记录 or None)
            如果解析成功，返回 (解析结果, None)
            如果解析失败（包括重试后仍失败），返回 (None, ParseErrorRecord)
        """
        # 先使用多层兜底提取JSON
        extracted_json = extract_json_with_fallback(response, target_field)

        # 第一次尝试解析
        try:
            result = parse_func(extracted_json)
            return result, None
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            error_message = str(e)
            logger.warning(f"[{stage}/{module}] JSON解析失败，尝试纠错: {error_message}")
            logger.debug(f"[{stage}/{module}] 原始响应:\n{response}")

        # 解析失败，进行纠错重试
        try:
            # 构建纠错消息（保留完整对话历史）
            retry_messages = original_messages.copy()
            retry_prompt = self._build_retry_prompt(response, error_message, expected_format)
            retry_messages.append({
                "role": "assistant",
                "content": response
            })
            retry_messages.append({
                "role": "user",
                "content": retry_prompt
            })

            # 调用LLM进行纠错
            retry_response = await llm.generate(retry_messages)

            # 使用多层兜底提取纠错后的JSON
            extracted_retry_json = extract_json_with_fallback(retry_response, target_field)

            # 尝试解析纠错后的响应
            try:
                result = parse_func(extracted_retry_json)
                logger.info(f"[{stage}/{module}] 纠错后解析成功")
                return result, None
            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as retry_e:
                # 纠错后仍失败
                retry_error = str(retry_e)
                logger.error(f"[{stage}/{module}] 纠错后仍解析失败: {retry_error}")
                logger.debug(f"[{stage}/{module}] 纠错响应:\n{retry_response}")

                # 创建错误记录
                error_record = ParseErrorRecord(
                    stage=stage,
                    module=module,
                    original_response=response,
                    error_message=error_message,
                    retry_response=retry_response,
                    retry_error=retry_error
                )
                return None, error_record

        except Exception as e:
            # 纠错调用本身失败
            logger.error(f"[{stage}/{module}] 纠错调用失败: {e}", exc_info=True)
            error_record = ParseErrorRecord(
                stage=stage,
                module=module,
                original_response=response,
                error_message=error_message,
                retry_response=None,
                retry_error=f"纠错调用失败: {str(e)}"
            )
            return None, error_record

    def _build_retry_prompt(
        self,
        original_response: str,
        error_message: str,
        expected_format: str
    ) -> str:
        """
        构建纠错提示

        Args:
            original_response: 原始响应
            error_message: 错误信息
            expected_format: 期望的JSON格式描述

        Returns:
            纠错提示文本
        """
        # 使用完整响应（不再截断）
        prompt = f"""你的上一次输出无法被正确解析为JSON格式。

错误信息：{error_message}

你的输出：
{original_response}

请重新输出，确保：
1. JSON格式正确，使用```json```代码块包裹
2. 符合期望的格式：{expected_format}

请直接输出修正后的JSON，不要添加其他解释。"""

        return prompt


async def retry_parse_with_llm(
    llm_client,
    messages: List[List[Dict[str, str]]],
    responses: List[str],
    parse_funcs: List[Callable[[str], Any]],
    temperature: float = 0.1,
    max_tokens: int = 4096
) -> tuple:
    """
    并行解析并对失败的项目进行重试（批量处理版本）

    Args:
        llm_client: LLM客户端
        messages: 消息列表，每个元素是消息列表
        responses: 第一轮响应列表
        parse_funcs: 解析函数列表
        temperature: 温度参数
        max_tokens: 最大token数

    Returns:
        (results, failed_indices)
        - results: 解析结果列表
        - failed_indices: 失败的项目索引列表
    """
    # 第一轮解析
    results = []
    failed_indices = []
    errors = []

    for i, (response, parse_func) in enumerate(zip(responses, parse_funcs)):
        try:
            result = parse_func(response)
            results.append(result)
            errors.append(None)
        except Exception as e:
            failed_indices.append(i)
            results.append(None)
            errors.append(str(e))
            logger.warning(f"[第一轮失败] 索引={i}, 错误: {str(e)[:200]}")
            logger.debug(f"[第一轮失败] 响应内容:\n{response}")

    # 如果没有失败的，直接返回
    if not failed_indices:
        return results, []

    # 第二轮：收集所有重试消息，然后批量调用
    logger.info(f"开始重试 {len(failed_indices)} 个失败项目...")
    retry_messages_list = []

    for idx in failed_indices:
        original_messages = messages[idx]
        previous_response = responses[idx]

        retry_messages = original_messages + [
            {"role": "assistant", "content": previous_response},
            {"role": "user", "content": f"你上一次的响应格式不正确，错误信息：{errors[idx]}。请重新输出为有效的JSON格式，确保JSON语法正确。"}
        ]
        retry_messages_list.append(retry_messages)

    # 一次性批量重试
    logger.info(f"发送 {len(retry_messages_list)} 个重试请求...")
    retry_results = await llm_client.generate_batch(
        retry_messages_list,
        temperature=temperature,
        max_tokens=max_tokens
    )

    # 更新失败项的结果
    still_failed = []
    for i, idx in enumerate(failed_indices):
        retry_text = retry_results[i]
        try:
            result = parse_funcs[idx](retry_text)
            results[idx] = result
            logger.info(f"[重试成功] 原始索引={idx}")
        except Exception as e:
            still_failed.append(idx)
            logger.warning(f"[重试失败] 原始索引={idx}, 错误: {str(e)}\n")
            logger.debug(f"[重试失败] 重试响应: {retry_text}\n\n")

    # 更新failed_indices，只保留真正失败的项目
    original_failed_count = len(failed_indices)
    failed_indices[:] = still_failed

    success_count = original_failed_count - len(still_failed)
    logger.info(f"重试完成: 成功 {success_count}/{original_failed_count}, 仍失败 {len(still_failed)}")

    return results, failed_indices
