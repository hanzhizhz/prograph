"""OpenAI API异步客户端"""

import asyncio
from typing import List, Dict, Optional
from openai import AsyncOpenAI
from .base import BaseLLM


def build_single_turn(system: Optional[str], user: str) -> List[Dict[str, str]]:
    """
    构建单轮对话的 messages

    Args:
        system: 系统消息（可选）
        user: 用户消息

    Returns:
        messages 列表
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return messages


class OpenAIClient(BaseLLM):
    """OpenAI API异步客户端，用于运行时调用"""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "qwen3",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.9,
        max_retries: int = 3,
        timeout: float = 180.0,
        concurrency: int = 100,
    ):
        """
        初始化OpenAI异步客户端

        Args:
            base_url: API基础URL
            api_key: API密钥
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大token数
            top_p: top-p采样
            max_retries: 最大重试次数
            timeout: 超时时间（秒）
            concurrency: 并发控制数量
        """
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.max_retries = max_retries
        self.timeout = timeout
        self.concurrency = concurrency

        self._client: Optional[AsyncOpenAI] = None
        self._semaphore: Optional[asyncio.Semaphore] = None

    def _ensure_client(self):
        """确保客户端已初始化"""
        if self._client is None:
            self._client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                max_retries=self.max_retries,
                timeout=self.timeout,
            )
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.concurrency)

    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """
        生成单个响应

        Args:
            messages: 输入消息列表，每条消息包含role和content字段
            temperature: 温度参数
            max_tokens: 最大token数
            top_p: top-p采样
            stop: 停止词列表
            **kwargs: 其他参数

        Returns:
            生成的文本响应
        """
        if not isinstance(messages, list):
            raise TypeError(f"messages must be a list of dicts, got {type(messages)}")

        return (await self.generate_batch(
            [messages], temperature, max_tokens, top_p, stop, **kwargs
        ))[0]

    async def generate_batch(
        self,
        messages: List[List[Dict[str, str]]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
        batch_size: Optional[int] = None,
        **kwargs
    ) -> List[str]:
        """
        批量生成响应（并发请求）

        Args:
            messages: 输入消息列表的列表，每个元素是一个消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            top_p: top-p采样
            stop: 停止词列表
            batch_size: 并发批次大小
            **kwargs: 其他参数

        Returns:
            生成的文本响应列表
        """
        self._ensure_client()

        # 使用配置的默认值
        if temperature is None:
            temperature = self.temperature
        if max_tokens is None:
            max_tokens = self.max_tokens
        if batch_size is None:
            batch_size = self.concurrency

        results = []

        # 分批处理以控制并发数
        for i in range(0, len(messages), batch_size):
            batch_messages = messages[i:i + batch_size]
            batch_tasks = [
                self._generate_single(msg_list, temperature, max_tokens, top_p, stop, **kwargs)
                for msg_list in batch_messages
            ]
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)

        return results

    async def _generate_single(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        top_p: float,
        stop: Optional[List[str]],
        **kwargs
    ) -> str:
        """单个请求的生成逻辑"""
        try:
            # 验证消息格式
            for msg in messages:
                if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                    raise ValueError(f"Invalid message format: {msg}. Expected dict with 'role' and 'content' keys.")

            async with self._semaphore:
                response = await self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stop=stop or [],
                    **kwargs
                )
            return response.choices[0].message.content
        except Exception as e:
            print(f"生成错误: {e}, messages: {str(messages)[:100]}...")
            raise

    async def close(self):
        """关闭客户端"""
        if self._client:
            await self._client.close()
