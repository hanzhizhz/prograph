"""
LLM 基础抽象接口
定义 LLM 和 Embedding 服务的统一接口
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """LLM 响应"""
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str


@dataclass
class EmbeddingResponse:
    """Embedding 响应"""
    embeddings: List[List[float]]
    model: str
    prompt_tokens: int


class BaseLLM(ABC):
    """LLM 基础抽象类"""

    @abstractmethod
    async def generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        生成单个响应

        Args:
            messages: 输入消息列表，每条消息包含role和content字段
            **kwargs: 其他参数（temperature, max_tokens, top_p, stop等）

        Returns:
            生成的文本响应
        """
        pass

    @abstractmethod
    async def generate_batch(
        self,
        messages: List[List[Dict[str, str]]],
        **kwargs
    ) -> List[str]:
        """
        批量生成响应

        Args:
            messages: 输入消息列表的列表，每个元素是一个消息列表
            **kwargs: 其他参数

        Returns:
            生成的文本响应列表
        """
        pass


class BaseEmbedding(ABC):
    """Embedding 基础抽象类"""

    def __init__(
        self,
        model_path: str,
        **kwargs
    ):
        self.model_path = model_path
        self.kwargs = kwargs

    @abstractmethod
    async def embed(
        self,
        texts: List[str],
        **kwargs
    ) -> EmbeddingResponse:
        """
        生成文本嵌入向量

        Args:
            texts: 文本列表
            **kwargs: 其他参数

        Returns:
            EmbeddingResponse
        """
        pass

    @abstractmethod
    async def embed_single(
        self,
        text: str,
        **kwargs
    ) -> List[float]:
        """
        生成单个文本的嵌入向量

        Args:
            text: 输入文本
            **kwargs: 其他参数

        Returns:
            嵌入向量
        """
        pass
