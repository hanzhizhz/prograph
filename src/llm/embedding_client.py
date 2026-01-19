"""
Embedding 客户端
用于生成文本嵌入向量
"""

import asyncio
import logging
from typing import List, Optional
from openai import AsyncOpenAI
import numpy as np
import torch
from .base import BaseEmbedding, EmbeddingResponse


logger = logging.getLogger(__name__)


class VLLMEmbeddingClient(BaseEmbedding):
    """
    vLLM Embedding 客户端（离线）

    使用 vLLM 的同步 LLM 类进行离线嵌入向量生成
    参考 ggagent3 的 VLLMEmbeddingService 实现
    """

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 2,
        gpu_memory_utilization: float = 0.05,
        trust_remote_code: bool = True,
        max_model_len: Optional[int] = None,
    ):
        """
        初始化 vLLM Embedding 客户端

        Args:
            model_path: 模型路径
            tensor_parallel_size: 张量并行大小
            gpu_memory_utilization: GPU 内存利用率
            trust_remote_code: 是否信任远程代码
            max_model_len: 模型最大长度
        """
        super().__init__(model_path=model_path)
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.trust_remote_code = trust_remote_code
        self.max_model_len = max_model_len

        # 构建 LLM 初始化参数
        llm_kwargs = {
            "model": model_path,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": trust_remote_code,
        }

        # 如果指定了 max_model_len，添加到参数中
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len

        # vLLM 模型加载是同步阻塞的，在初始化时完成
        logger.warning(f"正在加载 vLLM Embedding 模型: {model_path}")
        from vllm import LLM
        self.llm = LLM(**llm_kwargs)
        logger.info("vLLM Embedding 模型加载完成")

    def _to_list(self, embedding) -> List[float]:
        """
        将 embedding 转换为 Python 列表

        Args:
            embedding: 可能是 torch.Tensor、numpy.ndarray 或 list

        Returns:
            Python 列表
        """
        if isinstance(embedding, torch.Tensor):
            return embedding.cpu().numpy().tolist()
        elif isinstance(embedding, np.ndarray):
            return embedding.tolist()
        elif not isinstance(embedding, list):
            return list(embedding)
        return embedding

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
        if not texts:
            return EmbeddingResponse(
                embeddings=[],
                model=self.model_path,
                prompt_tokens=0
            )

        try:
            # 【性能修复】在线程池中执行同步的 vLLM embed 调用，避免阻塞事件循环
            loop = asyncio.get_event_loop()
            outputs = await loop.run_in_executor(None, self.llm.embed, texts)

            # 提取 embedding 向量
            embeddings = []
            for output in outputs:
                # 根据 vLLM API，embedding 在 output.outputs.embedding 中
                embedding = output.outputs.embedding
                embeddings.append(self._to_list(embedding))

            return EmbeddingResponse(
                embeddings=embeddings,
                model=self.model_path,
                prompt_tokens=sum(len(t.split()) for t in texts)  # 粗略估计
            )
        except Exception as e:
            logger.error(f"vLLM Embedding 生成错误: {e}")
            # 打印部分文本用于调试
            if texts:
                logger.warning(f"文本示例: {texts[0]}")
            raise

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
        response = await self.embed([text], **kwargs)
        return response.embeddings[0]


class OpenAIEmbeddingClient(BaseEmbedding):
    """
    OpenAI 兼容 Embedding API 客户端

    用于运行时嵌入向量生成
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "qwen3-embedding",
        max_retries: int = 3,
        timeout: float = 600.0,
        concurrency: int = 100,
    ):
        super().__init__(model_path=model)
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
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

    async def embed(
        self,
        texts: List[str],
        **kwargs
    ) -> EmbeddingResponse:
        """
        生成文本嵌入向量
        """
        self._ensure_client()

        async with self._semaphore:
            response = await self._client.embeddings.create(
                model=self.model,
                input=texts,
                **kwargs
            )

            embeddings = [item.embedding for item in response.data]
            prompt_tokens = response.usage.prompt_tokens

            return EmbeddingResponse(
                embeddings=embeddings,
                model=self.model,
                prompt_tokens=prompt_tokens
            )

    async def embed_single(
        self,
        text: str,
        **kwargs
    ) -> List[float]:
        """
        生成单个文本的嵌入向量
        """
        response = await self.embed([text], **kwargs)
        return response.embeddings[0]

    async def close(self):
        """关闭客户端"""
        if self._client:
            await self._client.close()


# 辅助函数
def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    计算余弦相似度

    Args:
        a: 向量 a
        b: 向量 b

    Returns:
        余弦相似度
    """
    a_arr = np.array(a)
    b_arr = np.array(b)

    dot_product = np.dot(a_arr, b_arr)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot_product / (norm_a * norm_b))
