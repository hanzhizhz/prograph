"""
嵌入向量缓存管理器

统一的嵌入缓存管理，提供：
1. 智能批处理：自动积累待嵌入文本，批量调用API
2. LRU内存缓存：最大50000条记录
3. 统计信息：缓存命中率、批处理效率
"""

import asyncio
import hashlib
import logging
from collections import OrderedDict
from typing import List, Dict

import numpy as np

from ..llm.base import BaseEmbedding


logger = logging.getLogger(__name__)


class EmbeddingCacheManager:
    """
    统一的嵌入缓存管理器

    功能：
    - 智能批处理：自动积累待嵌入文本，批量调用API
    - LRU内存缓存：最大50000条记录
    - 统计信息：缓存命中率、批处理效率
    """

    def __init__(
        self,
        embedding_client: BaseEmbedding,
        batch_size: int = 50,
        max_cache_size: int = 50000,
    ):
        """
        初始化缓存管理器

        Args:
            embedding_client: 嵌入客户端
            batch_size: 批量处理大小
            max_cache_size: 最大缓存条目数
        """
        self.embedding_client = embedding_client
        self.batch_size = batch_size
        self.max_cache_size = max_cache_size

        # 内存缓存：OrderedDict实现LRU
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

        # 批处理队列（用于延迟批处理）
        self._pending_texts: List[str] = []
        self._pending_futures: List[asyncio.Future] = []

        # 统计信息
        self._stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "batch_calls": 0,
            "single_calls": 0,
        }

    async def get_embedding(
        self,
        text: str,
        batch: bool = False
    ) -> np.ndarray:
        """
        获取文本嵌入（自动批处理优化）

        Args:
            text: 输入文本
            batch: 是否启用批处理（默认False，立即返回）

        Returns:
            嵌入向量
        """
        # 1. 检查内存缓存
        cache_key = self._hash_text(text)
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)  # LRU更新
            self._stats["cache_hits"] += 1
            return self._cache[cache_key]

        self._stats["cache_misses"] += 1

        # 2. 批处理模式：加入队列等待批量处理
        if batch:
            future = asyncio.Future()
            self._pending_texts.append(text)
            self._pending_futures.append(future)

            # 达到批量大小时触发处理
            if len(self._pending_texts) >= self.batch_size:
                await self._process_batch()

            return await future

        # 3. 单独处理模式（立即返回）
        else:
            embedding = await self._embed_single(text)
            self._cache[cache_key] = embedding

            # LRU淘汰
            if len(self._cache) > self.max_cache_size:
                self._cache.popitem(last=False)

            return embedding

    async def get_embeddings_batch(
        self,
        texts: List[str]
    ) -> List[np.ndarray]:
        """
        批量获取嵌入（显式批处理）

        优先使用缓存，未命中的统一批量调用API

        Args:
            texts: 文本列表

        Returns:
            嵌入向量列表
        """
        # 分离已缓存和未缓存的文本，同时保存 cache_key
        uncached_texts = []
        uncached_indices = []
        uncached_keys = []  # 保存未缓存文本的 cache_key
        results = [None] * len(texts)

        for i, text in enumerate(texts):
            cache_key = self._hash_text(text)
            if cache_key in self._cache:
                self._cache.move_to_end(cache_key)
                results[i] = self._cache[cache_key]
                self._stats["cache_hits"] += 1
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
                uncached_keys.append(cache_key)  # 保存 cache_key
                self._stats["cache_misses"] += 1

        # 批量处理未缓存的文本
        if uncached_texts:
            try:
                # 使用embedding_client的embed方法（已支持批量）
                response = await self.embedding_client.embed(uncached_texts)
                embeddings = [np.array(emb) for emb in response.embeddings]

                self._stats["batch_calls"] += 1

                # 更新缓存和结果
                for embedding, idx, cache_key in zip(embeddings, uncached_indices, uncached_keys):
                    self._cache[cache_key] = embedding

                    # LRU淘汰
                    if len(self._cache) > self.max_cache_size:
                        self._cache.popitem(last=False)

                    results[idx] = embedding

            except Exception as e:
                logger.error(f"批量嵌入失败: {e}")
                # 回退到单个调用
                for text, idx in zip(uncached_texts, uncached_indices):
                    try:
                        embedding = await self._embed_single(text)
                        results[idx] = embedding
                    except Exception as e2:
                        logger.error(f"单个嵌入也失败: {e2}")
                        results[idx] = np.zeros(768)  # 返回零向量作为fallback

        return results

    async def flush(self):
        """
        刷新批处理队列（处理剩余待处理文本）

        在使用批处理模式（batch=True）时，需要在处理完所有文本后调用此方法
        来确保队列中的剩余文本被处理。
        """
        if self._pending_texts:
            await self._process_batch()

    def get_stats(self) -> Dict[str, int]:
        """
        获取缓存统计信息

        Returns:
            统计信息字典
        """
        stats = self._stats.copy()
        stats["cache_size"] = len(self._cache)
        stats["cache_hit_rate"] = (
            stats["cache_hits"] / (stats["cache_hits"] + stats["cache_misses"])
            if (stats["cache_hits"] + stats["cache_misses"]) > 0
            else 0.0
        )
        return stats

    def clear_cache(self):
        """清空内存缓存"""
        self._cache.clear()
        logger.info("内存缓存已清空")

    async def _process_batch(self):
        """处理批量嵌入请求"""
        if not self._pending_texts:
            return

        texts = self._pending_texts
        futures = self._pending_futures

        # 清空队列
        self._pending_texts = []
        self._pending_futures = []

        try:
            # 批量调用API
            response = await self.embedding_client.embed(texts)
            embeddings = [np.array(emb) for emb in response.embeddings]

            self._stats["batch_calls"] += 1

            # 设置Future结果并更新缓存
            for text, embedding, future in zip(texts, embeddings, futures):
                cache_key = self._hash_text(text)
                self._cache[cache_key] = embedding

                # LRU淘汰
                if len(self._cache) > self.max_cache_size:
                    self._cache.popitem(last=False)

                future.set_result(embedding)

        except Exception as e:
            logger.error(f"批量处理失败: {e}")
            # 所有Future设置异常
            for future in futures:
                if not future.done():
                    future.set_exception(e)

    async def _embed_single(self, text: str) -> np.ndarray:
        """单个文本嵌入（内部方法）"""
        self._stats["single_calls"] += 1
        result = await self.embedding_client.embed_single(text)
        return np.array(result)

    def _hash_text(self, text: str) -> str:
        """
        生成文本缓存键（使用hash避免存储长文本）

        【性能优化】只哈希前100字符，减少大文本的哈希开销

        Args:
            text: 输入文本

        Returns:
            MD5哈希值
        """
        if len(text) > 100:
            text = text[:100]
        return hashlib.md5(text.encode()).hexdigest()
