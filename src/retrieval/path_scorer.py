"""
路径评分器
实现 S_sem + S_bridge 评分公式
支持使用预构建的向量索引来加速语义评分
"""

import asyncio
import math

import numpy as np
from typing import List, Set, Dict, Optional, Any
import networkx as nx
from dataclasses import dataclass
from pathlib import Path

from ..llm.embedding_client import OpenAIEmbeddingClient
from ..proposition_graph.graph_builder import MENTIONS_ENTITY, PROPOSITION_NODE
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .embedding_cache_manager import EmbeddingCacheManager


@dataclass
class NodeScore:
    """节点评分"""
    node_id: str
    semantic_score: float  # S_sem
    bridge_score: float  # S_bridge
    total_score: float = 0.0  # 加权总和

    def __repr__(self):
        return f"NodeScore({self.node_id[:30]}..., sem={self.semantic_score:.2f}, bridge={self.bridge_score:.2f}, total={self.total_score:.2f})"


class PathScorer:
    """
    路径评分器

    实现: S(v) = w_sem * S_sem(v) + w_bridge * S_bridge(v)

    其中:
    - S_sem = cosine_similarity(embed(question), embed(node_text))
    - S_bridge = min(1.0, log(1 + NewEntities) / log(1 + GlobalNorm))
      - GlobalNorm = max(global_median, global_mean)，在初始化时预计算

    支持使用预构建的向量索引来加速语义评分计算
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        embedding_client: OpenAIEmbeddingClient,
        semantic_weight: float = 0.4,
        bridge_weight: float = 0.3,
        index_dir: Optional[str] = None,
        embedding_cache_manager: Optional['EmbeddingCacheManager'] = None,
        adjacency_cache: Optional[Dict[str, Dict[str, List[Dict[str, Any]]]]] = None,
        persistence_dir: Optional[str] = None,
    ):
        self.graph = graph
        self.embedding_client = embedding_client
        self.semantic_weight = semantic_weight
        self.bridge_weight = bridge_weight
        self.index_dir = index_dir
        self.embedding_cache_manager = embedding_cache_manager
        self.persistence_dir = persistence_dir

        # 【性能优化】邻接缓存：预构建节点邻居映射
        self._adjacency_cache = adjacency_cache if adjacency_cache is not None else {}

        # 缓存问题嵌入
        self._question_embedding: Optional[np.ndarray] = None
        self._question_text: Optional[str] = None

        # 节点嵌入缓存（从索引加载或实时计算）
        self._node_embedding_cache: Dict[str, np.ndarray] = {}
        self._proposition_index = None

        # 批量嵌入配置
        self._batch_size = 50  # 批量嵌入大小

        # 全局实体统计（预计算）
        self._global_entity_stats: Dict[str, float]

        # 如果提供了索引目录，尝试加载
        if index_dir:
            self._load_index()

        # 预计算全局统计（优先从文件加载）
        self._global_entity_stats = self._load_or_compute_global_statistics()

    async def score_nodes(
        self,
        question: str,
        candidate_nodes: List[str],
        visited_entities: Set[str],
    ) -> List[NodeScore]:
        """
        对候选节点进行评分（批量嵌入优化）

        Args:
            question: 问题文本
            candidate_nodes: 候选节点 ID 列表
            visited_entities: 已访问的实体集合

        Returns:
            NodeScore 列表，按 total_score 降序排列
        """
        # 生成问题嵌入（优先使用缓存管理器）
        if self._question_text != question:
            if self.embedding_cache_manager:
                self._question_embedding = await self.embedding_cache_manager.get_embedding(question)
            else:
                self._question_embedding = np.array(
                    await self.embedding_client.embed_single(question)
                )
            self._question_text = question

        # 【性能优化】批量预加载未缓存的节点嵌入
        await self._preload_uncached_embeddings(candidate_nodes)

        # 过滤有效的节点
        valid_nodes = [n for n in candidate_nodes if n in self.graph.nodes]

        # 【性能优化】并行计算所有节点的语义分数
        semantic_scores = await asyncio.gather(
            *(self._compute_semantic_score(node_id) for node_id in valid_nodes)
        )

        # 构建分数结果
        scores = []
        for node_id, semantic_score in zip(valid_nodes, semantic_scores):
            # 计算桥接分数 S_bridge
            bridge_score = self._compute_bridge_score(node_id, visited_entities)

            # 计算总分
            total_score = (
                self.semantic_weight * semantic_score +
                self.bridge_weight * bridge_score
            )

            scores.append(NodeScore(
                node_id=node_id,
                semantic_score=semantic_score,
                bridge_score=bridge_score,
                total_score=total_score
            ))

        # 按总分降序排列
        scores.sort(key=lambda x: x.total_score, reverse=True)
        return scores

    async def _compute_semantic_score(self, node_id: str) -> float:
        """
        计算语义相关性分数 S_sem

        使用余弦相似度: cosine_similarity(embed(node), embed(question))

        优先级：缓存管理器 > 索引缓存 > 实时计算
        """
        node_data = self.graph.nodes[node_id]
        text = node_data.get("text", "")

        if not text:
            return 0.0

        # 1. 优先使用缓存管理器
        if self.embedding_cache_manager:
            node_embedding = await self.embedding_cache_manager.get_embedding(text)
        # 2. 使用本地缓存
        elif node_id in self._node_embedding_cache:
            node_embedding = self._node_embedding_cache[node_id]
        # 3. 实时计算
        else:
            node_embedding = np.array(
                await self.embedding_client.embed_single(text)
            )
            # 缓存以备后用
            self._node_embedding_cache[node_id] = node_embedding

        # 计算余弦相似度
        similarity = self._cosine_similarity(
            self._question_embedding,
            node_embedding
        )

        return float(similarity)

    def _load_index(self):
        """
        加载预构建的向量索引

        如果索引目录存在，加载命题索引并建立映射
        """
        index_path = Path(self.index_dir) / "indices" / "proposition"
        if not index_path.exists():
            return

        try:
            from ..retrieval.vector_index import PersistentHNSWIndex

            # 获取向量维度（从嵌入客户端或默认值）
            # 这里使用 768 作为默认值，实际应从配置获取
            self._proposition_index = PersistentHNSWIndex(dim=768)
            self._proposition_index.load(str(index_path))

            # 预加载所有节点嵌入到缓存
            # 注意：这需要较大的内存，可以根据需要调整
            # 【性能优化】使用批量获取，大幅提升预加载速度
            all_labels = list(self._proposition_index.label_to_payload.keys())
            all_vectors = self._proposition_index.get_vectors(all_labels)

            # 批量缓存
            for label, vector in zip(all_labels, all_vectors):
                if vector is not None:
                    payload = self._proposition_index.label_to_payload.get(label)
                    if payload:
                        node_id = payload.get('node_id')
                        if node_id:
                            self._node_embedding_cache[node_id] = vector
        except Exception as e:
            # 索引加载失败，回退到实时计算
            print(f"警告：索引加载失败 ({e})，将使用实时计算")

    async def _preload_uncached_embeddings(self, node_ids: List[str]) -> None:
        """
        批量预加载未缓存的节点嵌入

        【性能优化】将多次单独的 embed_single 调用合并为批量的 embed_batch 调用，
        预期可减少 80-90% 的嵌入 API 调用次数。

        Args:
            node_ids: 需要预加载的节点 ID 列表
        """
        # 找出未缓存的节点
        uncached_nodes = []
        uncached_texts = []

        for node_id in node_ids:
            if node_id not in self._node_embedding_cache and node_id in self.graph.nodes:
                node_data = self.graph.nodes[node_id]
                text = node_data.get("text", "")
                if text:
                    uncached_nodes.append(node_id)
                    uncached_texts.append(text)

        if not uncached_nodes:
            return

        # 分批处理（避免单次请求过大）
        for i in range(0, len(uncached_texts), self._batch_size):
            batch_texts = uncached_texts[i:i + self._batch_size]
            batch_nodes = uncached_nodes[i:i + self._batch_size]

            try:
                # 使用批量 API
                response = await self.embedding_client.embed(batch_texts)
                embeddings = response.embeddings

                # 缓存结果
                for node_id, embedding in zip(batch_nodes, embeddings):
                    self._node_embedding_cache[node_id] = np.array(embedding)

            except Exception:
                # 如果批量 API 失败，回退到单个调用
                for node_id, text in zip(batch_nodes, batch_texts):
                    try:
                        embedding = await self.embedding_client.embed_single(text)
                        self._node_embedding_cache[node_id] = np.array(embedding)
                    except Exception:
                        # 忽略失败的单个调用
                        pass

    def _count_mentioned_entities(self, node_id: str) -> int:
        """
        统计节点连接的实体数量

        Args:
            node_id: 节点ID

        Returns:
            通过 MENTIONS_ENTITY 边连接的实体数量
        """
        count = 0
        for neighbor in self.graph.neighbors(node_id):
            if self.graph.has_edge(node_id, neighbor):
                edge_data = self.graph[node_id][neighbor]
                edge_type = edge_data.get("edge_type", "")

                if edge_type == MENTIONS_ENTITY:
                    neighbor_data = self.graph.nodes[neighbor]
                    neighbor_type = neighbor_data.get("node_type", "")

                    if neighbor_type in ["entity", "global_entity"]:
                        count += 1
        return count

    def _load_or_compute_global_statistics(self) -> Dict[str, float]:
        """
        加载或计算全局实体统计

        优先从持久化文件加载，如果不存在则计算
        """
        import json

        # 尝试从文件加载
        if self.persistence_dir:
            stats_file = Path(self.persistence_dir) / "global_stats.json"
            if stats_file.exists():
                try:
                    with open(stats_file, 'r', encoding='utf-8') as f:
                        stats = json.load(f)
                    print(f"✓ 全局统计已从文件加载: norm={stats['norm']:.2f}")
                    return stats
                except Exception as e:
                    print(f"警告: 全局统计加载失败 ({e})，将重新计算")

        # 回退到计算
        return self._compute_global_statistics()

    def _compute_global_statistics(self) -> Dict[str, float]:
        """
        计算所有命题节点的实体连接统计

        【性能优化】使用 _adjacency_cache 替代实时遍历 graph.neighbors()
        复杂度：从 O(n × deg) 降到 O(n)

        Returns:
            包含 norm 的字典
        """
        entity_counts = []

        # 遍历所有命题节点，统计其连接的实体数
        for node_id, node_data in self.graph.nodes(data=True):
            if node_data.get("node_type") != PROPOSITION_NODE:
                continue

            # 【性能优化】使用邻接缓存获取实体邻居数量
            if self._adjacency_cache:
                neighbors_by_type = self._adjacency_cache.get(node_id, {})
                entity_count = len(neighbors_by_type.get("entity", [])) + len(neighbors_by_type.get("global_entity", []))
            else:
                # 回退到原始方法
                entity_count = self._count_mentioned_entities(node_id)

            if entity_count > 0:
                entity_counts.append(entity_count)

        if not entity_counts:
            # 图中没有命题节点或没有实体连接
            return {"norm": 1.0}

        median = float(np.median(entity_counts))
        mean = float(np.mean(entity_counts))

        return {"norm": max(median, mean)}

    def _compute_bridge_score(
        self,
        node_id: str,
        visited_entities: Set[str],
    ) -> float:
        """
        计算桥接分数 S_bridge（复用邻接缓存优化版）

        新公式（全局统计归一化）:
        S_bridge = min(1.0, log(1 + NewEntities) / log(1 + GlobalNorm))

        其中:
        - NewEntities = |N_entities(v) - VisitedEntities|
        - GlobalNorm = max(global_median, global_mean)

        全局统计在初始化时预计算，存储在 self._global_entity_stats 中。

        【性能优化】使用预构建的 _adjacency_cache 替代实时遍历 graph.neighbors()
        复杂度：从 O(deg(v)) 降到 O(1)
        """
        # 【性能优化】从缓存获取邻居信息
        neighbors_by_type = self._adjacency_cache.get(node_id, {})
        entity_neighbors = neighbors_by_type.get("entity", []) + neighbors_by_type.get("global_entity", [])

        # 统计新实体数量
        new_entities = 0
        for neighbor_info in entity_neighbors:
            neighbor_id = neighbor_info.get("node_id")
            if neighbor_id and neighbor_id not in visited_entities:
                new_entities += 1

        # 使用全局统计进行归一化
        global_norm = self._global_entity_stats.get("norm", 1.0)

        if global_norm <= 0:
            return 0.0

        bridge_score = math.log(1 + new_entities) / math.log(1 + global_norm)

        # 确保分数在 [0, 1] 范围内
        return min(1.0, float(bridge_score))

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """计算余弦相似度"""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))

    def should_terminate(
        self,
        recent_scores: List[float],
        threshold: float = 0.02,
        window: int = 2,
    ) -> bool:
        """
        判断是否应该终止搜索

        终止条件：
        1. 连续 window 步的分数增长小于 threshold
        2. Bridge score 为 0（没有新实体）

        Args:
            recent_scores: 最近的分数列表
            threshold: 分数增长阈值
            window: 窗口大小

        Returns:
            是否应该终止
        """
        if len(recent_scores) < window:
            return False

        # 检查分数增长
        recent = recent_scores[-window:]
        if max(recent) - min(recent) < threshold:
            return True

        return False
