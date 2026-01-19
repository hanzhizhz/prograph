"""
候选生成器
使用向量相似度生成实体链接候选对
"""

import asyncio
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
import numpy as np
from collections import defaultdict

from .embedding_index import EmbeddingIndex
from ..llm.base import BaseEmbedding


@dataclass
class CandidatePair:
    """候选对"""
    id1: str
    id2: str
    text1: str
    text2: str
    similarity: float
    type1: str  # "proposition" or "entity"
    type2: str
    # 上下文信息（用于实体链接判断）
    context1: str = ""  # 来源文档的完整内容
    context2: str = ""  # 来源文档的完整内容
    doc_id1: str = ""   # 文档 ID
    doc_id2: str = ""   # 文档 ID

    def __repr__(self):
        return f"CandidatePair({self.id1[:30]}... <-> {self.id2[:30]}..., sim={self.similarity:.2f})"


@dataclass
class EntityCandidateGroup:
    """实体候选组 - 一组可能需要融合的实体"""
    entity_ids: List[str]  # 实体 ID 列表
    entity_texts: List[str]  # 实体文本列表
    entity_types: List[str]  # 实体类型列表
    contexts: List[str]  # 每个实体的文档上下文
    doc_ids: List[str]  # 每个实体的文档 ID
    similarities: List[float]  # 相对于中心实体的相似度（用于排序）
    group_id: str  # 组 ID

    def __repr__(self):
        return f"EntityCandidateGroup(id={self.group_id}, count={len(self.entity_ids)})"


class CandidateGenerator:
    """
    候选生成器

    两阶段候选生成：
    1. 使用 HNSW 索引进行向量相似度检索
    2. 按类型分组
    """

    def __init__(
        self,
        embedding_client: BaseEmbedding,
        similarity_threshold: float = 0.85,
        vector_top_k: int = 10,
        entity_similarity_threshold: Optional[float] = None,
        proposition_similarity_threshold: Optional[float] = None,
        entity_fusion_group_size: int = 20,
        batch_search_size: int = 1000,
    ):
        """
        初始化候选生成器

        Args:
            embedding_client: 嵌入客户端
            similarity_threshold: 默认相似度阈值
            vector_top_k: 向量检索 top-k
            entity_similarity_threshold: 实体相似度阈值（如果为 None，使用 similarity_threshold）
            proposition_similarity_threshold: 命题相似度阈值（如果为 None，使用 similarity_threshold）
            entity_fusion_group_size: 实体融合每组最大数量
            batch_search_size: 批量搜索的批次大小
        """
        self.embedding_client = embedding_client
        self.vector_top_k = vector_top_k
        self.entity_fusion_group_size = entity_fusion_group_size
        self.batch_search_size = batch_search_size
        # 实体和命题使用不同的阈值
        self.entity_similarity_threshold = entity_similarity_threshold if entity_similarity_threshold is not None else similarity_threshold
        self.proposition_similarity_threshold = proposition_similarity_threshold if proposition_similarity_threshold is not None else similarity_threshold

    def _deduplicate_candidates(self, candidates: List[CandidatePair]) -> List[CandidatePair]:
        """去重候选对，避免 (A,B) 和 (B,A) 重复"""
        seen = set()
        deduplicated = []
        for pair in candidates:
            # 使用排序的元组作为唯一键
            key = (min(pair.id1, pair.id2), max(pair.id1, pair.id2))
            if key not in seen:
                seen.add(key)
                deduplicated.append(pair)
        return deduplicated

    async def generate_candidates(
        self,
        graph,
        proposition_index: Optional[EmbeddingIndex] = None,
        entity_index: Optional[EmbeddingIndex] = None,
    ) -> Tuple[List[CandidatePair], List[EntityCandidateGroup], List[EntityCandidateGroup]]:
        """
        为图中的节点生成候选链接对

        Args:
            graph: NetworkX 图
            proposition_index: 命题嵌入索引
            entity_index: 实体嵌入索引

        Returns:
            (proposition_candidates, auto_fuse_groups, llm_groups)
        """
        proposition_candidates = []
        auto_fuse_groups = []
        llm_groups = []

        # 收集节点信息
        propositions = []
        entities = []

        for node_id, node_data in graph.nodes(data=True):
            node_type = node_data.get("node_type", "")
            if node_type == "proposition":
                propositions.append((node_id, node_data))
            elif node_type == "entity":
                entities.append((node_id, node_data))

        print(f"节点统计: {len(propositions)} 个命题, {len(entities)} 个实体")

        # 生成命题候选对（使用命题专用阈值）
        if proposition_index and len(propositions) > 1:
            print("生成命题候选对...")
            proposition_candidates = await self._generate_proposition_candidates(
                graph, propositions, proposition_index
            )

        # 生成实体候选组（使用实体专用阈值）
        if entity_index and len(entities) > 1:
            print("生成实体候选组...")
            auto_fuse_groups, llm_groups, all_groups = await self._generate_entity_candidate_groups(
                graph, entities, entity_index
            )

            # 打印统计信息（全部组，不受 group_size 限制）
            self.print_entity_statistics(auto_fuse_groups, llm_groups, all_groups)

        print(f"候选统计: {len(proposition_candidates)} 个命题对")
        print(f"  - 自动融合组: {len(auto_fuse_groups)}")
        print(f"  - LLM判断组: {len(llm_groups)}")

        # 去重命题候选对
        proposition_candidates = self._deduplicate_candidates(proposition_candidates)
        print(f"去重后统计: {len(proposition_candidates)} 个命题对")

        return proposition_candidates, auto_fuse_groups, llm_groups

    async def _generate_proposition_candidates(
        self,
        graph,
        propositions: List[Tuple[str, dict]],
        index: EmbeddingIndex,
    ) -> List[CandidatePair]:
        """生成命题候选对（优化版）"""
        candidates = []

        if not propositions:
            return candidates

        # ========== 优化1: 缓存文档上下文 ==========
        # 按 doc_id 分组缓存文档内容，避免重复遍历图
        doc_contexts = self._build_doc_context_cache(graph, "proposition")

        # ========== 优化2: 批量生成嵌入 ==========
        # 提取所有文本
        node_ids = [node_id for node_id, _ in propositions]
        texts = [node_data.get("text", "") for _, node_data in propositions]
        doc_ids = [node_data.get("doc_id", "") for _, node_data in propositions]

        # 批量生成嵌入（一次调用处理所有）
        print(f"  批量生成 {len(texts)} 个命题的嵌入向量...")
        response = await self.embedding_client.embed(texts)
        embeddings = [np.array(emb, dtype=np.float32) for emb in response.embeddings]

        # ========== 批量搜索 ==========
        print(f"  批量搜索相似节点（{len(embeddings)} 个查询）...")
        batch_results = index.search_batch(
            np.array(embeddings),
            k=100
        )

        for i, (node_id, results) in enumerate(zip(node_ids, batch_results)):
            text = texts[i]
            doc_id = doc_ids[i]

            # 使用缓存的上下文
            context1 = doc_contexts.get(doc_id, "")

            # 过滤低相似度、自身和同文档内的节点
            for other_id, similarity in results:
                # 获取其他节点的数据（需要先获取才能过滤）
                other_data = graph.nodes.get(other_id, {})
                other_doc_id = other_data.get("doc_id", "")

                if (other_id != node_id and
                    doc_id != other_doc_id and  # 只链接不同文档的节点
                    similarity >= self.proposition_similarity_threshold):

                    other_text = other_data.get("text", "")

                    # 使用缓存的上下文
                    context2 = doc_contexts.get(other_doc_id, "")

                    candidates.append(CandidatePair(
                        id1=node_id,
                        id2=other_id,
                        text1=text,
                        text2=other_text,
                        similarity=similarity,
                        type1="proposition",
                        type2="proposition",
                        context1=context1,
                        context2=context2,
                        doc_id1=doc_id,
                        doc_id2=other_doc_id,
                    ))

        return candidates

    async def _generate_entity_candidate_groups(
        self,
        graph,
        entities: List[Tuple[str, dict]],
        index: EmbeddingIndex,
    ) -> Tuple[List[EntityCandidateGroup], List[EntityCandidateGroup], List[EntityCandidateGroup]]:
        """
        生成实体候选组，分离自动融合组和需要 LLM 判断的组

        Returns:
            (auto_fuse_groups, llm_groups, all_groups_before_split)
        """
        auto_fuse_groups = []  # 文本和类型完全一致的组
        llm_groups = []        # 需要 LLM 判断的组
        all_groups = []        # 分组前的全部组（用于统计）
        processed_entities = set()

        if not entities:
            return auto_fuse_groups, llm_groups, all_groups

        # 缓存文档上下文
        doc_contexts = self._build_doc_context_cache(graph, "entity")

        # 批量获取嵌入
        node_ids = [nid for nid, _ in entities]
        texts = [nd.get("text", "") for _, nd in entities]
        doc_ids_list = [nd.get("doc_id", "") for _, nd in entities]
        entity_types = [nd.get("entity_type", "") for _, nd in entities]

        print(f"  批量生成 {len(texts)} 个实体的嵌入向量...")
        response = await self.embedding_client.embed(texts)
        embeddings = [np.array(emb, dtype=np.float32) for emb in response.embeddings]

        # 创建 node_id 到 index 的映射
        node_id_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}

        # ========== 优化：预构建实体签名索引 ==========
        # 使用 (doc_id, text, type) 作为签名，用于 O(1) 查找同文档相同实体
        entity_signature_to_indices = defaultdict(list)
        for idx, (doc_id, text, entity_type) in enumerate(zip(doc_ids_list, texts, entity_types)):
            signature = (doc_id, text, entity_type)
            entity_signature_to_indices[signature].append(idx)

        # ========== 优化：使用队列实现 O(1) 获取下一个未处理实体 ==========
        from collections import deque
        unprocessed_queue = deque(range(len(node_ids)))

        print(f"  搜索相似实体并分组...")
        while unprocessed_queue:
            # O(1) - 直接从队列获取下一个未处理实体
            start_idx = unprocessed_queue.popleft()
            node_id = node_ids[start_idx]

            # 检查是否已被其他组处理
            if node_id in processed_entities:
                continue

            # 搜索相似实体
            embedding = embeddings[start_idx].reshape(1, -1)
            results = index.search_batch(embedding, k=self.vector_top_k)[0]

            # 收集相似的实体
            similar_entities = [{
                "id": node_id,
                "text": texts[start_idx],
                "type": entity_types[start_idx],
                "context": doc_contexts.get(doc_ids_list[start_idx], ""),
                "doc_id": doc_ids_list[start_idx],
                "similarity": 1.0  # 自己与自己
            }]

            for other_id, similarity in results:
                if other_id == node_id:
                    continue

                other_idx = node_id_to_idx.get(other_id, -1)
                if other_idx == -1:
                    continue

                other_doc_id = doc_ids_list[other_idx]
                # 只链接不同文档的实体
                if other_doc_id != doc_ids_list[start_idx] and similarity >= self.entity_similarity_threshold:
                    similar_entities.append({
                        "id": other_id,
                        "text": texts[other_idx],
                        "type": entity_types[other_idx],
                        "context": doc_contexts.get(other_doc_id, ""),
                        "doc_id": other_doc_id,
                        "similarity": float(similarity)
                    })

            # 如果有候选实体（至少2个）
            if len(similar_entities) >= 2:
                # 按相似度排序
                similar_entities.sort(key=lambda x: x["similarity"], reverse=True)

                # 检查是否文本和类型完全一致
                first_text = similar_entities[0]["text"]
                first_type = similar_entities[0]["type"]
                all_same = all(e["text"] == first_text and e["type"] == first_type
                               for e in similar_entities)

                # 创建全部组（用于统计，不受 group_size 限制）
                all_groups.append(self._create_entity_group(
                    similar_entities,
                    f"all_{len(all_groups)}"
                ))

                if all_same:
                    # 自动融合组
                    auto_fuse_groups.append(self._create_entity_group(
                        similar_entities,
                        f"auto_fuse_{len(auto_fuse_groups)}"
                    ))

                    # 标记已处理（使用预构建索引优化）
                    self._mark_entities_processed(
                        similar_entities, entity_signature_to_indices, node_ids, processed_entities
                    )
                else:
                    # 需要LLM判断，按 group_size 拆分
                    for j in range(0, len(similar_entities), self.entity_fusion_group_size):
                        chunk = similar_entities[j:j + self.entity_fusion_group_size]
                        if len(chunk) >= 2:  # 至少2个实体才需要判断
                            llm_groups.append(self._create_entity_group(
                                chunk,
                                f"llm_{len(llm_groups)}"
                            ))

                    # 标记已处理（使用预构建索引优化）
                    self._mark_entities_processed(
                        similar_entities, entity_signature_to_indices, node_ids, processed_entities
                    )

        return auto_fuse_groups, llm_groups, all_groups

    def _create_entity_group(
        self,
        entities: List[Dict],
        group_id: str
    ) -> EntityCandidateGroup:
        """
        从实体列表创建候选组

        Args:
            entities: 实体字典列表，每个字典包含 id/text/type/context/doc_id/similarity
            group_id: 组ID

        Returns:
            EntityCandidateGroup 实例
        """
        return EntityCandidateGroup(
            entity_ids=[e["id"] for e in entities],
            entity_texts=[e["text"] for e in entities],
            entity_types=[e["type"] for e in entities],
            contexts=[e["context"] for e in entities],
            doc_ids=[e["doc_id"] for e in entities],
            similarities=[e["similarity"] for e in entities],
            group_id=group_id
        )

    def _mark_entities_processed(
        self,
        similar_entities: List[Dict],
        entity_signature_to_indices: Dict,
        node_ids: List[str],
        processed_entities: Set[str],
    ) -> None:
        """
        标记实体为已处理，包括同文档相同文本和类型的实体

        使用预构建索引实现 O(m) 复杂度，m为similar_entities长度

        Args:
            similar_entities: 相似实体列表
            entity_signature_to_indices: 预构建的实体签名索引
            node_ids: 节点ID列表
            processed_entities: 已处理实体集合（就地修改）
        """
        for e in similar_entities:
            entity_id = e["id"]
            if entity_id not in processed_entities:
                processed_entities.add(entity_id)

            # 使用预构建索引 O(1) 查找同文档相同实体
            signature = (e["doc_id"], e["text"], e["type"])
            for other_idx in entity_signature_to_indices[signature]:
                other_id = node_ids[other_idx]
                if other_id not in processed_entities:
                    processed_entities.add(other_id)

    def print_entity_statistics(
        self,
        auto_fuse_groups: List[EntityCandidateGroup],
        llm_groups: List[EntityCandidateGroup],
        all_groups: List[EntityCandidateGroup]
    ) -> None:
        """打印实体候选组统计信息"""
        print("\n" + "=" * 60)
        print("实体候选组统计（分组前，不受group_size限制）")
        print("=" * 60)

        if not all_groups:
            print("未找到需要融合的实体候选组")
            return

        # 按组大小统计（全部组）
        size_distribution = defaultdict(int)
        total_entities = 0
        total_unique_entities = set()

        for group in all_groups:
            size = len(group.entity_ids)
            size_distribution[size] += 1
            total_entities += size
            total_unique_entities.update(group.entity_ids)

        print(f"总候选组数: {len(all_groups)}")
        print(f"涉及实体总数（含重复）: {total_entities}")
        print(f"涉及唯一实体数: {len(total_unique_entities)}")
        print(f"\n组大小分布（全部）:")
        for size in sorted(size_distribution.keys()):
            count = size_distribution[size]
            print(f"  {size} 个实体/组: {count} 组")

        # 显示最大的几组
        sorted_groups = sorted(all_groups, key=lambda g: len(g.entity_ids), reverse=True)
        print(f"\n最大的 {min(5, len(sorted_groups))} 个组:")
        for i, group in enumerate(sorted_groups[:5], 1):
            print(f"  组 {i}: {len(group.entity_ids)} 个实体")
            print(f"    实体文本: {', '.join(group.entity_texts[:3])}{'...' if len(group.entity_texts) > 3 else ''}")
            print(f"    相似度范围: {min(group.similarities):.2f} - {max(group.similarities):.2f}")

        # 自动融合 vs LLM 判断统计
        auto_fuse_count = sum(len(g.entity_ids) for g in auto_fuse_groups)
        llm_count = sum(len(g.entity_ids) for g in llm_groups)
        print(f"\n自动融合: {len(auto_fuse_groups)} 组，共 {auto_fuse_count} 个实体")
        print(f"LLM判断: {len(llm_groups)} 组，共 {llm_count} 个实体")
        print(f"节省LLM调用: {auto_fuse_count} 个实体（文本和类型完全一致）")

        print("=" * 60)

    def _build_doc_context_cache(self, graph, node_type: str) -> Dict[str, str]:
        """
        构建文档上下文缓存

        按文档 ID 缓存所有文档的内容，避免重复遍历图

        Args:
            graph: NetworkX 图
            node_type: 节点类型 ("proposition" 或 "entity")

        Returns:
            {doc_id: document_content} 的字典
        """
        from collections import defaultdict

        # 按 doc_id 分组收集命题
        doc_props = defaultdict(list)

        for node_id, node_data in graph.nodes(data=True):
            # 只收集命题节点（因为上下文是从命题构建的）
            if node_data.get("node_type") == "proposition":
                doc_id = node_data.get("doc_id", "")
                if doc_id:
                    prop_text = node_data.get("text", "")
                    sent_idx = node_data.get("sent_idx", -1)
                    if prop_text:
                        doc_props[doc_id].append((sent_idx, prop_text))

        # 为每个文档构建内容
        doc_contexts = {}
        for doc_id, props in doc_props.items():
            # 按句子索引排序
            props.sort(key=lambda x: x[0])
            # 拼接文档内容
            doc_content = "\n".join(f"[{idx}] {text}" for idx, text in props)
            doc_contexts[doc_id] = doc_content

        print(f"  缓存了 {len(doc_contexts)} 个文档的上下文")
        return doc_contexts


async def build_indices(
    graph,
    embedding_client: BaseEmbedding,
) -> Tuple[EmbeddingIndex, EmbeddingIndex]:
    """
    为图中的节点构建嵌入索引

    Args:
        graph: NetworkX 图
        embedding_client: 嵌入客户端

    Returns:
        (proposition_index, entity_index)
    """
    # 收集命题和实体
    proposition_texts = []
    proposition_ids = []
    entity_texts = []
    entity_ids = []

    for node_id, node_data in graph.nodes(data=True):
        node_type = node_data.get("node_type", "")
        text = node_data.get("text", "")

        if node_type == "proposition" and text:
            proposition_texts.append(text)
            proposition_ids.append(node_id)
        elif node_type == "entity" and text:
            entity_texts.append(text)
            entity_ids.append(node_id)

    # 生成嵌入向量
    print(f"生成命题嵌入向量: {len(proposition_texts)} 个")
    proposition_embeddings = []
    if proposition_texts:
        response = await embedding_client.embed(proposition_texts)
        proposition_embeddings = response.embeddings

    print(f"生成实体嵌入向量: {len(entity_texts)} 个")
    entity_embeddings = []
    if entity_texts:
        response = await embedding_client.embed(entity_texts)
        entity_embeddings = response.embeddings

    # 创建索引
    proposition_index = None
    entity_index = None

    if proposition_embeddings:
        proposition_index = EmbeddingIndex(
            dim=len(proposition_embeddings[0]),
            max_elements=len(proposition_embeddings) * 2,
        )
        proposition_index.add_items(
            np.array(proposition_embeddings, dtype=np.float32),
            proposition_ids,
        )
        print(f"命题索引已创建: {len(proposition_ids)} 个向量")

    if entity_embeddings:
        entity_index = EmbeddingIndex(
            dim=len(entity_embeddings[0]),
            max_elements=len(entity_embeddings) * 2,
        )
        entity_index.add_items(
            np.array(entity_embeddings, dtype=np.float32),
            entity_ids,
        )
        print(f"实体索引已创建: {len(entity_ids)} 个向量")

    return proposition_index, entity_index
