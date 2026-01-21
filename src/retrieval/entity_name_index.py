"""
实体名称向量索引

使用HNSW实现亚线性时间复杂度的模糊匹配，替代原有的O(n×L₁×L₂)编辑距离算法。

功能：
- 预构建所有实体名称的嵌入向量
- 使用HNSW实现亚线性时间复杂度的模糊匹配
- 支持持久化存储
"""

import json
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np

from .vector_index import PersistentHNSWIndex
from ..proposition_graph.graph_builder import ENTITY_NODE, GLOBAL_ENTITY_NODE


class EntityNameIndex:
    """实体名称向量索引（用于模糊匹配）
    
    将模糊匹配的时间复杂度从 O(n×L₁×L₂) 降低到 O(log n)，
    使用HNSW向量索引实现亚线性时间复杂度的相似度搜索。
    """
    
    def __init__(
        self,
        graph,
        embedding_client,
        index_path: Optional[str] = None,
        dim: Optional[int] = None
    ):
        """
        初始化实体名称向量索引
        
        Args:
            graph: NetworkX图
            embedding_client: 嵌入客户端
            index_path: 索引文件路径（不含扩展名）
            dim: 嵌入向量维度（None 表示自动检测）
        """
        self.graph = graph
        self.embedding_client = embedding_client
        self.index_path = index_path
        self.dim = dim  # 可以为 None，加载时从元数据获取，构建时从 embedding 推断
        self._index: Optional[PersistentHNSWIndex] = None
        
    async def build(self):
        """构建实体名称向量索引"""
        # 1. 收集所有实体节点名称
        entity_names = []
        entity_node_ids = []
        
        for node_id, node_data in self.graph.nodes(data=True):
            if node_data.get("node_type") in [ENTITY_NODE, GLOBAL_ENTITY_NODE]:
                text = node_data.get("text", "").strip()
                if text:
                    entity_names.append(text)
                    entity_node_ids.append(node_id)
        
        if not entity_names:
            print("警告: 图中没有找到实体节点，无法构建实体名称索引")
            return
        
        print(f"正在为 {len(entity_names)} 个实体构建名称向量索引...")
        
        # 2. 批量计算嵌入
        response = await self.embedding_client.embed(entity_names)
        vectors = np.array([emb for emb in response.embeddings])
        
        # 自动获取维度（从 embedding 结果推断）
        if self.dim is None:
            self.dim = vectors.shape[1]
        
        print(f"嵌入向量形状: {vectors.shape}，维度: {self.dim}")
        
        # 3. 构建HNSW索引
        self._index = PersistentHNSWIndex(dim=self.dim, index_path=self.index_path)
        
        # 准备payload
        payloads = [{"node_id": nid, "name": name} 
                    for nid, name in zip(entity_node_ids, entity_names)]
        
        self._index.add(vectors, payloads)
        
        # 4. 持久化
        if self.index_path:
            self._index.save(self.index_path)
            print(f"实体名称索引已保存到: {self.index_path}")
        
        print(f"实体名称索引构建完成，包含 {self._index.n_vectors} 个向量")
    
    async def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.7
    ) -> List[str]:
        """
        向量搜索最相似的实体
        
        Args:
            query: 查询文本
            top_k: 返回top-k结果
            threshold: 相似度阈值（0-1）
            
        Returns:
            匹配的实体节点ID列表（按相似度降序）
        """
        if self._index is None or self._index.n_vectors == 0:
            print("警告: 实体名称索引未构建或为空")
            return []
        
        # 1. 计算查询嵌入
        emb = await self.embedding_client.embed_single(query)
        query_vec = np.array([emb])
        
        # 2. HNSW搜索
        distances, indices, payloads = self._index.search(query_vec, k=top_k)
        
        # 3. 返回超过阈值的结果
        results = []
        for i in range(len(indices[0])):
            # 注意：distances是余弦距离，相似度 = 1 - distance
            similarity = 1.0 - distances[0][i]
            
            if similarity >= threshold:
                payload = payloads[0][i]
                if payload and "node_id" in payload:
                    results.append(payload["node_id"])
        
        return results
    
    async def search_batch(
        self,
        queries: List[str],
        top_k: int = 1,
        threshold: float = 0.7
    ) -> List[List[str]]:
        """
        批量向量搜索最相似的实体
        
        Args:
            queries: 查询文本列表
            top_k: 返回top-k结果
            threshold: 相似度阈值（0-1）
            
        Returns:
            每个查询的匹配实体节点ID列表（按相似度降序）
        """
        if not queries or self._index is None or self._index.n_vectors == 0:
            return [[] for _ in queries]
        
        # 批量计算查询嵌入
        response = await self.embedding_client.embed(queries)
        query_vecs = np.array([emb for emb in response.embeddings])
        
        # 批量 HNSW 搜索
        results = []
        for i, query_vec in enumerate(query_vecs):
            distances, indices, payloads = self._index.search(
                query_vec.reshape(1, -1), k=top_k
            )
            
            matched = []
            for j in range(len(indices[0])):
                # 注意：distances是余弦距离，相似度 = 1 - distance
                similarity = 1.0 - distances[0][j]
                if similarity >= threshold:
                    payload = payloads[0][j]
                    if payload and "node_id" in payload:
                        matched.append(payload["node_id"])
            results.append(matched)
        
        return results
    
    def load(self) -> bool:
        """加载已存在的索引"""
        if not self.index_path:
            return False
        
        # 检查索引文件是否存在（.bin 和 .meta.json）
        bin_file = Path(f"{self.index_path}.bin")
        meta_file = Path(f"{self.index_path}.meta.json")
        
        if not (bin_file.exists() and meta_file.exists()):
            return False
        
        # 从元数据自动获取维度
        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                meta_data = json.load(f)
            self.dim = meta_data.get('dim', self.dim)
        except Exception as e:
            print(f"警告: 读取元数据失败 ({e})，使用默认维度 {self.dim}")
        
        # 如果索引对象不存在，先创建
        if self._index is None:
            self._index = PersistentHNSWIndex(dim=self.dim, index_path=self.index_path)
        
        # 加载索引
        try:
            self._index.load(self.index_path)
            print(f"已加载实体名称索引，包含 {self._index.n_vectors} 个向量，维度 {self.dim}")
            return True
        except Exception as e:
            print(f"警告: 实体名称索引加载失败 ({e})")
            return False
    
    def is_built(self) -> bool:
        """检查索引是否已构建"""
        return self._index is not None and self._index.n_vectors > 0
