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
        dim: int = 768
    ):
        """
        初始化实体名称向量索引
        
        Args:
            graph: NetworkX图
            embedding_client: 嵌入客户端
            index_path: 索引文件路径（不含扩展名）
            dim: 嵌入向量维度
        """
        self.graph = graph
        self.embedding_client = embedding_client
        self.index_path = index_path
        self.dim = dim
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
        
        print(f"嵌入向量形状: {vectors.shape}")
        
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
    
    def load(self) -> bool:
        """加载已存在的索引"""
        if self.index_path and self._index and self._index._exists(self.index_path):
            self._index.load(self.index_path)
            print(f"已加载实体名称索引，包含 {self._index.n_vectors} 个向量")
            return True
        return False
    
    def is_built(self) -> bool:
        """检查索引是否已构建"""
        return self._index is not None and self._index.n_vectors > 0
