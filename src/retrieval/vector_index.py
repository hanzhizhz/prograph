"""持久化HNSW向量索引"""

import json
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import hnswlib


class PersistentHNSWIndex:
    """
    支持持久化的HNSW向量索引
    封装hnswlib的save_index/load_index功能，并保存元数据映射
    """

    def __init__(
        self,
        dim: int,
        index_path: Optional[str] = None,
        M: int = 16,
        ef_construction: int = 200,
        max_elements: int = 200000
    ):
        """
        初始化持久化HNSW索引

        Args:
            dim: 向量维度
            index_path: 索引文件路径（不含扩展名），如果提供则尝试加载
            M: 每个节点的最大连接数（默认16）
            ef_construction: 构建时的搜索范围（默认200）
            max_elements: 索引的最大元素数（默认200000）
        """
        self.dim = dim
        self.index_path = index_path
        self.max_elements = max_elements
        self.M = M
        self.ef_construction = ef_construction

        # 元数据映射：label_index -> payload
        # payload包含: node_id, node_type, field, retrieval_idx, text_digest
        self.label_to_payload: Dict[int, Dict[str, Any]] = {}
        self.n_vectors = 0
        self._index_initialized = False

        # 如果提供了路径且文件存在，直接加载（不先初始化）
        if index_path and self._exists(index_path):
            self.index = hnswlib.Index(space='cosine', dim=dim)
            self.load(index_path)
            self._index_initialized = True
        else:
            # 否则初始化空索引（用于构建新索引）
            self.index = hnswlib.Index(space='cosine', dim=dim)
            self.index.init_index(max_elements=max_elements, M=M, ef_construction=ef_construction)
            self._index_initialized = True

    def _exists(self, index_path: str) -> bool:
        """检查索引文件是否存在"""
        bin_path = f"{index_path}.bin"
        meta_path = f"{index_path}.meta.json"
        return os.path.exists(bin_path) and os.path.exists(meta_path)

    def add(
        self,
        vectors: np.ndarray,
        payloads: List[Dict[str, Any]]
    ):
        """
        添加向量到索引

        Args:
            vectors: 形状为 (n, dim) 的向量数组
            payloads: 对应的payload列表，每个包含 node_id, node_type, field, retrieval_idx, text
                     长度应与 vectors 的第一维相同
        """
        if len(vectors) == 0:
            return

        if len(payloads) != len(vectors):
            raise ValueError(f"payloads 长度 ({len(payloads)}) 与 vectors 长度 ({len(vectors)}) 不匹配")

        # 如果索引未初始化（从文件加载后可能未初始化），现在初始化
        if not self._index_initialized:
            if not hasattr(self, 'index') or self.index is None:
                self.index = hnswlib.Index(space='cosine', dim=self.dim)
            self.index.init_index(max_elements=self.max_elements, M=self.M, ef_construction=self.ef_construction)
            self._index_initialized = True

        # 【性能优化】直接使用向量，hnswlib 的 cosine 空间会自动归一化
        vectors_f32 = vectors.astype(np.float32)

        start_idx = self.n_vectors
        end_idx = start_idx + len(vectors)

        # 如果需要，扩展索引容量
        if end_idx > self.max_elements:
            new_max = int(self.max_elements * 1.5)
            self.index.resize_index(new_max)
            self.max_elements = new_max

        # 生成标签（使用连续整数）
        labels = np.arange(start_idx, end_idx, dtype=np.int64)

        # 添加向量到索引
        self.index.add_items(vectors_f32, labels)

        # 保存payload映射
        for label, payload in zip(labels, payloads):
            # 保存简化的payload（移除原始text以节省空间，只保留digest）
            payload_copy = payload.copy()
            if 'text' in payload_copy:
                # 使用text的前100个字符作为digest（或可以hash）
                text = payload_copy.pop('text')
                payload_copy['text_digest'] = text[:100] if len(text) > 100 else text
            self.label_to_payload[int(label)] = payload_copy

        self.n_vectors += len(vectors)

    def get_vector(self, label: int) -> Optional[np.ndarray]:
        """获取指定标签的向量"""
        try:
             # hnswlib 支持 get_items，返回 numpy array (1, dim)
             items = self.index.get_items([label])
             return items[0]
        except Exception:
             return None

    def get_vectors(self, labels: List[int]) -> List[Optional[np.ndarray]]:
        """
        批量获取向量（性能优化）

        Args:
            labels: 标签列表

        Returns:
            向量列表，如果某个向量不存在则对应位置为 None
        """
        if not labels:
            return []
        try:
            # 批量获取，hnswlib 内部优化
            return self.index.get_items(labels)
        except Exception:
            return [None] * len(labels)

    def search(
        self,
        query_vectors: np.ndarray,
        k: int
    ) -> Tuple[np.ndarray, np.ndarray, List[List[Dict[str, Any]]]]:
        """
        批量搜索最相似的向量（使用真正的批量查询优化）

        Args:
            query_vectors: 查询向量，形状为 (n_queries, dim)
            k: 返回 top-k 结果

        Returns:
            (distances, indices, payloads):
            - distances: 形状为 (n_queries, k) 的相似度分数数组
            - indices: 形状为 (n_queries, k) 的标签索引数组
            - payloads: 形状为 (n_queries, k) 的payload列表
        """
        if self.n_vectors == 0:
            n_queries = len(query_vectors)
            return (
                np.zeros((n_queries, k), dtype=np.float32),
                np.zeros((n_queries, k), dtype=np.int64),
                [[] for _ in range(n_queries)]
            )

        # 归一化查询向量
        query_norms = np.linalg.norm(query_vectors, axis=1, keepdims=True)
        query_norms = np.where(query_norms > 0, query_norms, 1.0)
        normalized_queries = (query_vectors / query_norms).astype(np.float32)

        n_queries = len(query_vectors)
        k = min(k, self.n_vectors)

        # 设置搜索参数
        ef_search = max(k * 2, 50)
        self.index.set_ef(ef_search)

        # 【性能优化】使用真正的批量查询，而非逐个循环
        labels, dists = self.index.knn_query(normalized_queries, k=k)

        # 转换为相似度（1 - distance），shape: (n_queries, k)
        similarities = 1.0 - dists

        distances = similarities.astype(np.float32)
        indices = labels.astype(np.int64)
        payloads = []

        # 构建payload列表
        for i in range(n_queries):
            payload_list = []
            for j, label in enumerate(labels[i]):
                payload = self.label_to_payload.get(int(label), {})
                payload_list.append(payload)
            payloads.append(payload_list)

        return distances, indices, payloads

    def save(self, index_path: Optional[str] = None):
        """
        保存索引到文件

        Args:
            index_path: 保存路径（不含扩展名），如果为None则使用初始化时的路径
        """
        if index_path is None:
            index_path = self.index_path

        if index_path is None:
            raise ValueError("必须提供index_path才能保存索引")

        # 确保目录存在
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)

        # 保存索引文件
        bin_path = f"{index_path}.bin"
        self.index.save_index(bin_path)

        # 保存元数据
        meta_path = f"{index_path}.meta.json"
        meta_data = {
            'dim': self.dim,
            'max_elements': self.max_elements,
            'n_vectors': self.n_vectors,
            'M': self.M,
            'ef_construction': self.ef_construction,
            'label_to_payload': self.label_to_payload
        }

        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta_data, f, ensure_ascii=False, indent=2)

    def load(self, index_path: Optional[str] = None):
        """
        从文件加载索引

        Args:
            index_path: 加载路径（不含扩展名），如果为None则使用初始化时的路径
        """
        if index_path is None:
            index_path = self.index_path

        if index_path is None:
            raise ValueError("必须提供index_path才能加载索引")

        if not self._exists(index_path):
            raise FileNotFoundError(f"索引文件不存在: {index_path}")

        # 加载元数据
        meta_path = f"{index_path}.meta.json"
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta_data = json.load(f)

        # 验证维度匹配
        if meta_data['dim'] != self.dim:
            raise ValueError(f"索引维度不匹配: 期望 {self.dim}, 实际 {meta_data['dim']}")

        # 加载索引文件
        bin_path = f"{index_path}.bin"
        self.index.load_index(bin_path, max_elements=meta_data['max_elements'])

        # 恢复元数据
        self.max_elements = meta_data['max_elements']
        self.n_vectors = meta_data['n_vectors']
        self.M = meta_data.get('M', self.M)
        self.ef_construction = meta_data.get('ef_construction', self.ef_construction)

        # 恢复label_to_payload（需要将key从字符串转换为int）
        self.label_to_payload = {
            int(k): v for k, v in meta_data['label_to_payload'].items()
        }

        self.index_path = index_path
