"""
HNSW 向量索引
用于高效的向量相似度搜索
"""

import pickle
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import hnswlib


class EmbeddingIndex:
    """
    HNSW 向量索引

    用于快速检索相似向量
    """

    def __init__(
        self,
        dim: int = 768,
        max_elements: int = 100000,
        ef_construction: int = 200,
        M: int = 16,
    ):
        """
        初始化索引

        Args:
            dim: 向量维度
            max_elements: 最大元素数量
            ef_construction: 构建时的搜索范围（越大越精确但越慢）
            M: 每个节点的最大连接数
        """
        self.dim = dim
        self.max_elements = max_elements

        # 初始化 HNSW 索引
        self.index = hnswlib.Index(space='cosine', dim=dim)

        # 初始化索引
        self.index.init_index(
            max_elements=max_elements,
            ef_construction=ef_construction,
            M=M
        )

        # 【性能优化】动态 ef 参数，根据 k 值调整
        self.ef = 50  # 默认值，可以在 search 时动态调整

        # 存储标签到 ID 的映射
        self.labels: List[str] = []
        self.label_to_idx: dict = {}

    def add_items(
        self,
        vectors: np.ndarray,
        labels: List[str],
    ) -> None:
        """
        添加向量到索引

        Args:
            vectors: 向量数组，shape (n, dim)
            labels: 标签列表
        """
        n = len(vectors)

        # 生成整数索引
        indices = list(range(len(self.labels), len(self.labels) + n))

        # 添加到 HNSW 索引
        self.index.add_items(vectors, indices)

        # 更新标签映射
        for label, idx in zip(labels, indices):
            self.labels.append(label)
            self.label_to_idx[label] = idx

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        搜索最相似的向量

        Args:
            query_vector: 查询向量，shape (dim,)
            k: 返回 top-k 结果

        Returns:
            [(label, similarity), ...] 按相似度降序排列
        """
        # 【性能优化】动态设置 ef，确保 ef >= k*2
        ef = max(k * 2, self.ef)
        self.index.set_ef(ef)

        # HNSW 返回 (indices, distances)
        indices, distances = self.index.knn_query(query_vector, k=k)

        # 转换为 (label, similarity)
        # 注意：HNSW 的 distance 是 1 - cosine_similarity
        results = []
        for idx, dist in zip(indices[0], distances[0]):  # hnswlib 返回的是 2D 数组
            if idx < len(self.labels):
                similarity = 1 - dist
                results.append((self.labels[idx], similarity))

        return results

    def search_batch(
        self,
        query_vectors: np.ndarray,
        k: int = 10,
    ) -> List[List[Tuple[str, float]]]:
        """
        批量搜索最相似的向量

        一次性处理所有查询，使用 HNSW 的批量查询优化

        Args:
            query_vectors: 查询向量数组，shape (n_queries, dim)
            k: 返回 top-k 结果

        Returns:
            [[(label, similarity), ...], ...]
            每个查询的结果列表，按相似度降序排列
        """
        n_queries = len(query_vectors)
        all_results = []

        # 【性能优化】动态设置 ef，确保 ef >= k*2
        ef = max(k * 2, self.ef)
        self.index.set_ef(ef)

        # 一次性处理所有查询
        indices, distances = self.index.knn_query(query_vectors, k=k)

        # 转换结果
        for i in range(n_queries):
            results = []
            for idx, dist in zip(indices[i], distances[i]):
                if idx < len(self.labels):
                    similarity = 1 - dist
                    results.append((self.labels[idx], similarity))
            all_results.append(results)

        return all_results

    def save(self, path: str) -> None:
        """
        保存索引到文件

        Args:
            path: 保存路径
        """
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)

        # 保存 HNSW 索引
        index_path = str(output).replace('.pkl', '_index.bin')
        self.index.save_index(index_path)

        # 保存元数据
        metadata = {
            'dim': self.dim,
            'max_elements': self.max_elements,
            'labels': self.labels,
            'label_to_idx': self.label_to_idx,
        }

        with open(output, 'wb') as f:
            pickle.dump(metadata, f)

        print(f"索引已保存: {output}")

    @classmethod
    def load(cls, path: str) -> 'EmbeddingIndex':
        """
        从文件加载索引

        Args:
            path: 索引文件路径

        Returns:
            EmbeddingIndex 实例
        """
        input_path = Path(path)

        # 加载元数据
        with open(input_path, 'rb') as f:
            metadata = pickle.load(f)

        # 创建实例
        instance = cls(
            dim=metadata['dim'],
            max_elements=metadata['max_elements'],
        )

        # 恢复标签映射
        instance.labels = metadata['labels']
        instance.label_to_idx = metadata['label_to_idx']

        # 加载 HNSW 索引
        index_path = str(input_path).replace('.pkl', '_index.bin')
        instance.index.load_index(index_path)

        # 【性能优化】移除固定 ef 设置，使用动态 ef

        print(f"索引已加载: {input_path}")
        return instance


class EmbeddingStore:
    """
    嵌入向量存储

    管理多个不同类型的索引
    """

    def __init__(self):
        self.indices: dict = {}

    def create_index(
        self,
        name: str,
        dim: int = 768,
        max_elements: int = 100000,
    ) -> EmbeddingIndex:
        """
        创建新索引

        Args:
            name: 索引名称
            dim: 向量维度
            max_elements: 最大元素数量

        Returns:
            EmbeddingIndex 实例
        """
        index = EmbeddingIndex(dim=dim, max_elements=max_elements)
        self.indices[name] = index
        return index

    def get_index(self, name: str) -> Optional[EmbeddingIndex]:
        """获取索引"""
        return self.indices.get(name)

    def save(self, output_dir: str) -> None:
        """保存所有索引"""
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        for name, index in self.indices.items():
            index.save(str(output / f"{name}.pkl"))

    @classmethod
    def load(cls, input_dir: str) -> 'EmbeddingStore':
        """
        从目录加载所有索引

        Args:
            input_dir: 输入目录

        Returns:
            EmbeddingStore 实例
        """
        store = cls()
        input_path = Path(input_dir)

        for index_file in input_path.glob("*.pkl"):
            # 排除 _index.bin 文件
            if "_index.bin" not in str(index_file):
                name = index_file.stem
                index = EmbeddingIndex.load(str(index_file))
                store.indices[name] = index

        return store
