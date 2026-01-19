"""
路径选择器
对搜索结果进行排序和多样性优化

支持两种 RankedPath 格式：
1. subgraph_structures.RankedPath（新格式，推荐）
2. path_structures.Path（旧格式，向后兼容）
"""

from typing import List, Set, Dict, Optional, Union
from dataclasses import dataclass

# 导入新的 RankedPath 格式
from .subgraph_structures import RankedPath as SubgraphRankedPath
# 保留旧的 Path 用于向后兼容
from .path_structures import Path


class PathSelector:
    """
    路径选择器

    对搜索结果进行：
    1. 支持归一化评分（SubgraphRankedPath 已内置）
    2. MMR 多样性重排序
    3. 文档覆盖统计
    """

    def __init__(
        self,
        graph,
        mmr_lambda: float = 0.7,  # MMR 相关性权重
    ):
        self.graph = graph
        self.mmr_lambda = mmr_lambda

    def select(
        self,
        paths: List[Union[SubgraphRankedPath, Path]],
        top_k: int = 5,
    ) -> List[SubgraphRankedPath]:
        """
        选择最优路径

        Args:
            paths: 候选路径列表（支持 SubgraphRankedPath 或 Path）
            top_k: 返回 top-k 路径

        Returns:
            SubgraphRankedPath 列表
        """
        if not paths:
            return []

        # 检测输入格式并转换为统一格式
        unified_paths = self._unify_paths(paths)

        # SubgraphRankedPath 已内置 normalized_score，直接使用
        # 按归一化分数排序（in-place）
        unified_paths.sort(key=lambda p: p.normalized_score, reverse=True)

        # MMR 多样性重排序（会创建新列表）
        reranked_paths = self._mmr_rerank(unified_paths)

        # 更新 top-k 路径的 metadata（避免创建新对象）
        for path in reranked_paths[:top_k]:
            path.metadata['document_coverage'] = self._get_document_coverage(path)
            path.metadata['diversity_score'] = 0.0  # TODO: 计算

        return reranked_paths[:top_k]

    def _unify_paths(
        self,
        paths: List[Union[SubgraphRankedPath, Path]]
    ) -> List[SubgraphRankedPath]:
        """将不同格式的路径统一为 SubgraphRankedPath"""
        unified = []

        for path in paths:
            if isinstance(path, SubgraphRankedPath):
                # 已经是新格式，直接使用
                unified.append(path)
            else:
                # 旧格式 Path，需要转换
                unified.append(self._convert_from_path(path))

        return unified

    def _convert_from_path(self, path: Path) -> SubgraphRankedPath:
        """将旧格式的 Path 转换为新格式"""
        # 计算归一化分数
        normalized = path.accumulated_score / max(len(path.nodes), 1)

        return SubgraphRankedPath(
            nodes=path.nodes.copy(),
            raw_score=path.accumulated_score,
            normalized_score=normalized,
            intent_label=getattr(path, 'intent_label', ''),
            metadata={
                'visited_entities': list(path.visited_entities),
                'gap': getattr(path.info_gap, 'gap_description', '') if hasattr(path, 'info_gap') and path.info_gap else ''
            }
        )

    # _normalize_scores 方法已删除
    # SubgraphRankedPath 已内置 normalized_score，无需额外归一化

    def _mmr_rerank(self, paths: List[SubgraphRankedPath]) -> List[SubgraphRankedPath]:
        """
        MMR (Maximal Marginal Relevance) 重排序

        【性能优化】惰性增量计算相似度，避免预构建完整的 n×n 矩阵

        平衡相关性和多样性
        优化点：
        1. 预计算端点 frozenset（避免重复创建）
        2. 惰性计算相似度（按需计算 + 缓存）
        3. 移除 numpy 依赖，改用纯 Python
        """
        if not paths:
            return []

        n = len(paths)
        if n <= 1:
            return paths.copy()

        # 限制最大参与MMR计算的路径数量（Top-50足够）
        MAX_MMR_PATHS = 50
        if n > MAX_MMR_PATHS:
            paths = paths[:MAX_MMR_PATHS]
            n = len(paths)

        # 【优化1】预计算端点 frozenset（避免在循环中重复创建）
        def get_endpoint_frozenset(path: SubgraphRankedPath) -> frozenset:
            """获取路径首尾节点的 frozenset"""
            if not path.nodes:
                return frozenset()
            endpoints = set()
            if path.nodes[0]:
                endpoints.add(path.nodes[0])
            if path.nodes[-1]:
                endpoints.add(path.nodes[-1])
            return frozenset(endpoints)

        path_endpoints = [get_endpoint_frozenset(p) for p in paths]

        # 【优化2】惰性相似度缓存：{(i, j): similarity}
        _similarity_cache: Dict[tuple, float] = {}

        def get_similarity(i: int, j: int) -> float:
            """惰性获取相似度（带缓存）"""
            key = (i, j) if i < j else (j, i)
            if key not in _similarity_cache:
                # 计算端点 Jaccard 相似度
                set_i = path_endpoints[i]
                set_j = path_endpoints[j]
                if not set_i or not set_j:
                    _similarity_cache[key] = 0.0
                else:
                    intersection = len(set_i & set_j)
                    union = len(set_i | set_j)
                    _similarity_cache[key] = intersection / union if union > 0 else 0.0
            return _similarity_cache[key]

        # 【优化3】贪心选择路径（纯 Python 实现）
        selected_indices: List[int] = []
        remaining: set = set(range(n))

        # 选择第一个（分数最高的）
        first_idx = 0
        selected_indices.append(first_idx)
        remaining.remove(first_idx)

        # 最多选择 MAX_SELECTED 条路径
        MAX_SELECTED = 10
        while remaining and len(selected_indices) < min(n, MAX_SELECTED):
            best_idx: Optional[int] = None
            best_mmr = float('-inf')

            for idx in remaining:
                relevance = paths[idx].normalized_score

                # 只计算与已选路径的最大相似度
                max_sim = 0.0
                for selected in selected_indices:
                    sim = get_similarity(idx, selected)
                    if sim > max_sim:
                        max_sim = sim

                mmr = self.mmr_lambda * relevance - (1 - self.mmr_lambda) * max_sim

                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = idx

            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining.remove(best_idx)

        return [paths[i] for i in selected_indices]

    def _compute_path_similarity(
        self,
        path1: SubgraphRankedPath,
        path2: SubgraphRankedPath
    ) -> float:
        """计算两条路径的相似度（基于节点重叠）"""
        set1 = set(path1.nodes)
        set2 = set(path2.nodes)

        if not set1 or not set2:
            return 0.0

        # Jaccard 相似度
        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def _get_document_coverage(
        self,
        path: SubgraphRankedPath
    ) -> Set[str]:
        """获取路径覆盖的文档"""
        docs = set()

        for node_id in path.nodes:
            if node_id in self.graph.nodes:
                node_data = self.graph.nodes[node_id]
                doc_id = node_data.get("doc_id", "")
                if doc_id:
                    docs.add(doc_id)

        return docs
