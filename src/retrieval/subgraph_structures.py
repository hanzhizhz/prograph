"""
子图结构定义

用于 MAP 阶段的独立意图子图构建与路径汇总。
"""

from typing import List, Set, Dict, Optional, Any, TYPE_CHECKING
from dataclasses import dataclass, field
import networkx as nx

if TYPE_CHECKING:
    from .path_trie import PathTrie


@dataclass
class IntentSubgraph:
    """单个意图的 DAG 子图

    每个意图分组独立构建自己的子图，互不干扰。
    """
    # 基本信息
    intent_label: str                    # 意图标签
    info_gap: 'InfoGap'                   # 关联的信息缺口

    # 图结构（只包含命题节点）
    graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    # 节点属性：{'node_id', 'text', 'score', 'step', 'new_entities'}
    # 边属性：{'score', 'step'}

    # 扩展状态
    frontier_nodes: Set[str] = field(default_factory=set)  # 当前末端节点（待扩展）
    visited_entities: Set[str] = field(default_factory=set)  # 已访问实体
    start_nodes: Set[str] = field(default_factory=set)  # 【性能优化】起点节点（入度为0）

    # 统计信息
    max_depth: int = 0                   # 最大深度


@dataclass
class RankedPath:
    """排序后的路径（新格式，用于子图汇总）"""
    nodes: List[str]                     # 节点序列
    raw_score: float                     # 原始分数（边分数之和）
    normalized_score: float              # 归一化分数 = raw_score / len(nodes)
    intent_label: str = ""               # 意图标签
    metadata: Dict[str, Any] = field(default_factory=dict)  # 其他元数据

    def __repr__(self):
        return f"RankedPath(len={len(self.nodes)}, score={self.normalized_score:.3f}, intent={self.intent_label})"


@dataclass
class AggregatedPaths:
    """
    汇总所有意图的最终路径

    【性能优化】使用 PathTrie 进行路径去重，替代 frozenset
    优势：
    1. 正确保持路径顺序（frozenset 会丢失顺序）
    2. 空间效率更高（共享公共前缀）
    3. 查找效率稳定（O(path_length)）
    """
    # 路径存储（key为路径节点的元组，保持顺序）
    unique_paths: Dict[tuple, RankedPath] = field(default_factory=dict)

    # Trie 树用于去重（延迟初始化）
    _path_trie: Optional['PathTrie'] = field(default=None, init=False, repr=False)

    total_paths: int = 0                  # 去重前路径数
    unique_path_count: int = 0            # 去重后路径数

    def __post_init__(self):
        """初始化后创建 PathTrie"""
        # 延迟导入避免循环依赖
        from .path_trie import PathTrie
        self._path_trie = PathTrie()

    def add_path(self, nodes: List[str], score: float, intent: str, gap_desc: str) -> bool:
        """
        添加路径，自动去重

        【性能优化】使用 PathTrie 检测重复路径

        Args:
            nodes: 节点序列
            score: 原始分数
            intent: 意图标签
            gap_desc: 信息缺口描述

        Returns:
            是否成功添加（如果已存在则返回 False）
        """
        if not nodes:
            return False

        self.total_paths += 1

        # 使用 PathTrie 检测重复
        if self._path_trie.add_path(nodes):
            # 新路径，添加到存储
            path_key = tuple(nodes)  # 使用元组保持顺序
            normalized = score / max(len(nodes), 1)
            self.unique_paths[path_key] = RankedPath(
                nodes=list(nodes),
                raw_score=score,
                normalized_score=normalized,
                intent_label=intent,
                metadata={'gap': gap_desc}
            )
            self.unique_path_count += 1
            return True
        return False

    def contains_path(self, nodes: List[str]) -> bool:
        """检查路径是否存在"""
        return self._path_trie.contains_path(nodes) if self._path_trie else False

    def get_top_k(self, k: int) -> List[RankedPath]:
        """
        获取按归一化分数排序的 top-k 路径

        Args:
            k: 返回的路径数量

        Returns:
            排序后的 top-k 路径列表
        """
        ranked = sorted(
            self.unique_paths.values(),
            key=lambda p: p.normalized_score,
            reverse=True
        )
        return ranked[:k]

    def get_trie_stats(self) -> Dict[str, int]:
        """获取 Trie 统计信息"""
        return self._path_trie.get_stats() if self._path_trie else {}
