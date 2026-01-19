"""
路径结构定义
"""

from typing import List, Set, Optional
from dataclasses import dataclass, field
import networkx as nx


@dataclass
class Path:
    """搜索路径"""
    nodes: List[str] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    visited_entities: Set[str] = field(default_factory=set)
    accumulated_score: float = 0.0
    graph: Optional[nx.DiGraph] = None  # 图引用

    # 关联的意图信息
    intent_label: str = ""  # 意图标签
    info_gap: Optional['InfoGap'] = None  # 关联的 InfoGap

    def copy(self) -> 'Path':
        """创建路径副本"""
        return Path(
            nodes=self.nodes.copy(),
            scores=self.scores.copy(),
            visited_entities=self.visited_entities.copy(),
            accumulated_score=self.accumulated_score,
            graph=self.graph
        )

    def add_node(self, node_id: str, score: float, new_entities: Set[str], graph: Optional[nx.DiGraph] = None):
        """添加节点到路径"""
        self.nodes.append(node_id)
        self.scores.append(score)
        self.visited_entities.update(new_entities)
        self.accumulated_score += score
        # 保存 graph 引用
        if graph is not None:
            self.graph = graph

    def get_last_proposition(self) -> Optional[str]:
        """获取路径中最后一个命题节点"""
        if self.graph is None:
            return None
        from ..proposition_graph.graph_builder import PROPOSITION_NODE
        for node_id in reversed(self.nodes):
            if node_id in self.graph.nodes:
                node_data = self.graph.nodes[node_id]
                if node_data.get("node_type") == PROPOSITION_NODE:
                    return node_id
        return None

    def __repr__(self):
        return f"Path(len={len(self.nodes)}, score={self.accumulated_score:.2f})"
