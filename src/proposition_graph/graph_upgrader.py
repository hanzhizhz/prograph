"""
图升级工具
将旧版本的图升级到新版本，添加缺失的反向边
"""

import networkx as nx
from pathlib import Path
from typing import Set
import pickle

from .rst_analyzer import MENTIONS_ENTITY


class GraphUpgrader:
    """图升级工具"""

    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        self.upgrade_stats = {
            "similarity_edges_added": 0,
            "mention_edges_added": 0,
            "bidirectional_rst_added": 0
        }

    def upgrade(self) -> nx.DiGraph:
        """升级图到新版本

        1. 为 SIMILARITY 边添加反向边
        2. 为 MENTIONS_ENTITY 边添加反向边
        3. 为 direction="1<->2" 的边添加反向边
        """
        # 记录已处理的边，避免重复
        processed_edges: Set[tuple] = set()

        # 遍历所有边，找出需要添加反向边的边
        for u, v, edge_data in self.graph.edges(data=True):
            edge_key = (u, v)
            if edge_key in processed_edges:
                continue

            edge_type = edge_data.get("edge_type", "")
            direction = edge_data.get("direction", "")

            # 情况 1: SIMILARITY 边强制双向
            if edge_type == "SIMILARITY":
                # 检查是否已有相同类型的反向边
                has_reverse = False
                if self.graph.has_edge(v, u):
                    reverse_data = self.graph.get_edge_data(v, u)
                    has_reverse = reverse_data.get("edge_type") == "SIMILARITY"

                if not has_reverse:
                    self.graph.add_edge(v, u, **{**edge_data, "direction": "bidirectional"})
                    self.upgrade_stats["similarity_edges_added"] += 1
                processed_edges.add((u, v))
                processed_edges.add((v, u))

            # 情况 2: MENTIONS_ENTITY 边添加反向边
            elif edge_type == MENTIONS_ENTITY:
                # 检查是否已有相同类型的反向边
                has_reverse = False
                if self.graph.has_edge(v, u):
                    reverse_data = self.graph.get_edge_data(v, u)
                    has_reverse = reverse_data.get("edge_type") == MENTIONS_ENTITY

                if not has_reverse:
                    self.graph.add_edge(v, u, **{**edge_data, "direction": "entity->prop"})
                    self.upgrade_stats["mention_edges_added"] += 1
                processed_edges.add((u, v))
                processed_edges.add((v, u))

            # 情况 3: direction="1<->2" 的边添加反向边
            elif direction == "1<->2":
                if not self.graph.has_edge(v, u):
                    self.graph.add_edge(v, u, **edge_data)
                    self.upgrade_stats["bidirectional_rst_added"] += 1
                processed_edges.add((u, v))
                processed_edges.add((v, u))

            else:
                processed_edges.add((u, v))

        return self.graph

    def print_stats(self):
        """打印升级统计信息"""
        print("图升级统计:")
        print(f"  SIMILARITY 反向边添加: {self.upgrade_stats['similarity_edges_added']}")
        print(f"  MENTIONS_ENTITY 反向边添加: {self.upgrade_stats['mention_edges_added']}")
        print(f"  双向 RST 边添加: {self.upgrade_stats['bidirectional_rst_added']}")
        print(f"  总计添加边: {sum(self.upgrade_stats.values())}")

    @staticmethod
    def upgrade_file(input_path: str, output_path: str = None) -> nx.DiGraph:
        """升级图文件

        Args:
            input_path: 输入图文件路径
            output_path: 输出图文件路径（可选，默认覆盖原文件）

        Returns:
            升级后的图
        """
        input_path = Path(input_path)
        if output_path is None:
            output_path = input_path
        else:
            output_path = Path(output_path)

        # 加载图
        print(f"加载图: {input_path}")
        with open(input_path, 'rb') as f:
            graph = pickle.load(f)

        # 升级
        upgrader = GraphUpgrader(graph)
        upgraded_graph = upgrader.upgrade()
        upgrader.print_stats()

        # 保存
        print(f"保存升级后的图: {output_path}")
        with open(output_path, 'wb') as f:
            pickle.dump(upgraded_graph, f)

        return upgraded_graph


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.proposition_graph.graph_upgrader <input_path> [output_path]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    GraphUpgrader.upgrade_file(input_path, output_path)
