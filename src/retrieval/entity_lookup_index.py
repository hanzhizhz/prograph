"""
实体查找索引

将实体名称到节点ID的映射从O(n)线性遍历优化为O(1)哈希查找
"""

import logging
from typing import Dict, List, Optional, Set

import networkx as nx


logger = logging.getLogger(__name__)


class EntityLookupIndex:
    """
    实体查找索引

    加速实体名称到节点ID的映射查找，从O(n)优化到O(1)

    功能：
    - 预构建 name -> node_id 的哈希索引
    - 支持 entity 和 global_entity 两种节点类型
    - global_entity 按 member_count 排序返回最优结果
    """

    def __init__(self, graph: nx.DiGraph, persistence_dir: Optional[str] = None):
        """
        初始化实体索引

        Args:
            graph: 命题图
            persistence_dir: 持久化数据目录
        """
        self.graph = graph
        self.persistence_dir = persistence_dir
        self._name_to_global_entities: Dict[str, List[str]] = {}
        self._name_to_entities: Dict[str, List[str]] = {}
        self._built = False

        # 从 graph_builder 导入节点类型常量
        self.ENTITY_NODE = "entity"
        self.GLOBAL_ENTITY_NODE = "global_entity"
        self.PROPOSITION_NODE = "proposition"

        # 尝试从模块获取常量
        try:
            from ..proposition_graph.graph_builder import ENTITY_NODE, GLOBAL_ENTITY_NODE
            self.ENTITY_NODE = ENTITY_NODE
            self.GLOBAL_ENTITY_NODE = GLOBAL_ENTITY_NODE
        except ImportError:
            pass

    def load(self) -> bool:
        """
        从文件加载实体查找索引

        Returns:
            True 如果成功加载，False 如果文件不存在或加载失败
        """
        import json
        from pathlib import Path

        if not self.persistence_dir:
            return False

        lookup_file = Path(self.persistence_dir) / "entity_lookup.json"
        if not lookup_file.exists():
            return False

        try:
            with open(lookup_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self._name_to_global_entities = data.get("global_entities", {})
            self._name_to_entities = data.get("entities", {})
            self._built = True

            stats = data.get("stats", {})
            print(f"✓ 实体查找索引已从文件加载: {stats.get('total_entities', 0)} 个实体")
            return True
        except Exception as e:
            print(f"警告: 实体查找索引加载失败 ({e})，将重新构建")
            return False

    def build(self) -> None:
        """
        构建索引

        遍历图中所有实体节点，建立名称到节点ID的映射
        """
        self._name_to_global_entities.clear()
        self._name_to_entities.clear()

        for node_id, node_data in self.graph.nodes(data=True):
            node_type = node_data.get("node_type")
            if node_type not in [self.ENTITY_NODE, self.GLOBAL_ENTITY_NODE]:
                continue

            text = node_data.get("text", "").lower()
            if not text:
                continue

            if node_type == self.GLOBAL_ENTITY_NODE:
                if text not in self._name_to_global_entities:
                    self._name_to_global_entities[text] = []
                self._name_to_global_entities[text].append(node_id)
            else:  # ENTITY_NODE
                if text not in self._name_to_entities:
                    self._name_to_entities[text] = []
                self._name_to_entities[text].append(node_id)

        # 按 member_count 排序 global_entities（member_count 越多越优先）
        for name in self._name_to_global_entities:
            self._name_to_global_entities[name].sort(
                key=lambda nid: self.graph.nodes[nid].get("member_count", 0),
                reverse=True
            )

        self._built = True

        logger.info(
            f"实体索引构建完成: "
            f"global_entity={len(self._name_to_global_entities)}, "
            f"entity={len(self._name_to_entities)}"
        )

    def lookup(
        self,
        entity_name: str,
        prefer_global: bool = True
    ) -> Optional[str]:
        """
        查找实体节点ID（O(1)复杂度）

        Args:
            entity_name: 实体名称
            prefer_global: 是否优先返回 global_entity

        Returns:
            匹配的节点ID，如果不存在则返回None
        """
        if not self._built:
            self.build()

        name_lower = entity_name.lower()

        # 按优先级构建查找顺序
        if prefer_global:
            candidates = [
                self._name_to_global_entities.get(name_lower, []),
                self._name_to_entities.get(name_lower, [])
            ]
        else:
            candidates = [
                self._name_to_entities.get(name_lower, []),
                self._name_to_global_entities.get(name_lower, [])
            ]

        # 返回第一个非空候选列表的第一个元素
        for candidate_list in candidates:
            if candidate_list:
                return candidate_list[0]

        return None

    def lookup_batch(
        self,
        entity_names: List[str],
        prefer_global: bool = True
    ) -> List[Optional[str]]:
        """
        批量查找实体节点ID

        Args:
            entity_names: 实体名称列表
            prefer_global: 是否优先返回 global_entity

        Returns:
            匹配的节点ID列表，不存在的位置为None
        """
        return [
            self.lookup(name, prefer_global)
            for name in entity_names
        ]

    def lookup_all(
        self,
        entity_name: str,
        prefer_global: bool = True
    ) -> List[str]:
        """
        查找所有匹配的实体节点ID

        Args:
            entity_name: 实体名称
            prefer_global: 是否优先返回 global_entity

        Returns:
            所有匹配的节点ID列表，如果不存在则返回空列表
        """
        if not self._built:
            self.build()

        name_lower = entity_name.lower()

        if prefer_global:
            if name_lower in self._name_to_global_entities:
                return self._name_to_global_entities[name_lower]
            if name_lower in self._name_to_entities:
                return self._name_to_entities[name_lower]
        else:
            if name_lower in self._name_to_entities:
                return self._name_to_entities[name_lower]
            if name_lower in self._name_to_global_entities:
                return self._name_to_global_entities[name_lower]

        return []

    def get_stats(self) -> Dict[str, int]:
        """
        获取索引统计信息

        Returns:
            统计信息字典
        """
        total_global = sum(len(v) for v in self._name_to_global_entities.values())
        total_entity = sum(len(v) for v in self._name_to_entities.values())

        return {
            "unique_global_names": len(self._name_to_global_entities),
            "unique_entity_names": len(self._name_to_entities),
            "total_global_nodes": total_global,
            "total_entity_nodes": total_entity,
            "built": self._built,
        }
