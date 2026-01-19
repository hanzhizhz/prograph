"""
图融合
合并链接的实体并创建全局节点
"""

import pickle
import json
from pathlib import Path
from typing import Dict, List, Set, Optional
import networkx as nx

from .entity_linker import EntityFusionResult, FusedEntity, LinkingDecision
from ..proposition_graph.graph_builder import (
    ENTITY_NODE,
    GLOBAL_ENTITY_NODE,
    MENTIONS_ENTITY
)


class GraphFusion:
    """
    图融合器

    将链接的实体合并为全局实体节点
    """

    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        self.entity_groups: Dict[str, Set[str]] = {}  # global_id -> local_ids
        self.local_to_global: Dict[str, str] = {}  # local_id -> global_id

    def fuse(
        self,
        entity_fusion_results: Optional[List[EntityFusionResult]] = None,
        proposition_decisions: Optional[List[LinkingDecision]] = None,
    ) -> nx.DiGraph:
        """
        融合图

        Args:
            entity_fusion_results: 实体融合结果
            proposition_decisions: 命题关系决策

        Returns:
            融合后的图
        """
        print("\n开始图融合...")

        # 1. 构建实体组
        print("1. 构建实体组...")
        if entity_fusion_results is not None:
            self._build_fusion_groups(entity_fusion_results)
        else:
            print("  没有实体融合决策，跳过")

        # 2. 创建全局实体节点
        print("2. 创建全局实体节点...")
        self._create_global_entities()

        # 3. 重定向包含边
        print("3. 重定向包含边...")
        self._redirect_mention_edges()

        # 4. 删除局部实体节点
        print("4. 删除局部实体节点...")
        self._remove_local_entities()

        # 5. 添加命题关系边（如果有）
        if proposition_decisions:
            print("5. 添加命题关系边...")
            self._add_proposition_relations(proposition_decisions)

        print("图融合完成！\n")
        return self.graph

    def _build_fusion_groups(self, fusion_results: List[EntityFusionResult]) -> None:
        """根据融合结果构建实体组（新格式）"""
        groups = {}
        total_fused = 0

        for result in fusion_results:
            for fused_entity in result.fused_entities:
                if len(fused_entity.original_ids) >= 2:
                    # 创建全局 ID
                    global_id = fused_entity.original_ids[0] + "_fused"
                    groups[global_id] = set(fused_entity.original_ids)
                    total_fused += len(fused_entity.original_ids)

        self.entity_groups = groups

        # 创建局部到全局的映射
        self.local_to_global = {}
        for global_id, local_ids in groups.items():
            for local_id in local_ids:
                self.local_to_global[local_id] = global_id

        print(f"  融合了 {len(groups)} 个实体组，共 {total_fused} 个实体")

    def _create_global_entities(self) -> None:
        """创建全局实体节点"""
        created_count = 0

        for global_id, local_ids in self.entity_groups.items():
            if len(local_ids) <= 1:
                # 只有一个成员，不需要创建全局节点
                for local_id in local_ids:
                    self.local_to_global[local_id] = local_id
                continue

            # 合并实体属性（从原始节点获取）
            merged_text = self._merge_entity_text(local_ids)
            merged_type = self._merge_entity_type(local_ids)
            merged_docs = self._get_entity_docs(local_ids)

            # 如果 global_id 是以 "_fused" 结尾的，说明是新的融合结果
            # 可以尝试从融合结果中获取更好的文本和类型
            # 但这里为了简单，还是使用合并逻辑

            # 创建全局实体节点
            self.graph.add_node(
                global_id,
                node_type=GLOBAL_ENTITY_NODE,
                text=merged_text,
                entity_type=merged_type,
                doc_ids=list(merged_docs),
                member_count=len(local_ids),
            )
            created_count += 1

        print(f"  创建了 {created_count} 个全局实体节点")

    def _merge_entity_text(self, local_ids: Set[str]) -> str:
        """合并实体文本"""
        texts = []
        for local_id in local_ids:
            if local_id in self.graph.nodes:
                text = self.graph.nodes[local_id].get("text", "")
                if text and text not in texts:
                    texts.append(text)

        # 选择最短的文本作为主要文本（通常最精确）
        if texts:
            texts.sort(key=len)
            return texts[0]
        return ""

    def _merge_entity_type(self, local_ids: Set[str]) -> str:
        """合并实体类型"""
        types = set()
        for local_id in local_ids:
            if local_id in self.graph.nodes:
                entity_type = self.graph.nodes[local_id].get("entity_type", "")
                if entity_type:
                    types.add(entity_type)

        # 优先级: PERSON > ORGANIZATION > LOCATION > DATE > NUMBER > MISC
        priority = ["PERSON", "ORGANIZATION", "LOCATION", "DATE", "NUMBER", "MISC"]
        for t in priority:
            if t in types:
                return t
        return "MISC"

    def _get_entity_docs(self, local_ids: Set[str]) -> Set[str]:
        """获取实体涉及的文档"""
        docs = set()
        for local_id in local_ids:
            if local_id in self.graph.nodes:
                doc_id = self.graph.nodes[local_id].get("doc_id", "")
                if doc_id:
                    docs.add(doc_id)
        return docs

    def _redirect_mention_edges(self) -> None:
        """重定向边（命题 -> 局部实体 -> 全局实体）"""
        redirect_count = 0

        for local_id, global_id in self.local_to_global.items():
            if local_id == global_id:
                continue

            # 1. 重定向入边（predecessors）
            predecessors = list(self.graph.predecessors(local_id))
            for pred_id in predecessors:
                if self.graph.has_edge(pred_id, local_id):
                    edge_data = self.graph[pred_id][local_id]
                    if not self.graph.has_edge(pred_id, global_id):
                        self.graph.add_edge(pred_id, global_id, **edge_data)
                        redirect_count += 1
                    self.graph.remove_edge(pred_id, local_id)

            # 2. 重定向出边（successors）
            successors = list(self.graph.successors(local_id))
            for succ_id in successors:
                if self.graph.has_edge(local_id, succ_id):
                    edge_data = self.graph[local_id][succ_id]
                    if not self.graph.has_edge(global_id, succ_id):
                        self.graph.add_edge(global_id, succ_id, **edge_data)
                        redirect_count += 1
                    self.graph.remove_edge(local_id, succ_id)

        print(f"  重定向了 {redirect_count} 条边")

    def _remove_local_entities(self) -> None:
        """删除局部实体节点"""
        removed_count = 0

        for local_id, global_id in self.local_to_global.items():
            if local_id != global_id and local_id in self.graph.nodes:
                self.graph.remove_node(local_id)
                removed_count += 1

        print(f"  删除了 {removed_count} 个局部实体节点")

    def _add_proposition_relations(self, decisions: List[LinkingDecision]) -> None:
        """添加命题关系边（根据 direction 正确添加方向）"""
        added_count = 0

        for decision in decisions:
            if decision.should_link and decision.relation_type:
                # SIMILARITY 边强制双向
                if decision.relation_type == "SIMILARITY":
                    self.graph.add_edge(
                        decision.id1, decision.id2,
                        edge_type=decision.relation_type,
                        direction="bidirectional",
                        confidence=decision.confidence,
                        reason=decision.reason
                    )
                    self.graph.add_edge(
                        decision.id2, decision.id1,
                        edge_type=decision.relation_type,
                        direction="bidirectional",
                        confidence=decision.confidence,
                        reason=decision.reason
                    )
                    added_count += 2
                else:
                    # 其他关系按 direction 字段添加，为每条边设置明确的 direction
                    direction = decision.direction or "1->2"
                    if direction in ["1->2", "1<->2"]:
                        self.graph.add_edge(
                            decision.id1, decision.id2,
                            edge_type=decision.relation_type,
                            direction="1->2",
                            confidence=decision.confidence,
                            reason=decision.reason
                        )
                        added_count += 1
                    if direction in ["2->1", "1<->2"]:
                        self.graph.add_edge(
                            decision.id2, decision.id1,
                            edge_type=decision.relation_type,
                            direction="2->1",
                            confidence=decision.confidence,
                            reason=decision.reason
                        )
                        added_count += 1

        print(f"  添加了 {added_count} 条命题关系边")

    def save_graph(self, output_path: str, format: str = "both") -> None:
        """
        保存融合后的图

        Args:
            output_path: 输出路径
            format: 保存格式
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        if format in ["pkl", "both"]:
            pkl_path = output.with_suffix(".pkl")
            with open(pkl_path, 'wb') as f:
                pickle.dump(self.graph, f)
            print(f"融合图已保存 (PKL): {pkl_path}")

        if format in ["json", "both"]:
            json_path = output.with_suffix(".json")
            graph_data = nx.node_link_data(self.graph)

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, ensure_ascii=False, indent=2)
            print(f"融合图已保存 (JSON): {json_path}")

    def get_statistics(self) -> Dict:
        """获取融合后的统计信息"""
        stats = {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "global_entities": 0,
            "local_entities": 0,
            "propositions": 0,
        }

        for _, node_data in self.graph.nodes(data=True):
            node_type = node_data.get("node_type", "")
            if node_type == GLOBAL_ENTITY_NODE:
                stats["global_entities"] += 1
            elif node_type == ENTITY_NODE:
                stats["local_entities"] += 1
            elif node_type == "proposition":
                stats["propositions"] += 1

        return stats
