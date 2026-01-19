"""
图构建器
构建 ProGraph 异构命题图
"""

import pickle
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, asdict
import networkx as nx

from .document_loader import Document
from .unified_extractor import UnifiedDocumentExtractor, PropositionWithEntities
from .proposition_extractor import Proposition
from .entity_extractor import Entity
from .rst_analyzer import (
    RSTRelation,
    PROPOSITION_EDGE_TYPES,
    ALL_EDGE_TYPES,
    EDGE_TYPE_SKELETON,
    EDGE_TYPE_DETAIL,
    MENTIONS_ENTITY,
    normalize_edge_type,
)


# 节点类型
PROPOSITION_NODE = "proposition"
ENTITY_NODE = "entity"
GLOBAL_ENTITY_NODE = "global_entity"  # 实体链接后使用


@dataclass
class GraphStatistics:
    """图统计信息"""
    total_nodes: int = 0
    proposition_nodes: int = 0
    entity_nodes: int = 0
    total_edges: int = 0
    proposition_edges: int = 0  # RST 边
    mention_edges: int = 0  # 包含边
    isolated_nodes: int = 0


class GraphBuilder:
    """
    ProGraph 图构建器

    使用统一文档提取器构建 NetworkX 有向图，包含：
    - 命题节点（proposition）
    - 实体节点（entity）
    - RST 关系边（proposition -> proposition）
    - 包含边（proposition -> entity）
    """

    def __init__(
        self,
        unified_extractor: UnifiedDocumentExtractor,
    ):
        """
        初始化图构建器

        Args:
            unified_extractor: 统一文档提取器
        """
        self.unified_extractor = unified_extractor

        self.graph = nx.DiGraph()
        self.documents: List[Document] = []
        self.propositions: List[Proposition] = []
        self.entities: List[Entity] = []
        self.rst_relations: List[RSTRelation] = []

        # 用于同文档实体去重：(doc_id, entity_text) -> entity_id
        self._entity_map: Dict[Tuple[str, str], str] = {}

    async def add_document(self, document: Document) -> None:
        """
        添加一个文档到图中

        Args:
            document: 文档对象
        """
        self.documents.append(document)
        doc_id = document.doc_id

        # 1. 使用统一提取器提取所有内容（单次 LLM 调用）
        print(f"  提取命题、实体和关系: {document.title}")
        propositions, entities, relations = await self.unified_extractor.extract_from_document(
            document=document
        )

        # 2. 记录命题起始索引
        prop_start_idx = len(self.propositions)
        self.propositions.extend(propositions)
        self.entities.extend(entities)
        self.rst_relations.extend(relations)

        # 3. 添加命题节点
        for prop in propositions:
            prop_id = f"{doc_id}_prop_{prop.prop_idx}"
            self.graph.add_node(
                prop_id,
                node_type=PROPOSITION_NODE,
                text=prop.text,
                doc_id=doc_id,
                sent_idx=prop.sent_idx,
                prop_idx=prop.prop_idx
            )

        # 4. 添加实体节点和包含边
        for entity in entities:
            # 使用 (doc_id, text) 作为去重键，避免同文档相同实体重复创建
            entity_key = (doc_id, entity.text)
            if entity_key not in self._entity_map:
                entity_id = f"{doc_id}_ent_{entity.text}"
                entity_id = entity_id.replace(" ", "_").replace('"', '').replace("'", '')
                self._entity_map[entity_key] = entity_id
                # 添加实体节点
                self.graph.add_node(
                    entity_id,
                    node_type=ENTITY_NODE,
                    text=entity.text,
                    entity_type=entity.type,
                    doc_id=doc_id
                )
            else:
                entity_id = self._entity_map[entity_key]

            # 添加包含边（双向：命题 <-> 实体）
            prop_id = f"{doc_id}_prop_{entity.prop_idx}"
            self._add_mentions_entity_edge(prop_id, entity_id)

        # 5. 添加 RST 关系边
        for rel in relations:
            # 调整索引到全局
            global_idx1 = prop_start_idx + rel.source_idx
            global_idx2 = prop_start_idx + rel.target_idx

            # 找到对应的命题
            if global_idx1 < len(self.propositions) and global_idx2 < len(self.propositions):
                prop1 = self.propositions[global_idx1]
                prop2 = self.propositions[global_idx2]

                prop1_id = f"{prop1.doc_id}_prop_{prop1.prop_idx}"
                prop2_id = f"{prop2.doc_id}_prop_{prop2.prop_idx}"

                self._add_rst_edge(prop1_id, prop2_id, rel)

    async def add_documents(self, documents: List[Document]) -> None:
        """
        批量添加文档（顺序处理，向后兼容方法）

        Args:
            documents: 文档列表
        """
        for doc in documents:
            await self.add_document(doc)

    async def add_documents_batch(
        self,
        documents: List[Document],
    ) -> Dict[str, int]:
        """
        批量添加文档（使用 vLLM 批量推理）

        利用 vLLM 的批量推理能力，一次性处理多个文档。

        Args:
            documents: 文档列表

        Returns:
            统计信息：{"success": 成功数, "failed": 失败数}
        """
        if not documents:
            return {"success": 0, "failed": 0}

        # 1. 使用批量提取器
        print(f"正在批量提取 {len(documents)} 个文档...")
        (
            all_propositions,
            all_entities,
            all_relations,
            success_flags
        ) = await self.unified_extractor.extract_from_documents(documents)

        # 2. 添加到图中
        success_count = 0
        prop_start_idx = len(self.propositions)

        for doc, props, ents, rels, success in zip(
            documents, all_propositions, all_entities, all_relations, success_flags
        ):
            if not success:
                continue

            self.documents.append(doc)
            doc_id = doc.doc_id

            # 记录当前文档的命题起始索引
            doc_prop_start = prop_start_idx

            # 累加到全局列表
            self.propositions.extend(props)
            self.entities.extend(ents)
            self.rst_relations.extend(rels)

            # 添加命题节点
            for prop in props:
                prop_id = f"{doc_id}_prop_{prop.prop_idx}"
                self.graph.add_node(
                    prop_id,
                    node_type=PROPOSITION_NODE,
                    text=prop.text,
                    doc_id=doc_id,
                    sent_idx=prop.sent_idx,
                    prop_idx=prop.prop_idx
                )

            # 添加实体节点和包含边
            for entity in ents:
                # 使用 (doc_id, text) 作为去重键，避免同文档相同实体重复创建
                entity_key = (doc_id, entity.text)
                if entity_key not in self._entity_map:
                    entity_id = f"{doc_id}_ent_{entity.text}"
                    entity_id = entity_id.replace(" ", "_").replace('"', '').replace("'", '')
                    self._entity_map[entity_key] = entity_id
                    # 添加实体节点
                    self.graph.add_node(
                        entity_id,
                        node_type=ENTITY_NODE,
                        text=entity.text,
                        entity_type=entity.type,
                        doc_id=doc_id
                    )
                else:
                    entity_id = self._entity_map[entity_key]

                # 添加包含边（双向：命题 <-> 实体）
                prop_id = f"{doc_id}_prop_{entity.prop_idx}"
                self._add_mentions_entity_edge(prop_id, entity_id)

            # 添加 RST 关系边
            for rel in rels:
                # 调整索引到全局
                global_idx1 = doc_prop_start + rel.source_idx
                global_idx2 = doc_prop_start + rel.target_idx

                # 找到对应的命题
                if global_idx1 < len(self.propositions) and global_idx2 < len(self.propositions):
                    prop1 = self.propositions[global_idx1]
                    prop2 = self.propositions[global_idx2]

                    prop1_id = f"{prop1.doc_id}_prop_{prop1.prop_idx}"
                    prop2_id = f"{prop2.doc_id}_prop_{prop2.prop_idx}"

                    self._add_rst_edge(prop1_id, prop2_id, rel)

            # 更新下一个文档的命题起始索引
            prop_start_idx += len(props)
            success_count += 1

        failed_count = len(documents) - success_count
        print(f"批量处理完成: 成功 {success_count} 个，失败 {failed_count} 个")

        return {"success": success_count, "failed": failed_count}

    def _add_mentions_entity_edge(self, prop_id: str, entity_id: str) -> None:
        """添加双向的命题-实体边

        Args:
            prop_id: 命题节点 ID
            entity_id: 实体节点 ID
        """
        if prop_id not in self.graph:
            return

        # 正向边：命题 -> 实体
        self.graph.add_edge(
            prop_id,
            entity_id,
            edge_type=MENTIONS_ENTITY,
            direction="prop->entity"
        )
        # 反向边：实体 -> 命题
        self.graph.add_edge(
            entity_id,
            prop_id,
            edge_type=MENTIONS_ENTITY,
            direction="entity->prop"
        )

    def _add_rst_edge(self, prop1_id: str, prop2_id: str, rel: RSTRelation) -> None:
        """添加 RST 关系边（根据 direction 正确添加方向）

        Args:
            prop1_id: 命题 1 的节点 ID
            prop2_id: 命题 2 的节点 ID
            rel: RST 关系对象
        """
        # 仅当两个节点都存在时添加边
        if prop1_id not in self.graph or prop2_id not in self.graph:
            return

        # SIMILARITY 边强制双向（相似关系是对称的）
        if rel.relation == "SIMILARITY":
            self.graph.add_edge(
                prop1_id, prop2_id,
                edge_type=rel.relation,
                direction="bidirectional",
                reason=rel.reason
            )
            self.graph.add_edge(
                prop2_id, prop1_id,
                edge_type=rel.relation,
                direction="bidirectional",
                reason=rel.reason
            )
        else:
            # 其他关系按 direction 字段添加，为每条边设置明确的 direction
            if rel.direction in ["1->2", "1<->2"]:
                self.graph.add_edge(
                    prop1_id, prop2_id,
                    edge_type=rel.relation,
                    direction="1->2",
                    reason=rel.reason
                )
            if rel.direction in ["2->1", "1<->2"]:
                self.graph.add_edge(
                    prop2_id, prop1_id,
                    edge_type=rel.relation,
                    direction="2->1",
                    reason=rel.reason
                )

    def get_graph(self) -> nx.DiGraph:
        """获取构建的图"""
        return self.graph

    def get_statistics(self) -> GraphStatistics:
        """获取图统计信息"""
        stats = GraphStatistics()

        stats.total_nodes = self.graph.number_of_nodes()
        stats.total_edges = self.graph.number_of_edges()

        for node_id, node_data in self.graph.nodes(data=True):
            node_type = node_data.get("node_type", "")
            if node_type == PROPOSITION_NODE:
                stats.proposition_nodes += 1
            elif node_type == ENTITY_NODE:
                stats.entity_nodes += 1

        for _, _, edge_data in self.graph.edges(data=True):
            edge_type = edge_data.get("edge_type", "")
            if edge_type == MENTIONS_ENTITY:
                stats.mention_edges += 1
            elif edge_type in PROPOSITION_EDGE_TYPES:
                stats.proposition_edges += 1

        # 计算孤立节点
        stats.isolated_nodes = sum(1 for node in self.graph.nodes() if self.graph.degree(node) == 0)

        return stats

    def save_graph(self, output_path: str, format: str = "both") -> None:
        """
        保存图到文件

        Args:
            output_path: 输出路径（不含扩展名）
            format: 保存格式 ("pkl", "json", "both")
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        if format in ["pkl", "both"]:
            pkl_path = output.with_suffix(".pkl")
            with open(pkl_path, 'wb') as f:
                pickle.dump(self.graph, f)
            print(f"图已保存 (PKL): {pkl_path}")

        if format in ["json", "both"]:
            json_path = output.with_suffix(".json")
            graph_data = nx.node_link_data(self.graph)

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, ensure_ascii=False, indent=2)
            print(f"图已保存 (JSON): {json_path}")

    def export_meta(self, output_dir: str) -> None:
        """
        导出 meta 目录产物

        Args:
            output_dir: 输出目录
        """
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        # 导出命题
        propositions_path = output / "propositions.jsonl"
        with open(propositions_path, 'w', encoding='utf-8') as f:
            for prop in self.propositions:
                f.write(json.dumps(asdict(prop), ensure_ascii=False) + "\n")
        print(f"命题已导出: {propositions_path}")

        # 导出实体（同文档内去重）
        entities_seen = set()
        entities_path = output / "entities.jsonl"
        dedup_count = 0
        with open(entities_path, 'w', encoding='utf-8') as f:
            for entity in self.entities:
                entity_key = (entity.doc_id, entity.text)
                if entity_key not in entities_seen:
                    entities_seen.add(entity_key)
                    f.write(json.dumps(asdict(entity), ensure_ascii=False) + "\n")
                else:
                    dedup_count += 1
        print(f"实体已导出: {entities_path}（去重后 {len(entities_seen)} 个，过滤重复 {dedup_count} 个）")

        # 导出 RST 关系
        if self.rst_relations:
            relations_path = output / "rst_relations.jsonl"
            with open(relations_path, 'w', encoding='utf-8') as f:
                for rel in self.rst_relations:
                    f.write(json.dumps(asdict(rel), ensure_ascii=False) + "\n")
            print(f"RST 关系已导出: {relations_path}")

        # 导出统计信息
        stats = self.get_statistics()
        stats_path = output / "statistics.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(stats), f, ensure_ascii=False, indent=2)
        print(f"统计信息已导出: {stats_path}")
