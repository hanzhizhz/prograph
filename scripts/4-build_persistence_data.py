#!/usr/bin/env python3
"""
ProGraph 数据持久化脚本
构建向量索引、节点映射、元数据缓存、邻接缓存、全局统计等，优化在线推理性能
"""

import asyncio
import pickle
import json
import sys
from pathlib import Path
from collections import defaultdict
import argparse
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm import VLLMEmbeddingClient
from src.config.model_config import ModelConfig
from src.config import set_model_config
from src.retrieval.vector_index import PersistentHNSWIndex


def build_adjacency_cache(graph):
    """
    构建邻接表缓存和前驱节点缓存

    预构建每个节点的邻居映射，避免重复图遍历
    缓存结构：{node_id: {node_type: [{"id": neighbor_id, "edge_type": edge_type}]}}
    前驱缓存：{node_id: [predecessor_ids]}
    """
    PROPOSITION_NODE = "proposition"
    ENTITY_NODE = "entity"
    GLOBAL_ENTITY_NODE = "global_entity"

    adjacency_cache = {}
    predecessors_cache = {}

    for node_id in graph.nodes():
        neighbors_by_type = {
            PROPOSITION_NODE: [],
            ENTITY_NODE: [],
            GLOBAL_ENTITY_NODE: []
        }

        for neighbor in graph.neighbors(node_id):
            if graph.has_edge(node_id, neighbor):
                edge_data = graph[node_id][neighbor]
                edge_type = edge_data.get("edge_type", "")
                neighbor_data = graph.nodes[neighbor]
                neighbor_type = neighbor_data.get("node_type", "")

                neighbor_info = {
                    "id": neighbor,
                    "edge_type": edge_type
                }

                if neighbor_type in neighbors_by_type:
                    neighbors_by_type[neighbor_type].append(neighbor_info)

        adjacency_cache[node_id] = neighbors_by_type

        # 构建前驱节点缓存
        predecessors_cache[node_id] = list(graph.predecessors(node_id))

    return adjacency_cache, predecessors_cache


def build_global_statistics(graph, adjacency_cache):
    """
    计算全局实体统计

    统计所有命题节点的实体连接数，计算中位数和平均值
    用于桥接分数的归一化
    """
    PROPOSITION_NODE = "proposition"
    entity_counts = []

    for node_id, node_data in graph.nodes(data=True):
        if node_data.get("node_type") != PROPOSITION_NODE:
            continue

        neighbors_by_type = adjacency_cache.get(node_id, {})
        entity_count = len(neighbors_by_type.get("entity", [])) + len(neighbors_by_type.get("global_entity", []))

        if entity_count > 0:
            entity_counts.append(entity_count)

    if not entity_counts:
        return {"norm": 1.0, "median": 0.0, "mean": 0.0, "count": 0}

    median = float(np.median(entity_counts))
    mean = float(np.mean(entity_counts))

    return {
        "norm": max(median, mean),
        "median": median,
        "mean": mean,
        "count": len(entity_counts)
    }


def build_entity_lookup_index(graph):
    """
    构建实体查找索引

    建立实体名称到节点ID的映射，支持O(1)查找
    """
    ENTITY_NODE = "entity"
    GLOBAL_ENTITY_NODE = "global_entity"

    name_to_global_entities = {}
    name_to_entities = {}

    for node_id, node_data in graph.nodes(data=True):
        node_type = node_data.get("node_type")
        if node_type not in [ENTITY_NODE, GLOBAL_ENTITY_NODE]:
            continue

        text = node_data.get("text", "").lower()
        if not text:
            continue

        if node_type == GLOBAL_ENTITY_NODE:
            if text not in name_to_global_entities:
                name_to_global_entities[text] = []
            name_to_global_entities[text].append(node_id)
        else:
            if text not in name_to_entities:
                name_to_entities[text] = []
            name_to_entities[text].append(node_id)

    # 按 member_count 排序 global_entities
    for name in name_to_global_entities:
        name_to_global_entities[name].sort(
            key=lambda nid: graph.nodes[nid].get("member_count", 0),
            reverse=True
        )

    total_global = sum(len(v) for v in name_to_global_entities.values())
    total_entity = sum(len(v) for v in name_to_entities.values())

    return {
        "global_entities": name_to_global_entities,
        "entities": name_to_entities,
        "stats": {
            "unique_global_names": len(name_to_global_entities),
            "unique_entity_names": len(name_to_entities),
            "total_global_nodes": total_global,
            "total_entity_nodes": total_entity,
            "total_entities": total_global + total_entity
        }
    }


async def build_persistence_data(
    graph_path: str,
    output_dir: str,
    config_path: str = "config.yaml"
):
    """
    构建完整的持久化数据

    包含：
    1. HNSW 向量索引（命题、实体）
    2. 节点关系映射
    3. 文档元数据
    4. 实体别名映射
    5. 邻接缓存（优化图遍历）
    6. 全局实体统计（优化桥接评分）
    7. 实体查找索引（优化实体链接）
    """
    print("=" * 60)
    print("ProGraph 数据持久化构建")
    print("=" * 60)
    print(f"输入图: {graph_path}")
    print(f"输出目录: {output_dir}")
    print()

    # 1. 加载配置
    print("加载配置...")
    model_config = ModelConfig.from_yaml(config_path)
    set_model_config(model_config)

    # 2. 加载图
    print("\n加载图...")
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)

    print(f"图已加载: {graph.number_of_nodes()} 个节点, {graph.number_of_edges()} 条边")

    # 3. 初始化嵌入客户端（离线 vLLM）
    print("\n初始化嵌入客户端（vLLM 离线）...")
    embedding_client = VLLMEmbeddingClient(
        model_path=model_config.vllm_embedding.model_path,
        tensor_parallel_size=model_config.vllm_embedding.tensor_parallel_size,
        gpu_memory_utilization=model_config.vllm_embedding.gpu_memory_utilization,
        trust_remote_code=model_config.vllm_embedding.trust_remote_code,
        max_model_len=model_config.vllm_embedding.max_model_len,
    )

    # 获取向量维度
    print("获取向量维度...")
    test_emb = await embedding_client.embed_single("test")
    dim = len(test_emb)
    print(f"向量维度: {dim}")

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    index_dir = output_path / "indices"
    index_dir.mkdir(exist_ok=True)

    # ========== 1. 构建命题向量索引 ==========
    print("\n" + "=" * 60)
    print("[1/5] 构建命题向量索引...")
    print("=" * 60)
    prop_texts = []
    prop_payloads = []
    prop_to_doc = {}
    doc_to_props = defaultdict(list)

    for node_id, data in graph.nodes(data=True):
        if data.get("node_type") == "proposition":
            text = data.get("text", "")
            if text:
                prop_texts.append(text)
                doc_id = data.get("doc_id", "")
                prop_payloads.append({
                    "node_id": node_id,
                    "node_type": "proposition",
                    "doc_id": doc_id,
                    "sent_idx": data.get("sent_idx", 0),
                    "text": text
                })
                prop_to_doc[node_id] = doc_id
                doc_to_props[doc_id].append(node_id)

    if prop_texts:
        print(f"  生成 {len(prop_texts)} 个命题嵌入...")
        prop_response = await embedding_client.embed(prop_texts)
        prop_embeddings = prop_response.embeddings

        prop_index = PersistentHNSWIndex(dim, max_elements=len(prop_texts) * 2)
        prop_index.add(np.array(prop_embeddings, dtype=np.float32), prop_payloads)
        prop_index.save(str(index_dir / "proposition"))
        print(f"  命题索引已保存: {len(prop_texts)} 个向量")

    # ========== 2. 构建实体向量索引 ==========
    print("\n" + "=" * 60)
    print("[2/5] 构建实体向量索引...")
    print("=" * 60)
    ent_texts = []
    ent_payloads = []
    ent_to_docs = defaultdict(set)
    doc_to_ents = defaultdict(set)

    for node_id, data in graph.nodes(data=True):
        if data.get("node_type") in ["entity", "global_entity"]:
            text = data.get("text", "")
            if text:
                ent_texts.append(text)
                doc_id = data.get("doc_id", "")
                ent_payloads.append({
                    "node_id": node_id,
                    "node_type": data.get("node_type"),
                    "doc_id": doc_id,
                    "entity_type": data.get("entity_type"),
                    "text": text
                })
                ent_to_docs[node_id].add(doc_id)
                doc_to_ents[doc_id].add(node_id)

    if ent_texts:
        print(f"  生成 {len(ent_texts)} 个实体嵌入...")
        ent_response = await embedding_client.embed(ent_texts)
        ent_embeddings = ent_response.embeddings

        ent_index = PersistentHNSWIndex(dim, max_elements=len(ent_texts) * 2)
        ent_index.add(np.array(ent_embeddings, dtype=np.float32), ent_payloads)
        ent_index.save(str(index_dir / "entity"))
        print(f"  实体索引已保存: {len(ent_texts)} 个向量")

    # ========== 3. 构建节点关系映射 ==========
    print("\n" + "=" * 60)
    print("[3/5] 构建节点关系映射...")
    print("=" * 60)
    node_mappings = {
        "proposition_to_doc": prop_to_doc,
        "doc_to_propositions": {k: v if isinstance(v, list) else list(v) for k, v in doc_to_props.items()},
        "entity_to_docs": {k: list(v) for k, v in ent_to_docs.items()},
        "doc_to_entities": {k: list(v) for k, v in doc_to_ents.items()},
    }

    mapping_file = output_path / "node_mappings.json"
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(node_mappings, f, ensure_ascii=False, indent=2)
    print(f"  节点映射已保存: {mapping_file}")
    print(f"    - proposition_to_doc: {len(prop_to_doc)} 条映射")
    print(f"    - doc_to_propositions: {len(doc_to_props)} 个文档")
    print(f"    - entity_to_docs: {len(ent_to_docs)} 个实体")
    print(f"    - doc_to_entities: {len(doc_to_ents)} 个文档")

    # ========== 4. 构建文档元数据 ==========
    print("\n" + "=" * 60)
    print("[4/5] 构建文档元数据...")
    print("=" * 60)
    doc_metadata = {}
    doc_titles = {}  # 收集文档标题

    # 首先收集所有文档标题
    for node_id, data in graph.nodes(data=True):
        if data.get("node_type") == "proposition":
            doc_id = data.get("doc_id", "")
            if doc_id:
                # 尝试从节点获取标题
                title = data.get("doc_title", "")
                if title and doc_id not in doc_titles:
                    doc_titles[doc_id] = title

    # 构建文档元数据
    for doc_id in set(list(doc_to_props.keys()) + list(doc_to_ents.keys())):
        doc_metadata[doc_id] = {
            "doc_id": doc_id,
            "title": doc_titles.get(doc_id, doc_id),  # 使用标题或 doc_id
            "proposition_count": len(doc_to_props.get(doc_id, [])),
            "entity_count": len(doc_to_ents.get(doc_id, set())),
        }

    # 保存文档元数据
    meta_file = output_path / "doc_metadata.json"
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(doc_metadata, f, ensure_ascii=False, indent=2)
    print(f"  文档元数据已保存: {len(doc_metadata)} 个文档")

    # ========== 5. 构建实体别名映射 ==========
    print("\n" + "=" * 60)
    print("[5/5] 构建实体别名映射...")
    print("=" * 60)
    entity_aliases = {}

    # 首先构建实体ID到节点ID的映射
    entity_name_to_node_id = {}

    for node_id, data in graph.nodes(data=True):
        if data.get("node_type") in ["entity", "global_entity"]:
            text = data.get("text", "")
            if text and text not in entity_name_to_node_id:
                # 优先使用 global_entity
                if data.get("node_type") == "global_entity":
                    entity_name_to_node_id[text] = node_id
                elif text not in entity_name_to_node_id:
                    entity_name_to_node_id[text] = node_id

    # 为每个实体生成别名
    for node_id, data in graph.nodes(data=True):
        if data.get("node_type") in ["entity", "global_entity"]:
            text = data.get("text", "")
            if text:
                # 主名称映射到节点ID
                if text not in entity_aliases:
                    entity_aliases[text] = node_id

                # 生成别名：姓氏/名字
                if " " in text:
                    parts = text.split()
                    if len(parts) > 1:
                        # 使用最后一个词作为别名（通常是姓氏）
                        surname = parts[-1]
                        if surname and surname not in entity_aliases:
                            entity_aliases[surname] = node_id
                        # 使用第一个词作为别名
                        firstname = parts[0]
                        if firstname and firstname not in entity_aliases:
                            entity_aliases[firstname] = node_id

    alias_file = output_path / "entity_aliases.json"
    with open(alias_file, 'w', encoding='utf-8') as f:
        json.dump(entity_aliases, f, ensure_ascii=False, indent=2)
    print(f"  实体别名已保存: {len(entity_aliases)} 个映射")

    # ========== 6. 构建邻接缓存 ==========
    print("\n" + "=" * 60)
    print("[6/9] 构建邻接缓存...")
    print("=" * 60)
    adjacency_cache, predecessors_cache = build_adjacency_cache(graph)

    cache_file = output_path / "adjacency_cache.pkl"
    with open(cache_file, 'wb') as f:
        pickle.dump(adjacency_cache, f)
    print(f"  邻接缓存已保存: {len(adjacency_cache)} 个节点")

    pred_cache_file = output_path / "predecessors_cache.pkl"
    with open(pred_cache_file, 'wb') as f:
        pickle.dump(predecessors_cache, f)
    print(f"  前驱缓存已保存: {len(predecessors_cache)} 个节点")

    # ========== 7. 构建全局实体统计 ==========
    print("\n" + "=" * 60)
    print("[7/9] 构建全局实体统计...")
    print("=" * 60)
    global_stats = build_global_statistics(graph, adjacency_cache)

    stats_file = output_path / "global_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(global_stats, f, ensure_ascii=False, indent=2)
    print(f"  全局统计已保存: norm={global_stats['norm']:.2f}")

    # ========== 8. 构建实体查找索引 ==========
    print("\n" + "=" * 60)
    print("[8/9] 构建实体查找索引...")
    print("=" * 60)
    entity_lookup = build_entity_lookup_index(graph)

    lookup_file = output_path / "entity_lookup.json"
    with open(lookup_file, 'w', encoding='utf-8') as f:
        json.dump(entity_lookup, f, ensure_ascii=False, indent=2)
    print(f"  实体查找索引已保存: {entity_lookup['stats']['total_entities']} 个实体")

    # ========== 9. 构建统计信息 ==========
    print("\n" + "=" * 60)
    print("[9/9] 构建统计信息...")
    print("=" * 60)
    stats = {
        "total_nodes": graph.number_of_nodes(),
        "total_edges": graph.number_of_edges(),
        "proposition_count": len(prop_to_doc),
        "entity_count": len(ent_to_docs),
        "document_count": len(doc_metadata),
        "vector_dim": dim,
        "proposition_index_size": len(prop_texts),
        "entity_index_size": len(ent_texts),
    }

    stats_file = output_path / "index_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # 输出统计信息
    print("\n" + "=" * 60)
    print("数据持久化构建完成")
    print("=" * 60)
    print(f"输出目录: {output_dir}")
    print(f"  向量索引: indices/")
    print(f"    - proposition.bin/.meta.json ({len(prop_texts)} 向量)")
    print(f"    - entity.bin/.meta.json ({len(ent_texts)} 向量)")
    print(f"  节点映射: node_mappings.json")
    print(f"  文档元数据: doc_metadata.json ({len(doc_metadata)} 文档)")
    print(f"  实体别名: entity_aliases.json ({len(entity_aliases)} 映射)")
    print(f"  邻接缓存: adjacency_cache.pkl ({len(adjacency_cache)} 节点)")
    print(f"  全局统计: global_stats.json (norm={global_stats['norm']:.2f})")
    print(f"  实体查找: entity_lookup.json ({entity_lookup['stats']['total_entities']} 实体)")
    print(f"  统计信息: index_stats.json")
    print("=" * 60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="ProGraph 数据持久化构建脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 为 HotpotQA 数据集构建持久化数据
  python scripts/4-build_persistence_data.py \\
    --graph output/HotpotQA/proposition_graph/linked_graph.pkl \\
    --output output/HotpotQA/persistence_data \\
    --config config.yaml

  # 使用默认路径
  python scripts/4-build_persistence_data.py --dataset HotpotQA
        """
    )

    parser.add_argument(
        '--dataset',
        type=str,
        help='数据集名称 (如 HotpotQA, 2WikiMultihopQA)'
    )

    parser.add_argument(
        '--graph',
        type=str,
        help='输入图文件路径 (.pkl)'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='输出目录'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='配置文件路径 (默认: config.yaml)'
    )

    args = parser.parse_args()

    # 如果使用 --dataset，设置默认路径
    if args.dataset:
        graph_path = args.graph or f"output/{args.dataset}/proposition_graph/linked_graph.pkl"
        output_dir = args.output or f"output/{args.dataset}/persistence_data"
    else:
        if not args.graph or not args.output:
            parser.error("--dataset 或 (--graph + --output) 必须提供")
        graph_path = args.graph
        output_dir = args.output

    # 运行
    asyncio.run(build_persistence_data(graph_path, output_dir, args.config))


if __name__ == "__main__":
    main()
