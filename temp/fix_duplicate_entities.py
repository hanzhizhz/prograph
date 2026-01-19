"""
后处理脚本：修复图中的同文档重复实体

对已生成的 raw_graph.pkl 进行去重：
- 按 (doc_id, text) 分组实体节点
- 保留每组ID最短的节点
- 重定向边并删除重复节点
"""

import pickle
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import networkx as nx


def fix_duplicate_entities(input_pkl: str, output_pkl: str) -> Dict:
    """
    修复图中的同文档重复实体

    Args:
        input_pkl: 输入图文件路径
        output_pkl: 输出图文件路径

    Returns:
        统计信息字典
    """
    print(f"加载图文件: {input_pkl}")
    with open(input_pkl, 'rb') as f:
        graph = pickle.load(f)

    original_nodes = graph.number_of_nodes()
    original_edges = graph.number_of_edges()

    # 找出所有实体节点并按 (doc_id, text) 分组
    print("分析实体节点...")
    entity_groups = defaultdict(list)
    for node_id, node_data in graph.nodes(data=True):
        if node_data.get('node_type') == 'entity':
            key = (node_data['doc_id'], node_data['text'])
            entity_groups[key].append(node_id)

    # 统计重复情况
    duplicates = {k: v for k, v in entity_groups.items() if len(v) > 1}
    duplicate_count = sum(len(v) - 1 for v in duplicates.values())
    print(f"发现 {len(duplicates)} 个重复实体组合，涉及 {sum(len(v) for v in duplicates.values())} 个节点")
    print(f"将删除 {duplicate_count} 个重复节点")

    # 对每组：保留ID最短的节点
    nodes_to_redirect: List[Tuple[str, str]] = []
    for (doc_id, text), nodes in duplicates.items():
        # 保留ID最短的（通常去掉后缀）
        keep_node = min(nodes, key=len)
        for node in nodes:
            if node != keep_node:
                nodes_to_redirect.append((node, keep_node))

    # 重定向边并删除节点
    print(f"重定向边并删除 {len(nodes_to_redirect)} 个重复节点...")
    removed = 0
    for old_node, new_node in nodes_to_redirect:
        # 将指向 old_node 的边重定向到 new_node
        for pred in list(graph.predecessors(old_node)):
            edge_data = graph.edges[pred, old_node]
            # 避免重复添加已存在的边
            if not graph.has_edge(pred, new_node):
                graph.add_edge(pred, new_node, **edge_data)
            graph.remove_edge(pred, old_node)

        for succ in list(graph.successors(old_node)):
            edge_data = graph.edges[old_node, succ]
            if not graph.has_edge(new_node, succ):
                graph.add_edge(new_node, succ, **edge_data)
            graph.remove_edge(old_node, succ)

        graph.remove_node(old_node)
        removed += 1

        if removed % 10000 == 0:
            print(f"  已处理 {removed}/{len(nodes_to_redirect)} 个节点...")

    # 保存修复后的图
    print(f"保存修复后的图: {output_pkl}")
    output_path = Path(output_pkl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_pkl, 'wb') as f:
        pickle.dump(graph, f)

    # 统计信息
    stats = {
        'original_nodes': original_nodes,
        'original_edges': original_edges,
        'fixed_nodes': graph.number_of_nodes(),
        'fixed_edges': graph.number_of_edges(),
        'nodes_removed': removed,
        'nodes_kept': original_nodes - removed,
    }

    print("\n" + "=" * 50)
    print("修复完成！")
    print("=" * 50)
    print(f"原始节点数: {stats['original_nodes']:,}")
    print(f"修复后节点数: {stats['fixed_nodes']:,}")
    print(f"删除节点数: {stats['nodes_removed']:,} ({stats['nodes_removed']/stats['original_nodes']*100:.1f}%)")
    print(f"原始边数: {stats['original_edges']:,}")
    print(f"修复后边数: {stats['fixed_edges']:,}")
    print("=" * 50)

    return stats


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description='修复图中的同文档重复实体')
    parser.add_argument('input', help='输入图文件路径 (.pkl)')
    parser.add_argument('output', help='输出图文件路径 (.pkl)')
    parser.add_argument('--dataset', help='数据集名称 (可选，用于统计)')

    args = parser.parse_args()

    stats = fix_duplicate_entities(args.input, args.output)

    # 可选：保存统计信息到 JSON
    if args.dataset:
        stats_path = Path(args.output).parent / f"fix_stats_{args.dataset}.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"统计信息已保存: {stats_path}")


if __name__ == "__main__":
    main()
