"""
验证去重功能的脚本

检查 fix_duplicate_entities.py 处理后的图是否正确：
1. 去重后是否还存在同文档同文本的重复实体
2. 边的连通性是否保持完整
3. 具体案例抽样验证
4. 对比原始图和修复后图的差异
"""

import pickle
import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple
import networkx as nx


def load_graph(pkl_path: str) -> nx.Graph:
    """加载图文件"""
    print(f"加载图文件: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def check_duplicate_entities(graph: nx.Graph) -> Dict:
    """
    检查图中是否存在同文档同文本的重复实体

    Returns:
        统计信息字典
    """
    print("\n" + "=" * 60)
    print("1. 检查重复实体")
    print("=" * 60)

    # 按 (doc_id, text) 分组实体节点
    entity_groups = defaultdict(list)
    for node_id, node_data in graph.nodes(data=True):
        if node_data.get('node_type') == 'entity':
            key = (node_data['doc_id'], node_data['text'])
            entity_groups[key].append(node_id)

    # 找出重复的
    duplicates = {k: v for k, v in entity_groups.items() if len(v) > 1}

    print(f"总实体节点数: {sum(len(v) for v in entity_groups.values())}")
    print(f"唯一实体数 (doc_id, text): {len(entity_groups)}")
    print(f"发现重复组数: {len(duplicates)}")

    if duplicates:
        print(f"\n⚠️  警告：发现 {len(duplicates)} 组重复实体！")
        # 显示前10个重复案例
        for i, ((doc_id, text), nodes) in enumerate(list(duplicates.items())[:10]):
            print(f"  [{i+1}] doc_id={doc_id}, text='{text}'")
            for node in nodes:
                print(f"      -> {node}")

        return {
            'has_duplicates': True,
            'duplicate_count': len(duplicates),
            'duplicate_groups': list(duplicates.items())[:20]  # 保存前20个案例
        }
    else:
        print("✓ 未发现重复实体，去重成功！")
        return {
            'has_duplicates': False,
            'duplicate_count': 0,
            'duplicate_groups': []
        }


def check_edge_connectivity(graph: nx.Graph) -> Dict:
    """
    检查边的连通性是否完整

    验证：
    - 所有命题节点到实体节点的边是否正常
    - 所有实体节点是否仍被命题节点引用
    - RST关系边是否保持
    """
    print("\n" + "=" * 60)
    print("2. 检查边连通性")
    print("=" * 60)

    stats = {
        'total_nodes': graph.number_of_nodes(),
        'total_edges': graph.number_of_edges(),
        'proposition_nodes': 0,
        'entity_nodes': 0,
        'prop_to_entity_edges': 0,
        'prop_to_prop_edges': 0,
        'isolated_propositions': 0,
        'isolated_entities': 0,
        'orphaned_entities': [],  # 没有命题指向的实体
    }

    # 统计节点和边类型
    entity_to_prop_count = defaultdict(int)
    prop_has_entity = set()

    for node_id, node_data in graph.nodes(data=True):
        if node_data.get('node_type') == 'proposition':
            stats['proposition_nodes'] += 1
            # 检查是否有出边到实体
            has_entity_edge = False
            for succ in graph.successors(node_id):
                succ_data = graph.nodes[succ]
                if succ_data.get('node_type') == 'entity':
                    stats['prop_to_entity_edges'] += 1
                    entity_to_prop_count[succ] += 1
                    has_entity_edge = True
                elif succ_data.get('node_type') == 'proposition':
                    stats['prop_to_prop_edges'] += 1

            if not has_entity_edge:
                stats['isolated_propositions'] += 1

        elif node_data.get('node_type') == 'entity':
            stats['entity_nodes'] += 1
            # 检查是否有命题指向这个实体
            has_incoming_prop = False
            for pred in graph.predecessors(node_id):
                pred_data = graph.nodes[pred]
                if pred_data.get('node_type') == 'proposition':
                    has_incoming_prop = True
                    break

            if not has_incoming_prop:
                stats['isolated_entities'] += 1
                stats['orphaned_entities'].append(node_id)

    # 输出统计
    print(f"总节点数: {stats['total_nodes']:,}")
    print(f"  - 命题节点: {stats['proposition_nodes']:,}")
    print(f"  - 实体节点: {stats['entity_nodes']:,}")
    print(f"总边数: {stats['total_edges']:,}")
    print(f"  - 命题->实体边: {stats['prop_to_entity_edges']:,}")
    print(f"  - 命题->命题边: {stats['prop_to_prop_edges']:,}")

    print(f"\n孤立命题数 (无实体边): {stats['isolated_propositions']}")
    print(f"孤立实体数 (无命题指向): {stats['isolated_entities']}")

    if stats['orphaned_entities']:
        print(f"\n前10个孤立实体:")
        for i, entity_id in enumerate(stats['orphaned_entities'][:10]):
            entity_data = graph.nodes[entity_id]
            print(f"  [{i+1}] {entity_id}")
            print(f"      doc_id={entity_data['doc_id']}, text='{entity_data['text']}'")

    # 检查每个实体平均被多少个命题引用
    if entity_to_prop_count:
        avg_refs = sum(entity_to_prop_count.values()) / len(entity_to_prop_count)
        print(f"\n每个实体平均被 {avg_refs:.2f} 个命题引用")

        # 找出引用最多的实体
        top_entities = sorted(entity_to_prop_count.items(),
                            key=lambda x: x[1], reverse=True)[:5]
        print(f"引用最多的5个实体:")
        for entity_id, count in top_entities:
            entity_data = graph.nodes[entity_id]
            print(f"  - {entity_data['text']}: {count} 次引用")

    return stats


def compare_graphs(original_graph: nx.Graph, fixed_graph: nx.Graph) -> Dict:
    """
    对比原始图和修复后图的差异
    """
    print("\n" + "=" * 60)
    print("3. 对比原始图和修复后图")
    print("=" * 60)

    original_nodes = set(original_graph.nodes())
    fixed_nodes = set(fixed_graph.nodes())

    removed_nodes = original_nodes - fixed_nodes
    added_nodes = fixed_nodes - original_nodes

    print(f"原始图节点数: {len(original_nodes):,}")
    print(f"修复后节点数: {len(fixed_nodes):,}")
    print(f"删除节点数: {len(removed_nodes):,}")
    print(f"新增节点数: {len(added_nodes):,}")

    # 分析删除的节点类型
    removed_entity_count = 0
    removed_prop_count = 0

    for node_id in removed_nodes:
        node_data = original_graph.nodes[node_id]
        if node_data.get('node_type') == 'entity':
            removed_entity_count += 1
        elif node_data.get('node_type') == 'proposition':
            removed_prop_count += 1

    print(f"\n删除的节点类型:")
    print(f"  - 实体节点: {removed_entity_count:,}")
    print(f"  - 命题节点: {removed_prop_count:,}")

    # 验证：删除的应该都是实体节点
    if removed_prop_count > 0:
        print(f"\n⚠️  警告：删除了 {removed_prop_count} 个命题节点！")
    else:
        print(f"✓ 正确：只删除了实体节点")

    return {
        'original_nodes': len(original_nodes),
        'fixed_nodes': len(fixed_nodes),
        'removed_nodes': len(removed_nodes),
        'removed_entity_count': removed_entity_count,
        'removed_prop_count': removed_prop_count
    }


def sample_check(graph: nx.Graph, sample_size: int = 10) -> List[Dict]:
    """
    抽样检查实体节点

    随机选择一些实体，检查它们的连接情况
    """
    print("\n" + "=" * 60)
    print("4. 抽样检查实体节点")
    print("=" * 60)

    entity_nodes = [n for n, d in graph.nodes(data=True)
                   if d.get('node_type') == 'entity']

    if len(entity_nodes) < sample_size:
        sample_size = len(entity_nodes)

    import random
    sampled = random.sample(entity_nodes, sample_size)

    results = []
    for i, entity_id in enumerate(sampled):
        entity_data = graph.nodes[entity_id]
        predecessors = list(graph.predecessors(entity_id))
        successors = list(graph.successors(entity_id))

        result = {
            'entity_id': entity_id,
            'doc_id': entity_data['doc_id'],
            'text': entity_data['text'],
            'entity_type': entity_data.get('entity_type', 'N/A'),
            'in_degree': len(predecessors),
            'out_degree': len(successors),
            'connected_propositions': [p for p in predecessors
                                      if graph.nodes[p].get('node_type') == 'proposition']
        }
        results.append(result)

        print(f"\n[{i+1}] 实体: {entity_data['text']}")
        print(f"    ID: {entity_id}")
        print(f"    文档: {entity_data['doc_id']}")
        print(f"    类型: {entity_data.get('entity_type', 'N/A')}")
        print(f"    入度: {len(predecessors)} (被多少节点引用)")
        print(f"    出度: {len(successors)} (引用多少节点)")
        print(f"    连接的命题数: {len(result['connected_propositions'])}")

    return results


def verify_dataset(dataset: str, output_dir: str = "/data/zhz/git/prograph/output") -> Dict:
    """
    验证单个数据集
    """
    print(f"\n\n{'#' * 70}")
    print(f"# 验证数据集: {dataset}")
    print(f"{'#' * 70}\n")

    base_path = Path(output_dir) / dataset / "proposition_graph"
    original_pkl = base_path / "raw_graph.pkl"
    fixed_pkl = base_path / "raw_graph_fixed.pkl"
    stats_json = base_path / f"fix_stats_{dataset}.json"

    # 检查文件是否存在
    if not fixed_pkl.exists():
        print(f"⚠️  跳过 {dataset}: 修复后的图文件不存在")
        return None

    # 加载图
    fixed_graph = load_graph(str(fixed_pkl))

    results = {
        'dataset': dataset,
        'checks': {}
    }

    # 1. 检查重复实体
    dup_check = check_duplicate_entities(fixed_graph)
    results['checks']['duplicate_entities'] = dup_check

    # 2. 检查边连通性
    edge_check = check_edge_connectivity(fixed_graph)
    results['checks']['edge_connectivity'] = edge_check

    # 3. 如果原始图存在，进行对比
    if original_pkl.exists():
        original_graph = load_graph(str(original_pkl))
        comp_check = compare_graphs(original_graph, fixed_graph)
        results['checks']['graph_comparison'] = comp_check

    # 4. 抽样检查
    sample_check_results = sample_check(fixed_graph, sample_size=5)
    results['checks']['sample_check'] = sample_check_results

    # 加载并保存统计信息
    if stats_json.exists():
        with open(stats_json, 'r') as f:
            results['fix_stats'] = json.load(f)

    return results


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='验证去重功能')
    parser.add_argument('--dataset', help='指定数据集 (HotpotQA, 2WikiMultihopQA, MuSiQue)')
    parser.add_argument('--output-dir', default='/data/zhz/git/prograph/output',
                       help='输出目录')
    parser.add_argument('--save-report', help='保存详细报告到指定文件')

    args = parser.parse_args()

    datasets = ['HotpotQA', '2WikiMultihopQA', 'MuSiQue']
    if args.dataset:
        datasets = [args.dataset]

    all_results = []

    for dataset in datasets:
        result = verify_dataset(dataset, args.output_dir)
        if result:
            all_results.append(result)

    # 生成总结报告
    print("\n\n" + "=" * 70)
    print("验证总结报告")
    print("=" * 70)

    for result in all_results:
        dataset = result['dataset']
        checks = result['checks']

        print(f"\n【{dataset}】")

        # 重复实体检查
        dup_check = checks['duplicate_entities']
        if dup_check['has_duplicates']:
            print(f"  ❌ 重复实体检查: 失败 (发现 {dup_check['duplicate_count']} 组重复)")
        else:
            print(f"  ✅ 重复实体检查: 通过")

        # 边连通性检查
        edge_check = checks['edge_connectivity']
        if edge_check['isolated_propositions'] > 100 or edge_check['isolated_entities'] > 100:
            print(f"  ⚠️  边连通性检查: 存在较多孤立节点")
        else:
            print(f"  ✅ 边连通性检查: 通过")

        # 图对比
        if 'graph_comparison' in checks:
            comp_check = checks['graph_comparison']
            if comp_check['removed_prop_count'] > 0:
                print(f"  ❌ 图对比检查: 删除了命题节点")
            else:
                print(f"  ✅ 图对比检查: 通过 (删除 {comp_check['removed_entity_count']} 个重复实体)")

    # 保存报告
    if args.save_report:
        with open(args.save_report, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n详细报告已保存: {args.save_report}")


if __name__ == "__main__":
    main()
