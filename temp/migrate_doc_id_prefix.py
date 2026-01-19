#!/usr/bin/env python3
"""
文档ID前缀迁移脚本

将已有数据从旧的文档ID格式（所有数据集都使用 hotpot_ 前缀）迁移到新格式：
- HotpotQA: 保持 hotpot_（无需修改）
- 2WikiMultihopQA: hotpot_ -> 2wiki_
- MuSiQue: hotpot_ -> musique_

直接覆盖替换现有数据文件。
"""

import pickle
import json
import sys
import re
import shutil
from pathlib import Path
from typing import Dict, Set, Tuple, Optional
import argparse
import networkx as nx
from collections import defaultdict

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def detect_dataset_type(path: str) -> str:
    """
    从路径检测数据集类型
    
    Args:
        path: 文件或目录路径
        
    Returns:
        数据集类型: 'hotpotqa', '2wiki', 'musique' 或 'unknown'
    """
    path_lower = str(path).lower()
    
    if 'hotpotqa' in path_lower or 'hotpot' in path_lower:
        return 'hotpotqa'
    elif '2wikimultihopqa' in path_lower or '2wiki' in path_lower:
        return '2wiki'
    elif 'musique' in path_lower:
        return 'musique'
    else:
        return 'unknown'


def get_new_prefix(dataset_type: str) -> Optional[str]:
    """
    获取新前缀
    
    Args:
        dataset_type: 数据集类型
        
    Returns:
        新前缀，如果不需要修改则返回 None
    """
    prefix_mapping = {
        'hotpotqa': None,  # 保持不变
        '2wiki': '2wiki',
        'musique': 'musique',
    }
    return prefix_mapping.get(dataset_type)


def migrate_doc_id_in_string(old_doc_id: str, old_prefix: str, new_prefix: str) -> str:
    """
    在字符串中迁移文档ID
    
    Args:
        old_doc_id: 旧的文档ID（如 "hotpot_42"）
        old_prefix: 旧前缀（"hotpot"）
        new_prefix: 新前缀（"2wiki" 或 "musique"）
        
    Returns:
        新的文档ID（如 "2wiki_42"）
    """
    # 匹配格式：hotpot_{数字}
    pattern = rf"^{re.escape(old_prefix)}_(\d+)$"
    match = re.match(pattern, old_doc_id)
    if match:
        idx = match.group(1)
        return f"{new_prefix}_{idx}"
    return old_doc_id


def migrate_node_id(old_node_id: str, old_doc_id: str, new_doc_id: str) -> str:
    """
    迁移节点ID
    
    节点ID格式：{doc_id}_prop_{idx} 或 {doc_id}_ent_{text}
    
    Args:
        old_node_id: 旧的节点ID（如 "hotpot_0_prop_0"）
        old_doc_id: 旧的文档ID（如 "hotpot_0"）
        new_doc_id: 新的文档ID（如 "musique_0"）
        
    Returns:
        新的节点ID（如 "musique_0_prop_0"）
    """
    # 如果节点ID以旧文档ID开头，替换为新文档ID
    if old_node_id.startswith(old_doc_id + '_'):
        return old_node_id.replace(old_doc_id + '_', new_doc_id + '_', 1)
    return old_node_id


def migrate_graph(graph_path: Path, dataset_type: str, old_prefix: str, new_prefix: str, dry_run: bool = False) -> Dict:
    """
    迁移图文件（.pkl 或 .json）
    
    Args:
        graph_path: 图文件路径
        dataset_type: 数据集类型
        old_prefix: 旧前缀
        new_prefix: 新前缀
        dry_run: 是否只检查不修改
        
    Returns:
        迁移统计信息
    """
    print(f"\n正在处理图文件: {graph_path}")
    
    stats = {
        'nodes_migrated': 0,
        'nodes_total': 0,
        'edges_migrated': 0,
        'edges_total': 0,
    }
    
    # 加载图
    if graph_path.suffix == '.pkl':
        with open(graph_path, 'rb') as f:
            graph = pickle.load(f)
    elif graph_path.suffix == '.json':
        with open(graph_path, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
            # NetworkX JSON 格式：使用 'edges' 键（新版本使用 edges，旧版本使用 links）
            try:
                # 新版本使用 'edges' 键
                graph = nx.node_link_graph(graph_data, edges='edges', directed=True, multigraph=False)
            except (KeyError, TypeError):
                # 旧版本可能使用 'links' 键
                try:
                    graph = nx.node_link_graph(graph_data, edges='links', directed=True, multigraph=False)
                except Exception as e:
                    print(f"  错误：无法加载 JSON 图文件: {e}")
                    return stats
    else:
        print(f"  跳过：不支持的文件格式 {graph_path.suffix}")
        return stats
    
    stats['nodes_total'] = graph.number_of_nodes()
    stats['edges_total'] = graph.number_of_edges()
    
    # 收集需要迁移的节点和映射关系
    old_to_new_doc_ids: Dict[str, str] = {}
    old_to_new_node_ids: Dict[str, str] = {}
    
    # 第一步：收集所有需要迁移的 doc_id 和节点ID映射
    for old_node_id, node_data in graph.nodes(data=True):
        old_doc_id = node_data.get('doc_id', '')
        
        # 检查是否是旧格式的文档ID
        if old_doc_id and old_doc_id.startswith(old_prefix + '_'):
            # 生成新的文档ID
            if old_doc_id not in old_to_new_doc_ids:
                new_doc_id = migrate_doc_id_in_string(old_doc_id, old_prefix, new_prefix)
                old_to_new_doc_ids[old_doc_id] = new_doc_id
            
            # 生成新的节点ID
            new_doc_id = old_to_new_doc_ids[old_doc_id]
            new_node_id = migrate_node_id(old_node_id, old_doc_id, new_doc_id)
            
            if old_node_id != new_node_id:
                old_to_new_node_ids[old_node_id] = new_node_id
                stats['nodes_migrated'] += 1
    
    print(f"  找到 {len(old_to_new_doc_ids)} 个文档ID需要迁移")
    print(f"  找到 {len(old_to_new_node_ids)} 个节点ID需要迁移")
    
    if dry_run:
        print(f"  [DRY RUN] 将迁移 {stats['nodes_migrated']} 个节点")
        return stats
    
    if not old_to_new_node_ids:
        print(f"  无需迁移")
        return stats
    
    # 第二步：创建新图并迁移节点
    new_graph = nx.DiGraph()
    
    # 迁移节点
    for old_node_id, node_data in graph.nodes(data=True):
        old_doc_id = node_data.get('doc_id', '')
        
        # 确定新的节点ID
        if old_node_id in old_to_new_node_ids:
            new_node_id = old_to_new_node_ids[old_node_id]
            new_doc_id = old_to_new_doc_ids[old_doc_id]
        else:
            new_node_id = old_node_id
            new_doc_id = old_doc_id
        
        # 复制节点数据并更新 doc_id
        new_node_data = dict(node_data)
        if new_doc_id:
            new_node_data['doc_id'] = new_doc_id
        
        new_graph.add_node(new_node_id, **new_node_data)
    
    # 第三步：迁移边
    for old_u, old_v, edge_data in graph.edges(data=True):
        new_u = old_to_new_node_ids.get(old_u, old_u)
        new_v = old_to_new_node_ids.get(old_v, old_v)
        
        if new_u != old_u or new_v != old_v:
            stats['edges_migrated'] += 1
        
        new_graph.add_edge(new_u, new_v, **edge_data)
    
    # 保存新图
    if graph_path.suffix == '.pkl':
        with open(graph_path, 'wb') as f:
            pickle.dump(new_graph, f)
    elif graph_path.suffix == '.json':
        # 使用与原始文件相同的边键名（edges 或 links）
        graph_data = nx.node_link_data(new_graph)
        with open(graph_path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)
    
    print(f"  ✓ 已迁移 {stats['nodes_migrated']} 个节点, {stats['edges_migrated']} 条边")
    return stats


def migrate_jsonl(jsonl_path: Path, old_prefix: str, new_prefix: str, field_name: str = 'doc_id', dry_run: bool = False) -> Dict:
    """
    迁移 JSONL 文件
    
    Args:
        jsonl_path: JSONL 文件路径
        old_prefix: 旧前缀
        new_prefix: 新前缀
        field_name: 需要迁移的字段名（默认为 'doc_id'）
        dry_run: 是否只检查不修改
        
    Returns:
        迁移统计信息
    """
    print(f"\n正在处理 JSONL 文件: {jsonl_path}")
    
    if not jsonl_path.exists():
        print(f"  跳过：文件不存在")
        return {'lines_total': 0, 'lines_migrated': 0}
    
    stats = {
        'lines_total': 0,
        'lines_migrated': 0,
    }
    
    # 读取所有行
    lines = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                lines.append(line)
                stats['lines_total'] += 1
    
    # 迁移每行
    migrated_lines = []
    for line in lines:
        data = json.loads(line)
        
        old_value = data.get(field_name, '')
        if old_value and isinstance(old_value, str) and old_value.startswith(old_prefix + '_'):
            new_value = migrate_doc_id_in_string(old_value, old_prefix, new_prefix)
            data[field_name] = new_value
            stats['lines_migrated'] += 1
            migrated_lines.append(json.dumps(data, ensure_ascii=False) + '\n')
        else:
            migrated_lines.append(line)
    
    print(f"  找到 {stats['lines_total']} 行，需要迁移 {stats['lines_migrated']} 行")
    
    if dry_run:
        print(f"  [DRY RUN] 将迁移 {stats['lines_migrated']} 行")
        return stats
    
    if stats['lines_migrated'] == 0:
        print(f"  无需迁移")
        return stats
    
    # 写回文件
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        f.writelines(migrated_lines)
    
    print(f"  ✓ 已迁移 {stats['lines_migrated']} 行")
    return stats


def migrate_dataset(dataset_dir: Path, dry_run: bool = False) -> Dict:
    """
    迁移单个数据集的所有文件
    
    Args:
        dataset_dir: 数据集目录（如 output/MuSiQue/proposition_graph/）
        dry_run: 是否只检查不修改
        
    Returns:
        迁移统计信息
    """
    print("\n" + "=" * 60)
    print(f"迁移数据集: {dataset_dir}")
    print("=" * 60)
    
    # 检测数据集类型
    dataset_type = detect_dataset_type(str(dataset_dir))
    print(f"检测到数据集类型: {dataset_type}")
    
    if dataset_type == 'unknown':
        print(f"警告：无法识别数据集类型，跳过")
        return {}
    
    # 获取新前缀
    new_prefix = get_new_prefix(dataset_type)
    if new_prefix is None:
        print(f"数据集 {dataset_type} 无需迁移（保持 hotpot_ 前缀）")
        return {}
    
    old_prefix = 'hotpot'
    print(f"迁移前缀: {old_prefix}_ -> {new_prefix}_")
    
    stats = {
        'files_processed': 0,
        'files_migrated': 0,
        'graphs': {},
        'jsonl_files': {},
    }
    
    # 需要处理的图文件
    graph_files = [
        dataset_dir / 'raw_graph.pkl',
        dataset_dir / 'raw_graph.json',
        dataset_dir / 'linked_graph.pkl',
        dataset_dir / 'linked_graph.json',
    ]
    
    # 迁移图文件
    for graph_file in graph_files:
        if graph_file.exists():
            stats['files_processed'] += 1
            try:
                graph_stats = migrate_graph(graph_file, dataset_type, old_prefix, new_prefix, dry_run)
                if graph_stats.get('nodes_migrated', 0) > 0:
                    stats['files_migrated'] += 1
                stats['graphs'][graph_file.name] = graph_stats
            except Exception as e:
                print(f"  错误：处理 {graph_file.name} 时出错: {e}")
    
    # 需要处理的 JSONL 文件
    jsonl_files = [
        (dataset_dir / 'propositions.jsonl', 'doc_id'),
        (dataset_dir / 'entities.jsonl', 'doc_id'),
    ]
    
    # 迁移 JSONL 文件
    for jsonl_file, field_name in jsonl_files:
        if jsonl_file.exists():
            stats['files_processed'] += 1
            try:
                jsonl_stats = migrate_jsonl(jsonl_file, old_prefix, new_prefix, field_name, dry_run)
                if jsonl_stats.get('lines_migrated', 0) > 0:
                    stats['files_migrated'] += 1
                stats['jsonl_files'][jsonl_file.name] = jsonl_stats
            except Exception as e:
                print(f"  错误：处理 {jsonl_file.name} 时出错: {e}")
    
    print(f"\n处理完成: {stats['files_processed']} 个文件, {stats['files_migrated']} 个文件需要迁移")
    return stats


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="文档ID前缀迁移脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 迁移所有数据集（干运行模式，只检查不修改）
  python temp/migrate_doc_id_prefix.py --all --dry-run

  # 迁移所有数据集（实际执行）
  python temp/migrate_doc_id_prefix.py --all

  # 迁移单个数据集
  python temp/migrate_doc_id_prefix.py --dataset output/MuSiQue/proposition_graph

  # 迁移指定数据集的输出目录
  python temp/migrate_doc_id_prefix.py --dataset-name MuSiQue
        """
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='迁移所有数据集'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        help='单个数据集目录路径（如 output/MuSiQue/proposition_graph）'
    )
    
    parser.add_argument(
        '--dataset-name',
        type=str,
        choices=['HotpotQA', '2WikiMultihopQA', 'MuSiQue'],
        help='数据集名称（自动构建路径）'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='输出目录（默认: output）'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='干运行模式：只检查需要迁移的内容，不实际修改文件'
    )
    
    args = parser.parse_args()
    
    # 确定需要处理的数据集
    datasets_to_process = []
    
    if args.all:
        # 处理所有数据集
        output_dir = Path(args.output_dir)
        for dataset_name in ['HotpotQA', '2WikiMultihopQA', 'MuSiQue']:
            dataset_path = output_dir / dataset_name / 'proposition_graph'
            if dataset_path.exists():
                datasets_to_process.append(dataset_path)
    elif args.dataset:
        # 处理指定的数据集目录
        datasets_to_process.append(Path(args.dataset))
    elif args.dataset_name:
        # 根据数据集名称构建路径
        dataset_path = Path(args.output_dir) / args.dataset_name / 'proposition_graph'
        if dataset_path.exists():
            datasets_to_process.append(dataset_path)
        else:
            print(f"错误：数据集目录不存在: {dataset_path}")
            sys.exit(1)
    else:
        parser.error("必须指定 --all, --dataset 或 --dataset-name 之一")
    
    if not datasets_to_process:
        print("错误：没有找到需要处理的数据集")
        sys.exit(1)
    
    # 显示摘要
    mode_str = "[DRY RUN]" if args.dry_run else "[实际执行]"
    print("=" * 60)
    print(f"文档ID前缀迁移脚本 {mode_str}")
    print("=" * 60)
    print(f"将处理 {len(datasets_to_process)} 个数据集:")
    for dataset_path in datasets_to_process:
        print(f"  - {dataset_path}")
    print()
    
    if args.dry_run:
        print("注意：这是干运行模式，不会实际修改文件")
        print()
    
    # 处理每个数据集
    all_stats = {}
    for dataset_path in datasets_to_process:
        try:
            stats = migrate_dataset(dataset_path, dry_run=args.dry_run)
            all_stats[dataset_path.name] = stats
        except Exception as e:
            print(f"错误：处理数据集 {dataset_path} 时出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 输出总结
    print("\n" + "=" * 60)
    print("迁移总结")
    print("=" * 60)
    
    total_files_processed = sum(s.get('files_processed', 0) for s in all_stats.values())
    total_files_migrated = sum(s.get('files_migrated', 0) for s in all_stats.values())
    
    print(f"处理的数据集数: {len(datasets_to_process)}")
    print(f"处理的文件总数: {total_files_processed}")
    print(f"需要迁移的文件数: {total_files_migrated}")
    
    for dataset_name, stats in all_stats.items():
        print(f"\n{dataset_name}:")
        print(f"  处理的文件: {stats.get('files_processed', 0)}")
        print(f"  需要迁移的文件: {stats.get('files_migrated', 0)}")
    
    print("\n" + "=" * 60)
    if args.dry_run:
        print("干运行完成。使用 --dry-run=false 或移除 --dry-run 参数来实际执行迁移。")
    else:
        print("迁移完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
