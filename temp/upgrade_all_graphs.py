#!/usr/bin/env python3
"""
批量升级所有数据集的图文件

修复问题：
1. SIMILARITY 边添加反向边
2. MENTIONS_ENTITY 边添加反向边
3. direction="1<->2" 的边添加反向边
"""

import sys
import pickle
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.proposition_graph.graph_upgrader import GraphUpgrader


# 数据集列表
DATASETS = ["HotpotQA", "MuSiQue", "2WikiMultihopQA"]
OUTPUT_DIR = project_root / "output"

# 需要升级的图文件
GRAPH_FILES = [
    "proposition_graph/raw_graph.pkl",
    "proposition_graph/linked_graph.pkl",
    "proposition_graph/raw_graph_fixed.pkl",
]


def upgrade_graph_file(input_path: Path, backup: bool = True, dry_run: bool = False) -> dict:
    """升级单个图文件

    Args:
        input_path: 图文件路径
        backup: 是否创建备份
        dry_run: 预览模式，不实际修改文件

    Returns:
        升级统计信息
    """
    print(f"\n{'='*60}")
    print(f"处理: {input_path.relative_to(project_root)}")

    if not input_path.exists():
        print(f"  ⚠ 文件不存在，跳过")
        return None

    # 加载图
    print(f"  加载图...")
    with open(input_path, 'rb') as f:
        graph = pickle.load(f)

    print(f"  原始图: {graph.number_of_nodes()} 节点, {graph.number_of_edges()} 边")

    if dry_run:
        print(f"  [预览模式] 不会修改文件")

    # 升级
    upgrader = GraphUpgrader(graph)
    upgraded_graph = upgrader.upgrade()

    # 保存（非预览模式）
    if not dry_run:
        # 创建备份
        if backup:
            backup_path = input_path.with_suffix('.pkl.bak')
            print(f"  创建备份: {backup_path.name}")
            with open(input_path, 'rb') as f:
                backup_data = f.read()
            with open(backup_path, 'wb') as f:
                f.write(backup_data)

        # 保存升级后的图
        print(f"  保存升级后的图...")
        with open(input_path, 'wb') as f:
            pickle.dump(upgraded_graph, f)

    print(f"  升级后: {upgraded_graph.number_of_nodes()} 节点, {upgraded_graph.number_of_edges()} 边")
    upgrader.print_stats()

    return upgrader.upgrade_stats


def upgrade_dataset(dataset: str, backup: bool = True, dry_run: bool = False) -> dict:
    """升级单个数据集的所有图文件

    Args:
        dataset: 数据集名称
        backup: 是否创建备份
        dry_run: 预览模式，不实际修改文件

    Returns:
        该数据集的升级统计
    """
    print(f"\n\n{'#'*60}")
    print(f"# 数据集: {dataset}")
    if dry_run:
        print(f"# [预览模式]")
    print(f"{'#'*60}")

    dataset_dir = OUTPUT_DIR / dataset
    if not dataset_dir.exists():
        print(f"⚠ 数据集目录不存在: {dataset_dir}")
        return None

    total_stats = {
        "similarity_edges_added": 0,
        "mention_edges_added": 0,
        "bidirectional_rst_added": 0
    }

    for graph_file in GRAPH_FILES:
        graph_path = dataset_dir / graph_file
        stats = upgrade_graph_file(graph_path, backup=backup, dry_run=dry_run)
        if stats:
            for key in total_stats:
                total_stats[key] += stats[key]

    return total_stats


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="ProGraph 图升级工具")
    parser.add_argument("--dry-run", action="store_true", help="预览模式，不实际修改文件")
    parser.add_argument("--no-backup", action="store_true", help="不创建备份文件")
    parser.add_argument("--dataset", nargs="+", choices=DATASETS + ["all"], default=["all"],
                        help="要处理的数据集，默认 all")

    args = parser.parse_args()

    print("="*60)
    print("ProGraph 图升级工具")
    print("="*60)

    if args.dry_run:
        print("[预览模式] 不会修改任何文件")

    # 确定要处理的数据集
    datasets = DATASETS if "all" in args.dataset else args.dataset

    # 是否创建备份
    backup = not args.no_backup

    # 总计
    grand_total = {
        "similarity_edges_added": 0,
        "mention_edges_added": 0,
        "bidirectional_rst_added": 0
    }

    # 处理每个数据集
    for dataset in datasets:
        stats = upgrade_dataset(dataset, backup=backup, dry_run=args.dry_run)
        if stats:
            for key in grand_total:
                grand_total[key] += stats[key]

    # 打印总计
    print("\n\n" + "="*60)
    print("总计")
    print("="*60)
    print(f"SIMILARITY 反向边添加: {grand_total['similarity_edges_added']}")
    print(f"MENTIONS_ENTITY 反向边添加: {grand_total['mention_edges_added']}")
    print(f"双向 RST 边添加: {grand_total['bidirectional_rst_added']}")
    print(f"总计添加边: {sum(grand_total.values())}")

    if args.dry_run:
        print("\n[预览模式完成] 使用 --no-dry-run 来实际执行升级")
    else:
        print("\n✓ 升级完成!")
        if backup:
            print("💾 备份文件已保存为 *.pkl.bak")
            print("   如需回滚，可以使用备份文件恢复")


if __name__ == "__main__":
    main()
