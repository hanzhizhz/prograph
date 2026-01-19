"""
PKL 转 JSON 脚本
将 NetworkX 图的 PKL 文件转换为 JSON 格式
"""

import pickle
import json
import argparse
from pathlib import Path
import networkx as nx


def convert_pkl_to_json(pkl_path: str, json_path: str = None) -> None:
    """
    将 PKL 格式的图文件转换为 JSON 格式

    Args:
        pkl_path: PKL 文件路径
        json_path: JSON 文件路径（可选，默认为 pkl_path 替换扩展名）
    """
    pkl_path = Path(pkl_path)
    if not pkl_path.exists():
        raise FileNotFoundError(f"PKL 文件不存在: {pkl_path}")

    # 默认输出路径
    if json_path is None:
        json_path = pkl_path.with_suffix('.json')
    else:
        json_path = Path(json_path)

    print(f"加载 PKL 文件: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        graph = pickle.load(f)

    # 转换为 JSON 格式
    print(f"转换图数据... 节点数: {graph.number_of_nodes()}, 边数: {graph.number_of_edges()}")
    graph_data = nx.node_link_data(graph)

    # 保存 JSON
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)

    print(f"JSON 文件已保存: {json_path}")
    print(f"文件大小: {json_path.stat().st_size / 1024 / 1024:.1f} MB")


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description='将 PKL 格式的图文件转换为 JSON 格式')
    parser.add_argument('pkl_file', nargs='?', help='PKL 文件路径（使用 --all 时可选）')
    parser.add_argument('-o', '--output', help='输出 JSON 文件路径（可选）')
    parser.add_argument('--all', action='store_true', help='转换所有数据集的图文件')

    args = parser.parse_args()

    if args.all:
        # 转换所有数据集
        datasets = ['2WikiMultihopQA', 'HotpotQA', 'MuSiQue']
        for dataset in datasets:
            pkl_path = Path(f"output/{dataset}/proposition_graph/raw_graph.pkl")
            if pkl_path.exists():
                print(f"\n转换 {dataset}...")
                convert_pkl_to_json(str(pkl_path))
            else:
                print(f"跳过 {dataset}: 文件不存在")
    else:
        if not args.pkl_file:
            parser.error("需要指定 pkl_file 或使用 --all")
        convert_pkl_to_json(args.pkl_file, args.output)


if __name__ == "__main__":
    main()
