#!/usr/bin/env python3
"""
ProGraph 数据集过滤脚本
在步骤2和3之间运行，过滤掉处理失败的文档

同时过滤：
1. full_docs.json - 根据图中成功处理的文档ID过滤
2. train_data.json - 根据图中成功处理的文档标题过滤，确保每个样本都能回答
"""

import pickle
import json
import sys
import re
from pathlib import Path
from typing import Set, List, Dict, Any, Tuple
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))


def extract_successful_doc_ids(graph_path: str) -> Set[str]:
    """
    从图中提取成功处理的文档ID集合
    
    Args:
        graph_path: 图文件路径 (.pkl)
        
    Returns:
        成功处理的文档ID集合
    """
    print(f"正在从图中提取文档ID: {graph_path}")
    
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)
    
    # 从所有节点中提取唯一的 doc_id
    doc_ids = set()
    for node_id, node_data in graph.nodes(data=True):
        doc_id = node_data.get("doc_id", "")
        if doc_id:
            doc_ids.add(doc_id)
    
    print(f"成功提取 {len(doc_ids)} 个文档ID")
    return doc_ids


def parse_doc_id_to_index(doc_id: str) -> int:
    """
    解析文档ID，提取索引部分
    
    支持的文档ID格式：
    - HotpotQA: hotpot_{idx} (例如: "hotpot_42")
    - 2WikiMultihopQA: 2wiki_{idx} (例如: "2wiki_42")
    - MuSiQue: musique_{idx} (例如: "musique_42")
    
    Args:
        doc_id: 文档ID
        
    Returns:
        文档索引
    """
    # 匹配格式：{prefix}_{数字}
    # 支持的前缀：hotpot, 2wiki, musique
    patterns = [
        r"hotpot_(\d+)",      # HotpotQA
        r"2wiki_(\d+)",       # 2WikiMultihopQA
        r"musique_(\d+)"      # MuSiQue
    ]
    
    for pattern in patterns:
        match = re.match(pattern, doc_id)
        if match:
            return int(match.group(1))
    
    raise ValueError(f"无法解析文档ID格式: {doc_id} (支持格式: hotpot_{{idx}}, 2wiki_{{idx}}, musique_{{idx}})")


def build_doc_id_to_title_mapping(
    full_docs_path: str,
    successful_doc_ids: Set[str]
) -> Dict[str, str]:
    """
    建立文档ID到标题的映射
    
    通过解析文档ID中的索引，从原始 full_docs.json 中提取对应的标题
    
    Args:
        full_docs_path: 原始 full_docs.json 路径
        successful_doc_ids: 成功处理的文档ID集合
        
    Returns:
        doc_id -> title 映射字典
    """
    print(f"正在建立文档ID到标题的映射...")
    
    # 加载原始文档
    with open(full_docs_path, 'r', encoding='utf-8') as f:
        original_docs = json.load(f)
    
    # 建立映射
    doc_id_to_title = {}
    for doc_id in successful_doc_ids:
        try:
            idx = parse_doc_id_to_index(doc_id)
            if 0 <= idx < len(original_docs):
                title = original_docs[idx].get("title", "")
                if title:
                    doc_id_to_title[doc_id] = title
        except ValueError as e:
            print(f"  警告: {e}")
            continue
    
    print(f"建立了 {len(doc_id_to_title)} 个文档ID到标题的映射")
    return doc_id_to_title


def filter_full_docs(
    full_docs_path: str,
    successful_doc_ids: Set[str],
    output_path: str
) -> Tuple[int, int]:
    """
    过滤 full_docs.json，只保留成功处理的文档
    
    Args:
        full_docs_path: 原始 full_docs.json 路径
        successful_doc_ids: 成功处理的文档ID集合
        output_path: 输出路径
        
    Returns:
        (原始文档数, 过滤后文档数)
    """
    print("\n" + "=" * 60)
    print("过滤 full_docs.json")
    print("=" * 60)
    
    # 加载原始文档
    print(f"正在加载原始文档: {full_docs_path}")
    with open(full_docs_path, 'r', encoding='utf-8') as f:
        original_docs = json.load(f)
    
    original_count = len(original_docs)
    print(f"原始文档数: {original_count}")
    
    # 建立索引到文档ID的映射
    index_to_doc_id = {}
    for doc_id in successful_doc_ids:
        try:
            idx = parse_doc_id_to_index(doc_id)
            if 0 <= idx < original_count:
                index_to_doc_id[idx] = doc_id
        except ValueError:
            continue
    
    # 过滤文档
    print("正在过滤文档...")
    filtered_docs = []
    skipped_indices = []
    
    for idx, doc in enumerate(original_docs):
        if idx in index_to_doc_id:
            filtered_docs.append(doc)
        else:
            skipped_indices.append({
                "index": idx,
                "title": doc.get("title", "")[:50]
            })
    
    filtered_count = len(filtered_docs)
    skipped_count = original_count - filtered_count
    
    # 保存过滤后的文档
    print(f"正在保存过滤后的文档...")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_docs, f, ensure_ascii=False, indent=2)
    
    # 保存统计信息
    stats_path = output.parent / "filter_docs_stats.json"
    stats = {
        "original_count": original_count,
        "filtered_count": filtered_count,
        "skipped_count": skipped_count,
        "filter_rate": skipped_count / original_count * 100 if original_count > 0 else 0,
        "skipped_docs": skipped_indices[:100]  # 只保存前100个用于检查
    }
    
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    # 输出统计信息
    print(f"\n过滤完成:")
    print(f"  原始文档数: {original_count}")
    print(f"  过滤后文档数: {filtered_count}")
    print(f"  跳过文档数: {skipped_count}")
    print(f"  过滤率: {stats['filter_rate']:.2f}%")
    print(f"  输出文件: {output_path}")
    print(f"  统计信息: {stats_path}")
    
    if skipped_indices:
        print(f"\n跳过的文档示例（前5个）:")
        for doc in skipped_indices[:5]:
            print(f"    - [{doc['index']}]: {doc['title']}")
    
    return original_count, filtered_count


def validate_supporting_facts(
    sample: Dict[str, Any],
    valid_titles: Set[str]
) -> bool:
    """
    验证样本的 supporting_facts 中的所有标题是否都在有效标题集合中
    
    Args:
        sample: 样本字典，包含 supporting_facts 字段
        valid_titles: 有效的文档标题集合
        
    Returns:
        如果所有标题都存在，返回 True；否则返回 False
    """
    supporting_facts = sample.get("supporting_facts", [])
    
    # 如果 supporting_facts 为空，保留该样本（视为有效）
    if not supporting_facts:
        return True
    
    # 提取所有标题并验证
    for fact in supporting_facts:
        if not isinstance(fact, list) or len(fact) < 1:
            continue
        title = fact[0]  # supporting_facts 格式：[[title, sent_idx], ...]
        if title not in valid_titles:
            return False
    
    return True


def filter_train_data(
    train_data_path: str,
    successful_titles: Set[str],
    output_path: str
) -> Tuple[int, int]:
    """
    过滤 train_data.json，只保留所有 supporting_facts 中的文档都存在的样本
    
    Args:
        train_data_path: 原始 train_data.json 路径
        successful_titles: 成功处理的文档标题集合
        output_path: 输出路径
        
    Returns:
        (原始样本数, 过滤后样本数)
    """
    print("\n" + "=" * 60)
    print("过滤 train_data.json")
    print("=" * 60)
    
    # 加载原始训练数据
    print(f"正在加载训练数据: {train_data_path}")
    with open(train_data_path, 'r', encoding='utf-8') as f:
        original_samples = json.load(f)
    
    original_count = len(original_samples)
    print(f"原始样本数: {original_count}")
    print(f"有效文档标题数: {len(successful_titles)}")
    
    # 过滤样本
    print("正在过滤样本...")
    filtered_samples = []
    skipped_samples = []
    
    for sample in original_samples:
        if validate_supporting_facts(sample, successful_titles):
            filtered_samples.append(sample)
        else:
            # 记录跳过的样本信息
            skipped_samples.append({
                "id": sample.get("id", "unknown"),
                "question": sample.get("question", "")[:100],
                "supporting_facts": sample.get("supporting_facts", [])
            })
    
    filtered_count = len(filtered_samples)
    skipped_count = original_count - filtered_count
    
    # 保存过滤后的样本
    print(f"正在保存过滤后的样本...")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_samples, f, ensure_ascii=False, indent=2)
    
    # 保存统计信息
    stats_path = output.parent / "filter_train_stats.json"
    stats = {
        "original_count": original_count,
        "filtered_count": filtered_count,
        "skipped_count": skipped_count,
        "filter_rate": skipped_count / original_count * 100 if original_count > 0 else 0,
        "skipped_samples": skipped_samples[:100]  # 只保存前100个用于检查
    }
    
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    # 输出统计信息
    print(f"\n过滤完成:")
    print(f"  原始样本数: {original_count}")
    print(f"  过滤后样本数: {filtered_count}")
    print(f"  跳过样本数: {skipped_count}")
    print(f"  过滤率: {stats['filter_rate']:.2f}%")
    print(f"  输出文件: {output_path}")
    print(f"  统计信息: {stats_path}")
    
    if skipped_samples:
        print(f"\n跳过的样本示例（前5个）:")
        for sample in skipped_samples[:5]:
            print(f"    - [{sample['id']}]: {sample['question']}")
            print(f"      supporting_facts: {sample['supporting_facts'][:2]}...")
    
    return original_count, filtered_count


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="ProGraph 数据集过滤脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 过滤 MuSiQue 数据集
  python scripts/2.5-filter_dataset.py \\
    --dataset dataset/MuSiQue/full_docs.json \\
    --train-data dataset/MuSiQue/train_data.json \\
    --graph output/MuSiQue/proposition_graph/linked_graph.pkl \\
    --output-docs dataset/MuSiQue/full_docs_filtered.json \\
    --output-train dataset/MuSiQue/train_data_filtered.json

  # 过滤 HotpotQA 数据集
  python scripts/2.5-filter_dataset.py \\
    --dataset dataset/HotpotQA/full_docs.json \\
    --train-data dataset/HotpotQA/train_data.json \\
    --graph output/HotpotQA/proposition_graph/linked_graph.pkl \\
    --output-docs dataset/HotpotQA/full_docs_filtered.json \\
    --output-train dataset/HotpotQA/train_data_filtered.json
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='原始数据集路径 (full_docs.json)'
    )
    
    parser.add_argument(
        '--train-data',
        type=str,
        required=True,
        help='原始训练数据路径 (train_data.json)'
    )
    
    parser.add_argument(
        '--graph',
        type=str,
        required=True,
        help='已处理的图文件路径 (.pkl)'
    )
    
    parser.add_argument(
        '--output-docs',
        type=str,
        required=True,
        help='输出过滤后数据集路径'
    )
    
    parser.add_argument(
        '--output-train',
        type=str,
        required=True,
        help='输出过滤后训练数据路径'
    )
    
    args = parser.parse_args()
    
    # 运行过滤流程
    print("=" * 60)
    print("ProGraph 数据集过滤")
    print("=" * 60)
    print(f"原始数据集: {args.dataset}")
    print(f"原始训练数据: {args.train_data}")
    print(f"输入图: {args.graph}")
    print(f"输出数据集: {args.output_docs}")
    print(f"输出训练数据: {args.output_train}")
    print()
    
    # 1. 从图中提取成功处理的文档ID
    successful_doc_ids = extract_successful_doc_ids(args.graph)
    
    # 2. 建立文档ID到标题的映射
    doc_id_to_title = build_doc_id_to_title_mapping(
        args.dataset,
        successful_doc_ids
    )
    successful_titles = set(doc_id_to_title.values())
    
    # 3. 过滤 full_docs.json
    filter_full_docs(
        args.dataset,
        successful_doc_ids,
        args.output_docs
    )
    
    # 4. 过滤 train_data.json
    filter_train_data(
        args.train_data,
        successful_titles,
        args.output_train
    )
    
    print("\n" + "=" * 60)
    print("所有过滤完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
