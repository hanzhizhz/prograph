#!/usr/bin/env python3
"""
ProGraph 多数据集评估脚本
评估 HotpotQA、2WikiMultihopQA 和 MuSiQue 三个数据集的问答结果

支持指标：
- Exact Match (EM): 精确匹配分数
- F1 Score: F1 分数
"""

import json
import re
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import argparse


def normalize_answer(text: str) -> str:
    """
    规范化答案文本用于评估

    遵循 MRQA 官方评估方法的规范化步骤：
    1. 转换为小写
    2. 移除标点符号（保留连字符和空格内的字符）
    3. 移除文章（a, an, the）
    4. 压缩空白字符

    Args:
        text: 原始答案文本

    Returns:
        规范化后的答案文本
    """
    if text is None or not isinstance(text, str):
        return ""

    # 转换为小写
    text = text.lower()

    # 移除标点符号（保留字母、数字、空格）
    # 保留连字符（如 "co-operate"）和缩略词（如 "u.s."）中的点
    text = re.sub(r'[^\w\s\.\-]', '', text)

    # 移除文章
    articles = ['a ', 'an ', 'the ']
    for article in articles:
        text = text.replace(article, ' ')

    # 压缩空白字符
    text = re.sub(r'\s+', ' ', text)

    # 去除首尾空白
    text = text.strip()

    return text


def compute_exact_match(gold: str, predicted: str) -> float:
    """
    计算精确匹配分数

    Args:
        gold: 标准答案
        predicted: 预测答案

    Returns:
        1.0 如果规范化后完全匹配，否则 0.0
    """
    gold_normalized = normalize_answer(gold)
    predicted_normalized = normalize_answer(predicted)

    return 1.0 if gold_normalized == predicted_normalized else 0.0


def compute_f1(gold: str, predicted: str) -> float:
    """
    计算 F1 分数

    Args:
        gold: 标准答案
        predicted: 预测答案

    Returns:
        F1 分数
    """
    gold_normalized = normalize_answer(gold)
    predicted_normalized = normalize_answer(predicted)

    gold_tokens = gold_normalized.split()
    predicted_tokens = predicted_normalized.split()

    if not gold_tokens and not predicted_tokens:
        return 1.0
    if not gold_tokens or not predicted_tokens:
        return 0.0

    # 计算公共 token
    common = Counter(predicted_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    # 计算 precision 和 recall
    precision = num_same / len(predicted_tokens)
    recall = num_same / len(gold_tokens)

    # 计算 F1
    f1 = 2 * (precision * recall) / (precision + recall)

    return f1


@dataclass
class EvalResult:
    """评估结果数据类"""
    dataset_name: str
    total_samples: int
    exact_match: float
    f1_score: float
    em_correct: int
    f1_sum: float
    details: List[Dict]


def evaluate_dataset(result_file: str, dataset_name: str = None) -> EvalResult:
    """
    评估单个数据集的问答结果

    Args:
        result_file: 结果文件路径（JSON 格式）
        dataset_name: 数据集名称（可选，从文件名推断）

    Returns:
        评估结果
    """
    # 加载结果文件
    with open(result_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    if not results:
        print(f"警告: {result_file} 为空")
        return EvalResult(
            dataset_name=dataset_name or Path(result_file).stem,
            total_samples=0,
            exact_match=0.0,
            f1_score=0.0,
            em_correct=0,
            f1_sum=0.0,
            details=[]
        )

    # 如果未指定数据集名称，从文件名推断
    if dataset_name is None:
        path = Path(result_file)
        dataset_name = path.parent.parent.name

    # 评估每个结果
    em_correct = 0
    f1_sum = 0.0
    details = []

    for item in results:
        question = item.get('question', '')
        answer = item.get('answer', '')
        reference_answer = item.get('reference_answer', '')

        # 处理可能的多个标准答案
        if isinstance(reference_answer, list):
            gold_answers = reference_answer
        else:
            gold_answers = [reference_answer]

        # 计算 EM（取所有 gold 答案中的最高分）
        em_scores = [compute_exact_match(gold, answer) for gold in gold_answers]
        em = max(em_scores)

        # 计算 F1（取所有 gold 答案中的最高分）
        f1_scores = [compute_f1(gold, answer) for gold in gold_answers]
        f1 = max(f1_scores)

        em_correct += int(em)
        f1_sum += f1

        details.append({
            'question': question,
            'answer': answer,
            'reference_answer': reference_answer,
            'exact_match': em,
            'f1': f1,
            'normalized_answer': normalize_answer(answer),
            'normalized_reference': normalize_answer(reference_answer) if not isinstance(reference_answer, list) else [normalize_answer(a) for a in reference_answer]
        })

    total = len(results)
    em_score = em_correct / total if total > 0 else 0.0
    f1_score = f1_sum / total if total > 0 else 0.0

    return EvalResult(
        dataset_name=dataset_name,
        total_samples=total,
        exact_match=em_score,
        f1_score=f1_score,
        em_correct=em_correct,
        f1_sum=f1_sum,
        details=details
    )


def print_eval_result(result: EvalResult, show_details: bool = False, max_details: int = 10):
    """
    打印评估结果

    Args:
        result: 评估结果
        show_details: 是否显示详细结果
        max_details: 最大显示的详细结果数量
    """
    print("=" * 70)
    print(f"数据集: {result.dataset_name}")
    print("=" * 70)
    print(f"总样本数: {result.total_samples}")
    print(f"Exact Match (EM): {result.exact_match:.4f} ({result.em_correct}/{result.total_samples})")
    print(f"F1 Score: {result.f1_score:.4f}")
    print("-" * 70)

    # 显示正确和错误的例子
    correct_examples = [d for d in result.details if d['exact_match'] == 1.0]
    wrong_examples = [d for d in result.details if d['exact_match'] == 0.0]

    print(f"正确示例: {len(correct_examples)} 个")
    print(f"错误示例: {len(wrong_examples)} 个")
    print()

    if show_details:
        print("-" * 70)
        print("正确示例 (最多显示 {} 个):".format(max_details))
        print("-" * 70)

        for i, ex in enumerate(correct_examples[:max_details]):
            print(f"\n[{i+1}] 问题: {ex['question'][:80]}...")
            print(f"    答案: {ex['answer'][:60]}...")
            print(f"    标准答案: {ex['reference_answer'][:60]}...")

        print("\n" + "-" * 70)
        print("错误示例 (最多显示 {} 个):".format(max_details))
        print("-" * 70)

        for i, ex in enumerate(wrong_examples[:max_details]):
            print(f"\n[{i+1}] 问题: {ex['question'][:80]}...")
            print(f"    答案: {ex['answer'][:60]}...")
            print(f"    标准答案: {ex['reference_answer'][:60]}...")
            print(f"    EM: {ex['exact_match']:.2f}, F1: {ex['f1']:.4f}")


def evaluate_all_datasets(
    datasets: List[str],
    output_dir: str = "output",
    show_details: bool = False,
    save_details: bool = True
) -> Dict[str, EvalResult]:
    """
    评估所有数据集

    Args:
        datasets: 数据集名称列表
        output_dir: 输出目录
        show_details: 是否显示详细结果
        save_details: 是否保存详细结果

    Returns:
        各数据集的评估结果
    """
    results = {}

    for dataset_name in datasets:
        result_file = Path(output_dir) / dataset_name / "result.json"

        if not result_file.exists():
            print(f"警告: 找不到结果文件 {result_file}")
            continue

        print(f"\n评估数据集: {dataset_name}")
        print(f"结果文件: {result_file}")

        eval_result = evaluate_dataset(str(result_file), dataset_name)
        results[dataset_name] = eval_result

        print_eval_result(eval_result, show_details=show_details)

        # 保存详细结果
        if save_details:
            details_file = Path(output_dir) / dataset_name / "eval_details.json"
            details_file.parent.mkdir(parents=True, exist_ok=True)

            with open(details_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'dataset_name': eval_result.dataset_name,
                    'total_samples': eval_result.total_samples,
                    'exact_match': eval_result.exact_match,
                    'f1_score': eval_result.f1_score,
                    'em_correct': eval_result.em_correct,
                    'details': eval_result.details
                }, f, ensure_ascii=False, indent=2)

            print(f"\n详细结果已保存到: {details_file}")

    return results


def print_summary(results: Dict[str, EvalResult]):
    """
    打印所有数据集的汇总统计

    Args:
        results: 各数据集的评估结果
    """
    print("\n" + "=" * 70)
    print("评估汇总")
    print("=" * 70)

    total_samples = 0
    total_em_correct = 0
    total_f1_sum = 0.0

    for dataset_name, result in sorted(results.items()):
        print(f"\n{dataset_name}:")
        print(f"  - 样本数: {result.total_samples}")
        print(f"  - EM: {result.exact_match:.4f}")
        print(f"  - F1: {result.f1_score:.4f}")

        total_samples += result.total_samples
        total_em_correct += result.em_correct
        total_f1_sum += result.f1_sum

    # 计算总体指标
    overall_em = total_em_correct / total_samples if total_samples > 0 else 0.0
    overall_f1 = total_f1_sum / total_samples if total_samples > 0 else 0.0

    print("\n" + "-" * 70)
    print("总体结果:")
    print(f"  - 总样本数: {total_samples}")
    print(f"  - 总体 EM: {overall_em:.4f}")
    print(f"  - 总体 F1: {overall_f1:.4f}")
    print("=" * 70)

    # 返回汇总结果
    return {
        'total_samples': total_samples,
        'overall_em': overall_em,
        'overall_f1': overall_f1,
        'by_dataset': {name: {
            'samples': r.total_samples,
            'em': r.exact_match,
            'f1': r.f1_score
        } for name, r in results.items()}
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="ProGraph 多数据集评估脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 评估所有三个数据集
  python scripts/5-evaluate_all_datasets.py

  # 评估指定数据集
  python scripts/5-evaluate_all_datasets.py --datasets HotpotQA 2WikiMultihopQA

  # 显示详细结果
  python scripts/5-evaluate_all_datasets.py --show-details

  # 保存详细结果到文件
  python scripts/5-evaluate_all_datasets.py --save-details

  # 从指定目录读取结果
  python scripts/5-evaluate_all_datasets.py --output-dir /path/to/output
        """
    )

    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        default=['HotpotQA', '2WikiMultihopQA', 'MuSiQue'],
        help='要评估的数据集列表 (默认: HotpotQA 2WikiMultihopQA MuSiQue)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='输出目录 (默认: output)'
    )

    parser.add_argument(
        '--show-details',
        action='store_true',
        help='显示每个问题的评估详情'
    )

    parser.add_argument(
        '--max-details',
        type=int,
        default=10,
        help='最大显示的详情数量 (默认: 10)'
    )

    parser.add_argument(
        '--no-save-details',
        action='store_true',
        help='不保存详细结果到文件'
    )

    parser.add_argument(
        '--result-file',
        type=str,
        default=None,
        help='指定单个结果文件路径（覆盖数据集参数）'
    )

    args = parser.parse_args()

    # 单文件模式
    if args.result_file:
        result = evaluate_dataset(args.result_file)
        print_eval_result(result, show_details=args.show_details, max_details=args.max_details)
        return

    # 多数据集模式
    results = evaluate_all_datasets(
        datasets=args.datasets,
        output_dir=args.output_dir,
        show_details=args.show_details,
        save_details=not args.no_save_details
    )

    # 打印汇总
    summary = print_summary(results)

    # 保存汇总结果
    summary_file = Path(args.output_dir) / "eval_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n汇总结果已保存到: {summary_file}")


if __name__ == '__main__':
    main()
