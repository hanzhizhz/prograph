#!/usr/bin/env python3
"""
ProGraph LLM-as-Judge 评估脚本
使用大语言模型作为评判者，对问答结果进行二分类评估（正确/错误）

支持功能：
- 从配置文件读取OpenAI API配置
- 并发评估（默认并发度50）
- 计算正确率百分比
"""

import json
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import argparse
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 直接导入，避免通过 __init__.py 触发 vllm 导入
from src.llm.openai_client import OpenAIClient, build_single_turn
from src.config.model_config import get_model_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class LLMJudgeResult:
    """LLM-as-Judge评估结果数据类"""
    dataset_name: str
    total_samples: int
    correct_count: int
    accuracy: float  # 正确率百分比
    details: List[Dict]


# LLM-as-Judge的评估提示词
JUDGE_SYSTEM_PROMPT = """你是一个答案评估专家，需要判断预测答案是否正确回答了问题。请仔细分析问题、标准答案和预测答案，判断预测答案是否正确。

评估标准：
1. 如果预测答案与标准答案意思相同（允许表达方式不同），则判断为正确
2. 如果预测答案包含了标准答案的关键信息，但表达更详细，也判断为正确
3. 如果预测答案明显错误或与标准答案意思相反，则判断为错误
4. 如果预测答案部分正确但关键信息缺失，则判断为错误

只回答"正确"或"错误"，不要添加任何其他内容。"""


def build_judge_prompt(question: str, reference_answer: str, predicted_answer: str) -> str:
    """
    构建LLM-as-Judge的评估提示词
    
    Args:
        question: 问题
        reference_answer: 标准答案
        predicted_answer: 预测答案
        
    Returns:
        用户提示词
    """
    prompt = f"""问题：{question}

标准答案：{reference_answer}

预测答案：{predicted_answer}

请判断预测答案是否正确回答了问题。只回答"正确"或"错误"。"""
    return prompt


async def judge_answer_with_llm(
    client: OpenAIClient,
    question: str,
    reference_answer: str,
    predicted_answer: str,
    temperature: float = 0.0
) -> int:
    """
    使用LLM判断答案是否正确（异步）
    
    Args:
        client: OpenAI客户端
        question: 问题
        reference_answer: 标准答案
        predicted_answer: 预测答案
        temperature: 温度参数（默认0.0确保结果稳定）
        
    Returns:
        1 表示正确，0 表示错误
    """
    try:
        # 构建消息
        user_prompt = build_judge_prompt(question, reference_answer, predicted_answer)
        messages = build_single_turn(JUDGE_SYSTEM_PROMPT, user_prompt)
        
        # 调用LLM
        response = await client.generate(
            messages=messages,
            temperature=temperature,
            max_tokens=10  # 只需要"正确"或"错误"
        )
        
        # 解析响应
        response = response.strip()
        if "正确" in response:
            return 1
        elif "错误" in response:
            return 0
        else:
            # 如果响应格式不符合预期，默认返回错误
            logger.warning(f"无法解析LLM响应: {response}，默认判断为错误")
            return 0
            
    except Exception as e:
        logger.error(f"LLM判断出错: {e}，问题: {question[:50]}...")
        # 发生错误时默认返回错误
        return 0


async def evaluate_dataset_with_llm_judge(
    result_file: str,
    dataset_name: Optional[str] = None,
    client: Optional[OpenAIClient] = None,
    concurrency: int = 50
) -> LLMJudgeResult:
    """
    使用LLM-as-Judge评估单个数据集的问答结果（异步）
    
    Args:
        result_file: 结果文件路径（JSON格式）
        dataset_name: 数据集名称（可选，从文件名推断）
        client: OpenAI客户端（可选，如果不提供则从配置创建）
        concurrency: 并发度（默认50）
        
    Returns:
        评估结果
    """
    # 加载结果文件
    with open(result_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    if not results:
        logger.warning(f"结果文件为空: {result_file}")
        return LLMJudgeResult(
            dataset_name=dataset_name or Path(result_file).stem,
            total_samples=0,
            correct_count=0,
            accuracy=0.0,
            details=[]
        )
    
    # 如果未指定数据集名称，从文件名推断
    if dataset_name is None:
        path = Path(result_file)
        dataset_name = path.parent.parent.name
    
    # 如果没有提供客户端，从配置创建
    client_created = False
    if client is None:
        model_config = get_model_config()
        llm_config = model_config.llm
        client = OpenAIClient(
            base_url=llm_config.base_url,
            api_key=llm_config.api_key,
            model=llm_config.model,
            temperature=llm_config.temperature,
            max_tokens=llm_config.max_tokens,
            concurrency=concurrency,
            max_retries=llm_config.max_retries,
            timeout=llm_config.timeout
        )
        client_created = True
    
    # 准备评估任务
    tasks = []
    details = []
    
    for item in results:
        question = item.get('question', '')
        answer = item.get('answer', '')
        reference_answer = item.get('reference_answer', '')
        
        # 处理可能的多个标准答案（取第一个进行判断，或可以后续改进为多答案处理）
        if isinstance(reference_answer, list):
            if reference_answer:
                gold_answer = reference_answer[0]
            else:
                gold_answer = ""
        else:
            gold_answer = reference_answer
        
        # 创建评估任务
        task = judge_answer_with_llm(client, question, gold_answer, answer)
        tasks.append(task)
        
        # 保存原始信息用于详情
        details.append({
            'question': question,
            'answer': answer,
            'reference_answer': reference_answer
        })
    
    # 并发执行评估（使用信号量控制并发度 + 分批提交）
    logger.info(f"开始并发评估 {len(tasks)} 个样本，并发度: {concurrency}")
    
    # 使用asyncio.gather分批执行以控制并发度
    semaphore = asyncio.Semaphore(concurrency)
    
    # 【性能优化】分批处理，避免瞬间高负载
    BATCH_SIZE = 100  # 每批任务数

    async def limited_judge(task, idx):
        async with semaphore:
            result = await task
            return idx, result

    async def process_batch(tasks_batch, indices_batch):
        """处理一批任务"""
        limited_tasks = [
            limited_judge(task, idx) 
            for task, idx in zip(tasks_batch, indices_batch)
        ]
        return await asyncio.gather(*limited_tasks)

    # 分批执行
    all_results = []
    for i in range(0, len(tasks), BATCH_SIZE):
        batch_tasks = tasks[i:i + BATCH_SIZE]
        batch_indices = list(range(i, min(i + BATCH_SIZE, len(tasks))))
        
        batch_results = await process_batch(batch_tasks, batch_indices)
        all_results.extend(batch_results)
        
        # 每批完成后短暂休息，避免瞬间高负载
        if i + BATCH_SIZE < len(tasks):
            await asyncio.sleep(0.1)
    
    judge_results = all_results
    
    # 处理结果
    correct_count = 0
    for idx, is_correct in judge_results:
        details[idx]['is_correct'] = is_correct
        correct_count += is_correct
    
    total = len(results)
    accuracy = (correct_count / total * 100.0) if total > 0 else 0.0
    
    # 关闭客户端（如果是我们创建的）
    if client_created and client is not None:
        await client.close()
    
    return LLMJudgeResult(
        dataset_name=dataset_name,
        total_samples=total,
        correct_count=correct_count,
        accuracy=accuracy,
        details=details
    )


def print_eval_result(result: LLMJudgeResult, show_details: bool = False, max_details: int = 10):
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
    print(f"正确数: {result.correct_count}")
    print(f"正确率: {result.accuracy:.2f}%")
    print("-" * 70)
    
    # 统计正确和错误的例子
    correct_examples = [d for d in result.details if d.get('is_correct', 0) == 1]
    wrong_examples = [d for d in result.details if d.get('is_correct', 0) == 0]
    
    print(f"正确示例: {len(correct_examples)} 个")
    print(f"错误示例: {len(wrong_examples)} 个")
    print()
    
    if show_details:
        print("-" * 70)
        print(f"正确示例 (最多显示 {max_details} 个):")
        print("-" * 70)
        
        for i, ex in enumerate(correct_examples[:max_details]):
            print(f"\n[{i+1}] 问题: {ex['question'][:80]}...")
            print(f"    答案: {ex['answer'][:60]}...")
            print(f"    标准答案: {ex['reference_answer'][:60] if not isinstance(ex['reference_answer'], list) else ex['reference_answer'][0][:60]}...")
        
        print("\n" + "-" * 70)
        print(f"错误示例 (最多显示 {max_details} 个):")
        print("-" * 70)
        
        for i, ex in enumerate(wrong_examples[:max_details]):
            print(f"\n[{i+1}] 问题: {ex['question'][:80]}...")
            print(f"    答案: {ex['answer'][:60]}...")
            print(f"    标准答案: {ex['reference_answer'][:60] if not isinstance(ex['reference_answer'], list) else ex['reference_answer'][0][:60]}...")


async def evaluate_all_datasets(
    datasets: List[str],
    output_dir: str = "output",
    concurrency: int = 50,
    show_details: bool = False,
    save_details: bool = True
) -> Dict[str, LLMJudgeResult]:
    """
    评估所有数据集
    
    Args:
        datasets: 数据集名称列表
        output_dir: 输出目录
        concurrency: 并发度
        show_details: 是否显示详细结果
        save_details: 是否保存详细结果
        
    Returns:
        各数据集的评估结果
    """
    results = {}
    
    # 从配置创建共享客户端
    model_config = get_model_config()
    llm_config = model_config.llm
    
    for dataset_name in datasets:
        result_file = Path(output_dir) / dataset_name / "result.json"
        
        if not result_file.exists():
            logger.warning(f"找不到结果文件: {result_file}")
            continue
        
        print(f"\n评估数据集: {dataset_name}")
        print(f"结果文件: {result_file}")
        
        # 为每个数据集创建客户端（或可以共享，这里为了安全起见分别创建）
        client = OpenAIClient(
            base_url=llm_config.base_url,
            api_key=llm_config.api_key,
            model=llm_config.model,
            temperature=llm_config.temperature,
            max_tokens=llm_config.max_tokens,
            concurrency=concurrency,
            max_retries=llm_config.max_retries,
            timeout=llm_config.timeout
        )
        
        try:
            eval_result = await evaluate_dataset_with_llm_judge(
                str(result_file),
                dataset_name,
                client=client,
                concurrency=concurrency
            )
            results[dataset_name] = eval_result
            
            print_eval_result(eval_result, show_details=show_details)
            
            # 保存详细结果
            if save_details:
                details_file = Path(output_dir) / dataset_name / "llm_judge_details.json"
                details_file.parent.mkdir(parents=True, exist_ok=True)
                
                with open(details_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'dataset_name': eval_result.dataset_name,
                        'total_samples': eval_result.total_samples,
                        'correct_count': eval_result.correct_count,
                        'accuracy': eval_result.accuracy,
                        'details': eval_result.details
                    }, f, ensure_ascii=False, indent=2)
                
                print(f"\n详细结果已保存到: {details_file}")
        finally:
            await client.close()
    
    return results


def print_summary(results: Dict[str, LLMJudgeResult]):
    """
    打印所有数据集的汇总统计
    
    Args:
        results: 各数据集的评估结果
    """
    print("\n" + "=" * 70)
    print("评估汇总")
    print("=" * 70)
    
    total_samples = 0
    total_correct = 0
    
    for dataset_name, result in sorted(results.items()):
        print(f"\n{dataset_name}:")
        print(f"  - 样本数: {result.total_samples}")
        print(f"  - 正确数: {result.correct_count}")
        print(f"  - 正确率: {result.accuracy:.2f}%")
        
        total_samples += result.total_samples
        total_correct += result.correct_count
    
    # 计算总体指标
    overall_accuracy = (total_correct / total_samples * 100.0) if total_samples > 0 else 0.0
    
    print("\n" + "-" * 70)
    print("总体结果:")
    print(f"  - 总样本数: {total_samples}")
    print(f"  - 总正确数: {total_correct}")
    print(f"  - 总体正确率: {overall_accuracy:.2f}%")
    print("=" * 70)
    
    # 返回汇总结果
    return {
        'total_samples': total_samples,
        'total_correct': total_correct,
        'overall_accuracy': overall_accuracy,
        'by_dataset': {
            name: {
                'samples': r.total_samples,
                'correct_count': r.correct_count,
                'accuracy': r.accuracy
            }
            for name, r in results.items()
        }
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="ProGraph LLM-as-Judge 评估脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 评估所有三个数据集
  python scripts/5.2-evaluate_with_llm_judge.py

  # 评估指定数据集
  python scripts/5.2-evaluate_with_llm_judge.py --datasets HotpotQA 2WikiMultihopQA

  # 显示详细结果
  python scripts/5.2-evaluate_with_llm_judge.py --show-details

  # 指定并发度
  python scripts/5.2-evaluate_with_llm_judge.py --concurrency 100

  # 评估单个结果文件
  python scripts/5.2-evaluate_with_llm_judge.py --result-file output/HotpotQA/result.json
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
        '--concurrency',
        type=int,
        default=50,
        help='并发度 (默认: 50)'
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
    
    parser.add_argument(
        '--config-path',
        type=str,
        default=None,
        help='配置文件路径（默认从环境变量或默认位置读取）'
    )
    
    args = parser.parse_args()
    
    # 单文件模式
    if args.result_file:
        async def eval_single():
            result = await evaluate_dataset_with_llm_judge(
                args.result_file,
                concurrency=args.concurrency
            )
            print_eval_result(result, show_details=args.show_details, max_details=args.max_details)
            return result
        
        asyncio.run(eval_single())
        return
    
    # 多数据集模式
    async def eval_all():
        results = await evaluate_all_datasets(
            datasets=args.datasets,
            output_dir=args.output_dir,
            concurrency=args.concurrency,
            show_details=args.show_details,
            save_details=not args.no_save_details
        )
        
        # 打印汇总
        summary = print_summary(results)
        
        # 保存汇总结果
        summary_file = Path(args.output_dir) / "llm_judge_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"\n汇总结果已保存到: {summary_file}")
    
    asyncio.run(eval_all())


if __name__ == '__main__':
    main()
