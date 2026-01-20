#!/usr/bin/env python3
"""
ProGraph 在线多跳问答脚本
基于 Intent-Group Beam Search 的多跳推理
"""

import asyncio
import sys
import pickle
import json
from pathlib import Path
from typing import Optional, List
import argparse

# 添加 src 路径到 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm import OpenAIClient
from src.config import get_model_config, set_model_config, get_retrieval_config, set_retrieval_config
from src.retrieval import (
    PathScorer,
    PathSelector,
    AgentStateMachine
)


async def initialize_resources(
    graph_path: str,
    config_path: str,
    index_dir: Optional[str] = None,
    persistence_dir: Optional[str] = None,
) -> tuple:
    """
    一次性初始化所有资源（仅加载一次）

    参考 ggagent3 的实现模式，在批量处理前集中初始化所有共享资源，
    然后在并发任务中传递使用，避免重复加载。

    Args:
        graph_path: 图文件路径
        config_path: 配置文件路径
        index_dir: 预构建索引目录（可选）
        persistence_dir: 持久化数据目录（可选）

    Returns:
        (graph, llm_client, embedding_client, agent_machine)
    """
    print("=" * 60)
    print("初始化共享资源（仅加载一次）")
    print("=" * 60)

    # 1. 加载配置
    print("加载配置...")
    model_config = get_model_config()
    retrieval_config = get_retrieval_config()
    print("✓ 配置已加载")

    # 2. 加载图（一次性）
    print(f"加载图: {graph_path}")
    import networkx as nx
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)
    print(f"✓ 图已加载: {graph.number_of_nodes()} 个节点")

    # 3. 初始化 LLM 客户端（一次性）
    print("初始化 LLM 客户端...")
    llm_client = OpenAIClient(
        base_url=model_config.llm.base_url,
        api_key=model_config.llm.api_key,
        model=model_config.llm.model,
        temperature=0.1,
        max_tokens=model_config.llm.max_tokens,
        max_retries=model_config.llm.max_retries,
        timeout=model_config.llm.timeout,
        concurrency=model_config.llm.concurrency,
    )
    print("✓ LLM 客户端已初始化")

    # 4. 初始化 Embedding 客户端（一次性，使用 OpenAI API）
    print("初始化 Embedding 客户端（OpenAI API）...")
    from src.llm import OpenAIEmbeddingClient
    embedding_client = OpenAIEmbeddingClient(
        base_url=model_config.embedding.base_url,
        api_key=model_config.embedding.api_key,
        model=model_config.embedding.model,
        max_retries=model_config.embedding.max_retries,
        timeout=model_config.embedding.timeout,
        concurrency=model_config.embedding.concurrency,
    )
    print("✓ Embedding 客户端已初始化（OpenAI API）")

    # 5. 初始化 Embedding 缓存管理器（仅内存缓存，无持久化）
    print("初始化 Embedding 缓存管理器...")
    from src.retrieval.embedding_cache_manager import EmbeddingCacheManager

    cache_manager = EmbeddingCacheManager(
        embedding_client=embedding_client,
        batch_size=50,
        max_cache_size=50000,  # 增大内存缓存
    )
    print("✓ Embedding 缓存管理器已初始化（仅内存缓存）")

    # 6. 加载持久化数据（如果可用）
    adjacency_cache = None
    if persistence_dir:
        cache_file = Path(persistence_dir) / "adjacency_cache.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    adjacency_cache = pickle.load(f)
                print(f"✓ 邻接缓存已加载: {len(adjacency_cache)} 个节点")
            except Exception as e:
                print(f"警告: 邻接缓存加载失败 ({e})")

    # 7. 初始化检索组件（一次性）
    print("初始化检索组件...")

    path_scorer = PathScorer(
        graph=graph,
        embedding_client=embedding_client,
        semantic_weight=retrieval_config.semantic_weight,
        bridge_weight=retrieval_config.bridge_weight,
        index_dir=index_dir,
        embedding_cache_manager=cache_manager,
        adjacency_cache=adjacency_cache,
        persistence_dir=persistence_dir,
    )

    path_selector = PathSelector(graph=graph)

    # 创建 AgentStateMachine（内部会创建 BeamSearch）
    agent_machine = AgentStateMachine(
        graph=graph,
        llm=llm_client,
        embedding_client=embedding_client,
        path_scorer=path_scorer,
        path_selector=path_selector,
        max_rounds=retrieval_config.max_rounds,
        semantic_weight=retrieval_config.semantic_weight,
        bridge_weight=retrieval_config.bridge_weight,
        map_beam_width=retrieval_config.beam_width,
        map_max_iterations=retrieval_config.max_path_depth,
        require_vector_index=False,
        embedding_cache_manager=cache_manager,
        persistence_dir=persistence_dir,
        index_dir=index_dir,
    )

    print("✓ 所有检索组件已初始化（使用 AgentStateMachine）")
    print("=" * 60)

    return (graph, llm_client, embedding_client, agent_machine)


async def process_single_question(
    question: str,
    graph,
    embedding_client,
    agent_machine,
    index_dir: Optional[str] = None,
) -> dict:
    """
    处理单个问题（使用 AgentStateMachine）

    使用新的 6 状态 AgentStateMachine 进行多跳问答。

    Args:
        question: 问题文本
        graph: 预加载的图对象
        embedding_client: 预初始化的 Embedding 客户端
        agent_machine: 预初始化的 AgentStateMachine
        index_dir: 预构建索引目录（可选）

    Returns:
        AgentResult（包含 answer / short_answer 等）
    """
    # 【性能优化】关闭 debug 和 verbose 以减少 I/O 阻塞
    result = await agent_machine.run(question=question, debug=False, verbose=False)

    return result


async def run_batch(
    dataset_file: str,
    graph_path: str,
    config_path: str,
    output_path: str,
    max_samples: Optional[int] = None,
    index_dir: Optional[str] = None,
    persistence_dir: Optional[str] = None,
    concurrency: int = 50,
):
    """
    批量处理问题（资源复用优化）

    参考 ggagent3 的实现，一次性初始化所有资源，
    然后在并发任务中共享使用，大幅降低内存和启动时间。

    Args:
        dataset_file: 数据集文件路径
        graph_path: 图文件路径
        config_path: 配置文件路径
        output_path: 输出文件路径
        max_samples: 最大处理样本数
        index_dir: 预构建索引目录（可选）
        concurrency: 并发数量（默认50）
    """
    print("=" * 60)
    print("ProGraph 批量问答（并发模式 - 资源复用）")
    print("=" * 60)

    # 【关键优化】一次性初始化所有资源
    (graph, llm_client, embedding_client, agent_machine) = await initialize_resources(
        graph_path=graph_path,
        config_path=config_path,
        index_dir=index_dir,
        persistence_dir=persistence_dir
    )

    # 加载数据集
    print(f"\n加载数据集: {dataset_file}")
    with open(dataset_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"总样本数: {len(data)}")

    if max_samples:
        data = data[:max_samples]
        print(f"限制处理: {max_samples} 个样本")

    print(f"并发度: {concurrency}")

    # 创建信号量控制并发
    sem = asyncio.Semaphore(concurrency)

    async def process_item(idx: int, item: dict) -> dict:
        """处理单个数据项（使用共享资源）"""
        async with sem:
            question = item.get("question", "")
            if not question:
                return None

            try:
                # 传递共享资源（使用 agent_machine）
                result = await process_single_question(
                    question=question,
                    graph=graph,
                    embedding_client=embedding_client,
                    agent_machine=agent_machine,
                    index_dir=index_dir,
                )

                return {
                    "question": question,
                    "answer": result.answer,
                    "short_answer": result.short_answer,
                    "reference_answer": item.get("answer", ""),
                }

            except Exception as e:
                print(f"[{idx + 1}] 处理失败: {e}")
                return {
                    "question": question,
                    "answer": f"错误: {e}",
                    "short_answer": "",
                    "reference_answer": item.get("answer", ""),
                }

    # 并发处理所有问题
    tasks = [process_item(idx, item) for idx, item in enumerate(data)]
    results_raw = await asyncio.gather(*tasks, return_exceptions=True)

    # 过滤 None 和异常
    results = []
    for r in results_raw:
        if r is None:
            continue
        if isinstance(r, Exception):
            print(f"任务异常: {r}")
            continue
        results.append(r)

    # 保存结果
    print(f"\n保存结果到: {output_path}")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"完成！处理 {len(results)}/{len(data)} 个问题")

    # 【新增】清理资源
    await llm_client.close()
    await embedding_client.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="ProGraph 多跳问答脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单问题模式
  python scripts/4-run_multi_hop_qa.py \\
    --question "What is the capital of France?" \\
    --graph output/HotpotQA/proposition_graph/linked_graph.pkl \\
    --config config.yaml

  # 批量模式
  python scripts/4-run_multi_hop_qa.py \\
    --dataset dataset/HotpotQA/train_data.json \\
    --graph output/HotpotQA/proposition_graph/linked_graph.pkl \\
    --output output/HotpotQA/result \\
    --config config.yaml
        """
    )

    # 单问题模式
    parser.add_argument(
        '--question',
        type=str,
        help='问题文本'
    )

    # 批量模式
    parser.add_argument(
        '--dataset',
        type=str,
        help='数据集文件路径'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='输出文件路径（批量模式）'
    )

    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='最大处理样本数（用于测试）'
    )

    parser.add_argument(
        '--concurrency',
        type=int,
        default=50,
        help='并发处理数量（默认50）'
    )

    # 公共参数
    parser.add_argument(
        '--graph',
        type=str,
        required=True,
        help='图文件路径 (.pkl)'
    )

    parser.add_argument(
        '--index-dir',
        type=str,
        default=None,
        help='预构建索引目录 (可选，用于加速检索)'
    )

    parser.add_argument(
        '--persistence-dir',
        type=str,
        default=None,
        help='持久化数据目录 (可选，用于加速初始化)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='配置文件路径 (默认: config.yaml)'
    )

    args = parser.parse_args()

    # 加载配置
    from src.config.model_config import ModelConfig
    from src.config.retrieval_config import RetrievalConfig

    model_config = ModelConfig.from_yaml(args.config)
    set_model_config(model_config)

    retrieval_config = RetrievalConfig.from_yaml(args.config)
    set_retrieval_config(retrieval_config)

    # 运行
    if args.question:
        # 单问题模式：先初始化资源，再处理
        async def run_single():
            # 初始化资源
            (graph, llm_client, embedding_client, agent_machine) = await initialize_resources(
                graph_path=args.graph,
                config_path=args.config,
                index_dir=args.index_dir,
                persistence_dir=args.persistence_dir
            )

            # 处理问题
            print("=" * 60)
            print("ProGraph 多跳问答（使用 AgentStateMachine）")
            print("=" * 60)
            print(f"问题: {args.question}")
            print()

            result = await agent_machine.run(question=args.question)

            # 输出答案
            print("\n" + "=" * 60)
            print("答案")
            print("=" * 60)
            print(result.answer)
            print("\n" + "=" * 60)
            print("简短答案（EM/F1）")
            print("=" * 60)
            print(result.short_answer)
            print(f"\n置信度: {result.confidence:.2f}")
            print(f"终止原因: {result.termination_reason}")

            # 清理资源
            await llm_client.close()
            await embedding_client.close()

        asyncio.run(run_single())

    elif args.dataset:
        # 批量模式
        if not args.output:
            print("错误：批量模式需要指定 --output 参数")
            return

        asyncio.run(run_batch(
            dataset_file=args.dataset,
            graph_path=args.graph,
            config_path=args.config,
            output_path=args.output,
            max_samples=args.max_samples,
            index_dir=args.index_dir,
            persistence_dir=args.persistence_dir,
            concurrency=args.concurrency,
        ))
    else:
        print("错误：需要指定 --question 或 --dataset 参数")
        parser.print_help()


if __name__ == '__main__':
    main()
