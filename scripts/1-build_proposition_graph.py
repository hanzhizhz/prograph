#!/usr/bin/env python3
"""
ProGraph 离线图构建脚本
从原始文档构建异构命题图（使用统一文档提取）
"""

import asyncio
import sys
import os
from pathlib import Path
from typing import Optional
import argparse

# 添加 src 路径到 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm import VLLMClient
from src.config import get_model_config, set_model_config
from src.proposition_graph import (
    DocumentLoader,
    UnifiedDocumentExtractor,
    GraphBuilder,
)


async def build_proposition_graph(
    dataset_path: str,
    output_path: str,
    max_documents: Optional[int] = None,
):
    """
    构建 ProGraph 命题图的主流程

    Args:
        dataset_path: 数据集文件路径
        output_path: 输出路径（不含扩展名）
        max_documents: 最大处理文档数（用于测试）
    """
    print("=" * 60)
    print("ProGraph 离线图构建（统一提取模式）")
    print("=" * 60)
    print(f"数据集路径: {dataset_path}")
    print(f"输出路径: {output_path}")
    if max_documents:
        print(f"最大文档数: {max_documents}")
    print()

    # 1. 加载配置
    print("加载配置...")
    model_config = get_model_config()

    # 2. 初始化 vLLM 客户端
    print("初始化 vLLM 客户端...")
    vllm_kwargs = {
        "model_path": model_config.vllm.model_path,
        "tensor_parallel_size": model_config.vllm.tensor_parallel_size,
        "gpu_memory_utilization": model_config.vllm.gpu_memory_utilization,
        "trust_remote_code": model_config.vllm.trust_remote_code,
        "max_model_len": model_config.vllm.max_model_len,
    }
    vllm_client = VLLMClient(**vllm_kwargs)

    # 3. 初始化统一提取器
    print("初始化统一文档提取器...")
    unified_extractor = UnifiedDocumentExtractor(
        llm=vllm_client,
        temperature=0.1,
        max_tokens=4096,
    )

    # 4. 初始化图构建器
    graph_builder = GraphBuilder(
        unified_extractor=unified_extractor,
    )

    # 5. 加载文档
    print("\n加载文档...")
    doc_loader = DocumentLoader()
    documents = doc_loader.load(dataset_path)
    print(f"加载了 {len(documents)} 个文档")

    if max_documents:
        documents = documents[:max_documents]
        print(f"限制处理 {max_documents} 个文档")

    # 6. 批量处理文档并构建图
    print(f"\n开始处理 {len(documents)} 个文档...")
    stats = await graph_builder.add_documents_batch(documents)
    print(f"成功处理: {stats['success']}/{len(documents)} 个文档")

    # 7. 保存图
    print("\n保存图...")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    graph_builder.save_graph(str(output), format='both')

    # 8. 导出 meta 产物
    print("\n导出 meta 产物...")
    graph_builder.export_meta(output.parent)

    # 9. 输出统计信息
    stats = graph_builder.get_statistics()
    print("\n" + "=" * 60)
    print("图统计信息")
    print("=" * 60)
    print(f"总节点数: {stats.total_nodes}")
    print(f"  - 命题节点: {stats.proposition_nodes}")
    print(f"  - 实体节点: {stats.entity_nodes}")
    print(f"总边数: {stats.total_edges}")
    print(f"  - RST 关系边: {stats.proposition_edges}")
    print(f"  - 包含边 (命题->实体): {stats.mention_edges}")
    print(f"孤立节点数: {stats.isolated_nodes}")

    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="ProGraph 离线图构建脚本（统一提取模式）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 处理 HotpotQA 数据集
  python scripts/1-build_proposition_graph.py \\
    --dataset dataset/HotpotQA/full_docs.json \\
    --output output/HotpotQA/proposition_graph/raw_graph \\
    --config config.yaml

  # 测试模式（只处理 5 个文档）
  python scripts/1-build_proposition_graph.py \\
    --dataset dataset/test/test_docs.json \\
    --output output/test/graph \\
    --max-documents 5
        """
    )

    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='数据集文件路径 (full_docs.json)'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='输出路径（不含扩展名，会自动添加 .pkl 和 .json）'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='配置文件路径 (默认: config.yaml)'
    )

    parser.add_argument(
        '--max-documents',
        type=int,
        default=None,
        help='最大处理文档数（用于测试）'
    )

    args = parser.parse_args()

    # 加载配置
    from src.config.model_config import ModelConfig

    model_config = ModelConfig.from_yaml(args.config)
    set_model_config(model_config)

    # 运行
    asyncio.run(build_proposition_graph(
        dataset_path=args.dataset,
        output_path=args.output,
        max_documents=args.max_documents,
    ))


if __name__ == '__main__':
    main()
