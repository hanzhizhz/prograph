#!/usr/bin/env python3
"""
ProGraph 实体链接第一阶段：生成候选对
仅加载向量模型进行向量化，生成候选实体组和命题对
"""

import asyncio
import sys
import os
import pickle
from pathlib import Path
import argparse

# 添加 src 路径到 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm import VLLMEmbeddingClient
from src.config import get_model_config, set_model_config
from src.config.retrieval_config import EntityLinkingConfig
from src.entity_linking import (
    build_indices,
    CandidateGenerator,
    EntityCandidateGroup,
)


async def generate_candidates(
    graph_path: str,
    output_path: str,
    config_path: str = "config.yaml",
):
    """
    生成候选对的主流程

    Args:
        graph_path: 输入图文件路径
        output_path: 输出目录路径
        config_path: 配置文件路径
    """
    print("=" * 60)
    print("ProGraph 实体链接第一阶段：生成候选对")
    print("=" * 60)
    print(f"输入图: {graph_path}")
    print(f"输出目录: {output_path}")
    print()

    # 1. 加载配置
    print("加载配置...")
    model_config = get_model_config()

    # 加载实体链接配置
    from src.config.config_loader import load_yaml
    config_dict = load_yaml(config_path)
    linking_config = EntityLinkingConfig.from_dict(config_dict)

    # 2. 加载图
    print("\n加载图...")
    with open(graph_path, 'rb') as f:
        import networkx as nx
        graph = nx.DiGraph()
        graph = pickle.load(f)

    print(f"图已加载: {graph.number_of_nodes()} 个节点, {graph.number_of_edges()} 条边")

    # 3. 初始化嵌入客户端（离线 vLLM）
    print("\n初始化嵌入客户端（vLLM 离线）...")
    embedding_client = VLLMEmbeddingClient(
        model_path=model_config.vllm_embedding.model_path,
        tensor_parallel_size=model_config.vllm_embedding.tensor_parallel_size,
        gpu_memory_utilization=model_config.vllm_embedding.gpu_memory_utilization,
        trust_remote_code=model_config.vllm_embedding.trust_remote_code,
        max_model_len=model_config.vllm_embedding.max_model_len,
    )

    # 4. 构建嵌入索引
    print("\n构建嵌入索引...")
    proposition_index, entity_index = await build_indices(graph, embedding_client)

    # 5. 生成候选对/组
    print("\n生成候选对/组...")
    candidate_generator = CandidateGenerator(
        embedding_client=embedding_client,
        similarity_threshold=linking_config.similarity_threshold,
        vector_top_k=linking_config.vector_top_k,
        entity_similarity_threshold=linking_config.entity_similarity_threshold,
        proposition_similarity_threshold=linking_config.proposition_similarity_threshold,
        entity_fusion_group_size=linking_config.entity_fusion_group_size,
        batch_search_size=linking_config.batch_search_size,
    )

    prop_candidates, auto_fuse_groups, llm_groups = await candidate_generator.generate_candidates(
        graph=graph,
        proposition_index=proposition_index,
        entity_index=entity_index,
    )

    print(f"\n生成候选完成:")
    print(f"  - 自动融合组: {len(auto_fuse_groups)}")
    print(f"  - LLM判断组: {len(llm_groups)}")
    print(f"  - 命题候选对: {len(prop_candidates)}")

    # 6. 保存候选对/组
    print(f"\n保存候选到 {output_path}...")
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)

    # 保存自动融合组
    auto_fuse_groups_path = output / "auto_fuse_groups.pkl"
    with open(auto_fuse_groups_path, 'wb') as f:
        pickle.dump(auto_fuse_groups, f)
    print(f"  - 自动融合组已保存: {auto_fuse_groups_path}")

    # 保存 LLM 判断组
    llm_groups_path = output / "llm_groups.pkl"
    with open(llm_groups_path, 'wb') as f:
        pickle.dump(llm_groups, f)
    print(f"  - LLM判断组已保存: {llm_groups_path}")

    # 保存命题候选对
    prop_candidates_path = output / "proposition_candidates.pkl"
    with open(prop_candidates_path, 'wb') as f:
        pickle.dump(prop_candidates, f)
    print(f"  - 命题候选对已保存: {prop_candidates_path}")

    print("\n" + "=" * 60)
    print("第一阶段完成！")
    print("=" * 60)
    print("\n接下来运行第二阶段进行链接和融合:")
    print(f"python scripts/2b-link_and_fuse.py \\")
    print(f"    --graph {graph_path} \\")
    print(f"    --temp_dir {output_path} \\")
    print(f"    --output <输出路径> \\")
    print(f"    --config {config_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="ProGraph 实体链接第一阶段：生成候选对",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 生成候选对
  python scripts/2a-generate_candidates.py \\
    --graph output/HotpotQA/proposition_graph/raw_graph.pkl \\
    --output output/HotpotQA/temp \\
    --config config.yaml
        """
    )

    parser.add_argument(
        '--graph',
        type=str,
        required=True,
        help='输入图文件路径 (.pkl)'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='输出目录路径'
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

    model_config = ModelConfig.from_yaml(args.config)
    set_model_config(model_config)

    # 运行
    try:
        asyncio.run(generate_candidates(
            graph_path=args.graph,
            output_path=args.output,
            config_path=args.config,
        ))
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)

    # 显式退出，避免C扩展库清理问题导致的错误退出码
    sys.exit(0)


if __name__ == '__main__':
    main()
