#!/usr/bin/env python3
"""
ProGraph 实体链接与图融合脚本
跨文档实体对齐并建立全局连接
"""

import asyncio
import sys
import os
import pickle
from pathlib import Path
import argparse

# 添加 src 路径到 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm import VLLMClient, VLLMEmbeddingClient
from src.config import get_model_config, set_model_config
from src.config.retrieval_config import EntityLinkingConfig
from src.entity_linking import (
    build_indices,
    CandidateGenerator,
    EntityLinker,
    GraphFusion,
    EntityCandidateGroup,
    EntityFusionResult,
    FusedEntity,
)


async def link_and_fuse(
    graph_path: str,
    output_path: str,
    config_path: str = "config.yaml",
):
    """
    实体链接与图融合的主流程

    Args:
        graph_path: 输入图文件路径
        output_path: 输出路径
        config_path: 配置文件路径
    """
    print("=" * 60)
    print("ProGraph 实体链接与图融合")
    print("=" * 60)
    print(f"输入图: {graph_path}")
    print(f"输出路径: {output_path}")
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

    # 6. 初始化链接器（离线 vLLM）
    print("\n初始化实体链接器（vLLM 离线）...")
    llm_client = VLLMClient(
        model_path=model_config.vllm.model_path,
        tensor_parallel_size=model_config.vllm.tensor_parallel_size,
        gpu_memory_utilization=model_config.vllm.gpu_memory_utilization,
        trust_remote_code=model_config.vllm.trust_remote_code,
        max_model_len=model_config.vllm.max_model_len,
    )

    entity_linker = EntityLinker(llm=llm_client)

    # 7. 融合实体（自动融合 + LLM 判断）
    all_fusion_results = []

    # 7.1 自动融合组（不需要调用 LLM）
    if auto_fuse_groups:
        print(f"\n自动融合 {len(auto_fuse_groups)} 组实体（文本和类型完全一致）...")
        for group in auto_fuse_groups:
            if len(group.entity_ids) >= 2:
                result = EntityFusionResult(fused_entities=[
                    FusedEntity(
                        original_ids=group.entity_ids,
                        fused_text=group.entity_texts[0],
                        fused_type=group.entity_types[0],
                        confidence=1.0,
                        reason="文本和类型完全一致，自动融合"
                    )
                ])
                all_fusion_results.append(result)

        auto_fused_count = sum(len(r.fused_entities) for r in all_fusion_results)
        auto_entities_count = sum(
            sum(len(fe.original_ids) for fe in r.fused_entities)
            for r in all_fusion_results
        )
        print(f"  自动融合完成: {auto_fused_count} 个融合实体，共 {auto_entities_count} 个原始实体")

    # 7.2 LLM 判断组
    if llm_groups:
        print(f"\n使用 LLM 判断 {len(llm_groups)} 组实体...")
        llm_fusion_results = await entity_linker.fuse_entity_groups(llm_groups)

        llm_fused_count = sum(len(r.fused_entities) for r in llm_fusion_results)
        llm_entities_count = sum(
            sum(len(fe.original_ids) for fe in r.fused_entities)
            for r in llm_fusion_results
        )
        print(f"  LLM融合完成: {llm_fused_count} 个融合实体，共 {llm_entities_count} 个原始实体")

        all_fusion_results.extend(llm_fusion_results)

    # 8. 链接命题
    proposition_decisions = []
    if prop_candidates:
        print(f"\n链接 {len(prop_candidates)} 个命题候选对...")
        proposition_decisions = await entity_linker.link_propositions(prop_candidates)

        # 统计链接结果
        link_count = sum(1 for d in proposition_decisions if d.should_link)
        print(f"命题链接完成: {link_count}/{len(proposition_decisions)} 对链接")

    # 9. 图融合
    print("\n开始图融合...")
    fusion = GraphFusion(graph)
    fused_graph = fusion.fuse(
        entity_fusion_results=all_fusion_results,
        proposition_decisions=proposition_decisions if proposition_decisions else None,
    )

    # 10. 保存融合后的图
    print("\n保存融合后的图...")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fusion.save_graph(str(output), format='both')

    # 11. 输出统计信息
    stats = fusion.get_statistics()
    print("\n" + "=" * 60)
    print("融合后统计信息")
    print("=" * 60)
    print(f"总节点数: {stats['total_nodes']}")
    print(f"  - 命题节点: {stats['propositions']}")
    print(f"  - 全局实体节点: {stats['global_entities']}")
    print(f"  - 局部实体节点: {stats['local_entities']}")
    print(f"总边数: {stats['total_edges']}")

    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)

    # 注意：vLLM 客户端不需要手动关闭，程序退出时会自动清理


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="ProGraph 实体链接与图融合脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 处理已构建的图（命题链接默认启用）
  python scripts/2-link_entities_fuse_graph.py \\
    --graph output/HotpotQA/proposition_graph/raw_graph.pkl \\
    --output output/HotpotQA/proposition_graph/linked_graph \\
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
        help='输出路径（不含扩展名）'
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
    asyncio.run(link_and_fuse(
        graph_path=args.graph,
        output_path=args.output,
        config_path=args.config,
    ))


if __name__ == '__main__':
    main()
