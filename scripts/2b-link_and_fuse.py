#!/usr/bin/env python3
"""
ProGraph 实体链接第二阶段：链接实体与图融合
仅加载大模型对候选实体进行判断，然后进行图融合
"""

import asyncio
import sys
import os
import pickle
from pathlib import Path
import argparse

# 添加 src 路径到 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm import VLLMClient
from src.config import get_model_config, set_model_config
from src.config.retrieval_config import EntityLinkingConfig
from src.entity_linking import (
    EntityLinker,
    EntityCandidateGroup,
    GraphFusion,
    EntityFusionResult,
    FusedEntity,
)


async def link_and_fuse(
    graph_path: str,
    temp_dir: str,
    output_path: str,
    config_path: str = "config.yaml",
):
    """
    链接实体与图融合的主流程

    Args:
        graph_path: 输入图文件路径
        temp_dir: 临时目录路径（包含候选对文件）
        output_path: 输出路径
        config_path: 配置文件路径
    """
    print("=" * 60)
    print("ProGraph 实体链接第二阶段：链接实体与图融合")
    print("=" * 60)
    print(f"输入图: {graph_path}")
    print(f"候选对目录: {temp_dir}")
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

    # 3. 加载候选对/组
    print(f"\n加载候选对/组...")
    temp_path = Path(temp_dir)

    # 加载分离格式（自动融合组 + LLM判断组）
    auto_fuse_groups_path = temp_path / "auto_fuse_groups.pkl"
    llm_groups_path = temp_path / "llm_groups.pkl"

    auto_fuse_groups = []
    llm_groups = []

    if auto_fuse_groups_path.exists():
        with open(auto_fuse_groups_path, 'rb') as f:
            auto_fuse_groups = pickle.load(f)
        print(f"  - 自动融合组: {len(auto_fuse_groups)} 组")
    else:
        print("  警告: 未找到自动融合组文件")

    if llm_groups_path.exists():
        with open(llm_groups_path, 'rb') as f:
            llm_groups = pickle.load(f)
        print(f"  - LLM判断组: {len(llm_groups)} 组")
    else:
        print("  警告: 未找到LLM判断组文件")

    prop_candidates_path = temp_path / "proposition_candidates.pkl"
    proposition_decisions = []
    if prop_candidates_path.exists():
        with open(prop_candidates_path, 'rb') as f:
            prop_candidates = pickle.load(f)
        print(f"  - 命题候选对: {len(prop_candidates)} 对")
    else:
        print("  - 未找到命题候选文件")
        prop_candidates = []

    # 4. 初始化链接器（离线 vLLM）
    print("\n初始化实体链接器（vLLM 离线）...")
    llm_client = VLLMClient(
        model_path=model_config.vllm.model_path,
        tensor_parallel_size=model_config.vllm.tensor_parallel_size,
        gpu_memory_utilization=model_config.vllm.gpu_memory_utilization,
        trust_remote_code=model_config.vllm.trust_remote_code,
        max_model_len=model_config.vllm.max_model_len,
    )

    entity_linker = EntityLinker(llm=llm_client)

    # 5. 融合实体
    all_fusion_results = []

    # 5.1 处理自动融合组（不调用LLM）
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

    # 5.2 处理LLM判断组
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

    # 6. 链接命题
    if prop_candidates:
        print(f"\n链接 {len(prop_candidates)} 个命题候选对...")
        proposition_decisions = await entity_linker.link_propositions(prop_candidates)

        # 统计链接结果
        link_count = sum(1 for d in proposition_decisions if d.should_link)
        print(f"命题链接完成: {link_count}/{len(proposition_decisions)} 对链接")

    # 6. 图融合
    print("\n开始图融合...")
    fusion = GraphFusion(graph)
    fused_graph = fusion.fuse(
        entity_fusion_results=all_fusion_results,
        proposition_decisions=proposition_decisions if proposition_decisions else None,
    )

    # 8. 保存融合后的图
    print("\n保存融合后的图...")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fusion.save_graph(str(output), format='both')

    # 9. 输出统计信息
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
        description="ProGraph 实体链接第二阶段：链接实体与图融合",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 链接实体并融合图
  python scripts/2b-link_and_fuse.py \\
    --graph output/HotpotQA/proposition_graph/raw_graph.pkl \\
    --temp_dir output/HotpotQA/temp \\
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
        '--temp_dir',
        type=str,
        required=True,
        help='临时目录路径（包含候选对文件）'
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
        temp_dir=args.temp_dir,
        output_path=args.output,
        config_path=args.config,
    ))


if __name__ == '__main__':
    main()
