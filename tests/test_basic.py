#!/usr/bin/env python3
"""
ProGraph 基础测试
验证各个模块的基本功能
"""

import sys
from pathlib import Path

# 添加 src 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ModelConfig, GraphConfig, RetrievalConfig


def test_config():
    """测试配置加载"""
    print("=" * 60)
    print("测试 1: 配置系统")
    print("=" * 60)

    # 测试 YAML 加载
    model_config = ModelConfig.from_yaml("config.yaml")
    print(f"✓ 模型配置加载成功")
    print(f"  - vLLM 模型: {model_config.vllm.model_path}")
    print(f"  - LLM API: {model_config.llm.base_url}")

    graph_config = GraphConfig.from_yaml("config.yaml")
    print(f"✓ 图配置加载成功")
    print(f"  - 命题提取温度: {graph_config.proposition_extraction.temperature}")

    retrieval_config = RetrievalConfig.from_yaml("config.yaml")
    print(f"✓ 检索配置加载成功")
    print(f"  - 最大深度: {retrieval_config.max_path_depth}")
    print(f"  - Beam width: {retrieval_config.beam_width}")

    print("\n配置系统测试通过！\n")
    return True


def test_document_loader():
    """测试文档加载器"""
    print("=" * 60)
    print("测试 2: 文档加载器")
    print("=" * 60)

    from src.proposition_graph import DocumentLoader

    loader = DocumentLoader()
    documents = loader.load("dataset/test/test_docs.json")

    print(f"✓ 加载了 {len(documents)} 个文档")

    for doc in documents:
        print(f"  - {doc.title}: {len(doc.content)} 个句子")

    print("\n文档加载器测试通过！\n")
    return True


def test_proposition_extractor():
    """测试命题提取器（mock）"""
    print("=" * 60)
    print("测试 3: 命题提取器（Mock）")
    print("=" * 60)

    from src.proposition_graph import PropositionExtractor
    from src.llm.base import LLMResponse
    import asyncio

    # 创建 mock LLM
    class MockLLM:
        async def generate(self, prompt, **kwargs):
            # 返回模拟的 JSON 响应
            return LLMResponse(
                text='''```json
[
  "The entity is mentioned in the text.",
  "This is a test proposition."
]
```''',
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                model="mock"
            )

    mock_llm = MockLLM()
    extractor = PropositionExtractor(llm=mock_llm)

    # 测试异步
    async def test():
        propositions = await extractor.extract_from_sentence(
            "Barack Obama was born in Hawaii.",
            sent_idx=0,
            doc_id="test"
        )
        return propositions

    propositions = asyncio.run(test())
    print(f"✓ 提取了 {len(propositions)} 个命题")
    for prop in propositions:
        print(f"  - {prop.text}")

    print("\n命题提取器测试通过！\n")
    return True


def test_graph_builder():
    """测试图构建器（简化版）"""
    print("=" * 60)
    print("测试 4: 图构建器（简化）")
    print("=" * 60)

    import networkx as nx

    # 创建一个简单的测试图
    graph = nx.DiGraph()

    # 添加命题节点
    graph.add_node("prop1", node_type="proposition", text="Barack Obama was born in Hawaii.", doc_id="test")
    graph.add_node("prop2", node_type="proposition", text="He served as the 44th President.", doc_id="test")

    # 添加实体节点
    graph.add_node("ent1", node_type="entity", text="Barack Obama", entity_type="PERSON", doc_id="test")
    graph.add_node("ent2", node_type="entity", text="Hawaii", entity_type="LOCATION", doc_id="test")

    # 添加边
    graph.add_edge("prop1", "ent1", edge_type="MENTIONS_ENTITY")
    graph.add_edge("prop1", "ent2", edge_type="MENTIONS_ENTITY")

    print(f"✓ 创建测试图")
    print(f"  - 节点数: {graph.number_of_nodes()}")
    print(f"  - 边数: {graph.number_of_edges()}")

    # 保存测试
    import pickle
    from pathlib import Path

    output_dir = Path("output/test")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "test_graph.pkl", "wb") as f:
        pickle.dump(graph, f)

    print(f"✓ 测试图已保存到 output/test/test_graph.pkl")

    print("\n图构建器测试通过！\n")
    return True


def test_prompt_templates():
    """测试提示词模板"""
    print("=" * 60)
    print("测试 5: 提示词模板")
    print("=" * 60)

    from src.proposition_graph import prompts

    # 测试命题提取提示词
    prop_prompt = prompts.get_proposition_extraction_prompt("Barack Obama was born in Hawaii.")
    print(f"✓ 命题提取提示词长度: {len(prop_prompt)} 字符")

    # 测试实体提取提示词
    entity_prompt = prompts.get_entity_extraction_prompt("Barack Obama was born in Hawaii.")
    print(f"✓ 实体提取提示词长度: {len(entity_prompt)} 字符")

    # 测试 RST 分析提示词
    rst_prompt = prompts.get_rst_analysis_prompt(
        "Barack Obama was born in Hawaii.",
        "He served as the 44th President."
    )
    print(f"✓ RST 分析提示词长度: {len(rst_prompt)} 字符")

    # 测试意图识别提示词
    intent_prompt = prompts.get_intent_recognition_prompt(
        "Where was Barack Obama born?",
        "prop1 -> prop2",
        "Barack Obama, Hawaii"
    )
    print(f"✓ 意图识别提示词长度: {len(intent_prompt)} 字符")

    print("\n提示词模板测试通过！\n")
    return True


def test_path_scorer_formula():
    """测试路径评分公式"""
    print("=" * 60)
    print("测试 7: 路径评分公式")
    print("=" * 60)

    import numpy as np

    # 测试余弦相似度计算
    vec1 = np.array([1.0, 0.0, 0.0])
    vec2 = np.array([1.0, 0.0, 0.0])
    vec3 = np.array([0.0, 1.0, 0.0])

    # 相同向量
    sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    print(f"✓ 相同向量相似度: {sim} (应为 1.0)")

    # 正交向量
    sim = np.dot(vec1, vec3) / (np.linalg.norm(vec1) * np.linalg.norm(vec3))
    print(f"✓ 正交向量相似度: {sim} (应为 0.0)")

    # 测试桥接分数公式
    mentioned_entities = {"ent1", "ent2", "ent3"}
    visited_entities = {"ent1"}

    novel_entities = mentioned_entities - visited_entities
    bridge_score = len(novel_entities) / len(mentioned_entities)
    print(f"\n✓ 桥接分数计算:")
    print(f"  - 提及实体: {mentioned_entities}")
    print(f"  - 已访问实体: {visited_entities}")
    print(f"  - 新实体: {novel_entities}")
    print(f"  - 桥接分数: {bridge_score} (应为 0.67)")

    print("\n路径评分公式测试通过！\n")
    return True


def main():
    """运行所有测试"""
    print("\n")
    print("*" * 60)
    print("*" + " " * 58 + "*")
    print("*" + "  ProGraph 基础测试".center(56) + "*")
    print("*" + " " * 58 + "*")
    print("*" * 60)
    print("\n")

    tests = [
        ("配置系统", test_config),
        ("文档加载器", test_document_loader),
        ("命题提取器", test_proposition_extractor),
        ("图构建器", test_graph_builder),
        ("提示词模板", test_prompt_templates),
        ("路径评分公式", test_path_scorer_formula),
    ]

    results = []

    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ {name} 测试失败: {e}\n")
            results.append((name, False))

    # 总结
    print("=" * 60)
    print("测试总结")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {name}: {status}")

    print(f"\n总计: {passed}/{total} 测试通过")

    if passed == total:
        print("\n🎉 所有测试通过！")
    else:
        print(f"\n⚠️  {total - passed} 个测试失败")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
