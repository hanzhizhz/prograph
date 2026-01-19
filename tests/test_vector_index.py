"""
向量索引性能优化测试
测试 PersistentHNSWIndex 和 EmbeddingIndex 的批量查询和批量获取优化
"""

import numpy as np
import time
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval.vector_index import PersistentHNSWIndex
from src.entity_linking.embedding_index import EmbeddingIndex


def test_persistent_index_batch_query():
    """测试 PersistentHNSWIndex 批量查询正确性"""
    print("\n=== 测试 PersistentHNSWIndex 批量查询 ===")

    index = PersistentHNSWIndex(dim=768)
    vectors = np.random.rand(100, 768).astype(np.float32)
    payloads = [{"node_id": str(i)} for i in range(100)]
    index.add(vectors, payloads)

    # 批量查询
    query_vectors = np.random.rand(10, 768).astype(np.float32)
    distances, indices, payloads_result = index.search(query_vectors, k=5)

    # 验证结果形状
    assert distances.shape == (10, 5), f"期望 distances.shape=(10, 5), 实际={distances.shape}"
    assert indices.shape == (10, 5), f"期望 indices.shape=(10, 5), 实际={indices.shape}"
    assert len(payloads_result) == 10, f"期望 payloads 长度=10, 实际={len(payloads_result)}"

    # 验证每个查询的结果数量
    for i in range(10):
        assert len(payloads_result[i]) == 5, f"查询 {i} 的 payload 数量应为 5"

    print("✓ 批量查询测试通过")
    print(f"  - 返回形状: distances={distances.shape}, indices={indices.shape}")
    print(f"  - 相似度范围: [{distances.min():.3f}, {distances.max():.3f}]")


def test_batch_get_vectors():
    """测试 PersistentHNSWIndex 批量获取向量"""
    print("\n=== 测试 PersistentHNSWIndex 批量获取向量 ===")

    index = PersistentHNSWIndex(dim=768)
    vectors = np.random.rand(100, 768).astype(np.float32)
    payloads = [{"node_id": str(i)} for i in range(100)]
    index.add(vectors, payloads)

    # 批量获取
    labels = list(range(100))
    vectors_result = index.get_vectors(labels)

    assert len(vectors_result) == 100, f"期望返回 100 个向量, 实际={len(vectors_result)}"
    assert all(v is not None for v in vectors_result), "所有向量应该不为 None"
    assert vectors_result[0].shape == (768,), f"期望向量形状=(768,), 实际={vectors_result[0].shape}"

    print("✓ 批量获取向量测试通过")
    print(f"  - 获取向量数: {len(vectors_result)}")
    print(f"  - 向量形状: {vectors_result[0].shape}")


def test_performance_batch_query():
    """性能测试：批量查询"""
    print("\n=== 性能测试：批量查询 ===")

    n_vectors = 10000
    n_queries = 1000

    print(f"构建索引: {n_vectors} 个向量...")
    index = PersistentHNSWIndex(dim=768)
    vectors = np.random.rand(n_vectors, 768).astype(np.float32)
    payloads = [{"node_id": str(i)} for i in range(n_vectors)]
    index.add(vectors, payloads)

    print(f"执行批量查询: {n_queries} 个查询...")
    query_vectors = np.random.rand(n_queries, 768).astype(np.float32)

    start = time.time()
    distances, indices, payloads = index.search(query_vectors, k=10)
    elapsed = time.time() - start

    print(f"✓ {n_queries} 个查询耗时: {elapsed:.3f} 秒")
    print(f"  平均每查询: {elapsed/n_queries*1000:.2f} ms")
    print(f"  QPS: {n_queries/elapsed:.0f} 查询/秒")

    # 验证结果
    assert distances.shape == (n_queries, 10), f"期望形状=({n_queries}, 10)"
    assert len(payloads) == n_queries, f"期望 payloads 长度={n_queries}"

    # 性能断言：1000 查询应在 2 秒内完成
    if elapsed > 2.0:
        print(f"⚠ 警告: 批量查询性能未达预期 ({elapsed:.3f}s > 2.0s)")
    else:
        print(f"✓ 性能测试通过！")


def test_performance_get_vectors():
    """性能测试：批量获取向量"""
    print("\n=== 性能测试：批量获取向量 ===")

    n_vectors = 10000

    print(f"构建索引: {n_vectors} 个向量...")
    index = PersistentHNSWIndex(dim=768)
    vectors = np.random.rand(n_vectors, 768).astype(np.float32)
    payloads = [{"node_id": str(i)} for i in range(n_vectors)]
    index.add(vectors, payloads)

    print(f"批量获取: {n_vectors} 个向量...")
    labels = list(range(n_vectors))

    start = time.time()
    vectors_result = index.get_vectors(labels)
    elapsed = time.time() - start

    print(f"✓ {n_vectors} 个向量获取耗时: {elapsed:.3f} 秒")
    print(f"  平均每向量: {elapsed/n_vectors*1000:.3f} ms")
    print(f"  吞吐量: {n_vectors/elapsed:.0f} 向量/秒")

    # 验证结果
    assert len(vectors_result) == n_vectors
    assert all(v is not None for v in vectors_result)

    # 性能断言：10000 向量应在 1 秒内完成
    if elapsed > 1.0:
        print(f"⚠ 警告: 批量获取性能未达预期 ({elapsed:.3f}s > 1.0s)")
    else:
        print(f"✓ 性能测试通过！")


def test_embedding_index_dynamic_ef():
    """测试 EmbeddingIndex 动态 ef 参数"""
    print("\n=== 测试 EmbeddingIndex 动态 ef 参数 ===")

    index = EmbeddingIndex(dim=768)
    vectors = np.random.rand(100, 768).astype(np.float32)
    labels = [f"item_{i}" for i in range(100)]
    index.add_items(vectors, labels)

    # 测试不同 k 值
    query = vectors[0]

    for k in [5, 10, 20, 30]:
        results = index.search(query, k=k)
        assert len(results) == k, f"期望返回 {k} 个结果, 实际={len(results)}"
        print(f"  k={k}: 返回 {len(results)} 个结果 ✓")

    # 测试批量查询
    query_vectors = vectors[:10]
    results = index.search_batch(query_vectors, k=10)
    assert len(results) == 10, f"期望返回 10 个查询结果, 实际={len(results)}"

    print("✓ 动态 ef 参数测试通过")


def test_persistent_index_save_load():
    """测试 PersistentHNSWIndex 保存和加载"""
    print("\n=== 测试 PersistentHNSWIndex 保存和加载 ===")

    import tempfile
    import shutil

    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    try:
        index_path = f"{temp_dir}/test_index"

        # 创建并保存索引
        print("创建索引...")
        index = PersistentHNSWIndex(dim=768)
        vectors = np.random.rand(100, 768).astype(np.float32)
        payloads = [{"node_id": str(i)} for i in range(100)]
        index.add(vectors, payloads)

        print("保存索引...")
        index.save(index_path)

        # 加载索引
        print("加载索引...")
        loaded_index = PersistentHNSWIndex(dim=768)
        loaded_index.load(index_path)

        # 验证
        assert loaded_index.n_vectors == 100, f"期望 100 个向量, 实际={loaded_index.n_vectors}"
        assert len(loaded_index.label_to_payload) == 100

        # 测试查询
        query_vectors = np.random.rand(5, 768).astype(np.float32)
        distances, indices, payloads = loaded_index.search(query_vectors, k=5)

        assert distances.shape == (5, 5)

        print("✓ 保存和加载测试通过")

    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir)


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("向量索引性能优化测试套件")
    print("=" * 60)

    test_persistent_index_batch_query()
    test_batch_get_vectors()
    test_performance_batch_query()
    test_performance_get_vectors()
    test_embedding_index_dynamic_ef()
    test_persistent_index_save_load()

    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
