#!/usr/bin/env python3
"""
测试简化的错误恢复机制（适配离线推理）
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock

# 添加 src 路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.json_parser import retry_parse_with_llm


async def test_retry_parse():
    """测试retry_parse_with_llm"""
    print("=" * 60)
    print("测试简化错误恢复机制（适配离线推理）")
    print("=" * 60)

    # 创建模拟LLM客户端
    mock_llm = AsyncMock()

    # 测试场景1: 正常解析成功
    print("\n场景1: 全部正常解析")
    print("-" * 40)

    import json

    def parse_func(text):
        return json.loads(text)

    mock_llm.generate = AsyncMock(return_value=type('obj', (object,), {'text': '{"result": "success"}'})())

    prompts = ["test prompt"]
    responses = ['{"result": "success"}']
    parse_funcs = [parse_func]

    results, failed_indices = await retry_parse_with_llm(
        llm_client=mock_llm,
        prompts=prompts,
        responses=responses,
        parse_funcs=parse_funcs
    )

    print(f"✓ 解析成功: {results[0]}")
    print(f"✓ 失败列表: {failed_indices}")
    assert results[0] == {"result": "success"}
    assert len(failed_indices) == 0

    # 测试场景2: 部分解析失败并重试成功
    print("\n场景2: 部分解析失败并重试")
    print("-" * 40)

    call_count = 0
    import json

    def parse_func_with_fail(text):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:  # 前两个会失败
            raise ValueError("JSON 解析失败")
        return json.loads(text)

    mock_llm.generate = AsyncMock(return_value=type('obj', (object,), {'text': '{"result": "retry success"}'})())

    prompts = ["test prompt 1", "test prompt 2", "test prompt 3"]
    responses = [
        '{"invalid": json}',  # 失败
        '{"invalid": json}',  # 失败
        '{"valid": true}'     # 成功
    ]
    parse_funcs = [parse_func_with_fail, parse_func_with_fail, parse_func]

    results, failed_indices = await retry_parse_with_llm(
        llm_client=mock_llm,
        prompts=prompts,
        responses=responses,
        parse_funcs=parse_funcs
    )

    print(f"✓ 解析结果: {results}")
    print(f"✓ 失败列表: {failed_indices}")
    assert len(results) == 3
    assert len(failed_indices) == 0  # 重试后都应该成功

    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)


async def main():
    """主函数"""
    try:
        await test_retry_parse()
        print("\n🎉 简化错误恢复机制测试通过！")
        return 0
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
