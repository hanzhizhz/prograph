#!/usr/bin/env python3
"""
测试 timing 功能
"""

import sys
import time
from pathlib import Path

# 添加 src 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.timing import TimingLogger, TimingContext


def test_timing_context():
    """测试 TimingContext 基本功能"""
    print("=" * 60)
    print("测试 1: TimingContext 基本功能")
    print("=" * 60)

    with TimingContext("Test operation", enabled=True):
        time.sleep(0.1)

    print("✓ TimingContext 测试通过\n")


def test_timing_logger():
    """测试 TimingLogger"""
    print("=" * 60)
    print("测试 2: TimingLogger")
    print("=" * 60)

    logger = TimingLogger(enabled=True)

    with logger.time("Operation 1"):
        time.sleep(0.05)
        with logger.time("  Sub-operation 1.1"):
            time.sleep(0.03)
        with logger.time("  Sub-operation 1.2"):
            time.sleep(0.02)

    with logger.time("Operation 2"):
        time.sleep(0.08)

    print("✓ TimingLogger 测试通过\n")


def test_timing_disabled():
    """测试禁用 timing"""
    print("=" * 60)
    print("测试 3: 禁用 Timing")
    print("=" * 60)

    logger = TimingLogger(enabled=False)

    print("(应该看不到任何 timing 输出)")
    with logger.time("This should not print"):
        time.sleep(0.05)

    print("✓ 禁用 Timing 测试通过\n")


def test_hierarchical_timing():
    """测试层级 timing"""
    print("=" * 60)
    print("测试 4: 层级 Timing")
    print("=" * 60)

    logger = TimingLogger(enabled=True)

    with logger.time("Level 1"):
        time.sleep(0.02)
        with logger.time("  Level 2"):
            time.sleep(0.02)
            with logger.time("    Level 3"):
                time.sleep(0.02)

    print("✓ 层级 Timing 测试通过\n")


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("ProGraph Timing 功能测试")
    print("=" * 60 + "\n")

    tests = [
        ("TimingContext 基本功能", test_timing_context),
        ("TimingLogger", test_timing_logger),
        ("禁用 Timing", test_timing_disabled),
        ("层级 Timing", test_hierarchical_timing),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {name} 测试失败: {e}\n")
            failed += 1

    print("=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"通过: {passed}/{len(tests)}")
    print(f"失败: {failed}/{len(tests)}")

    if failed == 0:
        print("\n✅ 所有测试通过！")
        return 0
    else:
        print(f"\n❌ {failed} 个测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
