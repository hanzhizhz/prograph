#!/bin/bash
# 串行运行所有测试
# 测试已经默认串行运行（无pytest，使用自定义测试脚本）

echo "========================================"
echo "Running tests serially..."
echo "========================================"
echo ""

# 记录失败的测试
FAILED_TESTS=()

# 0. Timing tests (new)
echo "0. Running timing tests..."
echo "----------------------------------------"
python tests/test_timing.py
if [ $? -ne 0 ]; then
    echo "❌ Timing tests failed"
    FAILED_TESTS+=("test_timing.py")
else
    echo "✅ Timing tests passed"
fi
echo ""

# 1. Basic tests
echo "1. Running basic tests..."
echo "----------------------------------------"
python tests/test_basic.py
if [ $? -ne 0 ]; then
    echo "❌ Basic tests failed"
    FAILED_TESTS+=("test_basic.py")
else
    echo "✅ Basic tests passed"
fi
echo ""

# 2. Vector index tests
echo "2. Running vector index tests..."
echo "----------------------------------------"
python tests/test_vector_index.py
if [ $? -ne 0 ]; then
    echo "❌ Vector index tests failed"
    FAILED_TESTS+=("test_vector_index.py")
else
    echo "✅ Vector index tests passed"
fi
echo ""

# 3. Error recovery tests
echo "3. Running error recovery tests..."
echo "----------------------------------------"
python tests/test_simple_error_recovery.py
if [ $? -ne 0 ]; then
    echo "❌ Error recovery tests failed"
    FAILED_TESTS+=("test_simple_error_recovery.py")
else
    echo "✅ Error recovery tests passed"
fi
echo ""

# 4. API tests
echo "4. Running API tests..."
echo "----------------------------------------"
python test_api.py
if [ $? -ne 0 ]; then
    echo "❌ API tests failed"
    FAILED_TESTS+=("test_api.py")
else
    echo "✅ API tests passed"
fi
echo ""

# 总结
echo "========================================"
echo "Test Summary"
echo "========================================"

if [ ${#FAILED_TESTS[@]} -eq 0 ]; then
    echo "✅ All tests passed!"
    exit 0
else
    echo "❌ ${#FAILED_TESTS[@]} test(s) failed:"
    for test in "${FAILED_TESTS[@]}"; do
        echo "  - $test"
    done
    exit 1
fi
