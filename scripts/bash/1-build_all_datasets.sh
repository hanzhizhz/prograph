#!/bin/bash

# 批量处理三个数据集的脚本
# 使用 vLLM 环境串行构建命题图

PYTHON_ENV="/home/ubuntu/miniconda3/envs/vllm/bin/python"
PROJECT_ROOT="/data/zhz/git/prograph"
LOG_DIR="${PROJECT_ROOT}/logs"

# 创建日志目录
mkdir -p "$LOG_DIR"

# 数据集配置
declare -A DATASETS=(
    ["HotpotQA"]="dataset/HotpotQA/full_docs.json|output/HotpotQA/proposition_graph/raw_graph"
    ["2WikiMultihopQA"]="dataset/2WikiMultihopQA/full_docs.json|output/2WikiMultihopQA/proposition_graph/raw_graph"
    ["MuSiQue"]="dataset/MuSiQue/full_docs.json|output/MuSiQue/proposition_graph/raw_graph"
)

# 串行处理每个数据集
for DATASET_NAME in "HotpotQA" "2WikiMultihopQA" "MuSiQue"; do
    IFS='|' read -r DATASET_PATH OUTPUT_PATH <<< "${DATASETS[$DATASET_NAME]}"

    echo "========================================"
    echo "开始处理: $DATASET_NAME"
    echo "数据集: $DATASET_PATH"
    echo "输出: $OUTPUT_PATH"
    echo "========================================"

    LOG_FILE="${LOG_DIR}/${DATASET_NAME}_build.log"

    # 使用 -u 参数（无缓冲）和 tee 同时输出到控制台和日志文件
    $PYTHON_ENV -u "${PROJECT_ROOT}/scripts/1-build_proposition_graph.py" \
        --dataset "$DATASET_PATH" \
        --output "$OUTPUT_PATH" \
        2>&1 | tee "$LOG_FILE"

    # 检查退出状态
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "$DATASET_NAME 处理完成"
    else
        echo "$DATASET_NAME 处理失败，请检查日志"
    fi
    echo ""
done

echo "========================================"
echo "所有数据集处理完成"
echo "========================================"
