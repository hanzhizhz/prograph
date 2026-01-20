#!/bin/bash

# 批量过滤三个数据集的脚本
# 在实体链接和图融合完成后运行，过滤掉处理失败的文档
# 使用 nohup 后台运行，日志输出到 ./logs 目录

PYTHON_ENV="/home/ubuntu/miniconda3/envs/vllm/bin/python"
PROJECT_ROOT="/data/zhz/git/prograph"
LOG_DIR="${PROJECT_ROOT}/logs"
CONFIG_FILE="${PROJECT_ROOT}/config.yaml"

# 切换到项目根目录
cd "$PROJECT_ROOT" || exit 1

# 创建日志目录
mkdir -p "$LOG_DIR"

# 数据集配置：数据集路径|训练数据路径|图路径|输出数据集路径|输出训练数据路径
declare -A DATASETS=(
    ["HotpotQA"]="dataset/HotpotQA/full_docs.json|dataset/HotpotQA/train_data.json|output/HotpotQA/proposition_graph/linked_graph.pkl|dataset/HotpotQA/full_docs_filtered.json|dataset/HotpotQA/train_data_filtered.json"
    ["2WikiMultihopQA"]="dataset/2WikiMultihopQA/full_docs.json|dataset/2WikiMultihopQA/train_data.json|output/2WikiMultihopQA/proposition_graph/linked_graph.pkl|dataset/2WikiMultihopQA/full_docs_filtered.json|dataset/2WikiMultihopQA/train_data_filtered.json"
    ["MuSiQue"]="dataset/MuSiQue/full_docs.json|dataset/MuSiQue/train_data.json|output/MuSiQue/proposition_graph/linked_graph.pkl|dataset/MuSiQue/full_docs_filtered.json|dataset/MuSiQue/train_data_filtered.json"
)

# 串行处理每个数据集（避免并发读取图文件的问题）
echo "========================================"
echo "开始批量过滤数据集"
echo "========================================"

for DATASET_NAME in "HotpotQA" "2WikiMultihopQA" "MuSiQue"; do
    IFS='|' read -r DATASET_PATH TRAIN_DATA_PATH GRAPH_PATH OUTPUT_DOCS_PATH OUTPUT_TRAIN_PATH <<< "${DATASETS[$DATASET_NAME]}"

    echo "========================================"
    echo "处理数据集: $DATASET_NAME"
    echo "  数据集: $DATASET_PATH"
    echo "  训练数据: $TRAIN_DATA_PATH"
    echo "  输入图: $GRAPH_PATH"
    echo "  输出数据集: $OUTPUT_DOCS_PATH"
    echo "  输出训练数据: $OUTPUT_TRAIN_PATH"
    echo "========================================"

    # 检查图文件是否存在
    if [ ! -f "$GRAPH_PATH" ]; then
        echo "警告: $DATASET_NAME 的图文件不存在，跳过"
        echo "  期望文件: $GRAPH_PATH"
        echo ""
        continue
    fi

    LOG_FILE="${LOG_DIR}/${DATASET_NAME}_filter.log"

    # 运行过滤脚本
    $PYTHON_ENV -u "${PROJECT_ROOT}/scripts/3-filter_dataset.py" \
        --dataset "$DATASET_PATH" \
        --train-data "$TRAIN_DATA_PATH" \
        --graph "$GRAPH_PATH" \
        --output-docs "$OUTPUT_DOCS_PATH" \
        --output-train "$OUTPUT_TRAIN_PATH" \
        2>&1 | tee "$LOG_FILE"

    # 检查退出状态
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✓ $DATASET_NAME 过滤完成"
        echo "  日志文件: $LOG_FILE"
    else
        echo "✗ $DATASET_NAME 过滤失败，请检查日志"
        echo "  日志文件: $LOG_FILE"
    fi
    echo ""

    # 短暂等待，确保资源释放
    sleep 2
done

echo "========================================"
echo "所有数据集过滤完成"
echo "========================================"
echo ""
echo "查看统计信息:"
echo "  cat output/HotpotQA/filter_docs_stats.json"
echo "  cat output/2WikiMultihopQA/filter_docs_stats.json"
echo "  cat output/MuSiQue/filter_docs_stats.json"
echo ""
