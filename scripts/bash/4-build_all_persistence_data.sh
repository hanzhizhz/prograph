#!/bin/bash

# 批量为三个数据集构建搜索索引的脚本
# 在实体链接和图融合完成后运行
# 使用 nohup 后台运行，日志输出到 ./logs 目录

PYTHON_ENV="/home/ubuntu/miniconda3/envs/vllm/bin/python"
PROJECT_ROOT="/data/zhz/git/prograph"
LOG_DIR="${PROJECT_ROOT}/logs"
CONFIG_FILE="${PROJECT_ROOT}/config.yaml"

# 切换到项目根目录
cd "$PROJECT_ROOT" || exit 1

# 创建日志目录
mkdir -p "$LOG_DIR"

# 数据集配置：图路径|持久化数据输出目录
declare -A DATASETS=(
    ["HotpotQA"]="output/HotpotQA/proposition_graph/linked_graph.pkl|output/HotpotQA/persistence_data"
    ["2WikiMultihopQA"]="output/2WikiMultihopQA/proposition_graph/linked_graph.pkl|output/2WikiMultihopQA/persistence_data"
    ["MuSiQue"]="output/MuSiQue/proposition_graph/linked_graph.pkl|output/MuSiQue/persistence_data"
)

# 串行处理每个数据集（避免 vLLM embedding 资源竞争）
echo "========================================"
echo "开始批量构建搜索索引"
echo "========================================"

for DATASET_NAME in "HotpotQA" "2WikiMultihopQA" "MuSiQue"; do
    IFS='|' read -r GRAPH_PATH OUTPUT_DIR <<< "${DATASETS[$DATASET_NAME]}"

    echo "========================================"
    echo "处理数据集: $DATASET_NAME"
    echo "  输入图: $GRAPH_PATH"
    echo "  输出目录: $OUTPUT_DIR"
    echo "========================================"

    # 检查图文件是否存在
    if [ ! -f "$GRAPH_PATH" ]; then
        echo "警告: $DATASET_NAME 的图文件不存在，跳过"
        echo "  期望文件: $GRAPH_PATH"
        echo ""
        continue
    fi

    LOG_FILE="${LOG_DIR}/${DATASET_NAME}_index.log"

    # 运行索引构建脚本
    $PYTHON_ENV -u "${PROJECT_ROOT}/scripts/4-build_persistence_data.py" \
        --graph "$GRAPH_PATH" \
        --output "$OUTPUT_DIR" \
        --config "$CONFIG_FILE" \
        2>&1 | tee "$LOG_FILE"

    # 检查退出状态
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✓ $DATASET_NAME 索引构建完成"
        echo "  日志文件: $LOG_FILE"
    else
        echo "✗ $DATASET_NAME 索引构建失败，请检查日志"
        echo "  日志文件: $LOG_FILE"
    fi
    echo ""

    # 短暂等待，确保资源释放
    sleep 5
done

echo "========================================"
echo "所有数据集索引构建完成"
echo "========================================"
echo ""
echo "查看索引统计:"
echo "  cat output/HotpotQA/persistence_data/index_stats.json"
echo "  cat output/2WikiMultihopQA/persistence_data/index_stats.json"
echo "  cat output/MuSiQue/persistence_data/index_stats.json"
echo ""
