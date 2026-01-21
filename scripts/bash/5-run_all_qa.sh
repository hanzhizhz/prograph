#!/bin/bash

# 批量运行三个数据集的多跳问答评估脚本
# 使用训练数据进行评估
# 串行运行三个数据集，每个数据集内部并行运行（并行度50）
# 输出位置: ./output/数据集名/result

PYTHON_ENV="/home/ubuntu/miniconda3/envs/vllm/bin/python"
PROJECT_ROOT="/data/zhz/git/prograph"
LOG_DIR="${PROJECT_ROOT}/logs"
CONFIG_FILE="${PROJECT_ROOT}/config.yaml"
CONCURRENCY=5  # 并行度
MAX_SAMPLES=5  # 最大处理样本数

# 切换到项目根目录
cd "$PROJECT_ROOT" || exit 1

# 创建日志目录
mkdir -p "$LOG_DIR"

# 数据集配置：训练数据路径|图路径|索引目录|持久化目录
declare -A DATASETS=(
    ["HotpotQA"]="dataset/HotpotQA/train_data_filtered.json|output/HotpotQA/proposition_graph/linked_graph.pkl|output/HotpotQA/persistence_data|output/HotpotQA/persistence_data"
    # ["2WikiMultihopQA"]="dataset/2WikiMultihopQA/train_data_filtered.json|output/2WikiMultihopQA/proposition_graph/linked_graph.pkl|output/2WikiMultihopQA/persistence_data|output/2WikiMultihopQA/persistence_data"
    # ["MuSiQue"]="dataset/MuSiQue/train_data_filtered.json|output/MuSiQue/proposition_graph/linked_graph.pkl|output/MuSiQue/persistence_data|output/MuSiQue/persistence_data"
)

# 串行处理每个数据集（数据集之间串行，内部并行）
echo "========================================"
echo "开始批量运行问答评估"
echo "并发度: $CONCURRENCY"
echo "========================================"

for DATASET_NAME in "HotpotQA" "2WikiMultihopQA" "MuSiQue"; do
    IFS='|' read -r TRAIN_DATA GRAPH_PATH INDEX_DIR PERSISTENCE_DIR <<< "${DATASETS[$DATASET_NAME]}"

    # 输出路径: ./output/数据集名/result.json
    OUTPUT_PATH="output/${DATASET_NAME}/result.json"

    echo "========================================"
    echo "处理数据集: $DATASET_NAME"
    echo "  训练数据: $TRAIN_DATA"
    echo "  输入图: $GRAPH_PATH"
    echo "  索引目录: $INDEX_DIR"
    echo "  持久化目录: $PERSISTENCE_DIR"
    echo "  输出结果: $OUTPUT_PATH"
    echo "========================================"

    # 检查必要文件是否存在
    if [ ! -f "$TRAIN_DATA" ]; then
        echo "警告: $DATASET_NAME 的训练数据文件不存在，跳过"
        echo "  期望文件: $TRAIN_DATA"
        echo ""
        continue
    fi

    if [ ! -f "$GRAPH_PATH" ]; then
        echo "警告: $DATASET_NAME 的图文件不存在，跳过"
        echo "  期望文件: $GRAPH_PATH"
        echo ""
        continue
    fi

    if [ ! -d "$INDEX_DIR" ]; then
        echo "警告: $DATASET_NAME 的索引目录不存在，跳过"
        echo "  期望目录: $INDEX_DIR"
        echo "  请先运行: ./scripts/4-build_all_persistence_data.sh"
        echo ""
        continue
    fi

    LOG_FILE="${LOG_DIR}/${DATASET_NAME}_qa.log"

    # 运行问答脚本（使用过滤后的训练数据，并发处理）
    $PYTHON_ENV -u "${PROJECT_ROOT}/scripts/5-run_multi_hop_qa.py" \
        --dataset "$TRAIN_DATA" \
        --graph "$GRAPH_PATH" \
        --index-dir "$INDEX_DIR" \
        --persistence-dir "$PERSISTENCE_DIR" \
        --output "$OUTPUT_PATH" \
        --concurrency "$CONCURRENCY" \
        --config "$CONFIG_FILE" \
        --max-samples "$MAX_SAMPLES" \
        2>&1 | tee "$LOG_FILE"

    # 检查退出状态
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✓ $DATASET_NAME 问答评估完成"
        echo "  日志文件: $LOG_FILE"
        echo "  结果文件: $OUTPUT_PATH"
    else
        echo "✗ $DATASET_NAME 问答评估失败，请检查日志"
        echo "  日志文件: $LOG_FILE"
    fi
    echo ""

    # 短暂等待，确保资源释放
    sleep 3
done

echo "========================================"
echo "所有数据集问答评估完成"
echo "========================================"
echo ""
echo "查看评估结果:"
echo "  cat output/HotpotQA/result.json"
echo "  cat output/2WikiMultihopQA/result.json"
echo "  cat output/MuSiQue/result.json"
echo ""
