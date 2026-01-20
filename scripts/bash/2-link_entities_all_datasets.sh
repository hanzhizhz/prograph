#!/bin/bash

# 批量处理三个数据集的实体链接与图融合脚本（两阶段）
# 阶段1：生成候选对（仅加载向量模型）
# 阶段2：链接实体与图融合（仅加载大模型）
# 使用 nohup 后台运行，日志输出到 ./logs 目录

PYTHON_ENV="/home/ubuntu/miniconda3/envs/vllm/bin/python"
PROJECT_ROOT="/data/zhz/git/prograph"
LOG_DIR="${PROJECT_ROOT}/logs"
CONFIG_FILE="${PROJECT_ROOT}/config.yaml"

# 切换到项目根目录
cd "$PROJECT_ROOT" || exit 1

# 创建日志目录
mkdir -p "$LOG_DIR"

# 数据集配置
declare -A DATASETS=(
    ["HotpotQA"]="output/HotpotQA/proposition_graph/raw_graph.pkl|output/HotpotQA/temp|output/HotpotQA/proposition_graph/linked_graph"
    ["2WikiMultihopQA"]="output/2WikiMultihopQA/proposition_graph/raw_graph.pkl|output/2WikiMultihopQA/temp|output/2WikiMultihopQA/proposition_graph/linked_graph"
    ["MuSiQue"]="output/MuSiQue/proposition_graph/raw_graph.pkl|output/MuSiQue/temp|output/MuSiQue/proposition_graph/linked_graph"
)

# 阶段1：生成候选对（串行处理，避免 vLLM 资源竞争）
echo "========================================"
echo "启动阶段1：生成候选对（串行处理）"
echo "========================================"

for DATASET_NAME in "HotpotQA" "2WikiMultihopQA" "MuSiQue"; do
    IFS='|' read -r GRAPH_PATH TEMP_DIR OUTPUT_PATH <<< "${DATASETS[$DATASET_NAME]}"

    echo "========================================"
    echo "处理阶段1任务: $DATASET_NAME"
    echo "  输入图: $GRAPH_PATH"
    echo "  临时目录: $TEMP_DIR"
    echo "========================================"

    LOG_FILE="${LOG_DIR}/${DATASET_NAME}_stage1.log"

    # 串行运行，等待每个任务完成
    $PYTHON_ENV -u "${PROJECT_ROOT}/scripts/2a-generate_candidates.py" \
        --graph "$GRAPH_PATH" \
        --output "$TEMP_DIR" \
        --config "$CONFIG_FILE" \
        > "$LOG_FILE" 2>&1

    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "错误: $DATASET_NAME 阶段1失败，退出码: $EXIT_CODE"
        echo "  日志文件: $LOG_FILE"
        echo ""
        continue
    fi

    echo "✓ $DATASET_NAME 阶段1完成"
    echo "  日志文件: $LOG_FILE"
    echo ""
    
    # 短暂等待，确保资源释放
    sleep 3
done

echo "========================================"
echo "阶段1全部完成！"
echo "========================================"
echo ""

# 短暂等待，确保文件写入完成
sleep 5

# 阶段2：链接实体与图融合（串行处理，避免 vLLM 资源竞争）
echo "========================================"
echo "启动阶段2：链接实体与图融合（串行处理）"
echo "========================================"

for DATASET_NAME in "HotpotQA" "2WikiMultihopQA" "MuSiQue"; do
    IFS='|' read -r GRAPH_PATH TEMP_DIR OUTPUT_PATH <<< "${DATASETS[$DATASET_NAME]}"

    # 检查阶段1输出文件是否存在
    if [ ! -f "${TEMP_DIR}/llm_groups.pkl" ] || [ ! -f "${TEMP_DIR}/proposition_candidates.pkl" ]; then
        echo "警告: $DATASET_NAME 的候选对文件不存在，跳过阶段2"
        echo "  期望文件: ${TEMP_DIR}/llm_groups.pkl, ${TEMP_DIR}/proposition_candidates.pkl"
        echo ""
        continue
    fi

    echo "========================================"
    echo "处理阶段2任务: $DATASET_NAME"
    echo "  输入图: $GRAPH_PATH"
    echo "  临时目录: $TEMP_DIR"
    echo "  输出: $OUTPUT_PATH"
    echo "========================================"

    LOG_FILE="${LOG_DIR}/${DATASET_NAME}_stage2.log"

    # 串行运行，等待每个任务完成
    $PYTHON_ENV -u "${PROJECT_ROOT}/scripts/2b-link_and_fuse.py" \
        --graph "$GRAPH_PATH" \
        --temp_dir "$TEMP_DIR" \
        --output "$OUTPUT_PATH" \
        --config "$CONFIG_FILE" \
        > "$LOG_FILE" 2>&1

    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "错误: $DATASET_NAME 阶段2失败，退出码: $EXIT_CODE"
        echo "  日志文件: $LOG_FILE"
        echo ""
        continue
    fi

    echo "✓ $DATASET_NAME 阶段2完成"
    echo "  日志文件: $LOG_FILE"
    echo ""
    
    # 短暂等待，确保资源释放
    sleep 3
done

echo "========================================"
echo "阶段2所有后台任务已启动"
echo "日志目录: $LOG_DIR"
echo "========================================"
echo ""
echo "查看运行状态:"
echo "  ps aux | grep 2b-link_and_fuse.py"
echo ""
echo "查看日志:"
echo "  tail -f $LOG_DIR/HotpotQA_stage2.log"
echo "  tail -f $LOG_DIR/2WikiMultihopQA_stage2.log"
echo "  tail -f $LOG_DIR/MuSiQue_stage2.log"
