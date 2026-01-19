#!/bin/bash

# 离线构建串行脚本
# 依次执行：1-构建数据集 -> 2-链接实体 -> 3-过滤数据集 -> 4-构建持久化数据
# 使用 nohup 后台运行，日志输出到 ./logs 目录

PROJECT_ROOT="/data/zhz/git/prograph"
LOG_DIR="${PROJECT_ROOT}/logs"
BASH_DIR="${PROJECT_ROOT}/scripts/bash"

# 切换到项目根目录
cd "$PROJECT_ROOT" || exit 1

# 创建日志目录
mkdir -p "$LOG_DIR"

# 主日志文件
MAIN_LOG="${LOG_DIR}/offline_build_all.log"

echo "========================================"
echo "开始离线构建流程"
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
echo "" | tee "$MAIN_LOG"

# 步骤1: 构建数据集
echo "[步骤 1/4] 构建数据集..." | tee -a "$MAIN_LOG"
bash "${BASH_DIR}/1-build_all_datasets.sh" 2>&1 | tee -a "$MAIN_LOG"
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "错误: 步骤1失败" | tee -a "$MAIN_LOG"
    exit 1
fi
echo "[步骤 1/4] 完成" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

# 步骤2: 链接实体
echo "[步骤 2/4] 链接实体..." | tee -a "$MAIN_LOG"
bash "${BASH_DIR}/2-link_entities_all_datasets.sh" 2>&1 | tee -a "$MAIN_LOG"
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "错误: 步骤2失败" | tee -a "$MAIN_LOG"
    exit 1
fi
echo "[步骤 2/4] 完成" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

# 步骤3: 过滤数据集
echo "[步骤 3/4] 过滤数据集..." | tee -a "$MAIN_LOG"
bash "${BASH_DIR}/3-filter_all_datasets.sh" 2>&1 | tee -a "$MAIN_LOG"
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "错误: 步骤3失败" | tee -a "$MAIN_LOG"
    exit 1
fi
echo "[步骤 3/4] 完成" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

# 步骤4: 构建持久化数据
echo "[步骤 4/4] 构建持久化数据..." | tee -a "$MAIN_LOG"
bash "${BASH_DIR}/4-build_all_persistence_data.sh" 2>&1 | tee -a "$MAIN_LOG"
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "错误: 步骤4失败" | tee -a "$MAIN_LOG"
    exit 1
fi
echo "[步骤 4/4] 完成" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

echo "========================================"
echo "离线构建流程全部完成"
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
