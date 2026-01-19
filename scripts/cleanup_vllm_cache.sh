#!/bin/bash
# vLLM 缓存清理脚本
# 用于清理因强制关闭导致的损坏缓存文件

set -e

echo "============================================================"
echo "清理 vLLM 缓存"
echo "============================================================"

# 1. 清理特定损坏的缓存目录（如果指定）
if [ -n "$1" ]; then
    CACHE_DIR="/home/ubuntu/.cache/vllm/torch_compile_cache/$1"
    if [ -d "$CACHE_DIR" ]; then
        echo "删除特定缓存目录: $CACHE_DIR"
        rm -rf "$CACHE_DIR"
        echo "✓ 已清理"
    else
        echo "目录不存在: $CACHE_DIR"
    fi
    exit 0
fi

# 2. 清理所有 vLLM torch_compile_cache（需要确认）
echo "当前 vLLM 缓存大小:"
du -sh /home/ubuntu/.cache/vllm/ 2>/dev/null || echo "缓存目录不存在"

echo ""
echo "选项："
echo "1. 清理所有 torch_compile_cache (会删除所有编译缓存，下次启动需要重新编译)"
echo "2. 仅清理最近修改的缓存目录"
echo "3. 仅清理损坏的缓存文件（检查 pickle 错误）"
echo "4. 退出"
read -p "请选择 [1-4]: " choice

case $choice in
    1)
        echo "清理所有 torch_compile_cache..."
        rm -rf /home/ubuntu/.cache/vllm/torch_compile_cache/
        echo "✓ 所有缓存已清理"
        ;;
    2)
        echo "查找最近修改的缓存目录（30天内）..."
        find /home/ubuntu/.cache/vllm/torch_compile_cache/ -maxdepth 1 -type d -mtime -30 -exec echo "{}" \;
        read -p "是否删除这些目录? [y/N]: " confirm
        if [[ $confirm == [yY] ]]; then
            find /home/ubuntu/.cache/vllm/torch_compile_cache/ -maxdepth 1 -type d -mtime -30 -exec rm -rf {} \;
            echo "✓ 最近修改的缓存已清理"
        fi
        ;;
    3)
        echo "检查损坏的 pickle 文件..."
        # 查找可能损坏的文件（通常是较小的文件）
        find /home/ubuntu/.cache/vllm/torch_compile_cache/ -name "*.pkl" -size -100c -print
        echo "注意：这可能需要手动检查"
        ;;
    4)
        echo "退出"
        exit 0
        ;;
    *)
        echo "无效选择"
        exit 1
        ;;
esac

echo ""
echo "清理完成！"
echo "提示：如果 GPU 内存仍被占用，可以尝试："
echo "  1. 等待几分钟让 GPU 自动释放内存"
echo "  2. 重启相关的 Python 进程"
echo "  3. 如果必要，可以重启系统"
