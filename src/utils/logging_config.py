"""统一日志配置模块"""

import logging
import sys
from pathlib import Path
from typing import Optional

# 日志级别映射
LOG_LEVELS = {
    'CRITICAL': logging.CRITICAL,
    'ERROR': logging.ERROR,
    'WARNING': logging.WARNING,
    'INFO': logging.INFO,
    'DEBUG': logging.DEBUG
}


def setup_logging(
    level: str = 'INFO',
    log_file: Optional[str] = None,
    log_dir: Optional[str] = None,
    quiet: bool = False
):
    """
    设置全局日志配置

    Args:
        level: 日志级别（CRITICAL, ERROR, WARNING, INFO, DEBUG）
        log_file: 日志文件路径（可选）
        log_dir: 日志目录（可选，如果指定会创建带时间戳的日志文件）
        quiet: 是否静默模式（不输出到控制台）
    """
    # 获取日志级别
    log_level = LOG_LEVELS.get(level.upper(), logging.INFO)

    # 配置vLLM日志，确保能看到模型加载进展
    vllm_logger = logging.getLogger('vllm')
    vllm_logger.setLevel(logging.INFO)
    vllm_logger.propagate = False

    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # 清除现有的处理器
    root_logger.handlers.clear()

    # 日志格式（包含模块名，便于调试）
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 控制台处理器（除非quiet模式）
    if not quiet:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # 文件处理器（如果指定了日志文件或目录）
    if log_file or log_dir:
        if log_dir:
            log_dir_path = Path(log_dir)
            log_dir_path.mkdir(parents=True, exist_ok=True)
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = str(log_dir_path / f'qa_{timestamp}.log')

        file_handler = logging.FileHandler(str(log_file), encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # 为vLLM添加控制台处理器（独立于根日志器）
    if not quiet:
        import sys
        vllm_console_handler = logging.StreamHandler(sys.stdout)
        vllm_console_handler.setLevel(logging.INFO)
        vllm_console_handler.setFormatter(formatter)
        vllm_logger.addHandler(vllm_console_handler)


def get_logger(name: str) -> logging.Logger:
    """
    获取指定名称的日志记录器

    Args:
        name: 日志记录器名称（通常使用__name__）

    Returns:
        日志记录器实例
    """
    return logging.getLogger(name)
