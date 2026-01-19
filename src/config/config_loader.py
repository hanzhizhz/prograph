"""
配置加载器
从 YAML 文件加载配置
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


# 默认配置文件路径（相对于项目根目录）
DEFAULT_CONFIG_PATH = "config.yaml"


def get_default_config_path() -> str:
    """
    获取默认配置文件路径

    查找顺序：
    1. 环境变量 PROGRAPH_CONFIG_PATH
    2. 当前目录下的 config.yaml
    3. 项目根目录下的 config.yaml（当从 scripts/ 运行时）
    4. 默认返回 "config.yaml"

    Returns:
        配置文件路径
    """
    # 先检查环境变量
    env_path = os.getenv('PROGRAPH_CONFIG_PATH')
    if env_path:
        return env_path

    # 检查当前目录
    current_path = Path(DEFAULT_CONFIG_PATH)
    if current_path.exists():
        return DEFAULT_CONFIG_PATH

    # 检查项目根目录（假设在 scripts/ 目录中运行）
    script_dir = Path(__file__).parent.parent
    root_config = script_dir / DEFAULT_CONFIG_PATH
    if root_config.exists():
        return str(root_config)

    # 默认返回
    return DEFAULT_CONFIG_PATH


def load_yaml(config_path: str) -> Dict[str, Any]:
    """
    加载 YAML 配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config if config else {}


def save_yaml(config: Dict[str, Any], output_path: str) -> None:
    """
    保存配置到 YAML 文件

    Args:
        config: 配置字典
        output_path: 输出文件路径
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并两个配置字典

    Args:
        base: 基础配置
        override: 覆盖配置

    Returns:
        合并后的配置
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result
