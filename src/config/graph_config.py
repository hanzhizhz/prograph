"""
图构建配置
管理异构图构建的相关配置
"""

from dataclasses import dataclass, field
from typing import Optional
from .config_loader import load_yaml, get_default_config_path


@dataclass
class PropositionExtractionConfig:
    """命题提取配置"""
    temperature: float = 0.1
    max_tokens: int = 2048

    @classmethod
    def from_dict(cls, config: dict) -> 'PropositionExtractionConfig':
        """从字典创建配置"""
        extraction_config = config.get('proposition_extraction', {})
        return cls(
            temperature=extraction_config.get('temperature', cls.temperature),
            max_tokens=extraction_config.get('max_tokens', cls.max_tokens),
        )


@dataclass
class EntityExtractionConfig:
    """实体提取配置"""
    temperature: float = 0.1
    max_tokens: int = 2048

    @classmethod
    def from_dict(cls, config: dict) -> 'EntityExtractionConfig':
        """从字典创建配置"""
        extraction_config = config.get('entity_extraction', {})
        return cls(
            temperature=extraction_config.get('temperature', cls.temperature),
            max_tokens=extraction_config.get('max_tokens', cls.max_tokens),
        )


@dataclass
class RSTAnalysisConfig:
    """RST 关系分析配置"""
    temperature: float = 0.2
    max_tokens: int = 1024

    @classmethod
    def from_dict(cls, config: dict) -> 'RSTAnalysisConfig':
        """从字典创建配置"""
        analysis_config = config.get('rst_analysis', {})
        return cls(
            temperature=analysis_config.get('temperature', cls.temperature),
            max_tokens=analysis_config.get('max_tokens', cls.max_tokens),
        )


@dataclass
class GraphConfig:
    """图构建配置总入口"""
    proposition_extraction: PropositionExtractionConfig = field(default_factory=PropositionExtractionConfig)
    entity_extraction: EntityExtractionConfig = field(default_factory=EntityExtractionConfig)
    rst_analysis: RSTAnalysisConfig = field(default_factory=RSTAnalysisConfig)

    @classmethod
    def from_yaml(cls, config_path: str) -> 'GraphConfig':
        """从 YAML 文件加载配置"""
        config_dict = load_yaml(config_path)
        graph_config_dict = config_dict.get('graph', {})
        return cls.from_dict(graph_config_dict)

    @classmethod
    def from_dict(cls, config: dict) -> 'GraphConfig':
        """从字典创建配置"""
        return cls(
            proposition_extraction=PropositionExtractionConfig.from_dict(config),
            entity_extraction=EntityExtractionConfig.from_dict(config),
            rst_analysis=RSTAnalysisConfig.from_dict(config),
        )


# 全局配置单例
_graph_config: Optional[GraphConfig] = None


def get_graph_config(config_path: Optional[str] = None) -> GraphConfig:
    """
    获取全局图配置

    Args:
        config_path: 可选的配置文件路径。如果为 None，使用默认路径查找

    Returns:
        图配置实例
    """
    global _graph_config
    if _graph_config is None:
        if config_path is None:
            config_path = get_default_config_path()
        config_dict = load_yaml(config_path).get('graph', {})
        _graph_config = GraphConfig.from_dict(config_dict)
    return _graph_config


def set_graph_config(config: GraphConfig) -> None:
    """设置全局图配置"""
    global _graph_config
    _graph_config = config
