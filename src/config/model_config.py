"""
模型配置
管理 LLM 和 Embedding 模型的配置
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from .config_loader import load_yaml, merge_configs, get_default_config_path


@dataclass
class VLLMConfig:
    """vLLM 离线推理配置"""
    model_path: str = "/data/zhz/model/Qwen3-30B-A3B-Instruct-2507-Int4-W4A16"
    tensor_parallel_size: int = 2
    gpu_memory_utilization: float = 0.5
    trust_remote_code: bool = True
    max_model_len: Optional[int] = None

    @classmethod
    def from_dict(cls, config: dict) -> 'VLLMConfig':
        """从字典创建配置"""
        vllm_config = config.get('vllm', {})
        return cls(
            model_path=os.getenv('VLLM_MODEL_PATH', vllm_config.get('model_path')),
            tensor_parallel_size=int(os.getenv('VLLM_TP_SIZE', str(vllm_config.get('tensor_parallel_size', cls.tensor_parallel_size)))),
            gpu_memory_utilization=float(os.getenv('VLLM_GPU_UTIL', str(vllm_config.get('gpu_memory_utilization', cls.gpu_memory_utilization)))),
            trust_remote_code=vllm_config.get('trust_remote_code', cls.trust_remote_code),
            max_model_len=vllm_config.get('max_model_len'),
        )


@dataclass
class VLLMEmbeddingConfig:
    """vLLM Embedding 配置"""
    model_path: str = "/data/zhz/model/Qwen3-Embedding-0.6B"
    tensor_parallel_size: int = 2
    gpu_memory_utilization: float = 0.05
    trust_remote_code: bool = True
    max_model_len: Optional[int] = None

    @classmethod
    def from_dict(cls, config: dict) -> 'VLLMEmbeddingConfig':
        """从字典创建配置"""
        embedding_config = config.get('vllm_embedding', {})
        return cls(
            model_path=os.getenv('VLLM_EMBEDDING_MODEL_PATH', embedding_config.get('model_path')),
            tensor_parallel_size=int(os.getenv('VLLM_EMBEDDING_TP_SIZE', str(embedding_config.get('tensor_parallel_size', cls.tensor_parallel_size)))),
            gpu_memory_utilization=float(os.getenv('VLLM_EMBEDDING_GPU_UTIL', str(embedding_config.get('gpu_memory_utilization', cls.gpu_memory_utilization)))),
            trust_remote_code=embedding_config.get('trust_remote_code', cls.trust_remote_code),
            max_model_len=embedding_config.get('max_model_len'),
        )


@dataclass
class LLMConfig:
    """运行时 LLM API 配置"""
    provider: str = "openai"
    model: str = "qwen3"
    base_url: str = "http://localhost:8901/v1"
    api_key: str = "123"
    temperature: float = 0.1
    max_tokens: int = 2000
    concurrency: int = 100
    max_retries: int = 3
    timeout: float = 180.0

    @classmethod
    def from_dict(cls, config: dict) -> 'LLMConfig':
        """从字典创建配置"""
        llm_config = config.get('llm', {})
        return cls(
            provider=llm_config.get('provider', cls.provider),
            model=llm_config.get('model', cls.model),
            base_url=llm_config.get('base_url', cls.base_url),
            api_key=os.getenv('OPENAI_API_KEY', llm_config.get('api_key', cls.api_key)),
            temperature=llm_config.get('temperature', cls.temperature),
            max_tokens=llm_config.get('max_tokens', cls.max_tokens),
            concurrency=llm_config.get('concurrency', cls.concurrency),
            max_retries=llm_config.get('max_retries', cls.max_retries),
            timeout=llm_config.get('timeout', cls.timeout),
        )


@dataclass
class EmbeddingConfig:
    """运行时 Embedding API 配置"""
    provider: str = "openai"
    model: str = "qwen3-embedding"
    base_url: str = "http://localhost:8902/v1"
    api_key: str = "123"
    temperature: float = 0.0
    max_tokens: int = 8191
    concurrency: int = 100
    max_retries: int = 3
    timeout: float = 600.0

    @classmethod
    def from_dict(cls, config: dict) -> 'EmbeddingConfig':
        """从字典创建配置"""
        embedding_config = config.get('embedding', {})
        return cls(
            provider=embedding_config.get('provider', cls.provider),
            model=embedding_config.get('model', cls.model),
            base_url=embedding_config.get('base_url', cls.base_url),
            api_key=os.getenv('OPENAI_API_KEY', embedding_config.get('api_key', cls.api_key)),
            temperature=embedding_config.get('temperature', cls.temperature),
            max_tokens=embedding_config.get('max_tokens', cls.max_tokens),
            concurrency=embedding_config.get('concurrency', cls.concurrency),
            max_retries=embedding_config.get('max_retries', cls.max_retries),
            timeout=embedding_config.get('timeout', cls.timeout),
        )


@dataclass
class ModelConfig:
    """模型配置总入口"""
    vllm: VLLMConfig = field(default_factory=VLLMConfig)
    vllm_embedding: VLLMEmbeddingConfig = field(default_factory=VLLMEmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)

    @classmethod
    def from_yaml(cls, config_path: str) -> 'ModelConfig':
        """从 YAML 文件加载配置"""
        config_dict = load_yaml(config_path)
        model_config_dict = config_dict.get('model', {})
        return cls.from_dict(model_config_dict)

    @classmethod
    def from_dict(cls, config: dict) -> 'ModelConfig':
        """从字典创建配置"""
        return cls(
            vllm=VLLMConfig.from_dict(config),
            vllm_embedding=VLLMEmbeddingConfig.from_dict(config),
            llm=LLMConfig.from_dict(config),
            embedding=EmbeddingConfig.from_dict(config),
        )


# 全局配置单例
_model_config: Optional[ModelConfig] = None


def get_model_config(config_path: Optional[str] = None) -> ModelConfig:
    """
    获取全局模型配置

    Args:
        config_path: 可选的配置文件路径。如果为 None，使用默认路径查找

    Returns:
        模型配置实例
    """
    global _model_config
    if _model_config is None:
        if config_path is None:
            config_path = get_default_config_path()
        _model_config = ModelConfig.from_yaml(config_path)
    return _model_config


def set_model_config(config: ModelConfig) -> None:
    """设置全局模型配置"""
    global _model_config
    _model_config = config