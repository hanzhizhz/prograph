"""
LLM 模块
"""

from .base import BaseLLM, BaseEmbedding, LLMResponse, EmbeddingResponse

# 尝试导入可选的客户端
try:
    from .vllm_client import VLLMClient
    _has_vllm = True
except ImportError:
    _has_vllm = False
    VLLMClient = None

from .openai_client import OpenAIClient, build_single_turn
from .embedding_client import VLLMEmbeddingClient, OpenAIEmbeddingClient, cosine_similarity

__all__ = [
    "BaseLLM",
    "BaseEmbedding",
    "LLMResponse",
    "EmbeddingResponse",
    "VLLMClient",
    "OpenAIClient",
    "build_single_turn",
    "VLLMEmbeddingClient",
    "OpenAIEmbeddingClient",
    "cosine_similarity",
]
