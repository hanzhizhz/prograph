"""
命题图模块
"""

from .document_loader import DocumentLoader, Document
from .proposition_extractor import Proposition
from .entity_extractor import Entity
from .rst_analyzer import RSTRelation, SKELETON_EDGE_EXAMPLES, DETAIL_EDGE_EXAMPLES, EDGE_TYPE_EXAMPLES
from .unified_extractor import (
    UnifiedDocumentExtractor,
    EntityWithReason,
    PropositionWithEntities,
    UnifiedExtractionResult,
)
from .graph_builder import GraphBuilder, GraphStatistics
from . import prompts

__all__ = [
    # 基础组件
    "DocumentLoader",
    "Document",
    # 数据类
    "Proposition",
    "Entity",
    "RSTRelation",
    "SKELETON_EDGE_EXAMPLES",
    "DETAIL_EDGE_EXAMPLES",
    "EDGE_TYPE_EXAMPLES",
    # 统一提取器
    "UnifiedDocumentExtractor",
    "EntityWithReason",
    "PropositionWithEntities",
    "UnifiedExtractionResult",
    # 图构建器
    "GraphBuilder",
    "GraphStatistics",
    # 提示词
    "prompts",
]
