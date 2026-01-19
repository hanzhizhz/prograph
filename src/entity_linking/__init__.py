"""
实体链接模块
"""

from .embedding_index import EmbeddingIndex, EmbeddingStore
from .candidate_generator import CandidateGenerator, CandidatePair, EntityCandidateGroup, build_indices
from .entity_linker import EntityLinker, LinkingDecision, EntityFusionResult, FusedEntity
from .graph_fusion import GraphFusion

__all__ = [
    "EmbeddingIndex",
    "EmbeddingStore",
    "CandidateGenerator",
    "CandidatePair",
    "EntityCandidateGroup",
    "build_indices",
    "EntityLinker",
    "LinkingDecision",
    "EntityFusionResult",
    "FusedEntity",
    "GraphFusion",
]
