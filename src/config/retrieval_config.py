"""
检索配置
管理在线检索的相关配置
"""

from dataclasses import dataclass, field
from typing import Optional, List
from .config_loader import load_yaml, get_default_config_path


@dataclass
class EntityLinkingConfig:
    """实体链接配置"""
    similarity_threshold: float = 0.8  # 与 config.yaml 保持一致
    entity_similarity_threshold: float = 0.9  # 实体相似度阈值
    proposition_similarity_threshold: float = 0.85  # 命题相似度阈值
    vector_top_k: int = 10
    entity_fusion_group_size: int = 20  # 实体融合每组最大数量
    batch_search_size: int = 1000  # 批量搜索的批次大小

    @classmethod
    def from_dict(cls, config: dict) -> 'EntityLinkingConfig':
        """从字典创建配置"""
        linking_config = config.get('entity_linking', {})
        return cls(
            similarity_threshold=linking_config.get('similarity_threshold', cls.similarity_threshold),
            entity_similarity_threshold=linking_config.get('entity_similarity_threshold', cls.entity_similarity_threshold),
            proposition_similarity_threshold=linking_config.get('proposition_similarity_threshold', cls.proposition_similarity_threshold),
            vector_top_k=linking_config.get('vector_top_k', cls.vector_top_k),
            entity_fusion_group_size=linking_config.get('entity_fusion_group_size', cls.entity_fusion_group_size),
            batch_search_size=linking_config.get('batch_search_size', cls.batch_search_size),
        )


@dataclass
class RetrievalConfig:
    """检索配置总入口"""
    # 图路径
    graph_path: str = ""
    meta_dir: str = ""
    index_dir: str = ""

    # 基础检索配置
    max_rounds: int = 5
    max_path_depth: int = 6
    beam_width: int = 4

    # 评分权重
    semantic_weight: float = 0.4
    bridge_weight: float = 0.3

    # 终止条件
    early_stop_threshold: float = 0.02
    max_plateau_steps: int = 2

    # ========== Agent 配置 ==========
    # 状态机配置
    agent_max_rounds: int = 5
    agent_max_evidence: int = 50

    # 锚点队列配置
    agent_anchor_queue_size: int = 100
    agent_anchor_duplicate_threshold: float = 0.9

    # MAP 阶段配置
    agent_map_max_iterations: int = 10
    agent_map_beam_width: int = 5
    agent_map_score_plateau_threshold: float = 0.02
    agent_map_score_plateau_window: int = 2

    # 批量执行配置
    agent_batch_concurrency: int = 10
    agent_top_k_paths: int = 10
    agent_top_k_docs: int = 5

    # ========== 信息缺口状态管理配置 ==========
    max_gap_attempts: int = 2                       # 单个缺口最大尝试次数
    gap_similarity_threshold: float = 0.85          # 缺口相似度阈值（防止微调描述绕过去重）
    enable_gap_evaluation: bool = True              # 是否启用缺口补全评估

    # ========== 双路检索配置 ==========
    retrieval_proposition_top_k: int = 10      # 命题检索 top-k
    retrieval_entity_top_k: int = 5            # 每个实体的关联命题 top-k
    retrieval_max_entities: int = 10           # 最多使用的 related_entities 数量
    retrieval_entity_enable: bool = True       # 是否启用实体检索路径

    # ========== 嵌入缓存配置 ==========
    enable_embedding_cache: bool = True        # 是否启用嵌入缓存管理器
    embedding_cache_size: int = 10000          # 内存缓存最大条目数
    embedding_batch_size: int = 50             # 批量嵌入大小
    embedding_cache_dir: Optional[str] = None  # 持久化缓存目录（None表示使用index_dir/cache）

    @classmethod
    def from_yaml(cls, config_path: str) -> 'RetrievalConfig':
        """从 YAML 文件加载配置"""
        config_dict = load_yaml(config_path)
        retrieval_config_dict = config_dict.get('retrieval', {})
        return cls.from_dict(retrieval_config_dict)

    @classmethod
    def from_dict(cls, config: dict) -> 'RetrievalConfig':
        """从字典创建配置"""
        return cls(
            graph_path=config.get('graph_path', cls.graph_path),
            meta_dir=config.get('meta_dir', cls.meta_dir),
            index_dir=config.get('index_dir', cls.index_dir),
            max_rounds=config.get('max_rounds', cls.max_rounds),
            max_path_depth=config.get('max_path_depth', cls.max_path_depth),
            beam_width=config.get('beam_width', cls.beam_width),
            semantic_weight=config.get('semantic_weight', cls.semantic_weight),
            bridge_weight=config.get('bridge_weight', cls.bridge_weight),
            early_stop_threshold=config.get('early_stop_threshold', cls.early_stop_threshold),
            max_plateau_steps=config.get('max_plateau_steps', cls.max_plateau_steps),
            # Agent 配置
            agent_max_rounds=config.get('agent_max_rounds', cls.agent_max_rounds),
            agent_max_evidence=config.get('agent_max_evidence', cls.agent_max_evidence),
            agent_anchor_queue_size=config.get('agent_anchor_queue_size', cls.agent_anchor_queue_size),
            agent_anchor_duplicate_threshold=config.get('agent_anchor_duplicate_threshold', cls.agent_anchor_duplicate_threshold),
            agent_map_max_iterations=config.get('agent_map_max_iterations', cls.agent_map_max_iterations),
            agent_map_beam_width=config.get('agent_map_beam_width', cls.agent_map_beam_width),
            agent_map_score_plateau_threshold=config.get('agent_map_score_plateau_threshold', cls.agent_map_score_plateau_threshold),
            agent_map_score_plateau_window=config.get('agent_map_score_plateau_window', cls.agent_map_score_plateau_window),
            agent_batch_concurrency=config.get('agent_batch_concurrency', cls.agent_batch_concurrency),
            agent_top_k_paths=config.get('agent_top_k_paths', cls.agent_top_k_paths),
            agent_top_k_docs=config.get('agent_top_k_docs', cls.agent_top_k_docs),
            # 信息缺口状态管理配置
            max_gap_attempts=config.get('max_gap_attempts', cls.max_gap_attempts),
            gap_similarity_threshold=config.get('gap_similarity_threshold', cls.gap_similarity_threshold),
            enable_gap_evaluation=config.get('enable_gap_evaluation', cls.enable_gap_evaluation),
            # 双路检索配置
            retrieval_proposition_top_k=config.get('retrieval_proposition_top_k', cls.retrieval_proposition_top_k),
            retrieval_entity_top_k=config.get('retrieval_entity_top_k', cls.retrieval_entity_top_k),
            retrieval_max_entities=config.get('retrieval_max_entities', cls.retrieval_max_entities),
            retrieval_entity_enable=config.get('retrieval_entity_enable', cls.retrieval_entity_enable),
            # 嵌入缓存配置
            enable_embedding_cache=config.get('enable_embedding_cache', cls.enable_embedding_cache),
            embedding_cache_size=config.get('embedding_cache_size', cls.embedding_cache_size),
            embedding_batch_size=config.get('embedding_batch_size', cls.embedding_batch_size),
            embedding_cache_dir=config.get('embedding_cache_dir', cls.embedding_cache_dir),
        )


# 全局配置单例
_retrieval_config: Optional[RetrievalConfig] = None


def get_retrieval_config(config_path: Optional[str] = None) -> RetrievalConfig:
    """
    获取全局检索配置

    Args:
        config_path: 可选的配置文件路径。如果为 None，使用默认路径查找

    Returns:
        检索配置实例
    """
    global _retrieval_config
    if _retrieval_config is None:
        if config_path is None:
            config_path = get_default_config_path()
        config_dict = load_yaml(config_path).get('retrieval', {})
        _retrieval_config = RetrievalConfig.from_dict(config_dict)
    return _retrieval_config


def set_retrieval_config(config: RetrievalConfig) -> None:
    """设置全局检索配置"""
    global _retrieval_config
    _retrieval_config = config
