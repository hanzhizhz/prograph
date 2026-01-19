"""
RST 关系数据类
"""

from typing import List, Optional
from dataclasses import dataclass


# ============================================================
# LLM 提示词示例（具体类型，用于帮助模型理解）
# ============================================================

# 骨架边示例 - Nucleus-Nucleus 关系（同等重要）
SKELETON_EDGE_EXAMPLES = ["SEQUENCE", "CONTRAST", "CONCESSION"]

# 细节边示例 - Nucleus-Satellite 关系（主从依赖）
DETAIL_EDGE_EXAMPLES = ["CAUSED_BY", "MOTIVATION", "ELABORATION", "BACKGROUND"]

# 所有示例类型（用于提示词）
EDGE_TYPE_EXAMPLES = SKELETON_EDGE_EXAMPLES + DETAIL_EDGE_EXAMPLES


# ============================================================
# 实际存储的边类型（简化版）
# ============================================================

# 骨架边 - 两个命题同等重要
EDGE_TYPE_SKELETON = "SKELETON"

# 细节边 - 主从依赖关系
EDGE_TYPE_DETAIL = "DETAIL"

# 相似边 - 软连接关系（存在相似内容和相关实体的命题）
EDGE_TYPE_SIMILARITY = "SIMILARITY"

# 命题到实体的包含边（特殊类型）
MENTIONS_ENTITY = "MENTIONS_ENTITY"

# 实际使用的命题边类型列表
PROPOSITION_EDGE_TYPES = [EDGE_TYPE_SKELETON, EDGE_TYPE_DETAIL, EDGE_TYPE_SIMILARITY]

# 所有边类型（包括特殊类型）
ALL_EDGE_TYPES = PROPOSITION_EDGE_TYPES + [MENTIONS_ENTITY]


# ============================================================
# 边类型映射（用于兼容旧格式）
# ============================================================

# 从具体类型映射到简化类型
EDGE_TYPE_MAPPING = {
    # 骨架边示例映射
    "SEQUENCE": EDGE_TYPE_SKELETON,
    "CONTRAST": EDGE_TYPE_SKELETON,
    "CONCESSION": EDGE_TYPE_SKELETON,
    # 细节边示例映射
    "CAUSED_BY": EDGE_TYPE_DETAIL,
    "MOTIVATION": EDGE_TYPE_DETAIL,
    "ELABORATION": EDGE_TYPE_DETAIL,
    "BACKGROUND": EDGE_TYPE_DETAIL,
    # 相似边映射
    "SIMILARITY": EDGE_TYPE_SIMILARITY,
    # 已经是简化类型
    EDGE_TYPE_SKELETON: EDGE_TYPE_SKELETON,
    EDGE_TYPE_DETAIL: EDGE_TYPE_DETAIL,
    EDGE_TYPE_SIMILARITY: EDGE_TYPE_SIMILARITY,
    "SKELETON": EDGE_TYPE_SKELETON,
    "DETAIL": EDGE_TYPE_DETAIL,
    "SIMILARITY": EDGE_TYPE_SIMILARITY,
}


def normalize_edge_type(edge_type: str) -> str:
    """
    将边类型规范化为简化类型

    Args:
        edge_type: 原始边类型（可以是具体类型或简化类型）

    Returns:
        规范化后的边类型（SKELETON、DETAIL 或 SIMILARITY）
    """
    return EDGE_TYPE_MAPPING.get(edge_type, EDGE_TYPE_SIMILARITY)


# ============================================================
# 数据类
# ============================================================

@dataclass
class RSTRelation:
    """RST 关系数据类"""
    source_idx: int  # 源命题索引
    target_idx: int  # 目标命题索引
    relation: str  # 关系类型
    direction: str  # "1->2", "2->1", 或 "1<->2"（双向）
    reason: str  # 选择理由

    def __post_init__(self):
        """初始化后处理：规范化边类型"""
        # 将 relation 规范化为简化类型
        if self.relation not in [EDGE_TYPE_SKELETON, EDGE_TYPE_DETAIL, EDGE_TYPE_SIMILARITY]:
            self.relation = normalize_edge_type(self.relation)

    def __repr__(self):
        return f"RSTRelation({self.source_idx}->{self.target_idx}, {self.relation})"
