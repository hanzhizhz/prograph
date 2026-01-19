"""
Agent 状态枚举

定义基于状态图驱动的多跳问答系统的5个核心状态。
"""

from enum import Enum


class AgentState(Enum):
    """Agent 状态枚举

    实现 5 个状态的循环：
    CHECK_PLAN -> RETRIEVE -> MAP -> UPDATE -> (CHECK_PLAN or ANSWER)
    """

    CHECK_PLAN = "CHECK_PLAN"
    RETRIEVE = "RETRIEVE"
    MAP = "MAP"
    UPDATE = "UPDATE"
    ANSWER = "ANSWER"
