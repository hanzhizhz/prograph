"""
Agent 上下文

定义基于状态图驱动的多跳问答系统的运行时上下文。
"""

from typing import List, Set, Optional
from dataclasses import dataclass, field
import networkx as nx

from .agent_states import AgentState
from .agent_data_structures import MapState, TraceLog, AnchorQueue


@dataclass
class AgentContext:
    """Agent 运行时上下文

    管理整个问答过程中的状态、证据、访问记录等。
    """

    # ========== 基础信息 ==========
    question: str                          # 原始问题
    current_state: AgentState              # 当前状态
    current_round: int = 0                 # 当前轮数
    max_rounds: int = 5                    # 最大轮数

    # ========== 检索状态 ==========
    collected_evidence: List[str] = field(default_factory=list)
    visited_entities: Set[str] = field(default_factory=set)

    # ========== MAP 阶段状态 ==========
    map_state: Optional[MapState] = None
    accumulated_paths: List['Path'] = field(default_factory=list)
    prev_leaf_nodes: List[str] = field(default_factory=list)  # 上一轮的叶子节点
    _prev_top_score: Optional[float] = None  # 上一轮 top-1 路径分数（用于自适应终止）

    # ========== 资源引用 ==========
    graph: nx.DiGraph = None
    anchor_queue: AnchorQueue = None
    trace_log: TraceLog = None

    # ========== PLAN 结果 ==========
    plan_result: Optional['PlanResult'] = None  # 当前轮的 PLAN 结果

    # ========== 配置 ==========
    max_evidence: int = 50

    def can_continue_round(self) -> bool:
        """是否可以继续下一轮"""
        return self.current_round < self.max_rounds

    def next_round(self):
        """进入下一轮"""
        self.current_round += 1

    def add_evidence(self, evidence: List[str]):
        """添加证据

        Args:
            evidence: 证据列表
        """
        self.collected_evidence.extend(evidence)
        # 限制证据数量
        if len(self.collected_evidence) > self.max_evidence:
            self.collected_evidence = self.collected_evidence[-self.max_evidence:]

    def get_recent_evidence(self, count: int = 20) -> List[str]:
        """获取最近的证据

        Args:
            count: 获取数量

        Returns:
            最近的证据列表
        """
        return self.collected_evidence[-count:] if self.collected_evidence else []

    def transition_to(self, new_state: AgentState):
        """转换到新状态

        Args:
            new_state: 新状态
        """
        self.current_state = new_state

    @classmethod
    def create(
        cls,
        question: str,
        graph: nx.DiGraph,
        max_rounds: int = 5,
        max_evidence: int = 50
    ) -> 'AgentContext':
        """创建 Agent 上下文

        Args:
            question: 问题
            graph: 图
            max_rounds: 最大轮数
            max_evidence: 最大证据数量

        Returns:
            Agent 上下文实例
        """
        from .agent_data_structures import TraceLog, AnchorQueue

        return cls(
            question=question,
            current_state=AgentState.CHECK_PLAN,
            current_round=0,
            max_rounds=max_rounds,
            graph=graph,
            anchor_queue=AnchorQueue(graph),
            trace_log=TraceLog(question=question),
            max_evidence=max_evidence
        )


# Path 类的前向引用
from .path_structures import Path
