"""
Agent 核心数据结构

定义基于状态图驱动的多跳问答系统的核心数据结构。
参考 ggagent3 的 gap-driven exploration 设计。
"""

from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import heapq
import networkx as nx

from .agent_states import AgentState


# ============== 信息缺口状态枚举 ==============

class GapStatus(Enum):
    """信息缺口状态枚举
    
    用于跟踪每个信息缺口的探索状态，避免重复无效检索。
    """
    PENDING = "pending"        # 待探索
    ACTIVE = "active"          # 正在探索
    EXHAUSTED = "exhausted"    # 尝试多次无果
    SATISFIED = "satisfied"    # 已补全


# ============== 信息缺口检索结果 ==============

@dataclass
class GapRetrievalResult:
    """单个意图缺口的检索结果
    
    记录每次检索的详细信息，用于指导后续检索策略。
    """
    gap_description: str                                      # 缺口描述
    status: GapStatus = GapStatus.PENDING                     # 当前状态
    attempt_count: int = 0                                    # 尝试次数
    retrieved_docs: List[str] = field(default_factory=list)   # 检索到的所有文档ID
    selected_evidence: List[str] = field(default_factory=list) # 模型选择的证据
    is_satisfied: bool = False                                # 是否补全
    failure_hints: List[str] = field(default_factory=list)    # 失败提示（用于下轮改写）
    last_rewritten_query: str = ""                            # 上次使用的改写查询
    
    def to_prompt_context(self) -> str:
        """转换为 Prompt 上下文格式"""
        status_desc = {
            GapStatus.PENDING: "待探索",
            GapStatus.ACTIVE: "正在探索",
            GapStatus.EXHAUSTED: "已耗尽（多次尝试无果）",
            GapStatus.SATISFIED: "已补全"
        }
        
        context = f"- 缺口: {self.gap_description}\n"
        context += f"  状态: {status_desc.get(self.status, '未知')}\n"
        context += f"  尝试次数: {self.attempt_count}\n"
        
        if self.retrieved_docs:
            context += f"  检索到的文档数: {len(self.retrieved_docs)}\n"
        
        if self.failure_hints:
            context += f"  失败提示: {'; '.join(self.failure_hints)}\n"
        
        if self.last_rewritten_query:
            context += f"  上次查询: {self.last_rewritten_query}\n"
        
        return context


# ============== 信息缺口结构 ==============

@dataclass
class InfoGap:
    """信息缺口

    PLAN 阶段输出的信息缺口，描述需要查找的信息、关联实体、意图标签等。
    """
    gap_description: str          # 需要查找的信息
    related_entities: List[str]   # 关联实体
    intent_label: str             # 意图标签（自由文本，不再限制为枚举值）
    active_edges: List[str]       # 激活的边类型
    rewritten_query: str = ""     # 改写后的查询文本（用于检索优化）
    
    # 新增：状态跟踪字段
    status: GapStatus = GapStatus.PENDING   # 当前状态
    attempt_count: int = 0                   # 尝试次数
    last_retrieval_result: Optional['GapRetrievalResult'] = None  # 上次检索结果

    def get_active_edges(self) -> Set[str]:
        """获取激活的边类型"""
        return set(self.active_edges)
    
    def is_exhausted(self, max_attempts: int = 2) -> bool:
        """检查是否已耗尽（超过最大尝试次数且未满足）"""
        return self.attempt_count >= max_attempts and self.status != GapStatus.SATISFIED
    
    def mark_active(self):
        """标记为正在探索"""
        self.status = GapStatus.ACTIVE
        self.attempt_count += 1
    
    def mark_satisfied(self):
        """标记为已补全"""
        self.status = GapStatus.SATISFIED
    
    def mark_exhausted(self):
        """标记为已耗尽"""
        self.status = GapStatus.EXHAUSTED


@dataclass
class PlanResult:
    """PLAN 阶段输出结果

    包含识别的信息缺口、摘要等。
    """
    info_gaps: List[InfoGap]                    # 识别的信息缺口列表
    summary: str = ""                           # 意图识别摘要
    visited_entities: Set[str] = field(default_factory=set)  # 已访问的实体集合
    current_knowledge: List[str] = field(default_factory=list)  # 当前已知信息

    def get_active_edges(self) -> Set[str]:
        """获取所有激活的边类型"""
        edges = set()
        for gap in self.info_gaps:
            edges.update(gap.active_edges)
        return edges

    def get_intent_labels(self) -> List[str]:
        """获取所有意图标签"""
        return [gap.intent_label for gap in self.info_gaps]


# ============== 2. MAP 阶段状态相关 ==============

@dataclass
class MapState:
    """MAP 阶段状态

    管理 MAP 阶段的探索过程，包括叶子节点、探索路径、访问实体等。

    使用 RankedPath 替代 Path，支持去重、归一化评分和意图跟踪。
    """
    initial_nodes: List[str] = field(default_factory=list)  # 初始节点
    leaf_nodes: List[str] = field(default_factory=list)    # 当前叶子节点（用于下一轮扩展起点）
    explored_paths: List['RankedPath'] = field(default_factory=list)  # 本轮探索的所有路径（RankedPath 格式）
    top_paths: List['RankedPath'] = field(default_factory=list)  # 本轮 top-k 路径（RankedPath 格式）
    visited_entities: Set[str] = field(default_factory=set)  # 本轮访问的实体
    avg_score: float = 0.0  # 本轮平均分数

    # 按 InfoGap 分组的叶子节点（使用 gap_description 作为键）
    leaf_nodes_by_gap: Dict[str, List[str]] = field(default_factory=dict)

    # 新增：MAP阶段收集的文档详细信息列表 [{doc_id, title, content}]
    top_docs: List[Dict[str, Any]] = field(default_factory=list)
    top_doc_ids: List[str] = field(default_factory=list)


# ============== 3. 历史记录相关 ==============

@dataclass
class RoundTrace:
    """单轮追踪记录（增强版）

    记录单轮的状态转换、意图分配、锚点、迭代次数等。
    """
    round_num: int
    state_transitions: List[AgentState]    # 经历的状态
    plan_result: Optional[PlanResult]      # PLAN 结果
    anchor_count: int                      # 锚点数量
    map_iterations: int                    # MAP 迭代次数
    evidence_count: int                    # 收集的证据数量
    decision: str                          # 关键决策

    # 新增：动作历史详情
    intent_allocation: Dict[str, str] = field(default_factory=dict)  # {entity: intent}
    exploration_paths: List[Dict[str, Any]] = field(default_factory=list)  # 路径详情
    top_1_score: float = 0.0               # top-1 路径分数
    termination_reason: str = ""           # 终止原因

    def to_prompt_context(self) -> str:
        """转换为 Prompt 上下文格式（用于 LLM）"""
        context = f"## Round {self.round_num}\n"

        if self.plan_result:
            context += f"- Plan: {self.plan_result.summary}\n"
            context += f"- Info Gaps ({len(self.plan_result.info_gaps)}):\n"
            for gap in self.plan_result.info_gaps:
                context += f"  * {gap.gap_description} (意图: {gap.intent_label})\n"

        context += f"- Anchors: {self.anchor_count}\n"
        context += f"- MAP iterations: {self.map_iterations}\n"
        context += f"- Top-1 score: {self.top_1_score:.3f}\n"

        if self.termination_reason:
            context += f"- Termination: {self.termination_reason}\n"

        context += f"- Evidence collected: {self.evidence_count}\n"

        return context


@dataclass
class TraceLog:
    """完整的追踪日志

    记录整个问答过程的探索历史，用于生成答案和调试。
    """
    question: str
    rounds: List[RoundTrace] = field(default_factory=list)

    def add_round(self, round_trace: RoundTrace):
        """添加一轮记录"""
        self.rounds.append(round_trace)

    def get_recent_context(self, num_rounds: int = 3) -> str:
        """获取最近几轮的上下文（用于 Prompt）

        Args:
            num_rounds: 获取最近几轮

        Returns:
            格式化的上下文字符串
        """
        recent = self.rounds[-num_rounds:]
        context = f"# Question: {self.question}\n\n"
        context += "# Recent Exploration History:\n"
        for round_trace in recent:
            context += round_trace.to_prompt_context() + "\n"
        return context

    def get_summary(self) -> Dict[str, Any]:
        """获取追踪日志摘要"""
        return {
            "question": self.question,
            "total_rounds": len(self.rounds),
            "decisions": [r.decision for r in self.rounds]
        }


# ============== 4. 锚点队列管理 ==============

class AnchorQueue:
    """锚点队列管理器

    功能：
    1. 缓存历史锚点（避免重复探索）
    2. 按优先级排序锚点（使用 heapq 实现优先队列）
    3. 支持批量补充锚点

    性能优化：
    - 使用 heapq 实现优先队列，add_anchors 从 O(n log n) → O(n log k)
    - get_top_k 从 O(1) → O(k log k)，但不需要全排序
    - 移除 O(n×k) 的词集合相似度重复检测，仅使用 _visited 集合
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        max_queue_size: int = 100,
    ):
        self.graph = graph
        self.max_queue_size = max_queue_size
        # 使用最小堆（优先级取负值实现最大堆）：[(priority, node_id), ...]
        self._queue: List[Tuple[float, str]] = []
        self._visited: Set[str] = set()

    async def add_anchors(self, node_ids: List[str], priorities: List[float]):
        """批量添加锚点 - O(n log k) 其中 n 是添加数量

        【性能优化】移除 O(n×k) 的词集合相似度重复检测，
        仅使用 _visited 集合进行精确去重。

        Args:
            node_ids: 节点ID列表
            priorities: 优先级列表（与node_ids一一对应）
        """
        for node_id, priority in zip(node_ids, priorities):
            if node_id in self._visited:
                continue

            # 使用负优先级实现最大堆（heapq 是最小堆）
            heapq.heappush(self._queue, (-priority, node_id))
            self._visited.add(node_id)

        # 限制队列大小 - O(k log k)
        while len(self._queue) > self.max_queue_size:
            heapq.heappop(self._queue)  # 弹出最小（优先级最低）

    async def get_top_k(self, k: int) -> List[str]:
        """获取 top-k 锚点 - O(k log k)

        Args:
            k: 获取数量

        Returns:
            top-k 节点ID列表（按优先级降序）
        """
        # heapq 的 pop 已经是按优先级排序的
        result = []
        # 使用临时副本，避免影响原队列
        temp_queue = self._queue.copy()

        for _ in range(min(k, len(temp_queue))):
            if temp_queue:
                neg_priority, node_id = heapq.heappop(temp_queue)
                result.append(node_id)

        return result

    async def refill_from_entities(
        self,
        entity_ids: List[str],
        embedding_client,
        top_k_per_entity: int = 3
    ):
        """从实体节点补充锚点

        Args:
            entity_ids: 实体节点ID列表
            embedding_client: 嵌入客户端
            top_k_per_entity: 每个实体获取的锚点数
        """
        from ..proposition_graph.graph_builder import MENTIONS_ENTITY, PROPOSITION_NODE

        new_anchors = []
        priorities = []

        for entity_id in entity_ids:
            # 获取实体的邻居命题节点
            neighbors = list(self.graph.neighbors(entity_id))
            proposition_neighbors = [
                n for n in neighbors
                if self.graph.nodes[n].get("node_type") == PROPOSITION_NODE
            ]

            # 基于相似度选择 top-k
            for prop_id in proposition_neighbors[:top_k_per_entity]:
                if prop_id not in self._visited:
                    new_anchors.append(prop_id)
                    # 使用节点的度作为优先级
                    priorities.append(float(self.graph.degree[prop_id]))

        if new_anchors:
            await self.add_anchors(new_anchors, priorities)

    def clear(self):
        """清空队列"""
        self._queue.clear()
        self._visited.clear()

    def __len__(self) -> int:
        return len(self._queue)


# ============== 5. Agent 结果相关 ==============

@dataclass
class AgentResult:
    """Agent 最终结果

    包含生成的答案、置信度、追踪日志、证据和路径等信息。
    """
    answer: str                          # 生成的答案
    short_answer: str = ""               # 适合 EM/F1 的简短答案
    confidence: float = 0.0              # 置信度
    trace_log: Optional[TraceLog] = None # 完整追踪日志
    collected_evidence: List[str] = field(default_factory=list)        # 收集的所有证据
    final_documents: List[Dict[str, Any]] = field(default_factory=list) # 最终引用的文档
    final_paths: List['RankedPath'] = field(default_factory=list)      # 最终的 top-k 路径
    termination_reason: str = ""         # 终止原因

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "answer": self.answer,
            "short_answer": self.short_answer,
            "confidence": self.confidence,
            "termination_reason": self.termination_reason,
            "trace_summary": self.trace_log.get_summary() if self.trace_log else {},
            "evidence_count": len(self.collected_evidence),
            "doc_count": len(self.final_documents),
            "top_docs": self.final_documents,
            "path_count": len(self.final_paths)
        }


# Path 类的前向引用
from .path_structures import Path
# 使用 subgraph_structures.RankedPath 作为新的路径格式
from .subgraph_structures import RankedPath
