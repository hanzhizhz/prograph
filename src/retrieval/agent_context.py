"""
Agent 上下文

定义基于状态图驱动的多跳问答系统的运行时上下文。
"""

from typing import List, Set, Optional, Dict, Any
from dataclasses import dataclass, field
import networkx as nx

from .agent_states import AgentState
from .agent_data_structures import MapState, TraceLog, AnchorQueue, GapStatus, GapRetrievalResult


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
    context_documents: List[Dict[str, Any]] = field(default_factory=list)  # 文档级上下文记忆
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

    # ========== 信息缺口历史 ==========
    gap_history: Dict[str, GapRetrievalResult] = field(default_factory=dict)
    # key = gap_id, value = 该缺口的累计检索结果

    # ========== 配置 ==========
    max_documents: int = 20  # 最大文档数量
    max_gap_attempts: int = 2  # 单个缺口最大尝试次数

    def can_continue_round(self) -> bool:
        """是否可以继续下一轮"""
        return self.current_round < self.max_rounds

    def next_round(self):
        """进入下一轮"""
        self.current_round += 1

    def add_documents(self, docs: List[Dict[str, Any]]):
        """添加文档到上下文记忆（去重）
        
        Args:
            docs: 文档列表，每个文档包含 doc_id, title, content
        """
        existing_ids = {d["doc_id"] for d in self.context_documents}
        
        for doc in docs:
            if doc["doc_id"] not in existing_ids:
                self.context_documents.append(doc)
                existing_ids.add(doc["doc_id"])
        
        # 限制文档数量
        if len(self.context_documents) > self.max_documents:
            self.context_documents = self.context_documents[-self.max_documents:]

    def get_context_documents(self) -> List[Dict[str, Any]]:
        """获取所有上下文文档（不限制数量）
            
        Returns:
            文档列表
        """
        return self.context_documents

    def transition_to(self, new_state: AgentState):
        """转换到新状态

        Args:
            new_state: 新状态
        """
        self.current_state = new_state

    # ========== 信息缺口管理方法 ==========
    
    def get_gap_status(self, gap_id: str) -> Optional[GapStatus]:
        """获取指定缺口的状态
        
        Args:
            gap_id: 缺口ID
            
        Returns:
            缺口状态，如果不存在则返回 None
        """
        if gap_id in self.gap_history:
            return self.gap_history[gap_id].status
        return None
    
    def get_gap_result(self, gap_id: str) -> Optional[GapRetrievalResult]:
        """获取指定缺口的检索结果
        
        Args:
            gap_id: 缺口ID
            
        Returns:
            检索结果，如果不存在则返回 None
        """
        return self.gap_history.get(gap_id)
    
    def update_gap_result(self, result: GapRetrievalResult):
        """更新缺口的检索结果
        
        Args:
            result: 检索结果
        """
        self.gap_history[result.gap_id] = result
    
    def is_gap_exhausted(self, gap_id: str) -> bool:
        """检查缺口是否已耗尽
        
        Args:
            gap_id: 缺口ID
            
        Returns:
            是否已耗尽
        """
        result = self.gap_history.get(gap_id)
        if result is None:
            return False
        return result.status == GapStatus.EXHAUSTED
    
    def is_gap_satisfied(self, gap_id: str) -> bool:
        """检查缺口是否已补全
        
        Args:
            gap_id: 缺口ID
            
        Returns:
            是否已补全
        """
        result = self.gap_history.get(gap_id)
        if result is None:
            return False
        return result.status == GapStatus.SATISFIED
    
    def get_active_gaps(self) -> List[GapRetrievalResult]:
        """获取所有活跃状态的缺口
        
        Returns:
            活跃缺口列表
        """
        return [
            result for result in self.gap_history.values()
            if result.status == GapStatus.ACTIVE
        ]
    
    def get_exhausted_gaps(self) -> List[GapRetrievalResult]:
        """获取所有耗尽状态的缺口
        
        Returns:
            耗尽缺口列表
        """
        return [
            result for result in self.gap_history.values()
            if result.status == GapStatus.EXHAUSTED
        ]
    
    def get_gap_history_prompt(self) -> str:
        """生成缺口历史的 Prompt 上下文
        
        Returns:
            格式化的缺口历史字符串
        """
        if not self.gap_history:
            return "无历史缺口记录"
        
        context = ""
        for gap_desc, result in self.gap_history.items():
            context += result.to_prompt_context() + "\n"
        
        return context.strip()
    
    def find_similar_gap(self, gap_description: str, threshold: float = 0.85) -> Optional[GapRetrievalResult]:
        """查找与给定描述相似的历史缺口
        
        简单实现：基于词集合的 Jaccard 相似度
        
        Args:
            gap_description: 缺口描述
            threshold: 相似度阈值
            
        Returns:
            相似的历史缺口结果，如果没有则返回 None
        """
        def jaccard_similarity(s1: str, s2: str) -> float:
            """计算两个字符串的 Jaccard 相似度"""
            set1 = set(s1.lower().split())
            set2 = set(s2.lower().split())
            if not set1 or not set2:
                return 0.0
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            return intersection / union if union > 0 else 0.0
        
        for existing_desc, result in self.gap_history.items():
            if jaccard_similarity(gap_description, existing_desc) >= threshold:
                return result
        
        return None

    @classmethod
    def create(
        cls,
        question: str,
        graph: nx.DiGraph,
        max_rounds: int = 5,
        max_documents: int = 20,
        max_gap_attempts: int = 2
    ) -> 'AgentContext':
        """创建 Agent 上下文

        Args:
            question: 问题
            graph: 图
            max_rounds: 最大轮数
            max_documents: 最大文档数量
            max_gap_attempts: 单个缺口最大尝试次数

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
            max_documents=max_documents,
            max_gap_attempts=max_gap_attempts
        )


# Path 类的前向引用
from .path_structures import Path
