"""
Agent 状态机核心

实现基于状态图驱动的多跳问答系统的状态机核心。
"""

import json
import asyncio
import re
from typing import List, Set, Optional, Dict, Any
import networkx as nx
import numpy as np

from .agent_states import AgentState
from .agent_context import AgentContext
from .agent_data_structures import (
    InfoGap, PlanResult, MapState, TraceLog,
    RoundTrace, AnchorQueue, AgentResult,
    GapStatus, GapRetrievalResult
)
from .agent_prompts import (
    get_check_plan_prompt, get_anchor_judge_prompt,
    get_extract_from_docs_prompt, get_gap_satisfaction_prompt,
    get_forced_answer_prompt
)
from .path_scorer import PathScorer, NodeScore
from .path_structures import Path
from .path_selector import PathSelector
from .subgraph_structures import IntentSubgraph, AggregatedPaths, RankedPath
from ..llm.base import BaseLLM
from ..llm.embedding_client import OpenAIEmbeddingClient
from ..proposition_graph.graph_builder import PROPOSITION_NODE, ENTITY_NODE, GLOBAL_ENTITY_NODE, MENTIONS_ENTITY
from ..utils.json_parser import extract_json_with_fallback
from ..utils.llm_response_parser import LLMResponseParser
from ..utils.timing import TimingLogger


class AgentStateMachine:
    """Agent 状态机

    实现 5 个状态的循环：
    CHECK_PLAN -> RETRIEVE -> MAP -> UPDATE -> (CHECK_PLAN or ANSWER)
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        llm: BaseLLM,
        embedding_client: OpenAIEmbeddingClient,
        path_scorer: PathScorer,
        path_selector: PathSelector,
        max_rounds: int = 5,
        semantic_weight: float = 0.3,
        bridge_weight: float = 0.5,
        map_beam_width: int = 5,
        map_max_iterations: int = 10,
        map_score_plateau_threshold: float = 0.02,
        map_score_plateau_window: int = 2,
        require_vector_index: bool = False,
        embedding_cache_manager: Optional[Any] = None,
        persistence_dir: Optional[str] = None,
        index_dir: Optional[str] = None,
        enable_timing: bool = True,
        top_k_docs: int = 5,
        # 【单例优化】预加载的资源参数
        adjacency_cache: Optional[Dict[str, Dict[str, List[Dict[str, Any]]]]] = None,
        predecessors_cache: Optional[Dict[str, List[str]]] = None,
        proposition_index: Optional[Any] = None,
        entity_lookup_index: Optional[Any] = None,
    ):
        self.graph = graph
        self.llm = llm
        self.embedding_client = embedding_client
        self.path_scorer = path_scorer
        self.path_selector = path_selector
        self.embedding_cache_manager = embedding_cache_manager
        self.persistence_dir = persistence_dir
        self._index_dir = index_dir

        # 配置
        self.max_rounds = max_rounds
        self.semantic_weight = semantic_weight
        self.bridge_weight = bridge_weight
        self.map_beam_width = map_beam_width
        self.map_max_iterations = map_max_iterations
        self.map_score_plateau_threshold = map_score_plateau_threshold
        self.map_score_plateau_window = map_score_plateau_window
        self.top_k_docs = top_k_docs

        # 【性能优化】向量索引配置
        self._require_vector_index = require_vector_index
        self._vector_index_loaded = False

        # 权重传递给 path_scorer
        self.path_scorer.semantic_weight = semantic_weight
        self.path_scorer.bridge_weight = bridge_weight

        # JSON 解析器
        self.response_parser = LLMResponseParser(llm)

        # 【性能监控】Timing logger
        self.timing_logger = TimingLogger(enabled=enable_timing)

        # 【单例优化】优先使用传入的命题索引，避免重复加载
        self._proposition_index: Optional[Any] = proposition_index
        self._question_embedding_cache: Dict[str, np.ndarray] = {}

        # 【单例优化】优先使用传入的邻接缓存，避免重复加载
        if adjacency_cache is not None:
            self._adjacency_cache = adjacency_cache
            self._predecessors_cache = predecessors_cache if predecessors_cache is not None else {}
        else:
            # 回退到加载或构建
            self._adjacency_cache: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
            self._load_or_build_adjacency_cache()

        # 【单例优化】优先使用传入的实体查找索引，避免重复加载
        if entity_lookup_index is not None:
            self._entity_lookup_index = entity_lookup_index
        else:
            # 回退到加载或构建
            from .entity_lookup_index import EntityLookupIndex
            self._entity_lookup_index = EntityLookupIndex(graph, persistence_dir=persistence_dir)
            if not self._entity_lookup_index.load():
                self._entity_lookup_index.build()

        # 【性能优化】实体名称向量索引：用于模糊匹配（O(log n)复杂度）
        from .entity_name_index import EntityNameIndex
        index_path = None
        if persistence_dir:
            from pathlib import Path
            # 使用与预构建文件一致的路径
            index_path = str(Path(persistence_dir) / "indices" / "entity")
        self._entity_name_index = EntityNameIndex(
            graph=graph,
            embedding_client=embedding_client,
            index_path=index_path,
            dim=None  # 自动从元数据获取维度
        )

        # 【资源加载】加载节点映射和文档元数据
        self._node_mappings: Dict[str, Any] = {}
        self._doc_metadata: Dict[str, Any] = {}
        self._load_mappings_and_metadata()

    def _load_mappings_and_metadata(self):
        """加载节点映射和文档元数据"""
        if not self.persistence_dir:
            return

        from pathlib import Path
        mapping_file = Path(self.persistence_dir) / "node_mappings.json"
        meta_file = Path(self.persistence_dir) / "doc_metadata.json"

        if mapping_file.exists():
            try:
                with open(mapping_file, 'r', encoding='utf-8') as f:
                    self._node_mappings = json.load(f)
                print(f"✓ 节点映射已加载: {len(self._node_mappings.get('doc_to_propositions', {}))} 个文档")
            except Exception as e:
                print(f"警告: 节点映射加载失败 ({e})")

        if meta_file.exists():
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    self._doc_metadata = json.load(f)
                print(f"✓ 文档元数据已加载: {len(self._doc_metadata)} 个文档")
            except Exception as e:
                print(f"警告: 文档元数据加载失败 ({e})")

    async def run(
        self,
        question: str,
        initial_anchors: Optional[List[str]] = None,
        verbose: bool = True,
        debug: bool = False,
    ) -> AgentResult:
        """运行完整的 Agent 流程

        Args:
            question: 问题
            initial_anchors: 初始锚点（可选，如果不提供将自动查找）
            verbose: 是否输出 INFO 级别日志
            debug: 是否输出 DEBUG 级别日志（状态输入输出、LLM 调用）

        Returns:
            Agent 结果
        """
        # Truncate question for timing display
        question_display = question

        with self.timing_logger.time(f"Total QA: {question_display}"):
            self._verbose = verbose
            self._debug = debug

            if self._verbose:
                print("=" * 60)
                print("ProGraph Agent 开始运行")
                print(f"问题: {question}")
                print(f"最大轮数: {self.max_rounds}")
                if initial_anchors:
                    print(f"初始锚点: {len(initial_anchors)} 个（用户提供）")
                else:
                    print(f"初始锚点: 自动查找")
                print("=" * 60)

            # 1. 初始化上下文
            context = AgentContext.create(
                question=question,
                graph=self.graph,
                max_rounds=self.max_rounds
            )

            # 添加初始锚点
            if initial_anchors:
                await context.anchor_queue.add_anchors(
                    initial_anchors,
                    [1.0] * len(initial_anchors)
                )

            # 2. 主循环：状态转换
            termination_reason = "max_rounds_reached"

            while context.can_continue_round():
                state = context.current_state

                if self._verbose:
                    print(f"\n--- Round {context.current_round + 1}/{self.max_rounds} ---")
                    print(f"当前状态: {state.value}")

                if state == AgentState.CHECK_PLAN:
                    with self.timing_logger.time(f"Round {context.current_round + 1} - CHECK_PLAN"):
                        can_answer = await self._state_check_plan(context)
                    if can_answer:
                        if self._verbose:
                            print(f"✓ 信息充足，进入 ANSWER 阶段")
                        context.transition_to(AgentState.ANSWER)
                        termination_reason = "information_sufficient"
                        break
                    else:
                        if self._verbose:
                            print(f"→ 信息不足，进入 RETRIEVE 阶段")

                        # 记录追踪日志
                        round_trace = RoundTrace(
                            round_num=context.current_round,
                            state_transitions=[AgentState.CHECK_PLAN],
                            plan_result=context.plan_result,
                            anchor_count=0,
                            map_iterations=0,
                            document_count=len(context.context_documents),
                            decision="Check and plan completed"
                        )
                        context.trace_log.add_round(round_trace)

                        context.transition_to(AgentState.RETRIEVE)

                elif state == AgentState.RETRIEVE:
                    with self.timing_logger.time(f"Round {context.current_round + 1} - RETRIEVE"):
                        gap_to_anchors = await self._state_retrieve(context)
                    # 统计所有有效锚点总数
                    total_anchors = sum(len(anchors) for anchors in gap_to_anchors.values())
                    if not gap_to_anchors or total_anchors == 0:
                        if self._verbose:
                            print(f"  RETRIEVE: 未找到有效锚点，终止")
                        context.transition_to(AgentState.ANSWER)
                        termination_reason = "no_valid_anchors"
                        break
                    else:
                        if self._verbose:
                            print(f"  RETRIEVE: 获得 {total_anchors} 个有效锚点，分布在 {len(gap_to_anchors)} 个信息缺口")
                            # 显示每个gap的锚点信息
                            for gap_id, anchors in gap_to_anchors.items():
                                print(f"    Gap ID: {gap_id}")
                                print(f"      锚点数量: {len(anchors)}")
                                for anchor_data in anchors:  # 显示每个gap的前2个锚点
                                    print(f"      • {anchor_data.get('text', '')}")

                        # 注意：MapState 已在 _state_retrieve 中创建，不需要重复创建
                        context.transition_to(AgentState.MAP)

                elif state == AgentState.MAP:
                    with self.timing_logger.time(f"Round {context.current_round + 1} - MAP"):
                        await self._state_map(context)

                    if self._verbose and context.map_state:
                        top_path = context.map_state.top_paths[0] if context.map_state.top_paths else None
                        top_score = getattr(top_path, 'normalized_score', getattr(top_path, 'accumulated_score', 0)) if top_path else 0
                        print(f"  MAP: 探索 {len(context.map_state.explored_paths)} 条路径, Top-1: {top_score:.3f}")

                    context.transition_to(AgentState.UPDATE)

                elif state == AgentState.UPDATE:
                    with self.timing_logger.time(f"Round {context.current_round + 1} - UPDATE"):
                        await self._state_update(context)

                        # 记录本轮动作历史
                        self._record_round_actions(context)

                    if self._verbose:
                        doc_count = len(context.context_documents)
                        print(f"  UPDATE: 上下文文档 {doc_count} 个")
                        print(f"    访问实体: {len(context.visited_entities)}")

                    context.next_round()
                    context.transition_to(AgentState.CHECK_PLAN)

                elif state == AgentState.ANSWER:
                    termination_reason = "answer_generated"
                    break

            # 3. 生成最终答案（一次 LLM 调用同时生成 answer + short_answer）
            is_forced = (termination_reason != "information_sufficient" and termination_reason != "answer_generated")
            if is_forced and self._verbose:
                print(f"\n[强制回答] 触发强制回答模式 (原因: {termination_reason})")
                
            with self.timing_logger.time("ANSWER - Generate final answer"):
                answer, short_answer = await self._state_answer(context, is_forced=is_forced)

            # 4. 返回结果
            if self._verbose:
                print("\n" + "=" * 60)
                print("Agent 运行完成")
                print(f"终止原因: {termination_reason}")
                print(f"总轮数: {context.current_round + 1}")
                print(f"上下文文档: {len(context.context_documents)} 个")
                print(f"访问实体: {len(context.visited_entities)} 个")
                print(f"累计路径: {len(context.accumulated_paths)} 条")
                print(f"回答: {short_answer}")
                print("=" * 60)

            return AgentResult(
                answer=answer,
                short_answer=short_answer,
                confidence=0.7,  # 可以从 answer_generator 中获取
                trace_log=context.trace_log,
                final_documents=context.context_documents,
                final_paths=[],  # 可以从 context.accumulated_paths 中生成
                termination_reason=termination_reason
            )

    async def _state_check_plan(self, context: AgentContext) -> bool:
        """CHECK_PLAN: 判断可答性并识别信息缺口

        一次 LLM 调用同时完成：
        1. 判断当前信息是否足够回答
        2. 如果不够，识别信息缺口并分配意图

        Returns:
            bool: 是否可以回答
        """
        # 直接使用文档格式（不限制数量和内容长度）
        recent_docs = context.get_context_documents()
        evidence_list = [
            f"【{doc.get('title', '未知文档')}】\n{doc.get('content', '')}"
            for doc in recent_docs
        ]
        
        # 获取历史缺口状态信息
        gap_history = context.get_gap_history_prompt()

        if self._debug:
            print(f"    [DEBUG] CHECK_PLAN 输入: {len(recent_docs)} 个文档, {len(context.visited_entities)} 个实体")
            if context.gap_history:
                print(f"    [DEBUG] 历史缺口数: {len(context.gap_history)}")
                for gap_id, result in context.gap_history.items():
                    print(f"      - {gap_id} : {result.status.value}, attempts={result.attempt_count}")

        # 生成 prompt（包含历史缺口状态）
        prompt = get_check_plan_prompt(
            question=context.question,
            documents=evidence_list,
            visited_entities=context.visited_entities,
            gap_history=gap_history
        )

        try:
            messages = [{"role": "user", "content": prompt}]
            with self.timing_logger.time("  LLM: Check plan"):
                response = await self.llm.generate(
                    messages=messages,
                    temperature=0.2,  # 介于 CHECK(0.1) 和 PLAN(0.3) 之间
                    max_tokens=512
                )

            # 解析响应
            result = await self._parse_json_response(response, target_field="can_answer")

            if self._debug:
                print(f"    [DEBUG] 原始响应:\n{response}")
                print(f"    [DEBUG] 解析结果: {result}")

            can_answer = result.get("can_answer", False)

            # 如果不能回答，解析 info_gaps 并存储
            if not can_answer and "info_gaps" in result:
                info_gaps = []
                for i, gap_data in enumerate(result.get("info_gaps", [])):
                    # 获取或生成 gap_id
                    gap_id = gap_data.get("gap_id", f"gap_{i+1}")
                    
                    # 意图标签现在使用自由文本，不再限制为枚举值
                    intent_label_str = gap_data.get("intent_label", "Bridge_Entity")
                    gap_description = gap_data.get("gap_description", "")
                    
                    # 检查是否与历史缺口相似
                    similar_gap = context.find_similar_gap(gap_description)
                    
                    # 如果找到相似的已耗尽缺口，跳过该缺口
                    if similar_gap and similar_gap.status == GapStatus.EXHAUSTED:
                        if self._debug:
                            print(f"    [DEBUG] 跳过已耗尽的相似缺口: {gap_description[:40]}...")
                        continue
                    
                    # 如果找到相似的活跃缺口，继承其状态
                    new_gap = InfoGap(
                        gap_id=gap_id,
                        gap_description=gap_description,
                        related_entities=gap_data.get("related_entities", []),
                        intent_label=intent_label_str,
                        active_edges=gap_data.get("active_edges", [])
                    )
                    
                    if similar_gap:
                        # 继承历史状态
                        new_gap.status = similar_gap.status
                        new_gap.attempt_count = similar_gap.attempt_count
                        new_gap.last_retrieval_result = similar_gap
                    
                    info_gaps.append(new_gap)

                # 存储 PlanResult
                context.plan_result = PlanResult(
                    info_gaps=info_gaps,
                    summary=result.get("summary", ""),
                    visited_entities=context.visited_entities.copy(),
                    current_knowledge=recent_docs.copy()  # 直接使用文档列表
                )

                if self._verbose:
                    print(f"  CHECK_PLAN: 识别 {len(info_gaps)} 个信息缺口")
                    for gap in info_gaps:
                        status_str = f"[{gap.status.value}]" if gap.status != GapStatus.PENDING else ""
                        print(f"    - {gap.gap_description} [{gap.intent_label}] {status_str} active_edges={gap.active_edges}")

            return can_answer

        except Exception as e:
            print(f"CHECK_PLAN 阶段失败: {e}")
            # 如果有足够文档，认为可以回答
            return len(context.context_documents) >= 5

    async def _rewrite_queries(
        self,
        question: str,
        info_gaps: List[InfoGap],
        context: Optional[AgentContext] = None
    ) -> List[str]:
        """使用 LLM 改写查询文本

        Args:
            question: 原始问题
            info_gaps: 信息缺口列表
            context: Agent 上下文（用于获取重试信息）

        Returns:
            List[改写后的查询]，按 info_gaps 顺序对应
        """
        from .agent_prompts import get_query_rewrite_prompt, get_query_rewrite_retry_prompt

        if not info_gaps:
            return []

        # 检查是否有重试缺口
        retry_info = {}
        if context:
            for i, gap in enumerate(info_gaps):
                # 检查该缺口是否有历史记录
                gap_result = context.get_gap_result(gap.gap_id)
                if gap_result and gap_result.attempt_count > 0:
                    retry_info[i] = {
                        "last_query": gap_result.last_rewritten_query,
                        "failure_hints": gap_result.failure_hints,
                        "retrieved_docs": gap_result.retrieved_docs
                    }
                # 也检查相似缺口
                elif gap.attempt_count > 0 and gap.last_retrieval_result:
                    retry_info[i] = {
                        "last_query": gap.last_retrieval_result.last_rewritten_query,
                        "failure_hints": gap.last_retrieval_result.failure_hints,
                        "retrieved_docs": gap.last_retrieval_result.retrieved_docs
                    }

        # 根据是否有重试缺口选择不同的 prompt
        if retry_info:
            prompt = get_query_rewrite_retry_prompt(question, info_gaps, retry_info)
            if self._debug:
                print(f"    [DEBUG] QUERY_REWRITE: 使用增强版 prompt，{len(retry_info)} 个重试缺口")
        else:
            prompt = get_query_rewrite_prompt(question, info_gaps)

        try:
            messages = [{"role": "user", "content": prompt}]
            response = await self.llm.generate(
                messages=messages,
                temperature=0.1 if not retry_info else 0.3,  # 重试时稍高温度以增加多样性
                max_tokens=512
            )

            # 解析响应
            result = await self._parse_json_response(response, target_field="queries")
            queries = result.get("queries", [])

            # 构建索引映射
            rewritten_list = [None] * len(info_gaps)
            for item in queries:
                index = item.get("index", -1)
                rewritten = item.get("query", "")
                if 0 <= index < len(info_gaps) and rewritten:
                    rewritten_list[index] = rewritten

            # 填充缺失的项（使用原始描述）
            for i, gap in enumerate(info_gaps):
                if rewritten_list[i] is None:
                    rewritten_list[i] = gap.gap_description

            if self._debug:
                print(f"    [DEBUG] QUERY_REWRITE: 改写了 {len(rewritten_list)} 个查询")
                for i, (gap, query) in enumerate(zip(info_gaps, rewritten_list)):
                    is_retry = "[重试]" if i in retry_info else ""
                    print(f"      [{i}] {is_retry} {gap.gap_description[:40]}... → {query}")

            return rewritten_list

        except Exception as e:
            print(f"查询改写失败: {e}，使用原始缺口描述")
            # 失败时返回原始缺口描述
            return [gap.gap_description for gap in info_gaps]

    async def _state_retrieve(self, context: AgentContext) -> Dict[str, List[Dict]]:
        """RETRIEVE: 按意图分组并行检索锚点

        1. 使用 LLM 改写查询（基于当前 info_gaps）
        2. 按意图分组并行检索锚点
        3. LLM 验证并创建/更新 MapState
        
        Returns:
            Dict[str, List[Dict]]: gap_id -> 验证通过的锚点节点数据列表
                每个节点数据字典包含: node_id, text, node_type, doc_id 等属性
        """
        from .agent_data_structures import MapState

        # 1. 查询改写（传入 context 以支持重试缺口的增强改写）
        rewritten_queries = await self._rewrite_queries(
            question=context.question,
            info_gaps=context.plan_result.info_gaps,
            context=context
        )

        for i, gap in enumerate(context.plan_result.info_gaps):
            if i < len(rewritten_queries):
                gap.rewritten_query = rewritten_queries[i]
            else:
                gap.rewritten_query = gap.gap_description

        # 2. 分组并行检索（拆分计时）
        with self.timing_logger.time("  Vector search: Propositions"):
            gap_to_anchors = await self._retrieve_anchors_by_gaps(
                context=context,
                use_rewritten_queries=True
            )

        # 3. 收集所有锚点用于后续处理（从节点数据字典中提取ID）
        all_anchors = []
        for anchors in gap_to_anchors.values():
            for anchor_data in anchors:
                all_anchors.append(anchor_data['node_id'])

        # 去重
        seen = set()
        unique_anchors = []
        for anchor_id in all_anchors:
            if anchor_id not in seen:
                seen.add(anchor_id)
                unique_anchors.append(anchor_id)

        if not unique_anchors:
            return {}

        # 4. 对锚点按优先级排序（保持现有逻辑）
        prioritized_anchors = await self._prioritize_anchors(
            question=context.question,
            candidates=unique_anchors,
            info_gaps=context.plan_result.info_gaps
        )

        # 5. LLM 验证锚点（保持现有逻辑）
        with self.timing_logger.time("  LLM: Validate anchors"):
            valid_anchor_ids = await self._validate_anchors(context, prioritized_anchors)

        # 6. 创建或更新 MapState（按意图分组）
        # 对于本轮的每个 InfoGap，保留验证通过的锚点
        # gap_to_anchors 现在是 {gap_id: [节点数据字典列表]}
        validated_gap_to_anchors = {}
        
        # 【修复】跨gap去重：跟踪全局已分配的节点，避免同一节点被多个gap重复使用
        global_assigned_nodes = set()
        
        for gap_id, anchors in gap_to_anchors.items():
            # 过滤出验证通过的锚点（保留完整数据结构）
            valid_for_gap = []
            for anchor_data in anchors:
                node_id = anchor_data['node_id']
                # 仅添加验证通过且尚未分配给其他gap的节点
                if node_id in valid_anchor_ids and node_id not in global_assigned_nodes:
                    valid_for_gap.append(anchor_data)
                    global_assigned_nodes.add(node_id)
            
            validated_gap_to_anchors[gap_id] = valid_for_gap

        # 提取ID用于 MapState
        validated_ids_by_gap = {gap_id: [a['node_id'] for a in anchors] 
                               for gap_id, anchors in validated_gap_to_anchors.items()}

        # 如果存在上一轮的 leaf_nodes_by_gap，合并（增量式探索）
        if context.map_state and context.map_state.leaf_nodes_by_gap:
            merged_leaf_nodes_by_gap = {}
            # 合并上一轮和本轮的结果
            all_gap_ids = set(context.map_state.leaf_nodes_by_gap.keys()) | set(validated_ids_by_gap.keys())
            for gap_id in all_gap_ids:
                prev_nodes = context.map_state.leaf_nodes_by_gap.get(gap_id, [])
                curr_nodes = validated_ids_by_gap.get(gap_id, [])
                merged = list(set(prev_nodes + curr_nodes))
                merged_leaf_nodes_by_gap[gap_id] = merged

            # 创建新 MapState，继承上一轮的状态
            context.map_state = MapState(
                initial_nodes=list(set(context.prev_leaf_nodes + valid_anchor_ids)),
                explored_paths=[],
                top_paths=[],
                visited_entities=set(),
                avg_score=0.0,
                leaf_nodes_by_gap=merged_leaf_nodes_by_gap
            )
        else:
            # 首次创建 MapState
            if self._debug:
                print(f"    [DEBUG] RETRIEVE: 首次创建 MapState")
            context.map_state = MapState(
                initial_nodes=[],
                explored_paths=[],
                top_paths=[],
                visited_entities=set(),
                avg_score=0.0,
                leaf_nodes_by_gap=validated_ids_by_gap
            )

        if self._debug:
            print(f"    [DEBUG] RETRIEVE: {len(valid_anchor_ids)} 个有效锚点, {len(validated_gap_to_anchors)} 个分组")
            for gap_id, nodes in validated_gap_to_anchors.items():
                print(f"      - Gap ID: {gap_id}")
                print(f"        锚点数量: {len(nodes)}")
                for node_data in nodes[:3]:  # 显示前3个节点
                    print(f"        * [{node_data['node_id']}] {node_data.get('text', '')}")

        return validated_gap_to_anchors

    async def _prioritize_anchors(
        self,
        question: str,
        candidates: List[str],
        info_gaps: List[InfoGap]
    ) -> List[str]:
        """对锚点按优先级排序

        基于与问题和信息缺口的相似度计算优先级。

        Args:
            question: 问题文本
            candidates: 候选锚点节点ID列表
            info_gaps: 信息缺口列表

        Returns:
            按优先级排序的锚点ID列表
        """
        if not candidates:
            return []

        # 如果没有信息缺口，直接基于问题相似度排序
        if not info_gaps:
            return await self._prioritize_by_question(question, candidates)

        # 【批量优化】收集所有需要 embed 的文本
        texts_to_embed = [question]
        anchor_texts = []
        
        for anchor_id in candidates:
            anchor_text = self.graph.nodes[anchor_id].get("text", "")
            anchor_texts.append(anchor_text)
            if anchor_text:
                texts_to_embed.append(anchor_text)
        
        # 添加 gap descriptions
        gap_texts = [gap.gap_description for gap in info_gaps]
        texts_to_embed.extend(gap_texts)
        
        # 【批量优化】一次性批量计算所有 embeddings
        all_embeddings = await self.embedding_client.embed(texts_to_embed)
        all_embeddings = [np.array(emb) for emb in all_embeddings.embeddings]
        
        # 解析结果
        question_embedding = all_embeddings[0]
        anchor_embeddings = {}
        emb_idx = 1
        for anchor_id, anchor_text in zip(candidates, anchor_texts):
            if anchor_text:
                anchor_embeddings[anchor_id] = all_embeddings[emb_idx]
                emb_idx += 1
        
        gap_embeddings = all_embeddings[emb_idx:]
        
        # 计算优先级
        priorities = []
        for anchor_id, anchor_text in zip(candidates, anchor_texts):
            if not anchor_text:
                priorities.append((anchor_id, 0.0))
                continue
            
            # 问题相似度
            question_sim = self._cosine_similarity(question_embedding, anchor_embeddings[anchor_id])
            
            # Gap 相似度（取最大值）
            gap_sim = 0.0
            if gap_embeddings:
                gap_sims = [self._cosine_similarity(anchor_embeddings[anchor_id], gap_emb) 
                           for gap_emb in gap_embeddings]
                gap_sim = max(gap_sims)
            
            # 综合优先级：问题相似度占 60%，缺口相似度占 40%
            priority = 0.6 * question_sim + 0.4 * gap_sim
            priorities.append((anchor_id, priority))

        # 按优先级降序排序
        priorities.sort(key=lambda x: x[1], reverse=True)
        return [anchor_id for anchor_id, _ in priorities]

    async def _prioritize_by_question(
        self,
        question: str,
        candidates: List[str]
    ) -> List[str]:
        """仅基于问题相似度排序（无信息缺口时的后备方案）【批量优化】"""
        # 【批量优化】收集所有需要 embed 的文本
        texts_to_embed = [question]
        anchor_texts = []
        
        for anchor_id in candidates:
            anchor_text = self.graph.nodes[anchor_id].get("text", "")
            anchor_texts.append(anchor_text)
            if anchor_text:
                texts_to_embed.append(anchor_text)
        
        # 【批量优化】一次性批量计算所有 embeddings
        all_embeddings = await self.embedding_client.embed(texts_to_embed)
        all_embeddings = [np.array(emb) for emb in all_embeddings.embeddings]
        
        # 解析结果
        question_embedding = all_embeddings[0]
        anchor_embeddings = {}
        emb_idx = 1
        for anchor_id, anchor_text in zip(candidates, anchor_texts):
            if anchor_text:
                anchor_embeddings[anchor_id] = all_embeddings[emb_idx]
                emb_idx += 1
        
        # 计算相似度
        similarities = []
        for anchor_id, anchor_text in zip(candidates, anchor_texts):
            if not anchor_text:
                similarities.append((anchor_id, 0.0))
                continue
            
            sim = self._cosine_similarity(question_embedding, anchor_embeddings[anchor_id])
            similarities.append((anchor_id, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return [anchor_id for anchor_id, _ in similarities]

    async def _state_map(self, context: AgentContext):
        """MAP: 独立构建意图子图 + 汇总去重

        新架构：
        1. 每个意图独立构建 DAG 子图（独立的 beam_width）
        2. 并行扩展每个子图（互不干扰）
        3. 汇总所有子图的路径，去重和归一化评分
        """
        if not context.map_state or not context.map_state.leaf_nodes_by_gap:
            if self._debug:
                print(f"    [DEBUG] MAP: map_state={context.map_state}, leaf_nodes_by_gap={context.map_state.leaf_nodes_by_gap if context.map_state else None}")
            return

        map_state = context.map_state

        if self._debug:
            print(f"    [DEBUG] MAP: leaf_nodes_by_gap has {len(map_state.leaf_nodes_by_gap)} groups")
            for gap_id, nodes in map_state.leaf_nodes_by_gap.items():
                print(f"      - {gap_id} : {len(nodes)} nodes")

        # 1. 构建 gap_id -> InfoGap 查找字典
        gap_lookup = {gap.gap_id: gap for gap in context.plan_result.info_gaps}

        # 2. 为每个意图构建独立的子图
        subgraphs = await self._build_intent_subgraphs(context, gap_lookup)

        if not subgraphs:
            if self._debug:
                print(f"    [WARN] MAP: 没有有效的子图")
            return

        if self._debug:
            print(f"    [DEBUG] MAP: 构建了 {len(subgraphs)} 个独立子图")
            for sg in subgraphs:
                print(f"      - [{sg.intent_label}] {len(sg.frontier_nodes)} 个起点, active_edges={sg.info_gap.active_edges}")

        # 3. 并行扩展每个子图（独立 beam search）
        with self.timing_logger.time("  Subgraph expansion (beam search)"):
            expand_tasks = []
            for subgraph in subgraphs:
                # 使用意图改写后的查询（如果有）
                target_query = subgraph.info_gap.rewritten_query or context.question
                task = self._expand_subgraph(
                    subgraph=subgraph,
                    question=target_query,
                    max_depth=self.map_max_iterations,
                    beam_width=self.map_beam_width
                )
                expand_tasks.append(task)

            await asyncio.gather(*expand_tasks)

        if self._debug:
            print(f"    [DEBUG] MAP: 子图扩展完成")
            for sg in subgraphs:
                print(f"      - [{sg.intent_label}] 深度={sg.max_depth}, 节点数={sg.graph.number_of_nodes()}")

        # 4. 汇总所有子图的路径：先去重，再返回 top-k
        # 步骤1：收集并去重所有路径
        deduplicated_paths = self._collect_and_deduplicate_paths(subgraphs)

        # 步骤2：按路径分数排序
        sorted_paths = sorted(
            deduplicated_paths,
            key=lambda p: p.normalized_score,
            reverse=True
        )

        # 步骤3：更新 map_state（直接使用 RankedPath，无需转换）
        map_state.explored_paths = sorted_paths
        map_state.top_paths = sorted_paths[:self.map_beam_width]
        map_state.avg_score = sum(rp.normalized_score for rp in sorted_paths) / len(sorted_paths) if sorted_paths else 0.0

        # 更新叶子节点（用于下一轮）
        map_state.leaf_nodes = [p.nodes[-1] for p in map_state.top_paths if p.nodes]

        # 6. 更新访问实体
        for subgraph in subgraphs:
            context.visited_entities.update(subgraph.visited_entities)
            map_state.visited_entities.update(subgraph.visited_entities)

    async def _state_update(self, context: AgentContext):
        """UPDATE: 路径级打分排序，统一评估意图满足情况，选择文档添加到上下文"""
        if not context.map_state or not context.map_state.explored_paths:
            return

        # 1. 使用路径选择器进行排序
        with self.timing_logger.time("  Path ranking"):
            ranked_paths = self.path_selector.select(
                context.map_state.explored_paths,
            )

        context.accumulated_paths.extend(ranked_paths)

        if self._debug:
            print(f"    [DEBUG] UPDATE: 排序后路径 {len(ranked_paths)} 条")

        # 2. 顺序遍历路径，提取top-k文档
        with self.timing_logger.time("  Document mapping"):
            top_docs = []
            seen_docs = set()
            for path in ranked_paths:
                for node_id in path.nodes:
                    if node_id in self.graph.nodes:
                        doc_id = self.graph.nodes[node_id].get("doc_id", "")
                        if doc_id and doc_id not in seen_docs:
                            seen_docs.add(doc_id)
                            # 获取标题
                            title = self._doc_metadata.get(doc_id, {}).get("title", doc_id)
                            # 重建内容
                            content = self._reconstruct_document(doc_id)
                            top_docs.append({
                                "doc_id": doc_id,
                                "title": title,
                                "content": content
                            })
                            if len(top_docs) >= self.top_k_docs:
                                break
                if len(top_docs) >= self.top_k_docs:
                    break
            
            context.map_state.top_docs = top_docs
            context.map_state.top_doc_ids = [d["doc_id"] for d in top_docs]

        if self._debug:
            print(f"    [DEBUG] UPDATE: 提取文档 {len(top_docs)} 个")

        # 3. 统一评估所有意图 + 选择需要添加的文档
        with self.timing_logger.time("  Unified evaluation and selection"):
            selected_docs, next_hints = await self._unified_evaluate_and_select(
                context, top_docs
            )

        if self._debug:
            print(f"    [DEBUG] UPDATE: 选择添加 {len(selected_docs)} 个文档到上下文")
            if next_hints:
                print(f"    [DEBUG] 未满足意图的下一步提示: {len(next_hints)} 个")

        # 4. 添加选中的文档到上下文记忆
        if selected_docs:
            context.add_documents(selected_docs)

        # 5. 更新叶子节点供下一轮使用
        context.prev_leaf_nodes = context.map_state.leaf_nodes.copy()

    def _reconstruct_document(self, doc_id: str) -> str:
        """从图节点重建文档内容"""
        # 优先使用 pre-built mapping
        prop_ids = []
        if self._node_mappings and "doc_to_propositions" in self._node_mappings:
            prop_ids = self._node_mappings["doc_to_propositions"].get(doc_id, [])
        
        if not prop_ids:
            # 回退到遍历图
            prop_ids = [node_id for node_id, data in self.graph.nodes(data=True)
                        if data.get("node_type") == PROPOSITION_NODE and data.get("doc_id") == doc_id]

        # 获取文本和句子索引
        props = []
        for pid in prop_ids:
            if pid in self.graph.nodes:
                data = self.graph.nodes[pid]
                text = data.get("text", "")
                sent_idx = data.get("sent_idx", -1)
                if text:
                    props.append((sent_idx, text))
        
        # 按句子索引排序并拼接
        props.sort(key=lambda x: x[0])
        return "\n".join(text for idx, text in props)

    async def _unified_evaluate_and_select(
        self,
        context: AgentContext,
        retrieved_docs: List[Dict[str, Any]]
    ) -> tuple[List[Dict[str, Any]], Dict[str, str]]:
        """统一评估所有信息缺口并选择需要添加的文档
        
        Args:
            context: Agent 上下文
            retrieved_docs: 本轮检索到的文档列表
            
        Returns:
            (selected_docs, next_hints): 选中的文档列表和未满足意图的下一步提示
        """
        if not context.plan_result or not context.plan_result.info_gaps:
            # 如果没有明确的信息缺口，返回所有文档
            return retrieved_docs[:self.top_k_docs], {}
        
        # 格式化信息缺口列表
        intent_list_text = ""
        for i, gap in enumerate(context.plan_result.info_gaps):
            status_text = ""
            if gap.status != GapStatus.PENDING:
                status_text = f" [{gap.status.value}]"
            intent_list_text += f"{i+1}. 【{gap.intent_label}】{status_text}\n"
            intent_list_text += f"   ID: {gap.gap_id}\n"
            intent_list_text += f"   描述: {gap.gap_description}\n"
            intent_list_text += f"   关联实体: {', '.join(gap.related_entities)}\n"
            if gap.attempt_count > 0:
                intent_list_text += f"   尝试次数: {gap.attempt_count}\n"
            intent_list_text += "\n"
        
        # 获取已有上下文文档
        existing_docs = context.get_context_documents()
        
        # 生成 prompt
        from .agent_prompts import get_unified_evaluate_prompt
        prompt = get_unified_evaluate_prompt(
            question=context.question,
            intent_list=intent_list_text.strip(),
            existing_docs=existing_docs,
            retrieved_docs=retrieved_docs,
        )
        
        try:
            messages = [{"role": "user", "content": prompt}]
            with self.timing_logger.time("  LLM: Unified evaluation"):
                response = await self.llm.generate(
                    messages=messages,
                    temperature=0.2,
                    max_tokens=1024
                )
            
            # 解析响应
            result = await self._parse_json_response(response, target_field="intent_evaluation")
            
            if self._debug:
                print(f"    [DEBUG] 统一评估结果:")
                for eval_item in result.get("intent_evaluation", []):
                    gap_id = eval_item.get("gap_id", "unknown")
                    status_str = eval_item.get("status", "continue")
                    print(f"      - {gap_id}: {status_str}")
            
            # 更新缺口状态
            next_hints = {}
            for eval_item in result.get("intent_evaluation", []):
                gap_id = eval_item.get("gap_id", "")
                status_str = eval_item.get("status", "continue")
                reason = eval_item.get("reason", "")
                next_hint = eval_item.get("next_hint", "")
                
                # 通过 gap_id 查找对应的 gap
                matched_gap = None
                for gap in context.plan_result.info_gaps:
                    if gap.gap_id == gap_id:
                        matched_gap = gap
                        break
                
                if not matched_gap:
                    if self._debug:
                        print(f"    [DEBUG] 警告: 未找到 gap_id={gap_id} 对应的缺口")
                    continue
                
                # 更新缺口状态
                existing_result = context.get_gap_result(matched_gap.gap_id)
                if existing_result:
                    attempt_count = existing_result.attempt_count + 1
                else:
                    attempt_count = 1
                
                # 根据 status_str 设置状态
                if status_str == "satisfied":
                    status = GapStatus.SATISFIED
                    matched_gap.mark_satisfied()
                elif status_str == "partially_satisfied":
                    status = GapStatus.PARTIALLY_SATISFIED
                    matched_gap.mark_partially_satisfied()
                elif status_str == "manually_closed":
                    status = GapStatus.MANUALLY_CLOSED
                    matched_gap.mark_manually_closed()
                elif attempt_count >= context.max_gap_attempts:
                    # 两次检索后仍然失败，自动转化为枯竭
                    status = GapStatus.EXHAUSTED
                    matched_gap.mark_exhausted()
                else:
                    # continue 状态，标记为 ACTIVE
                    status = GapStatus.ACTIVE
                    matched_gap.mark_active()
                
                # 更新到 gap_history
                gap_result = GapRetrievalResult(
                    gap_id=matched_gap.gap_id,
                    gap_description=matched_gap.gap_description,
                    status=status,
                    attempt_count=attempt_count,
                    retrieved_docs=[d["doc_id"] for d in retrieved_docs],
                    selected_evidence=[],  # 不再使用证据级别
                    is_satisfied=(status in {GapStatus.SATISFIED, GapStatus.PARTIALLY_SATISFIED, GapStatus.MANUALLY_CLOSED}),
                    failure_hints=[next_hint] if next_hint else []
                )
                context.update_gap_result(gap_result)
                
                # 记录下一步提示（仅 continue 状态时）
                if status_str in ["continue", "partially_satisfied"] and next_hint:
                    next_hints[matched_gap.gap_id] = next_hint
            
            # 选择需要添加的文档
            docs_to_add_ids = result.get("docs_to_add", [])
            selected_docs = []
            
            # 建立 doc_id -> doc 映射
            doc_map = {d["doc_id"]: d for d in retrieved_docs}
            
            for doc_id in docs_to_add_ids:
                if doc_id in doc_map:
                    selected_docs.append(doc_map[doc_id])
            
            if self._debug:
                add_reason = result.get("add_reason", "")
                print(f"    [DEBUG] 选择添加 {len(selected_docs)} 个文档: {docs_to_add_ids}")
                if add_reason:
                    print(f"    [DEBUG] 理由: {add_reason}")
            
            return selected_docs, next_hints
            
        except Exception as e:
            print(f"统一评估失败: {e}")
            # 失败时返回前几个文档
            return retrieved_docs[:min(3, len(retrieved_docs))], {}

    def _clean_short_answer(self, text: str) -> str:
        """清洗 short_answer（尽量满足 EM/F1 的比较习惯）"""
        if not text:
            return ""
        short = text.strip().strip('"').strip("'").strip()
        short = re.sub(r"\s+", " ", short)
        short = re.sub(r"[。．\.!！\?？;；,:，]+$", "", short).strip()
        return short

    async def _state_answer(self, context: AgentContext, is_forced: bool = False) -> tuple[str, str]:
        """ANSWER: 一次生成自然语言 answer + 可评测 short_answer"""
        # 直接使用文档格式（不限制数量和内容长度）
        recent_docs = context.get_context_documents()
        evidence_list = [
            f"【文档 {i+1}: {doc.get('title', f'文档{i+1}')}】\n{doc.get('content', '')}"
            for i, doc in enumerate(recent_docs)
        ]
        evidence = evidence_list
        
        history = context.trace_log.get_recent_context(3)

        if self._debug:
            print(f"    [DEBUG] ANSWER 输入: {len(recent_docs)} 个文档 ({len(evidence)} 条证据), is_forced={is_forced}")

        # 格式化证据
        evidence_text = self._format_evidence(evidence)

        # 构造 prompt
        if is_forced:
            prompt = get_forced_answer_prompt(
                question=context.question,
                evidence=evidence,
                history=history
            )
        else:
            prompt = f"""你是一个专业的问答助手。请根据以下证据回答问题。

**问题：**
{context.question}

**证据：**
{evidence_text}
"""

            # 添加探索历史
            if history:
                prompt = f"{prompt}\n\n**探索历史：**\n{history}"

            # 要求 JSON 格式输出（同时包含 short_answer）
            prompt = f"""{prompt}

**输出格式：**
请以 JSON 格式返回答案，格式如下（必须同时包含两个字段）：
```json
{{
  "answer": "自然语言回答（允许完整句子）",
  "short_answer": "用于 EM/F1 的简短答案"
}}
```

**short_answer 规则：**
- 只输出最终答案本身，不要解释
- 不要在末尾加句号/逗号等标点
- 使用英文回答

请确保返回有效的 JSON 格式。"""

        try:
            messages = [{"role": "user", "content": prompt}]
            with self.timing_logger.time("  LLM: Generate answer"):
                response = await self.llm.generate(
                    messages=messages,
                    temperature=0.2 if is_forced else 0.1,
                    max_tokens=1024
                )

            # 解析答案
            result = await self._parse_json_response(response, target_field=None)
            answer = result.get("answer", "无法确定")
            short_answer = self._clean_short_answer(result.get("short_answer", ""))

            if not short_answer:
                # 不再额外二次调用 LLM；用 answer 做弱清洗兜底
                short_answer = self._clean_short_answer(answer)

            if self._debug:
                print(f"    [DEBUG] 答案: {answer}")
                print(f"    [DEBUG] short_answer: {short_answer}")

            return answer, short_answer

        except Exception as e:
            print(f"ANSWER 阶段失败: {e}")
            fallback = self._fallback_answer(context)
            return fallback, self._clean_short_answer(fallback)

    async def _extract_short_answer(self, context: AgentContext, natural_answer: str) -> str:
        """提取适合 EM/F1 的简短答案（尽量为实体/短词/yes-no/数字）"""
        recent_docs = context.get_context_documents()
        evidence_list = [
            f"【{doc.get('title', '未知文档')}】\n{doc.get('content', '')}"
            for doc in recent_docs
        ]
        evidence_text = self._format_evidence(evidence_list)

        prompt = f"""As an advanced reading comprehension assistant, your task to analyze text passages and corresponding questions meticulously.

Question:
{context.question}

Evidence:
{evidence_text}

You MUST follow this format exactly:
Thought: <brief reasoning, 1-3 sentences>
Answer: <a concise final answer suitable for EM/F1>

Rules for Answer:
- Output ONLY the final answer text after 'Answer:'
- Keep it as short as possible: entity/name/number/date/yes/no.
- Do NOT add explanations, punctuation at the end, or extra words.
- If the question expects multiple items, separate with comma.
"""

        try:
            response = await self.llm.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=256,
            )
        except Exception as e:
            if self._debug:
                print(f"    [DEBUG] short_answer LLM 调用失败: {e}")
            response = ""

        # 解析 "Answer:" 行
        short = ""
        if response:
            m = re.search(r"(?is)\bAnswer\s*:\s*(.+?)\s*$", response.strip())
            if m:
                short = m.group(1).strip()
            else:
                # 容错：如果没有 Answer:，就取最后一行
                lines = [ln.strip() for ln in response.splitlines() if ln.strip()]
                short = lines[-1] if lines else ""

        # 进一步清洗（去掉引号/尾部标点/多余空白）
        short = short.strip().strip('"').strip("'").strip()
        short = re.sub(r"\s+", " ", short)
        short = re.sub(r"[。．\.!！\?？;；,:，]+$", "", short).strip()

        # 最后兜底：如果还是空，用自然语言答案做一个弱清洗版
        if not short:
            fallback = (natural_answer or "").strip()
            fallback = fallback.strip().strip('"').strip("'").strip()
            fallback = re.sub(r"\s+", " ", fallback)
            fallback = re.sub(r"[。．\.!！\?？;；,:，]+$", "", fallback).strip()
            short = fallback

        if self._debug:
            print(f"    [DEBUG] short_answer: {short}")

        return short

    def _format_evidence(self, evidence: List[str]) -> str:
        """格式化证据为文本"""
        if not evidence:
            return "（无证据）"

        # 限制证据数量
        max_evidence = 20
        if len(evidence) > max_evidence:
            evidence = evidence[:max_evidence]

        formatted = []
        for i, ev in enumerate(evidence, 1):
            formatted.append(f"{i}. {ev}")

        return "\n".join(formatted)

    # ========== 辅助方法 ==========

    def _load_vector_index(self) -> Optional[Any]:
        """
        加载向量索引（懒加载，只加载一次）

        【性能优化】跟踪向量索引加载状态，提供更好的错误反馈。
        """
        if self._proposition_index is not None:
            return self._proposition_index

        try:
            from pathlib import Path

            # 使用构造函数传入的 index_dir，而不是从配置读取
            index_dir = self._index_dir

            if not index_dir:
                if self._require_vector_index:
                    raise RuntimeError("配置要求使用向量索引，但未指定 index_dir")
                return None

            index_path = Path(index_dir) / "indices" / "proposition"
            # 检查索引文件是否存在（.bin 文件）
            index_file = index_path / f"{index_path.name}.bin"
            if not index_file.exists():
                if self._require_vector_index:
                    raise RuntimeError(f"配置要求使用向量索引，但索引路径不存在: {index_path}")
                return None

            from .vector_index import PersistentHNSWIndex

            # 获取向量维度（从元数据文件读取实际维度）
            meta_path = index_path / f"{index_path.name}.meta.json"
            if meta_path.exists():
                import json
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                    dim = meta.get('dim', 768)
            else:
                dim = 768  # 默认维度

            self._proposition_index = PersistentHNSWIndex(dim=dim)
            # load() 需要索引文件的路径（不含扩展名），不是目录路径
            self._proposition_index.load(str(index_path / index_path.name))
            self._vector_index_loaded = True
            return self._proposition_index

        except Exception as e:
            if self._require_vector_index:
                raise RuntimeError(f"强制向量索引加载失败: {e}") from e
            print(f"警告：向量索引加载失败 ({e})，将使用 O(n) 遍历方式（性能较差）")
            return None

    async def _retrieve_initial_anchors(
        self,
        context: AgentContext,
        use_rewritten_queries: bool = True
    ) -> List[str]:
        """
        双路检索：命题向量检索 + 实体检索

        Args:
            context: Agent 上下文
            use_rewritten_queries: 是否使用改写后的查询

        Returns:
            融合后的锚点节点 ID 列表
        """
        from ..config.retrieval_config import get_retrieval_config

        config = get_retrieval_config()

        # 1. 构建查询文本列表
        if use_rewritten_queries and context.plan_result.info_gaps:
            query_texts = [
                gap.rewritten_query
                for gap in context.plan_result.info_gaps
                if gap.rewritten_query
            ]
        else:
            query_texts = [context.question]

        # 2. 路径1：向量索引批量检索（命题）
        index = self._load_vector_index()
        if index is None:
            raise RuntimeError("向量索引不可用，无法执行检索")

        # 为每个查询文本独立检索，然后合并结果
        proposition_anchors = await self._batch_retrieve_from_vector_index(
            context=context,
            index=index,
            query_texts=query_texts,
            top_k=config.retrieval_proposition_top_k
        )

        # 3. 路径2：实体检索（如果启用且有相关实体）
        entity_anchors = []
        if config.retrieval_entity_enable:
            # 收集所有 info_gaps 中的 related_entities
            all_entities = []
            for gap in context.plan_result.info_gaps:
                all_entities.extend(gap.related_entities)

            # 去重
            unique_entities = list(set(all_entities))

            if unique_entities:
                # 使用合并的查询文本进行实体检索
                merged_query = " ".join(query_texts)
                query_embedding = await self._get_query_embedding(merged_query)

                entity_anchors = await self._retrieve_from_entities(
                    context=context,
                    entity_ids=unique_entities,
                    query_embedding=query_embedding,
                    top_k_per_entity=config.retrieval_entity_top_k,
                    max_entities=config.retrieval_max_entities
                )

        # 4. 融合去重
        fused_anchors = self._fuse_anchors(proposition_anchors, entity_anchors)

        if self._debug:
            print(f"    [DEBUG] 查询数: {len(query_texts)}, 命题检索: {len(proposition_anchors)}, 实体检索: {len(entity_anchors)}, 融合后: {len(fused_anchors)}")

        return fused_anchors

    async def _get_query_embedding(self, query_text: str) -> np.ndarray:
        """获取查询嵌入（带缓存）

        Args:
            query_text: 查询文本

        Returns:
            查询向量嵌入
        """
        if query_text in self._question_embedding_cache:
            return self._question_embedding_cache[query_text]
        else:
            embedding = np.array(await self.embedding_client.embed_single(query_text))
            self._question_embedding_cache[query_text] = embedding
            return embedding

    async def _batch_retrieve_from_vector_index(
        self,
        context: AgentContext,
        index,
        query_texts: List[str],
        top_k: int = 10
    ) -> List[str]:
        """批量向量索引检索：使用批量 embedding 和批量搜索优化性能

        Args:
            context: Agent 上下文
            index: 向量索引
            query_texts: 查询文本列表
            top_k: 每个查询返回 top-k 结果

        Returns:
            合并去重后的命题节点 ID 列表
        """
        if not query_texts:
            return []

        # 【性能优化】批量获取所有 embeddings（一次 API 调用）
        all_embeddings = await self.embedding_client.embed(query_texts)
        embedding_matrix = np.array(all_embeddings.embeddings)

        # 【性能优化】批量搜索（真正的批量查询，HNSW 原生支持）
        distances, indices, payloads = index.search(
            query_vectors=embedding_matrix,
            k=top_k
        )

        # 合并结果并去重
        seen = set()
        unique_anchors = []

        for i in range(len(query_texts)):
            for j in range(top_k):
                payload = payloads[i][j]
                node_id = payload.get('node_id')
                if node_id and node_id not in seen:
                    # 确保节点在图中存在
                    if node_id in self.graph.nodes:
                        seen.add(node_id)
                        unique_anchors.append(node_id)

        return unique_anchors

    async def _retrieve_for_single_gap(
        self,
        context: AgentContext,
        gap: 'InfoGap',
        index,
        top_k: int = 10
    ) -> List[Dict]:
        """为单个 InfoGap 执行完整检索流程

        Args:
            context: Agent 上下文
            gap: 单个信息缺口
            index: 向量索引
            top_k: 每个路径返回的锚点数量

        Returns:
            该 InfoGap 对应的锚点节点数据字典列表，每个字典包含 node_id, text, node_type 等属性
        """
        from .agent_data_structures import InfoGap

        # 1. 获取查询文本
        query = gap.rewritten_query if gap.rewritten_query else gap.gap_description

        # 2. 向量检索（命题）
        query_embedding = await self._get_query_embedding(query)
        proposition_anchors = await self._retrieve_from_vector_index(
            context=context,
            index=index,
            query_embedding=query_embedding,
            top_k=top_k
        )

        # 3. 实体检索（如果有相关实体，使用独立计时）
        entity_anchors = []
        if gap.related_entities:
            with self.timing_logger.time("    Entity retrieval"):
                entity_anchors = await self._retrieve_from_entities(
                    context=context,
                    entity_ids=gap.related_entities,
                    query_embedding=query_embedding,
                    top_k_per_entity=3,
                    max_entities=5
                )

        # 4. 融合去重
        fused_anchors = self._fuse_anchors(proposition_anchors, entity_anchors)

        # 5. 转换为节点数据字典列表
        node_data_list = []
        for node_id in fused_anchors:
            node_data = dict(self.graph.nodes[node_id])
            node_data['node_id'] = node_id  # 确保包含node_id
            node_data_list.append(node_data)

        return node_data_list

    async def _retrieve_anchors_by_gaps(
        self,
        context: AgentContext,
        use_rewritten_queries: bool = True
    ) -> Dict[str, List[Dict]]:
        """按 InfoGap 分组并行检索锚点

        Args:
            context: Agent 上下文
            use_rewritten_queries: 是否使用改写后的查询

        Returns:
            {gap_id: [锚点节点数据字典列表]} 的字典
            每个节点数据字典包含 node_id, text, node_type, doc_id 等属性
        """
        from ..config.retrieval_config import get_retrieval_config
        from .agent_data_structures import InfoGap

        config = get_retrieval_config()

        # 加载向量索引
        index = self._load_vector_index()
        if index is None:
            raise RuntimeError("向量索引不可用，无法执行检索")

        info_gaps = context.plan_result.info_gaps
        if not info_gaps:
            return {}

        # 为每个 InfoGap 创建检索任务
        tasks = [
            self._retrieve_for_single_gap(
                context=context,
                gap=gap,
                index=index,
                top_k=config.retrieval_proposition_top_k
            )
            for gap in info_gaps
        ]

        # 并行执行所有检索任务
        if self._debug:
            print(f"    [DEBUG] 并行检索 {len(tasks)} 个 InfoGap...")

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 构建返回结果（使用 gap_id 作为键）
        gap_to_anchors = {}
        for gap, result in zip(info_gaps, results):
            if isinstance(result, Exception):
                print(f"    [WARN] InfoGap '{gap.gap_id}' 检索失败: {result}")
                gap_to_anchors[gap.gap_id] = []
            else:
                gap_to_anchors[gap.gap_id] = result

        if self._debug:
            total_anchors = sum(len(anchors) for anchors in gap_to_anchors.values())
            # 显示节点文本信息便于调试
            debug_info = []
            for gap_id, anchors in gap_to_anchors.items():
                anchor_texts = [a.get('text', '')[:50] for a in anchors[:3]]  # 只显示前3个
                debug_info.append(f"  {gap_id}: {len(anchors)}个锚点")
                for text in anchor_texts:
                    if text:
                        debug_info.append(f"    - {text}...")
            print(f"    [DEBUG] 分组检索完成: {len(gap_to_anchors)} 个分组, 共 {total_anchors} 个锚点")
            if debug_info:
                print("\n".join(debug_info))

        return gap_to_anchors

    async def _retrieve_from_vector_index(
        self,
        context: AgentContext,
        index,
        query_embedding: np.ndarray,
        top_k: int = 10
    ) -> List[str]:
        """使用向量索引检索命题节点

        Args:
            context: Agent 上下文
            index: 向量索引
            query_embedding: 查询向量嵌入
            top_k: 返回 top-k 结果

        Returns:
            命题节点 ID 列表
        """
        # 使用 HNSW 索引搜索
        distances, indices, payloads = index.search(
            query_vectors=query_embedding.reshape(1, -1),
            k=top_k
        )

        # 提取 node_id
        node_ids = []
        for payload in payloads[0]:
            node_id = payload.get('node_id')
            if node_id and node_id in self.graph.nodes:
                node_ids.append(node_id)

        return node_ids

    async def _retrieve_from_entities(
        self,
        context: AgentContext,
        entity_ids: List[str],
        query_embedding: np.ndarray,
        top_k_per_entity: int = 5,
        max_entities: int = 10
    ) -> List[str]:
        """从关联实体检索命题节点（实体路径）

        逻辑：
        1. 将实体名称匹配到图中的实体节点ID
        2. 遍历每个实体节点，获取关联的命题节点
        3. 计算每个命题与查询的相似度
        4. 按相似度全局排序，返回 top-k

        Args:
            context: Agent 上下文
            entity_ids: 实体名称列表（来自 InfoGap.related_entities）
            query_embedding: 问题/查询的向量嵌入
            top_k_per_entity: 每个实体获取的关联命题数量
            max_entities: 最多处理的实体数量

        Returns:
            命题节点 ID 列表（按相似度排序）
        """
        if not entity_ids:
            return []

        # 【新增】将实体名称匹配到节点ID
        entity_node_ids = await self._match_entity_names_to_node_ids(entity_ids)

        if not entity_node_ids:
            if self._debug:
                print(f"    [DEBUG] 实体匹配失败: {entity_ids}")
            return []

        if self._debug:
            print(f"    [DEBUG] 实体名称→节点ID: {len(entity_ids)} → {len(entity_node_ids)}")

        # 限制实体数量
        entity_node_ids = entity_node_ids[:max_entities]

        # 【性能优化】先收集所有关联命题（避免在循环中重复处理）
        all_prop_ids = set()
        for entity_node_id in entity_node_ids:
            # 检查实体是否存在（现在 entity_node_id 是真正的节点ID）
            if entity_node_id not in self.graph:
                continue

            # 获取实体节点类型并验证
            entity_data = self.graph.nodes[entity_node_id]
            if entity_data.get("node_type") not in [ENTITY_NODE, GLOBAL_ENTITY_NODE]:
                continue

            # 获取指向该实体的命题节点（MENTIONS_ENTITY 边: proposition -> entity）
            # 使用 predecessors 缓存查找指向该实体的命题节点
            proposition_neighbors = [
                n for n in self._predecessors_cache.get(entity_node_id, [])
                if self.graph.nodes[n].get("node_type") == PROPOSITION_NODE
            ]
            
            # 限制每个实体的命题数量，避免处理过多文本
            max_props_per_entity = 50  # 限制每个实体最多处理50个命题
            all_prop_ids.update(proposition_neighbors[:max_props_per_entity])

        if not all_prop_ids:
            return []

        # 【性能优化】批量收集所有需要 embedding 的命题文本（一次性处理）
        prop_texts_to_embed = []
        prop_ids_to_embed = []
        for prop_id in all_prop_ids:
            if prop_id not in self.graph:
                continue

            prop_text = self.graph.nodes[prop_id].get("text", "")
            if not prop_text:
                continue

            prop_texts_to_embed.append(prop_text)
            prop_ids_to_embed.append(prop_id)

        # 【性能优化】批量获取所有 embeddings（一次 API 调用，而不是每个实体一次）
        all_propositions = {}  # {prop_id: similarity}
        if prop_texts_to_embed:
            # 使用缓存管理器如果可用
            if hasattr(self, 'embedding_cache_manager') and self.embedding_cache_manager:
                prop_embeddings = await self.embedding_cache_manager.get_embeddings_batch(prop_texts_to_embed)
                prop_embeddings = [np.array(emb) for emb in prop_embeddings]
            else:
                response = await self.embedding_client.embed(prop_texts_to_embed)
                prop_embeddings = [np.array(emb) for emb in response.embeddings]

            # 计算每个命题与查询的相似度
            for prop_id, prop_embedding in zip(prop_ids_to_embed, prop_embeddings):
                similarity = self._cosine_similarity(query_embedding, np.array(prop_embedding))
                all_propositions[prop_id] = similarity

        # 按相似度排序
        sorted_props = sorted(all_propositions.items(), key=lambda x: x[1], reverse=True)

        # 返回 top-k
        total_limit = min(len(entity_node_ids) * top_k_per_entity, len(sorted_props))
        return [prop_id for prop_id, _ in sorted_props[:total_limit]]

    def _fuse_anchors(
        self,
        proposition_anchors: List[str],
        entity_anchors: List[str]
    ) -> List[str]:
        """融合两路检索结果并去重（命题优先）

        Args:
            proposition_anchors: 命题检索结果
            entity_anchors: 实体检索结果

        Returns:
            融合后的锚点列表（去重）
        """
        seen = set()
        fused = []

        # 先添加命题检索结果（优先）
        for anchor in proposition_anchors:
            if anchor not in seen:
                seen.add(anchor)
                fused.append(anchor)

        # 再添加实体检索结果
        for anchor in entity_anchors:
            if anchor not in seen:
                seen.add(anchor)
                fused.append(anchor)

        return fused

    async def _match_entity_names_to_node_ids(
        self,
        entity_names: List[str]
    ) -> List[str]:
        """将实体名称匹配到图中的实体节点ID（批量向量检索）

        Args:
            entity_names: 实体名称列表（来自 LLM/InfoGap.related_entities）

        Returns:
            匹配到的实体节点ID列表
        """
        if not entity_names or not self._entity_name_index:
            return []
        
        if not self._entity_name_index.is_built():
            return []
        
        # 批量向量检索
        batch_results = await self._entity_name_index.search_batch(
            queries=entity_names,
            top_k=1,
            threshold=0.7
        )
        
        # 展平结果（每个查询取第一个匹配）
        matched_node_ids = []
        for results in batch_results:
            if results:
                matched_node_ids.append(results[0])
        
        return matched_node_ids


    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """计算余弦相似度"""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))

    # ========== 分组 Beam Search 辅助方法 ==========

    def _load_or_build_adjacency_cache(self):
        """
        加载或构建邻接表缓存

        优先从持久化文件加载，如果不存在则构建
        """
        import pickle
        from pathlib import Path

        # 尝试从文件加载
        if self.persistence_dir:
            cache_file = Path(self.persistence_dir) / "adjacency_cache.pkl"
            pred_cache_file = Path(self.persistence_dir) / "predecessors_cache.pkl"

            if cache_file.exists() and pred_cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        self._adjacency_cache = pickle.load(f)
                    with open(pred_cache_file, 'rb') as f:
                        self._predecessors_cache = pickle.load(f)
                    print(f"✓ 邻接缓存已从文件加载: {len(self._adjacency_cache)} 个节点")
                    print(f"✓ 前驱缓存已从文件加载: {len(self._predecessors_cache)} 个节点")
                    return
                except Exception as e:
                    print(f"警告: 缓存加载失败 ({e})，将重新构建")

        # 回退到构建
        self._build_adjacency_cache()

    def _build_adjacency_cache(self):
        """
        构建邻接表缓存（包含前驱和后继）

        【性能优化】预构建每个节点的邻居映射，避免重复图遍历
        缓存结构：{node_id: {node_type: [(neighbor_id, edge_type), ...]}}
        """
        self._adjacency_cache.clear()
        self._predecessors_cache: Dict[str, List[str]] = {}

        for node_id in self.graph.nodes():
            # 按节点类型分组邻居
            neighbors_by_type: Dict[str, List[Dict[str, Any]]] = {
                PROPOSITION_NODE: [],
                ENTITY_NODE: [],
                GLOBAL_ENTITY_NODE: []
            }

            for neighbor in self.graph.neighbors(node_id):
                if self.graph.has_edge(node_id, neighbor):
                    edge_data = self.graph[node_id][neighbor]
                    edge_type = edge_data.get("edge_type", "")
                    neighbor_data = self.graph.nodes[neighbor]
                    neighbor_type = neighbor_data.get("node_type", "")

                    # 【修复】规范化边类型（处理遗留的 RST 边类型）
                    from ..proposition_graph.rst_analyzer import normalize_edge_type
                    if edge_type != MENTIONS_ENTITY and neighbor_type == PROPOSITION_NODE:
                        # 只规范化命题之间的边类型
                        normalized_edge_type = normalize_edge_type(edge_type)
                    else:
                        normalized_edge_type = edge_type

                    # 存储邻居信息（使用规范化后的边类型）
                    neighbor_info = {
                        "id": neighbor,
                        "edge_type": normalized_edge_type,
                        "original_edge_type": edge_type  # 保留原始类型以备需要
                    }

                    if neighbor_type in neighbors_by_type:
                        neighbors_by_type[neighbor_type].append(neighbor_info)

            self._adjacency_cache[node_id] = neighbors_by_type
            
            # 【性能优化】构建前驱节点缓存
            predecessors = list(self.graph.predecessors(node_id))
            self._predecessors_cache[node_id] = predecessors

    def _get_candidate_nodes(
        self,
        current_node: str,
        active_edges: Set[str],
        visited_entities: Set[str],
    ) -> List[tuple]:
        """
        获取候选节点（通过指定的 active_edges）

        【性能优化】使用邻接表缓存，O(1)查找

        同时支持：
        1. 通过 active_edges 直接连接的命题节点
        2. 通过 MENTIONS_ENTITY 边连接的实体节点（作为跳转到下游命题的桥梁）

        Args:
            current_node: 当前节点ID
            active_edges: 激活的边类型集合
            visited_entities: 已访问的实体集合

        Returns:
            List of (node_id, node_type) tuples，仅包含命题节点
        """
        from ..proposition_graph.rst_analyzer import normalize_edge_type
        
        candidates = []
        seen_nodes = set()
        neighbors_by_type = self._adjacency_cache[current_node]
        
        # 规范化 active_edges 以匹配规范化后的边类型
        normalized_active_edges = {normalize_edge_type(e) if e != MENTIONS_ENTITY else e for e in active_edges}

        # 1. 命题节点：区分不同边类型的处理
        # SIMILARITY 边始终可用，SKELETON 和 DETAIL 边需要检查 active_edges
        from ..proposition_graph.rst_analyzer import EDGE_TYPE_SIMILARITY
        
        for neighbor_info in neighbors_by_type.get(PROPOSITION_NODE, []):
            edge_type = neighbor_info["edge_type"]
            
            # SIMILARITY 边始终可用，不受 active_edges 限制
            if edge_type == EDGE_TYPE_SIMILARITY:
                node_id = neighbor_info["id"]
                if node_id not in seen_nodes:
                    candidates.append((node_id, PROPOSITION_NODE))
                    seen_nodes.add(node_id)
            # SKELETON 和 DETAIL 边需要检查 active_edges
            elif edge_type in normalized_active_edges:
                node_id = neighbor_info["id"]
                if node_id not in seen_nodes:
                    candidates.append((node_id, PROPOSITION_NODE))
                    seen_nodes.add(node_id)

        # 2. 实体节点跳转逻辑：
        #    跨过实体寻找其他引用该实体的命题（反向边连接的命题）
        #    实现 Proposition -> Entity -> Proposition 的一步跨越
        for node_type in [ENTITY_NODE, GLOBAL_ENTITY_NODE]:
            for neighbor_info in neighbors_by_type.get(node_type, []):
                if neighbor_info["edge_type"] == MENTIONS_ENTITY:
                    entity_id = neighbor_info["id"]
                    
                    # 寻找所有引用该实体的其他命题
                    for pred in self._predecessors_cache.get(entity_id, []):
                        if pred == current_node or pred in seen_nodes:
                            continue
                        pred_data = self.graph.nodes[pred]
                        if pred_data.get("node_type") == PROPOSITION_NODE:
                            if self.graph[pred][entity_id].get("edge_type") == MENTIONS_ENTITY:
                                candidates.append((pred, PROPOSITION_NODE))
                                seen_nodes.add(pred)
        
        # 3. 处理当前节点本身是实体的情况（例如锚点是实体）
        current_node_data = self.graph.nodes[current_node]
        if current_node_data.get("node_type") in [ENTITY_NODE, GLOBAL_ENTITY_NODE]:
            for pred in self._predecessors_cache.get(current_node, []):
                if pred in seen_nodes:
                    continue
                pred_data = self.graph.nodes[pred]
                if pred_data.get("node_type") == PROPOSITION_NODE:
                    if self.graph[pred][current_node].get("edge_type") == MENTIONS_ENTITY:
                        candidates.append((pred, PROPOSITION_NODE))
                        seen_nodes.add(pred)

        return candidates

    def _get_new_entities(self, node_id: str, visited_entities: Set[str]) -> Set[str]:
        """获取节点引入的新实体

        Args:
            node_id: 节点ID
            visited_entities: 已访问的实体集合

        Returns:
            新发现的实体ID集合
        """
        new_entities = set()

        # 【性能优化】使用邻接表缓存避免图遍历
        neighbors_by_type = self._adjacency_cache.get(node_id, {})
        entity_neighbors = neighbors_by_type.get(ENTITY_NODE, []) + neighbors_by_type.get(GLOBAL_ENTITY_NODE, [])

        for neighbor_info in entity_neighbors:
            if neighbor_info.get("edge_type") == MENTIONS_ENTITY:
                neighbor_id = neighbor_info["id"]
                if neighbor_id not in visited_entities:
                    new_entities.add(neighbor_id)

        return new_entities

    def _should_terminate_early(self, paths: List[Path]) -> bool:
        """判断是否应该提前终止搜索

        Args:
            paths: 当前路径列表

        Returns:
            是否应该终止
        """
        if not paths:
            return True

        # 如果还是初始状态（所有路径长度 < 2），继续探索
        if all(len(path.scores) < 2 for path in paths):
            return False

        # 检查是否所有路径都没有新实体
        for path in paths:
            # 检查最后一步是否引入了新实体
            if len(path.scores) >= 2:
                recent_bridge = self.path_scorer._compute_bridge_score(
                    path.nodes[-1],
                    path.visited_entities - {path.nodes[-1]}
                )
                if recent_bridge > 0:
                    return False

        return True

    async def _create_initial_paths(
        self,
        context: AgentContext,
        anchors: List[str],
        gap: 'InfoGap'
    ) -> List[Path]:
        """为锚点创建初始路径

        Args:
            context: Agent 上下文
            anchors: 锚点节点ID列表
            gap: 关联的 InfoGap

        Returns:
            初始路径列表
        """
        paths = []

        # 获取初始分数
        anchor_scores = await self.path_scorer.score_nodes(
            question=context.question,
            candidate_nodes=anchors,
            visited_entities=set(),
        )

        for score in anchor_scores:
            path = Path(graph=self.graph)
            new_entities = self._get_new_entities(score.node_id, set())
            path.add_node(score.node_id, score.total_score, new_entities, graph=self.graph)
            # 设置路径所属的分组信息
            path.intent_label = gap.intent_label
            path.info_gap = gap
            paths.append(path)

        return paths

    async def _expand_group_paths(
        self,
        paths: List[Path],
        active_edges: Set[str],
        question: str
    ) -> List[Path]:
        """扩展单个分组的路径（使用该分组的 active_edges）

        支持通过实体节点跳转到下游命题，使用拼接文本评分。

        Args:
            paths: 当前路径列表
            active_edges: 该分组激活的边类型
            question: 问题

        Returns:
            扩展后的新路径列表
        """
        if not paths:
            return []

        new_paths = []

        for path in paths:
            if not path.nodes:
                continue

            current_node = path.nodes[-1]

            # 获取候选节点 (返回 (node_id, node_type) tuples)
            candidates = self._get_candidate_nodes(
                current_node=current_node,
                active_edges=active_edges,
                visited_entities=path.visited_entities
            )

            if not candidates:
                continue

            # 评分命题节点（直接评分）
            prop_node_ids = [nid for nid, _ in candidates]
            all_scores = await self.path_scorer.score_nodes(
                question=question,
                candidate_nodes=prop_node_ids,
                visited_entities=path.visited_entities,
            ) if prop_node_ids else []

            # 扩展路径（添加的都是命题节点ID）
            for score in all_scores:
                # 【防止循环】跳过已经在路径中的节点
                if score.node_id in path.nodes:
                    if self._debug:
                        print(f"    [DEBUG] 跳过路径中的重复节点: {score.node_id}")
                    continue

                new_path = path.copy()
                new_entities = self._get_new_entities(score.node_id, path.visited_entities)
                new_path.add_node(score.node_id, score.total_score, new_entities, graph=self.graph)
                # 保持路径的分组信息
                new_path.intent_label = path.intent_label
                new_path.info_gap = path.info_gap
                new_paths.append(new_path)

        return new_paths

    def _global_prune(self, paths: List[Path], beam_width: int) -> List[Path]:
        """全局剪枝：按分数排序，保留 top-k

        Args:
            paths: 路径列表
            beam_width: 束宽度

        Returns:
            剪枝后的路径列表
        """
        paths.sort(key=lambda p: p.accumulated_score, reverse=True)
        return paths[:beam_width]

    def _path_belongs_to_group(self, path: Path, gap: 'InfoGap') -> bool:
        """判断路径是否属于某个分组

        Args:
            path: 路径
            gap: InfoGap

        Returns:
            是否属于该分组
        """
        return path.info_gap == gap if hasattr(path, 'info_gap') else False

    async def _grouped_beam_search(
        self,
        groups: List[Dict],
        question: str,
        max_depth: int,
        beam_width: int
    ) -> List[Path]:
        """分组 Beam Search：并行扩展 + 全局剪枝

        Args:
            groups: 分组列表，每项包含 {gap, active_edges, paths}
            question: 问题
            max_depth: 最大深度
            beam_width: 束宽度

        Returns:
            最终 top-k 路径
        """
        # 收集所有初始路径
        all_paths = []
        for group in groups:
            all_paths.extend(group['paths'])

        if not all_paths:
            return []

        # 分数收敛检测初始化
        prev_best_score = 0.0
        plateau_counter = 0

        # 迭代搜索
        for depth in range(max_depth):
            if self._debug:
                print(f"  深度 {depth + 1}/{max_depth}, 当前路径数: {len(all_paths)}")

            # 检查提前终止（无新实体）
            if self._should_terminate_early(all_paths):
                if self._debug:
                    print(f"  提前终止：没有新实体可探索")
                break

            # 【关键】并行扩展每个分组
            expand_tasks = [
                self._expand_group_paths(
                    group['paths'],
                    group['active_edges'],
                    question
                )
                for group in groups
                if group['paths']  # 只处理有路径的分组
            ]

            if not expand_tasks:
                break

            # 并行执行扩展
            group_results = await asyncio.gather(*expand_tasks, return_exceptions=True)

            # 收集所有新路径
            new_paths = []
            for result in group_results:
                if isinstance(result, Exception):
                    if self._debug:
                        print(f"    [ERROR] 分组扩展失败: {result}")
                    continue
                new_paths.extend(result)

            if not new_paths:
                if self._debug:
                    print(f"  无法继续扩展，终止搜索")
                break

            if self._debug:
                print(f"  扩展后路径数: {len(new_paths)}")

            # 【关键】全局剪枝：所有分组的路径一起排序
            all_paths = self._global_prune(new_paths, beam_width)

            if self._debug:
                print(f"  剪枝后路径数: {len(all_paths)}")

            # 【新增】分数收敛检测（在剪枝后立即检查）
            if depth >= 2:  # 至少2轮后再检测
                current_best = all_paths[0].accumulated_score if all_paths else 0

                # 计算边际收益
                marginal_gain = current_best - prev_best_score

                if marginal_gain < self.map_score_plateau_threshold:
                    plateau_counter += 1
                    if self._debug:
                        print(f"  深度 {depth + 1}: 边际收益={marginal_gain:.4f} < {self.map_score_plateau_threshold}, "
                              f"plateau_counter={plateau_counter}/{self.map_score_plateau_window}")
                    if plateau_counter >= self.map_score_plateau_window:
                        if self._debug:
                            print(f"  提前终止：分数收敛（连续{plateau_counter}轮边际收益 < {self.map_score_plateau_threshold}）")
                        break
                else:
                    plateau_counter = 0  # 有显著改进，重置计数器

                prev_best_score = current_best
            else:
                # 前两轮初始化
                if all_paths:
                    prev_best_score = all_paths[0].accumulated_score

            # 更新每个分组的路径（用于下一轮扩展）
            # 将路径按所属分组重新分配
            for group in groups:
                group['paths'] = [
                    p for p in all_paths
                    if self._path_belongs_to_group(p, group['gap'])
                ]

            if not all_paths:
                break

        # 最终全局排序
        all_paths.sort(key=lambda p: p.accumulated_score, reverse=True)
        return all_paths[:beam_width]

    async def _validate_anchors(self, context: AgentContext, candidates: List[str]) -> List[str]:
        """使用 LLM 验证锚点（优化：支持批量处理）"""
        if not candidates:
            return []

        # 如果候选数量较少，使用批处理
        if len(candidates) <= 15:
            return await self._batch_validate_anchors(context, candidates)

        # 否则分批处理
        return await self._batch_validate_anchors_large(context, candidates)

    async def _batch_validate_anchors(
        self,
        context: AgentContext,
        candidates: List[str]
    ) -> List[str]:
        """批量验证锚点（单次 LLM 调用）"""
        # 准备候选信息
        candidate_info = []
        for node_id in candidates:
            node_data = self.graph.nodes[node_id]
            text = node_data.get("text", "")
            candidate_info.append({
                "node_id": node_id,
                "text": text
            })

        if self._debug:
            print(f"    [DEBUG] 候选锚点: {[c['node_id'] for c in candidate_info]}")

        prompt = get_anchor_judge_prompt(
            question=context.question,
            info_gaps=[
                {
                    "gap_description": gap.gap_description,
                    "related_entities": gap.related_entities
                }
                for gap in context.plan_result.info_gaps
            ],
            candidates=candidate_info
        )

        try:
            messages = [{"role": "user", "content": prompt}]
            response = await self.llm.generate(
                messages=messages,
                temperature=0.1,
                max_tokens=256
            )

            result = await self._parse_json_response(response, target_field="valid_anchors")
            valid_ids = result.get("valid_anchors", [])

            if self._debug:
                print(f"    [DEBUG] LLM 返回有效锚点: {valid_ids}")

            # 过滤有效节点（确保在候选列表中）
            filtered = [aid for aid in valid_ids if aid in candidates]

            if self._debug:
                print(f"    [DEBUG] 过滤后有效锚点: {filtered}")

            return filtered

        except Exception as e:
            print(f"锚点验证失败: {e}")
            # 返回所有候选
            return candidates

    async def _batch_validate_anchors_large(
        self,
        context: AgentContext,
        candidates: List[str],
        batch_size: int = 10
    ) -> List[str]:
        """批量验证锚点（并发多个 LLM 调用）

        对于大量候选，分批并发验证以提高吞吐量
        """
        # 分批处理
        batches = [candidates[i:i+batch_size] for i in range(0, len(candidates), batch_size)]

        # 并发调用 LLM
        tasks = [self._batch_validate_anchors(context, batch) for batch in batches]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 合并结果
        valid_anchors = []
        for result in results:
            if isinstance(result, Exception):
                print(f"批处理验证失败: {result}")
                continue
            valid_anchors.extend(result)

        # 去重
        return list(set(valid_anchors))

    def _fallback_answer(self, context: AgentContext) -> str:
        """答案生成的后备方案"""
        recent_docs = context.get_context_documents()
        if not recent_docs:
            return "抱歉，根据现有信息无法回答该问题。"
        
        doc_summaries = [
            f"{doc.get('title', '未知文档')}: {doc.get('content', '')[:200]}..."
            for doc in recent_docs[:3]  # 后备方案只取前3个的摘要
        ]
        return f"根据相关文档，{' '.join(doc_summaries)}"

    async def _parse_json_response(self, response: str, target_field: Optional[str] = None) -> Dict[str, Any]:
        """解析 JSON 响应（使用 LLMResponseParser）

        Args:
            response: LLM 响应文本
            target_field: 期望包含的字段名（如 "can_answer", "info_gaps"）

        Returns:
            解析后的字典，失败返回空字典
        """
        if self._debug:
            print(f"    [DEBUG] 原始响应:\n{response}")

        def parse_dict(json_str: str) -> Dict[str, Any]:
            """解析 JSON 字典"""
            result = json.loads(json_str)
            # 验证目标字段存在
            if target_field and target_field not in result:
                raise ValueError(f"缺少必需字段: {target_field}")
            return result

        # 使用批量解析器（带重试）
        results = await self.response_parser.parse_batch(
            responses=[response],
            parse_funcs=[parse_dict],
            enable_retry=True
        )

        result, error = results[0]
        if error:
            if self._debug:
                print(f"    [DEBUG] 解析失败（包含重试）: {error}")
            return {}

        if self._debug:
            print(f"    [DEBUG] 解析结果: {result}")
        return result

    # ========== 新 MAP 阶段方法：独立意图子图构建 ==========

    async def _initialize_subgraph(
        self,
        subgraph: IntentSubgraph,
        start_nodes: List[str],
        context: AgentContext
    ) -> None:
        """初始化子图：添加起点节点并设置 frontier

        Args:
            subgraph: 意图子图
            start_nodes: 起点节点列表
            context: Agent 上下文
        """
        # 获取起点分数
        if start_nodes:
            # 使用意图改写后的查询（如果有）
            target_query = subgraph.info_gap.rewritten_query or context.question
            scores = await self.path_scorer.score_nodes(
                question=target_query,
                candidate_nodes=start_nodes,
                visited_entities=set(),
            )

            for score in scores:
                # 添加节点到子图
                subgraph.graph.add_node(
                    score.node_id,
                    text=self.graph.nodes[score.node_id].get("text", ""),
                    score=score.total_score,
                    step=0
                )
                # 设置为 frontier
                subgraph.frontier_nodes.add(score.node_id)
                # 【性能优化】跟踪起点节点
                subgraph.start_nodes.add(score.node_id)

                # 更新访问实体
                new_entities = self._get_new_entities(score.node_id, set())
                subgraph.visited_entities.update(new_entities)

    async def _build_intent_subgraphs(
        self,
        context: AgentContext,
        gap_lookup: Dict[str, 'InfoGap']
    ) -> List[IntentSubgraph]:
        """为每个意图构建独立的子图

        Args:
            context: Agent 上下文
            gap_lookup: gap_id -> InfoGap 查找字典

        Returns:
            意图子图列表
        """
        subgraphs = []

        for gap_id, anchors in context.map_state.leaf_nodes_by_gap.items():
            gap = gap_lookup.get(gap_id)
            if not gap:
                continue

            # 合并历史叶子和新增锚点
            start_nodes = list(set(context.prev_leaf_nodes + anchors))

            # 创建子图
            subgraph = IntentSubgraph(
                intent_label=gap.intent_label,
                info_gap=gap,
                graph=nx.DiGraph(),
                frontier_nodes=set(),
                visited_entities=set()
            )

            # 初始化子图
            await self._initialize_subgraph(subgraph, start_nodes, context)

            if subgraph.frontier_nodes:  # 只有有起点的子图才保留
                subgraphs.append(subgraph)

        return subgraphs

    async def _expand_subgraph(
        self,
        subgraph: IntentSubgraph,
        question: str,
        max_depth: int,
        beam_width: int
    ) -> None:
        """
        扩展单个子图（独立 beam search）

        【性能优化】深度级并行扩展：
        1. 预先收集所有候选节点
        2. 批量评分所有节点
        3. 使用 asyncio.gather() 并行构建路径扩展

        每个子图独立进行扩展，使用自己的 beam_width，
        不会受到其他子图的干扰。

        Args:
            subgraph: 意图子图
            question: 问题
            max_depth: 最大深度
            beam_width: 束宽度
        """
        for depth in range(max_depth):
            if not subgraph.frontier_nodes:
                break

            # 【性能优化】1. 预先收集所有候选节点和待评分节点
            all_prop_candidates = []  # 所有命题候选

            for node_id in list(subgraph.frontier_nodes):
                candidates = self._get_candidate_nodes(
                    current_node=node_id,
                    active_edges=set(subgraph.info_gap.active_edges),
                    visited_entities=subgraph.visited_entities
                )

                if self._debug and depth == 0:
                    print(f"    [DEBUG] 扩展节点 {node_id}: {len(candidates)} 个候选")

                # 收集命题候选
                all_prop_candidates.extend([(node_id, nid, nt) for nid, nt in candidates])

            # 【性能优化】2. 批量评分所有命题节点
            all_expansions = []  # (from_node, to_node, score)

            if all_prop_candidates:
                # 提取节点ID
                prop_node_ids = [nid for _, nid, _ in all_prop_candidates]

                # 批量评分
                scores = await self.path_scorer.score_nodes(
                    question=question,
                    candidate_nodes=prop_node_ids,
                    visited_entities=subgraph.visited_entities,
                )

                # 构建扩展候选（保持 from_node 信息）
                score_dict = {s.node_id: s for s in scores}
                for from_node, to_node, _ in all_prop_candidates:
                    if to_node in score_dict:
                        all_expansions.append((from_node, to_node, score_dict[to_node].total_score))

            if self._debug and depth == 0:
                print(f"    [DEBUG] 深度 {depth}: {len(all_expansions)} 个扩展候选")

            # 4. 添加到子图（按分数排序后剪枝）
            all_expansions.sort(key=lambda x: x[2], reverse=True)

            new_frontier = set()
            for from_node, to_node, score in all_expansions[:beam_width]:
                # 【防止循环】跳过已经在子图中的节点
                # 注意：这里跳过所有已存在节点，确保子图保持树形结构（每个节点只有一个父节点）
                # 如果需要 DAG（允许多父节点），需要额外检查 nx.has_path(subgraph.graph, to_node, from_node)
                if to_node in subgraph.graph:
                    if self._debug:
                        print(f"    [DEBUG] 跳过子图中的重复节点: {to_node}")
                    continue

                # 添加边
                subgraph.graph.add_edge(from_node, to_node, score=score, step=depth + 1)
                # 确保节点存在
                if to_node not in subgraph.graph:
                    subgraph.graph.add_node(
                        to_node,
                        text=self.graph.nodes[to_node].get("text", ""),
                        score=score,
                        step=depth + 1
                    )
                new_frontier.add(to_node)

                # 更新访问实体
                new_entities = self._get_new_entities(to_node, subgraph.visited_entities)
                subgraph.visited_entities.update(new_entities)

            # 5. 更新 frontier
            subgraph.frontier_nodes = new_frontier
            subgraph.max_depth = depth + 1

            if self._debug:
                print(f"    [DEBUG] 深度 {depth+1}: frontier_nodes={len(new_frontier)}, 总节点数={subgraph.graph.number_of_nodes()}")

            if not new_frontier:
                if self._debug:
                    print(f"    [DEBUG] 深度 {depth+1}: new_frontier 为空，终止扩展")
                break

    def _collect_all_paths(
        self,
        graph: nx.DiGraph,
        start: str
    ) -> List[List[str]]:
        """前向遍历 DAG，收集所有从 start 开始的路径

        Args:
            graph: DAG 子图
            start: 起始节点

        Returns:
            所有路径的节点序列列表
        """
        all_paths = []

        def dfs(current: str, path: List[str]):
            successors = list(graph.successors(current))
            if not successors:
                # 叶子节点，保存路径
                all_paths.append(path.copy())
                return

            # 继续扩展
            valid_successors = []
            for next_node in successors:
                # 循环检测：next_node 不能在当前路径中（防止环）
                if next_node in path:
                    continue
                valid_successors.append(next_node)
                path.append(next_node)
                dfs(next_node, path)
                path.pop()

            # 如果所有后继都被跳过（都在 path 中），当前路径也是有效终点
            if not valid_successors and successors:
                all_paths.append(path.copy())

        dfs(start, [start])

        # 去重前缀路径（只保留最长路径）
        return self._deduplicate_prefix_paths(all_paths)

    def _deduplicate_prefix_paths(
        self,
        paths: List[List[str]]
    ) -> List[List[str]]:
        """去重前缀路径，只保留最长路径

        例如：如果存在路径 [A, B, C] 和 [A, B]，则只保留 [A, B, C]

        Args:
            paths: 路径列表

        Returns:
            去重后的路径列表
        """
        if not paths or len(paths) < 2:
            return paths

        # 【性能优化】按长度降序排序，使用集合累积避免O(n²)比较
        paths_sorted = sorted(paths, key=len, reverse=True)
        result = []
        seen_nodes = set()

        for path in paths_sorted:
            path_set = frozenset(path)
            # 只检查当前路径是否是已见节点的子集
            if not path_set.issubset(seen_nodes):
                result.append(path)
                seen_nodes.update(path_set)

        return result

    def _aggregate_subgraph_paths(
        self,
        subgraphs: List[IntentSubgraph],
        top_k: int
    ) -> List[RankedPath]:
        """汇总所有子图的路径，去重和归一化

        Args:
            subgraphs: 意图子图列表
            top_k: 返回的 top-k 路径数量

        Returns:
            排序后的去重路径列表
        """
        aggregated = AggregatedPaths()

        for subgraph in subgraphs:
            # 【性能优化】使用缓存的起点节点，避免重新计算入度
            start_nodes = list(subgraph.start_nodes)

            # 前向遍历所有路径
            for start in start_nodes:
                paths = self._collect_all_paths(subgraph.graph, start)

                for path_nodes in paths:
                    # 计算路径分数（边分数之和）
                    path_score = 0.0
                    for i in range(len(path_nodes) - 1):
                        edge_data = subgraph.graph[path_nodes[i]][path_nodes[i + 1]]
                        path_score += edge_data.get('score', 0.0)

                    aggregated.add_path(
                        path_nodes,
                        path_score,
                        subgraph.intent_label,
                        subgraph.info_gap.gap_description
                    )

        # 按归一化分数排序，返回 top-k
        return aggregated.get_top_k(top_k)

    def _collect_and_deduplicate_paths(
        self,
        subgraphs: List[IntentSubgraph]
    ) -> List[RankedPath]:
        """收集所有子图的路径并进行去重

        Args:
            subgraphs: 意图子图列表

        Returns:
            去重后的路径列表（未排序）
        """
        aggregated = AggregatedPaths()

        for subgraph in subgraphs:
            # 【性能优化】使用缓存的起点节点，避免重新计算入度
            start_nodes = list(subgraph.start_nodes)

            # 前向遍历所有路径
            for start in start_nodes:
                paths = self._collect_all_paths(subgraph.graph, start)

                for path_nodes in paths:
                    # 计算路径分数（边分数之和）
                    path_score = 0.0
                    for i in range(len(path_nodes) - 1):
                        edge_data = subgraph.graph[path_nodes[i]][path_nodes[i + 1]]
                        path_score += edge_data.get('score', 0.0)

                    aggregated.add_path(
                        path_nodes,
                        path_score,
                        subgraph.intent_label,
                        subgraph.info_gap.gap_description
                    )

        # 返回所有去重后的路径
        return list(aggregated.unique_paths.values())

    def _collect_documents_from_paths(
        self,
        paths: List[RankedPath],
        k: int
    ) -> List[str]:
        """从路径收集文档ID（按路径分数排序，去重）
        
        逻辑：
        1. 按路径分数从高到低遍历所有路径
        2. 从每条路径的命题节点中提取 doc_id
        3. 使用集合去重，返回前 k 个唯一文档ID
        
        Args:
            paths: 去重后的路径列表
            k: 返回的文档数量上限
        
        Returns:
            按优先级排序的文档ID列表
        """
        # 1. 按路径分数排序（从高到低）
        sorted_paths = sorted(paths, key=lambda p: p.normalized_score, reverse=True)
        
        # 2. 收集文档
        collected_docs: List[str] = []  # 保持顺序
        seen_docs: Set[str] = set()     # 用于去重
        
        for path in sorted_paths:
            if len(collected_docs) >= k:
                break
            
            for node_id in path.nodes:
                if len(collected_docs) >= k:
                    break
                
                # 从图中获取节点数据
                if node_id not in self.graph.nodes:
                    continue
                
                node_data = self.graph.nodes[node_id]
                if node_data.get("node_type") == PROPOSITION_NODE:
                    doc_id = node_data.get("doc_id")
                    if doc_id and doc_id not in seen_docs:
                        seen_docs.add(doc_id)
                        collected_docs.append(doc_id)
        
        return collected_docs

    def _record_round_actions(self, context: AgentContext):
        """记录每轮的动作历史到 RoundTrace

        记录意图分配、探索路径、top-1 分数等信息。

        Args:
            context: Agent 上下文
        """
        if not context.trace_log.rounds:
            return

        latest_round = context.trace_log.rounds[-1]

        # 记录意图分配
        if context.plan_result:
            for gap in context.plan_result.info_gaps:
                for entity in gap.related_entities:
                    latest_round.intent_allocation[entity] = gap.intent_label

        # 记录探索路径
        if context.map_state and context.map_state.explored_paths:
            for path in context.map_state.explored_paths:
                # 使用 RankedPath 格式的归一化分数
                score = path.normalized_score
                # 从 metadata 获取 visited_entities
                visited_entities = path.metadata.get('visited_entities', [])

                latest_round.exploration_paths.append({
                    "nodes": path.nodes,
                    "score": score,
                    "visited_entities": visited_entities
                })

            # 记录 top-1 分数（使用归一化分数）
            if context.map_state.top_paths:
                top_path = context.map_state.top_paths[0]
                latest_round.top_1_score = top_path.normalized_score
        
        # 记录文档数量
        latest_round.document_count = len(context.context_documents)
