"""
Agent 批量执行器

实现问题间并发执行，同时共享 LLM/Embedding 资源。
"""

import asyncio
from typing import List, Optional, Callable
import logging

from .agent_state_machine import AgentStateMachine
from .agent_data_structures import AgentResult


logger = logging.getLogger(__name__)


class AgentBatchExecutor:
    """Agent 批量执行器

    实现问题间并发执行，同时共享 LLM/Embedding 资源。
    """

    def __init__(
        self,
        agent_factory: Callable[[], AgentStateMachine],
        max_concurrency: int = 10
    ):
        """初始化批量执行器

        Args:
            agent_factory: Agent 工厂函数
            max_concurrency: 最大并发数
        """
        self.agent_factory = agent_factory
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.max_concurrency = max_concurrency

    async def run_batch(
        self,
        questions: List[str],
        initial_anchors_list: Optional[List[List[str]]] = None,
        show_progress: bool = True
    ) -> List[AgentResult]:
        """批量执行 Agent

        Args:
            questions: 问题列表
            initial_anchors_list: 初始锚点列表（与 questions 一一对应）
            show_progress: 是否显示进度

        Returns:
            Agent 结果列表
        """
        if initial_anchors_list is None:
            initial_anchors_list = [None] * len(questions)

        if len(questions) != len(initial_anchors_list):
            raise ValueError("questions 和 initial_anchors_list 长度不一致")

        total = len(questions)
        if show_progress:
            logger.info(f"开始批量处理 {total} 个问题，并发度: {self.max_concurrency}")

        # 创建任务
        tasks = [
            self._run_with_semaphore(question, anchors, i, total, show_progress)
            for i, (question, anchors) in enumerate(zip(questions, initial_anchors_list))
        ]

        # 并发执行
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理异常结果
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"问题 {i} 处理失败: {result}")
                # 创建一个失败结果
                processed_results.append(AgentResult(
                    answer=f"处理失败: {str(result)}",
                    confidence=0.0,
                    trace_log=None,
                    collected_evidence=[],
                    final_paths=[],
                    termination_reason="error"
                ))
            else:
                processed_results.append(result)

        if show_progress:
            logger.info(f"批量处理完成: 成功 {sum(1 for r in processed_results if r.confidence > 0)} / {total}")

        return processed_results

    async def _run_with_semaphore(
        self,
        question: str,
        anchors: Optional[List[str]],
        index: int,
        total: int,
        show_progress: bool
    ) -> AgentResult:
        """使用信号量控制并发执行"""
        async with self.semaphore:
            if show_progress:
                logger.info(f"[{index + 1}/{total}] 开始处理: {question}")

            try:
                # 创建新的 Agent 实例
                agent = self.agent_factory()

                # 运行
                result = await agent.run(question, anchors)

                if show_progress:
                    logger.info(f"[{index + 1}/{total}] 完成: 置信度={result.confidence:.2f}")

                return result

            except Exception as e:
                logger.error(f"[{index + 1}/{total}] 处理失败: {e}")
                raise

    async def run_single(
        self,
        question: str,
        initial_anchors: Optional[List[str]] = None
    ) -> AgentResult:
        """运行单个问题

        Args:
            question: 问题
            initial_anchors: 初始锚点

        Returns:
            Agent 结果
        """
        agent = self.agent_factory()
        return await agent.run(question, initial_anchors)
