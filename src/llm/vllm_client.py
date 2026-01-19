"""vLLM离线加载客户端"""

from typing import List, Dict, Any, Optional
import logging
from vllm import LLM, SamplingParams
from .base import BaseLLM
from transformers import AutoTokenizer



logger = logging.getLogger(__name__)
class VLLMClient(BaseLLM):
    """vLLM离线加载客户端，用于离线图构建"""

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        trust_remote_code: bool = True,
        max_model_len: Optional[int] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.9,
        **kwargs
    ):
        """
        初始化vLLM客户端

        Args:
            model_path: 模型路径
            tensor_parallel_size: 张量并行大小
            gpu_memory_utilization: GPU内存利用率
            trust_remote_code: 是否信任远程代码
            max_model_len: 模型最大长度（可选）
            temperature: 温度参数
            max_tokens: 最大token数
            top_p: top-p采样
            **kwargs: 其他vLLM参数
        """
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.trust_remote_code = trust_remote_code
        self.max_model_len = max_model_len
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

        # 构建LLM初始化参数
        llm_kwargs = {
            "model": model_path,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": trust_remote_code,
        }

        # 如果指定了 max_model_len，添加到参数中
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len

        # 合并其他参数
        llm_kwargs.update(kwargs)

        logger.warning(f"正在加载vLLM模型: {model_path}")
        self.llm = LLM(**llm_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        logger.info("模型加载完成")

    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None,
        top_p: float = None,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """
        生成单个响应（异步包装）

        Args:
            messages: 输入消息列表，每条消息包含role和content字段
            temperature: 温度参数
            max_tokens: 最大token数
            top_p: top-p采样
            stop: 停止词列表
            **kwargs: 其他参数

        Returns:
            生成的文本响应
        """
        # 使用实例默认值
        if temperature is None:
            temperature = self.temperature
        if max_tokens is None:
            max_tokens = self.max_tokens
        if top_p is None:
            top_p = self.top_p

        return (await self.generate_batch(
            [messages], temperature, max_tokens, top_p, stop, **kwargs
        ))[0]

    async def generate_batch(
        self,
        messages: List[List[Dict[str, str]]],
        temperature: float = None,
        max_tokens: int = None,
        top_p: float = None,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> List[str]:
        """
        批量生成响应
        vLLM内部已经实现了高效的批量推理，直接使用其同步批量方法
        虽然函数定义为async，但内部直接同步调用vLLM的批量推理，无需asyncio包装

        Args:
            messages: 输入消息列表列表
            temperature: 温度参数
            max_tokens: 最大token数
            top_p: top-p采样
            stop: 停止词列表
            **kwargs: 其他参数

        Returns:
            生成的文本响应列表
        """
        # 使用实例默认值
        if temperature is None:
            temperature = self.temperature
        if max_tokens is None:
            max_tokens = self.max_tokens
        if top_p is None:
            top_p = self.top_p

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop or [],
            **kwargs
        )
        prompts = [self._format_prompt(message) for message in messages]
        # vLLM的generate方法本身支持批量推理，直接同步调用
        outputs = self.llm.generate(prompts, sampling_params)

        # 提取生成的文本
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text
            results.append(generated_text)

        return results

    def _format_prompt(self, message: List[Dict[str, str]]) -> str:
        """
        格式化消息列表为提示词

        Args:
            message: 消息列表（包含多条消息，每条消息有role和content字段）

        Returns:
            格式化后的提示词字符串
        """
        prompt = self.tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True
        )
        return prompt
