"""工具模块类型定义"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ParseErrorRecord:
    """JSON解析错误记录

    用于记录LLM响应解析失败的详细信息，便于后续分析和调试。
    """
    stage: str                                 # 发生错误的阶段（如 "EXTRACT", "LINK"）
    module: str                                # 发生错误的模块名
    original_response: str                     # LLM原始响应
    error_message: str                         # 错误信息
    retry_response: Optional[str] = None       # 纠错后的响应（如果有）
    retry_error: Optional[str] = None          # 纠错后的错误（如果有）
