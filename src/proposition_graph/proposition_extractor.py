"""
命题数据类
"""

from dataclasses import dataclass


@dataclass
class Proposition:
    """命题数据类"""
    text: str
    sent_idx: int  # 来源句子索引
    prop_idx: int  # 命题索引
    doc_id: str  # 来源文档 ID

    def __repr__(self):
        return f"Proposition({self.text[:50]}...)"
